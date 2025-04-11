import os
import re
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as tv_models
import torch.nn.functional as F
from skimage import exposure
import random
import cv2

class CustomCIFAR100(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = self._load_images()
        self.classes = list(set([label for _, label in self.imgs]))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.feature_extractor = tv_models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()  

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._filter_generated_images()

    def _load_images(self):
        imgs = []
        for fname in os.listdir(self.root):
            if fname.endswith('.png') and not any(
                    suffix in fname for suffix in ["Autumn", "snowy", "sunset", "rainbow", "mosaic"]):
                match = re.match(r'.*_label_(\d+)(_.*)?\.png$', fname)
                if match:
                    label = int(match.group(1))
                    path = os.path.join(self.root, fname)
                    imgs.append((path, label))
        return imgs

    def _filter_generated_images(self):
        filtered_imgs = []

        generated_suffixes = ["Autumn", "snowy", "sunset", "rainbow", "mosaic"]

        total_images = len(self.imgs)
        print(f"Number of original images: {total_images}")    

        threshold=0.3
        img_account=0

        print(f"Start comparing the generated image with the original image.")

        for idx, (original_path, label) in enumerate(self.imgs):
            original_features = self._extract_features(original_path)

            original_base_name = os.path.basename(original_path).split('_label_')[0]
            original_label = int(os.path.basename(original_path).split('_label_')[1].replace('.png', ''))

            print(f"Processing the {idx + 1}/{total_images}th original image: {original_path}")

            found_similar_images = False
            for fname in os.listdir(self.root):
                if any(fname.endswith(f"{suffix}.png") for suffix in generated_suffixes):

                    parts = os.path.basename(fname).split('_label_')
                    if len(parts) < 2:
                        continue  

                    generated_base_name = parts[0]
                    generated_label_part = parts[1].replace('.png', '')

                    try:
                        generated_label = int(generated_label_part.split('_')[0])
                    except ValueError:
                        continue  

                    if original_base_name == generated_base_name and original_label == generated_label:
                        generated_path = os.path.join(self.root, fname)
                        generated_features = self._extract_features(generated_path)
                        similarity = self._calculate_similarity(original_features, generated_features)

                        print(f"Comparison generated image: {generated_path}, similarity: {similarity:.2f}")

                        if similarity >= threshold:
                            print(f" Similarity is higher than the threshold, apply ACE processing to the generated image: {generated_path}.")
                            img_account+=1
                            self._apply_ace_to_image(generated_path)
                            generated_features = self._extract_features(generated_path) 
                            similarity = self._calculate_similarity(original_features, generated_features)
                            print(f" Similarity after applying DVM: {similarity:.2f}")

                        filtered_imgs.append((generated_path, label))
                        print(f"A generated image has been selected: {generated_path}, similarity: {similarity:.2f}")
                        found_similar_images = True

            if not found_similar_images:
                print(f" The current original image {original_path} does not find a matching generated image or the similarity does not meet the standard.")

        print(f"Comparison completed, {len(filtered_imgs)} generated images were filtered out.")
        print(f"{img_account} generated images processed.")
        self.imgs.extend(filtered_imgs)  

    def _extract_features(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img).unsqueeze(0)  
        with torch.no_grad():  
            features = self.feature_extractor(img)
        return features.flatten()  

    def _apply_ace_to_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        img = self._apply_augmentation(img)

        img = Image.fromarray(img)
        img.save(img_path) 

    def _apply_augmentation(self, img):

        img = self._cut_mix(img, blur=True, blur_level=5) 
        return img


    def _cut_mix(self, img, blur=False, blur_level=15):
        """Apply CutMix augmentation."""
        h, w, _ = img.shape

        cut_ratio = random.uniform(0.1, 0.5)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        x_start = random.randint(0, w - cut_w)
        y_start = random.randint(0, h - cut_h)

        img_cutmix = img.copy()

        random_index = random.randint(0, len(self.imgs) - 1)
        other_img_path, other_label = self.imgs[random_index]
        other_img = Image.open(other_img_path).convert('RGB')
        other_img = np.array(other_img) 

        other_img = cv2.resize(other_img, (cut_w, cut_h))

        if blur:
            blur_level = blur_level if blur_level % 2 == 1 else blur_level + 1
            other_img = cv2.GaussianBlur(other_img, (blur_level, blur_level), 0)

        img_cutmix[y_start:y_start + cut_h, x_start:x_start + cut_w] = other_img

        return img_cutmix

    def _calculate_similarity(self, features1, features2):
        cos_sim = F.cosine_similarity(features1, features2, dim=0)
        return cos_sim.item() 

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.imgs)

    def get_cls_num_list(self):
        cls_num_list = [0] * len(self.classes)
        for _, label in self.imgs:
            cls_num_list[self.class_to_idx[label]] += 1
        return cls_num_list

    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        total_samples = sum(cls_num_list)
        class_weights = [total_samples / cls_num for cls_num in cls_num_list]
        sample_weights = [class_weights[self.class_to_idx[label]] for _, label in self.imgs]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return sampler


class CustomCIFAR10(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = self._load_images()
        self.classes = list(set([label for _, label in self.imgs]))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.feature_extractor = tv_models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval() 

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._filter_generated_images()

    def _load_images(self):
        imgs = []
        for fname in os.listdir(self.root):
            if fname.endswith('.png') and not any(
                    suffix in fname for suffix in ["autumn", "snowy", "sunset", "rainbow", "mosaic"]):
                match = re.match(r'.*_label_(\d+)(_.*)?\.png$', fname)
                if match:
                    label = int(match.group(1))
                    path = os.path.join(self.root, fname)
                    imgs.append((path, label))
        return imgs

    def _filter_generated_images(self):
        filtered_imgs = []

        generated_suffixes = ["autumn", "snowy", "sunset", "rainbow", "mosaic"]

        total_images = len(self.imgs)
        print(f"Number of original images: {total_images}")

        threshold = 0.3  
        img_account = 0

        print(f"Start comparing the generated image with the original image.")

        for idx, (original_path, label) in enumerate(self.imgs):
            original_features = self._extract_features(original_path)

            original_base_name = os.path.basename(original_path).split('_label_')[0]
            original_label = int(os.path.basename(original_path).split('_label_')[1].replace('.png', ''))

            print(f"Processing the {idx + 1}/{total_images}th original image: {original_path}")

            found_similar_images = False
            for fname in os.listdir(self.root):
                if any(fname.endswith(f"{suffix}.png") for suffix in generated_suffixes):
                    parts = os.path.basename(fname).split('_label_')
                    if len(parts) < 2:
                        continue 

                    generated_base_name = parts[0]
                    generated_label_part = parts[1].replace('.png', '')

                    try:
                        generated_label = int(generated_label_part.split('_')[0])
                    except ValueError:
                        continue 

                    if original_base_name == generated_base_name and original_label == generated_label:
                        generated_path = os.path.join(self.root, fname)
                        generated_features = self._extract_features(generated_path)
                        similarity = self._calculate_similarity(original_features, generated_features)

                        print(f"Comparison generated image: {generated_path}, similarity: {similarity:.2f}")

                        if similarity >= threshold:
                            print(f" Similarity is higher than the threshold, apply ACE processing to the generated image: {generated_path}.")
                            img_account += 1
                            self._apply_ace_to_image(generated_path)
                            generated_features = self._extract_features(generated_path)  
                            similarity = self._calculate_similarity(original_features, generated_features)
                            print(f" Similarity after applying DVM: {similarity:.2f}")

                        filtered_imgs.append((generated_path, label))
                        print(f"A generated image has been selected: {generated_path}, similarity: {similarity:.2f}")
                        found_similar_images = True

            if not found_similar_images:
                print(f" The current original image {original_path} does not find a matching generated image or the similarity does not meet the standard.")

        print(f"Comparison completed, {len(filtered_imgs)} generated images were filtered out.")
        print(f"{img_account} generated images processed.")
        self.imgs.extend(filtered_imgs) 

    def _extract_features(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img).unsqueeze(0)  
        with torch.no_grad():  
            features = self.feature_extractor(img)
        return features.flatten() 

    def _apply_ace_to_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        img = self._apply_augmentation(img)

        img = Image.fromarray(img)
        img.save(img_path)

    def _apply_augmentation(self, img):

        img = self._cut_mix(img, blur=True, blur_level=5)  
        return img

    def _cut_mix(self, img, blur=False, blur_level=15):
        """Apply CutMix augmentation."""
        h, w, _ = img.shape

        cut_ratio = random.uniform(0.1, 0.5)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        x_start = random.randint(0, w - cut_w)
        y_start = random.randint(0, h - cut_h)

        img_cutmix = img.copy()

        random_index = random.randint(0, len(self.imgs) - 1)
        other_img_path, other_label = self.imgs[random_index]
        other_img = Image.open(other_img_path).convert('RGB')
        other_img = np.array(other_img) 

        other_img = cv2.resize(other_img, (cut_w, cut_h))


        if blur:
 
            blur_level = blur_level if blur_level % 2 == 1 else blur_level + 1
            other_img = cv2.GaussianBlur(other_img, (blur_level, blur_level), 0)

        img_cutmix[y_start:y_start + cut_h, x_start:x_start + cut_w] = other_img

        return img_cutmix

    def _calculate_similarity(self, features1, features2):
        cos_sim = F.cosine_similarity(features1, features2, dim=0)
        return cos_sim.item()  

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.imgs)

    def get_cls_num_list(self):
        cls_num_list = [0] * len(self.classes)
        for _, label in self.imgs:
            cls_num_list[self.class_to_idx[label]] += 1
        return cls_num_list

    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        total_samples = sum(cls_num_list)
        class_weights = [total_samples / cls_num for cls_num in cls_num_list]
        sample_weights = [class_weights[self.class_to_idx[label]] for _, label in self.imgs]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return sampler
