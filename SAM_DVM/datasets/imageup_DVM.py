import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import transforms,models
import torch.nn as nn
import re
from PIL import Image, ImageFilter, ImageOps
import torchvision.models as tv_models
import torch.nn.functional as F
from skimage import exposure
import random


class UP_Datasets(Dataset):
    num_classes = 1000
    processed_count = 0  # 处理的生成图片数量计数
    generated_keywords = ["sunset", "Autumn"]  # 生成图片关键词

    def __init__(self, root, transform=None):
        self.root = root
        self.imgs = self._load_images()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])  # 移除最后的分类层
        self.resnet18.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._apply_ace_to_all_images()
        cls_num_list_old = [np.sum(np.array([label for _, label in self.imgs]) == i) for i in range(self.num_classes)]

        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i

        self.targets = np.array(self.class_map)[[label for _, label in self.imgs]].tolist()

        self.class_data = [[] for _ in range(self.num_classes)]
        for i, label in enumerate(self.targets):
            self.class_data[label].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

    def _load_images(self):
        imgs = []
        for class_index in range(self.num_classes):
            class_folder = os.path.join(self.root, str(class_index))
            if os.path.isdir(class_folder):
                print(f"The category folder is being processed: {class_folder}")
                for fname in os.listdir(class_folder):
                    if fname.lower().endswith(('.png', '.jpeg', '.jpg')):
                        # 处理所有图片
                        path = os.path.join(class_folder, fname)
                        imgs.append((path, class_index))
            else:
                print(f"The category folder does not exist: {class_folder}")
        return imgs

    def _apply_ace_to_all_images(self):
        print("Start applying DVM processing to the generated picture.")
        total_images = len(self.imgs)
        print(f"The total number of images：{total_images}")

        generated_keywords = ["sunset", "Autumn"]  #prompts
        processed_images = 0

        for idx, (img_path, label) in enumerate(self.imgs):

            if any(keyword in img_path for keyword in generated_keywords):
                print(f"{idx + 1}/{total_images} image is being processed：{img_path}")

                # 提取生成图片的原始文件名 (不使用路径，仅提取名称部分)
                file_name = os.path.basename(img_path)
                parts = file_name.split('_')
                original_img_name = '_'.join(parts[:3]) + ".JPEG"  # 通过命名规则构造原始图片文件名

                # 使用与生成图片相同的路径构造原始图片路径
                original_img_path = os.path.join(os.path.dirname(img_path), original_img_name)

                if os.path.exists(original_img_path):
                    # 读取生成图片和原始图片
                    gen_img = Image.open(img_path).convert('RGB')
                    ori_img = Image.open(original_img_path).convert('RGB')

                    # 计算生成图片和原始图片的特征值
                    gen_features = self._extract_features(gen_img)
                    ori_features = self._extract_features(ori_img)

                    # 计算特征值的余弦相似度
                    similarity = self._calculate_similarity(gen_features, ori_features)
                    print(f"Generate images: {file_name}, Original image: {original_img_name}, Cosine Similarity: {similarity:.4f}")

                    # 如果相似度低于阈值，应用ACE处理
                    if similarity > 0.8:
                        print(f"Similarity is higher than the threshold, DVM processing is applied: {file_name}")
                        self.upup(img_path)
                        # 重新提取处理后的生成图片特征
                        processed_gen_img = Image.open(img_path).convert('RGB')  # 重新读取处理后的图片
                        processed_gen_features = self._extract_features(processed_gen_img)

                        # 重新计算处理后的相似度
                        processed_similarity = self._calculate_similarity(processed_gen_features, ori_features)
                        print(f"Similarity after processing: {processed_similarity:.4f}")
                        processed_images += 1
                else:
                    print(f"Original image not found: {original_img_name} in path: {original_img_path}")

        print(f"The generated image processing is complete, and {processed_images} generated images have been processed.")

    def _extract_features(self, img):
        img = self.transform(img).unsqueeze(0)  # 添加批量维度
        with torch.no_grad():
            features = self.resnet18(img).squeeze()  # 提取特征并去除批量维度
        return features

    def _calculate_similarity(self, features1, features2):
        cos_sim = nn.functional.cosine_similarity(features1, features2, dim=0)
        return cos_sim.item()


    def upup(self, image_path):
        print(f"Apply DVM processing to {image_path}")

        # 打开图片并应用ACE和滤波处理
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        img = self._apply_augmentation(img)


        img = Image.fromarray(img)
        img.save(image_path)

    def _apply_augmentation(self, img):
        img = self._vagmix(img, blur=False, blur_level=0)
        return img

    def _vagmix(self, img, blur=False, blur_level=5):
        if isinstance(img, Image.Image):
            img = np.array(img)

        h, w, c = img.shape

        cut_ratio = random.uniform(0.1, 0.2)
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

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, label = self.imgs[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
