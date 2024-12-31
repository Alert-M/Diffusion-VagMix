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

        # 使用 torchvision.models 中的 resnet18 模型，通过别名 tv_models 引用
        self.feature_extractor = tv_models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()  # 设置为评估模式

        # 预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 适配ResNet输入大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 筛选生成的图片
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

        # 匹配生成图片的后缀单词
        generated_suffixes = ["Autumn", "snowy", "sunset", "rainbow", "mosaic"]

        total_images = len(self.imgs)
        print(f"原始图片数量：{total_images}")

        # 计算特征相似度的阈值
        threshold = 0.3  # 设置相似度阈值
        img_account=0

        print(f"开始对比生成图片和原始图片。")

        for idx, (original_path, label) in enumerate(self.imgs):
            original_features = self._extract_features(original_path)

            # 提取原始图片文件名的基础部分和标签
            original_base_name = os.path.basename(original_path).split('_label_')[0]
            original_label = int(os.path.basename(original_path).split('_label_')[1].replace('.png', ''))

            # 显示当前处理的原始图片进度
            print(f"正在处理第 {idx + 1}/{total_images} 张原始图片：{original_path}")

            # 对比当前原始图片与所有生成图片
            found_similar_images = False
            for fname in os.listdir(self.root):
                if any(fname.endswith(f"{suffix}.png") for suffix in generated_suffixes):
                    # 提取生成图片的基础部分和标签
                    parts = os.path.basename(fname).split('_label_')
                    if len(parts) < 2:
                        continue  # 文件名格式不正确，跳过

                    generated_base_name = parts[0]
                    generated_label_part = parts[1].replace('.png', '')

                    # 提取标签部分，忽略生成图片的后缀
                    try:
                        generated_label = int(generated_label_part.split('_')[0])
                    except ValueError:
                        continue  # 标签部分格式不正确，跳过

                    # 确保生成图片的基础部分和标签都与原始图片匹配
                    if original_base_name == generated_base_name and original_label == generated_label:
                        generated_path = os.path.join(self.root, fname)
                        generated_features = self._extract_features(generated_path)
                        similarity = self._calculate_similarity(original_features, generated_features)

                        # 输出与原图像对比的相似度
                        print(f"    对比生成图片：{generated_path}，相似度：{similarity:.2f}")

                        # 如果相似度高于阈值，应用ACE处理
                        if similarity >= threshold:
                            print(f"    相似度高于阈值，应用ACE处理到生成图片：{generated_path}。")
                            img_account+=1
                            self._apply_ace_to_image(generated_path)
                            generated_features = self._extract_features(generated_path)  # 重新提取处理后的特征
                            similarity = self._calculate_similarity(original_features, generated_features)
                            print(f"    应用ACE后相似度：{similarity:.2f}")

                        # 记录生成图片（无论是否应用ACE）
                        filtered_imgs.append((generated_path, label))
                        print(f"    已筛选出一张生成图片：{generated_path}，相似度：{similarity:.2f}")
                        found_similar_images = True

            if not found_similar_images:
                print(f"    当前原始图片 {original_path} 没有找到匹配的生成图片或相似度未达标。")

        print(f"对比完成，筛选出 {len(filtered_imgs)} 张生成图片。")
        print(f"已处理 {img_account} 张生成图片。")
        self.imgs.extend(filtered_imgs)  # 加入筛选后的生成图片

    def _extract_features(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img).unsqueeze(0)  # 添加批次维度
        with torch.no_grad():  # 不计算梯度
            features = self.feature_extractor(img)
        return features.flatten()  # 展平特征向量

    def _apply_ace_to_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # 应用图像增强方法
        img = self._apply_augmentation(img)

        # 将处理后的图像保存为新文件（注意不影响磁盘中的数据集）
        img = Image.fromarray(img)
        img.save(img_path)  # 保存覆盖原图像

    def _apply_augmentation(self, img):
        # 随机应用图像增强方法
        img = self._cut_mix(img, blur=True, blur_level=5)  # Replace CutBlur with CutMix
        return img


    def _cut_mix(self, img, blur=False, blur_level=15):
        """Apply CutMix augmentation."""
        h, w, _ = img.shape

        # Randomly select CutMix region size
        cut_ratio = random.uniform(0.1, 0.5)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        # Randomly select CutMix region position
        x_start = random.randint(0, w - cut_w)
        y_start = random.randint(0, h - cut_h)

        # Create a copy of the original image
        img_cutmix = img.copy()

        # Select a random image from the dataset
        random_index = random.randint(0, len(self.imgs) - 1)
        other_img_path, other_label = self.imgs[random_index]
        other_img = Image.open(other_img_path).convert('RGB')
        other_img = np.array(other_img)  # 转换为 NumPy 数组

        # 调整另一张图像的大小以匹配 CutMix 区域
        other_img = cv2.resize(other_img, (cut_w, cut_h))

        # 如果启用了模糊，则对另一张图像进行模糊处理
        if blur:
            # 确保 blur_level 是奇数，因为 GaussianBlur 需要奇数核尺寸
            blur_level = blur_level if blur_level % 2 == 1 else blur_level + 1
            other_img = cv2.GaussianBlur(other_img, (blur_level, blur_level), 0)

        # 将另一张图像的裁剪区域粘贴到原始图像中
        img_cutmix[y_start:y_start + cut_h, x_start:x_start + cut_w] = other_img

        return img_cutmix

    def _calculate_similarity(self, features1, features2):
        cos_sim = F.cosine_similarity(features1, features2, dim=0)
        return cos_sim.item()  # 返回相似度值

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

        # 使用 torchvision.models 中的 resnet18 模型，通过别名 tv_models 引用
        self.feature_extractor = tv_models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()  # 设置为评估模式

        # 预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 适配ResNet输入大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 筛选生成的图片
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

        # 匹配生成图片的后缀单词
        generated_suffixes = ["autumn", "snowy", "sunset", "rainbow", "mosaic"]

        total_images = len(self.imgs)
        print(f"原始图片数量：{total_images}")

        # 计算特征相似度的阈值
        threshold = 0.3  # 设置相似度阈值
        img_account = 0

        print(f"开始对比生成图片和原始图片。")

        for idx, (original_path, label) in enumerate(self.imgs):
            original_features = self._extract_features(original_path)

            # 提取原始图片文件名的基础部分和标签
            original_base_name = os.path.basename(original_path).split('_label_')[0]
            original_label = int(os.path.basename(original_path).split('_label_')[1].replace('.png', ''))

            # 显示当前处理的原始图片进度
            print(f"正在处理第 {idx + 1}/{total_images} 张原始图片：{original_path}")

            # 对比当前原始图片与所有生成图片
            found_similar_images = False
            for fname in os.listdir(self.root):
                if any(fname.endswith(f"{suffix}.png") for suffix in generated_suffixes):
                    # 提取生成图片的基础部分和标签
                    parts = os.path.basename(fname).split('_label_')
                    if len(parts) < 2:
                        continue  # 文件名格式不正确，跳过

                    generated_base_name = parts[0]
                    generated_label_part = parts[1].replace('.png', '')

                    # 提取标签部分，忽略生成图片的后缀
                    try:
                        generated_label = int(generated_label_part.split('_')[0])
                    except ValueError:
                        continue  # 标签部分格式不正确，跳过

                    # 确保生成图片的基础部分和标签都与原始图片匹配
                    if original_base_name == generated_base_name and original_label == generated_label:
                        generated_path = os.path.join(self.root, fname)
                        generated_features = self._extract_features(generated_path)
                        similarity = self._calculate_similarity(original_features, generated_features)

                        # 输出与原图像对比的相似度
                        print(f"    对比生成图片：{generated_path}，相似度：{similarity:.2f}")

                        # 如果相似度高于阈值，应用ACE处理
                        if similarity >= threshold:
                            print(f"    相似度高于阈值，应用ACE处理到生成图片：{generated_path}。")
                            img_account += 1
                            self._apply_ace_to_image(generated_path)
                            generated_features = self._extract_features(generated_path)  # 重新提取处理后的特征
                            similarity = self._calculate_similarity(original_features, generated_features)
                            print(f"    应用ACE后相似度：{similarity:.2f}")

                        # 记录生成图片（无论是否应用ACE）
                        filtered_imgs.append((generated_path, label))
                        print(f"    已筛选出一张生成图片：{generated_path}，相似度：{similarity:.2f}")
                        found_similar_images = True

            if not found_similar_images:
                print(f"    当前原始图片 {original_path} 没有找到匹配的生成图片或相似度未达标。")

        print(f"对比完成，筛选出 {len(filtered_imgs)} 张生成图片。")
        print(f"已处理 {img_account} 张生成图片。")
        self.imgs.extend(filtered_imgs)  # 加入筛选后的生成图片

    def _extract_features(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img).unsqueeze(0)  # 添加批次维度
        with torch.no_grad():  # 不计算梯度
            features = self.feature_extractor(img)
        return features.flatten()  # 展平特征向量

    def _apply_ace_to_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # 应用图像增强方法
        img = self._apply_augmentation(img)

        # 将处理后的图像保存为新文件（注意不影响磁盘中的数据集）
        img = Image.fromarray(img)
        img.save(img_path)  # 保存覆盖原图像

    def _apply_augmentation(self, img):
        # 随机应用图像增强方法
        img = self._cut_mix(img, blur=True, blur_level=5)  # Replace CutBlur with CutMix
        return img

    def _cut_mix(self, img, blur=False, blur_level=15):
        """Apply CutMix augmentation."""
        h, w, _ = img.shape

        # Randomly select CutMix region size
        cut_ratio = random.uniform(0.1, 0.5)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        # Randomly select CutMix region position
        x_start = random.randint(0, w - cut_w)
        y_start = random.randint(0, h - cut_h)

        # Create a copy of the original image
        img_cutmix = img.copy()

        # Select a random image from the dataset
        random_index = random.randint(0, len(self.imgs) - 1)
        other_img_path, other_label = self.imgs[random_index]
        other_img = Image.open(other_img_path).convert('RGB')
        other_img = np.array(other_img)  # 转换为 NumPy 数组

        # 调整另一张图像的大小以匹配 CutMix 区域
        other_img = cv2.resize(other_img, (cut_w, cut_h))

        # 如果启用了模糊，则对另一张图像进行模糊处理
        if blur:
            # 确保 blur_level 是奇数，因为 GaussianBlur 需要奇数核尺寸
            blur_level = blur_level if blur_level % 2 == 1 else blur_level + 1
            other_img = cv2.GaussianBlur(other_img, (blur_level, blur_level), 0)

        # 将另一张图像的裁剪区域粘贴到原始图像中
        img_cutmix[y_start:y_start + cut_h, x_start:x_start + cut_w] = other_img

        return img_cutmix

    def _calculate_similarity(self, features1, features2):
        cos_sim = F.cosine_similarity(features1, features2, dim=0)
        return cos_sim.item()  # 返回相似度值

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