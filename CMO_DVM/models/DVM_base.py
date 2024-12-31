import os
import re
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset


class CustomCIFAR10(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, weighted_alpha=1.0):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.imgs = self._load_images(root)
        self.classes = list(set(label for _, label in self.imgs))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.cls_num_list = self.get_cls_num_list()
        self.targets = [label for _, label in self.imgs]  # 用于get_weighted_sampler中的targets
        self.weighted_alpha = weighted_alpha  # 用于控制权重计算的参数

    def _load_images(self, root):
        imgs = []
        for fname in os.listdir(root):
            if fname.endswith('.png'):
                match = re.match(r'.*_label_(\d+)(_.*)?\.png$', fname)
                if match:
                    label = int(match.group(1))
                    path = os.path.join(root, fname)
                    imgs.append((path, label))
        return imgs

    def get_cls_num_list(self):
        cls_num_list = [0] * len(self.classes)
        for _, label in self.imgs:
            cls_num_list[label] += 1
        return cls_num_list

    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        # 计算每个类别的权重，alpha控制平滑
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        # 为每个样本分配权重
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        # 创建WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler

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

class CustomCIFAR100(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, weighted_alpha=1.0):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.imgs = self._load_images(root)
        self.classes = list(set(label for _, label in self.imgs))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.cls_num_list = self.get_cls_num_list()
        self.targets = [label for _, label in self.imgs]  # 用于get_weighted_sampler中的targets
        self.weighted_alpha = weighted_alpha  # 用于控制权重计算的参数

    def _load_images(self, root):
        imgs = []
        for fname in os.listdir(root):
            if fname.endswith('.png'):
                match = re.match(r'.*_label_(\d+)(_.*)?\.png$', fname)
                if match:
                    label = int(match.group(1))
                    path = os.path.join(root, fname)
                    imgs.append((path, label))
        return imgs

    def get_cls_num_list(self):
        cls_num_list = [0] * len(self.classes)
        for _, label in self.imgs:
            cls_num_list[label] += 1
        return cls_num_list

    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        # 计算每个类别的权重，alpha控制平滑
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        # 为每个样本分配权重
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        # 创建WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler

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