import os
import re
from PIL import Image
from torchvision.datasets import VisionDataset


class CustomCIFAR100(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.imgs = self._load_images(root)
        self.classes = list(set(label for _, label in self.imgs))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.cls_num_list = self.get_cls_num_list()

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
    
class CustomCIFAR10(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.imgs = self._load_images(root)
        self.classes = list(set(label for _, label in self.imgs))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.cls_num_list = self.get_cls_num_list()

    def _load_images(self, root):
        imgs = []
        for fname in os.listdir(root):
            if fname.endswith('.png'):
                # Adjust regex pattern as needed for CIFAR-10 naming convention
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