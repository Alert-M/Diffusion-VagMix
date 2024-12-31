import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset
from .sampler import ClassAwareSampler
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .sampler import ClassAwareSampler
from datasets.imageup1   import UP_Datasets

class LT_Dataset(Dataset):
    num_classes = 1000

    def __init__(self, root, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform

        for class_index in range(self.num_classes):
            class_folder = os.path.join(root, str(class_index))
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.img_path.append(img_path)
                    self.targets.append(class_index)

        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i

        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class LT_Dataset_Eval(Dataset):
    num_classes = 1000

    def __init__(self, root, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map

        for img_name in os.listdir(root):
            if "_" in img_name:
                img_path = os.path.join(root, img_name)
                class_index = int(img_name.split('_')[1].split('.')[0])
                self.img_path.append(img_path)
                self.targets.append(class_index)

        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class ImageNet_LT(object):
    def __init__(self, distributed, root="", batch_size=60, num_works=40):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = LT_Dataset(os.path.join(root, 'train'), transform=transform_train)
        eval_dataset = LT_Dataset_Eval(os.path.join(root, 'test'), class_map=train_dataset.class_map, transform=transform_test)

        self.cls_num_list = train_dataset.cls_num_list

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)

