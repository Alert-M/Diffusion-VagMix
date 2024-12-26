import os
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = os.listdir(root_dir)  # 列出根目录下的所有子文件夹
        self.classes.sort()  # 确保类的顺序是一致的
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  # 类别名到索引的映射

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

