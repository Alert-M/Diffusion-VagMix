import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.imageup_DVM import UP_Datasets

root_dir = '/root/autodl-tmp/root/autodl-tmp/2p/train'

# 实例化数据集
dataset = UP_Datasets(root=root_dir)