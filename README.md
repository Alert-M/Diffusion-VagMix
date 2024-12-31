# Diffusion-VagMix
This is the reproduction code of the paper *Data Enhancement for Long-tailed Tasks: Diffusion Model with Optimized Quality Filter*.

# Requirements

    conda create -n DVM python=3.9
    conda activate DVM
    pip install -r requirements.txt

# Dataset
CIFAR_10_LT/CIFAR_100_LT: Use the following command to download the CIFAR dataset

    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

ImageNet_LT: Download at this link: [ImageNet_LT](https://github.com/zhmiao/OpenLongTailRecognition-OLTR).

 
# Directory structure of the dataset
First, after preparing the dataset, we need to extract the dataset into a file. Label and arrange the dataset as follows, ready to use the diffusion model for diffusion.
    
    │   ├── Cifar100     
    
    │       ├── origin
    
    │           ├── 10003_label_22.png

    │           ├── 10010_label_1.png
    
    │           └── ......
    
    │   └── Cifar10         
    
    │       ├── origin     
    
    │           ├── 01_label_0.png

    │           ├── 1093_label_2.png
    
    │           └── ......
    
    │   └── ImageNet         

    │       ├── train
    
    │           ├── 0     
    
    │               ├── n01440764_10027_0.JPEG
    
    │               └── ......
    
    │           ├── 1     
    
    │               ├── n01443537_10025_1.JPEG
    
    │               └── ......
    
    │           └── ......

    │       ├── test

    │           ├── 00000001_65.JPEG

    │           ├── 00000002_970.JPEG
    
    │           └── ......
 
# Training
 
 
 

