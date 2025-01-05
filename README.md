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
Secondly, the DVM is employed to expand the datasets.

    python DVM-Diffusion/main.py --prompts "-" --train_dir PATH --fractal_dir PATH
    
Each time you can select one from the following prompts.

    prompts = ["Autumn", "snowy","sunset", "rainbow", "aurora", "mosaic"]

Then we can start training.

##CIFAR
Since the dataset is already expanded through diffusion, the DVM module does not need to be employed during subsequent uses. It is sufficient to make minor adjustments to the dataset loading method. Consequently, the complete DVM module is integrated solely within the LDAM-DRW approach. Training can be conducted using this method initially, after which the enhanced dataset can be directly applied to other methods.

- LDAM-DRW-DVM
For the first training, use train_DVM.py, and then you can use train_base.py for training.

        python LDAM-DRW-DVM/train_DVM.py --gpu 0 --imb_type exp --loss_type CE --train_rule None --dataset cifar10/cifar100
        python LDAM-DRW-DVM/train_DVM.py --gpu 0 --imb_type exp --loss_type CE --train_rule DRW --dataset cifar10/cifar100
        python LDAM-DRW-DVM/train_DVM.py --gpu 0 --imb_type exp --loss_type LDAM --train_rule DRW --dataset cifar10/cifar100

- CMO_DVM

        python CMO_DVM/train_DVM_CIFAR10.py --dataset cifar10 --loss_type CE --train_rule DRW --epochs 200 --data_aug CMO --num_classes 10
        python CMO_DVM/train_DVM_CIFAR100.py --dataset cifar100 --loss_type CE --train_rule DRW --epochs 200 --data_aug CMO --num_classes 100

##ImageNet
    

 
 

