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

ImageNet_LT: Download at this link: [ImageNet_LT]([https://github.com](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)).

 
# 目录结构描述
    ├── ReadMe.md           // 帮助文档
    
    ├── AutoCreateDDS.py    // 合成DDS的 python脚本文件
    
    ├── DDScore             // DDS核心文件库，包含各版本的include、src、lib文件夹，方便合并
    
    │   ├── include_src     // 包含各版本的include、src文件夹
    
    │       ├── V1.0
    
    │           ├── include
    
    │           └── src
    
    │       └── V......
    
    │   └── lib             // 包含各版本的lib文件夹
    
    │       ├── arm64       // 支持arm64系统版本的lib文件夹
    
    │           ├── V1.0
    
    │           └── V......
    
    │       └── x86         // 支持x86系统版本的lib文件夹
    
    │           ├── V1.0
    
    │           └── V......
    
    ├── target              // 合成结果存放的文件夹
    
    └── temp                // 存放待合并的服务的服务文件夹
 
# 使用说明
 
 
 
# 版本内容更新
###### v1.0.0: 
    1.实现gen文件的拷贝、合并
    
    2.实现common文件的合并
    
    3.实现指定版本的include、src、lib文件的拷贝
 
 v
