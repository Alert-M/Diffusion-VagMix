import argparse
from torchvision import datasets
from augment.handler import ModelHandler
from augment.utils import Utils
from augment.diffuseMix import DiffuseMix
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate an augmented dataset from original images and fractal patterns.")
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to the directory containing the original training images.')
    parser.add_argument('--fractal_dir', type=str, required=True,
                        help='Path to the directory containing the fractal images.')
    parser.add_argument('--prompts', type=str, required=True,
                        help='Comma-separated list of prompts for image generation.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    prompts = args.prompts.split(',')  # This will give you a list of prompts

    # Initialize the model
    model_id = "timbrooks/instruct-pix2pix"
    model_initialization = ModelHandler(model_id=model_id, device='cuda')

    # Load the original dataset
    train_dataset = datasets.ImageFolder(root=args.train_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    class_to_idx = train_dataset.class_to_idx  # Used for mapping class names to indices

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(args.fractal_dir)

    # Create the augmented dataset
    augmented_train_dataset = DiffuseMix(
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=1,
        guidance_scale=4,
        idx_to_class=idx_to_class,
        prompts=prompts,
        model_handler=model_initialization
    )

    base_directory = '/root/autodl-tmp/diffuseMix-main/augmented_images'

    # Ensure base directory exists
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    print(f"Number of images in augmented_train_dataset: {len(augmented_train_dataset)}")

    # 遍历数据集并保存图像
    for idx, (image, label) in enumerate(augmented_train_dataset):
        try:
            # 获取源文件的路径和名称
            source_image_path, _ = augmented_train_dataset.original_dataset.samples[idx]
            source_image_name = os.path.basename(source_image_path)
            source_image_base, source_image_ext = os.path.splitext(source_image_name)
        except IndexError:
            print(f"IndexError: Could not access image at index {idx}.")
            continue

        # Ensure label is an integer index
        if isinstance(label, str):
            try:
                # Convert label from string to its corresponding index using class_to_idx
                label = class_to_idx[label]
            except KeyError:
                print(f"KeyError: Label '{label}' not found in class_to_idx.")
                continue

        # Ensure class_name is a string and valid directory name
        if label in idx_to_class:
            class_name = idx_to_class[label]
        else:
            print(f"KeyError: Label index '{label}' not found in idx_to_class.")
            continue

        # Create the class directory if it does not exist
        class_dir = os.path.join(base_directory, class_name)
        if not os.path.exists(class_dir):
            try:
                os.makedirs(class_dir)
                print(f"Created directory: {class_dir}")
            except OSError as e:
                print(f"OSError: Failed to create directory {class_dir}. {e}")
                continue

        # 为每个提示词生成唯一的文件名并保存
        for prompt in prompts:
            save_path = os.path.join(class_dir, f"{source_image_base}_{prompt}{source_image_ext}")
            try:
                image.save(save_path)
                print(f"Saved image to: {save_path}")
            except IOError as e:
                print(f"IOError: Failed to save image to {save_path}. {e}")
                continue


if __name__ == '__main__':
    main()
