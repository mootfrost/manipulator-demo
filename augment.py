import os
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

IMAGE_SIZE = (224, 224)  
AUGMENTED_IMAGES_PER_INPUT = 5 

INPUT_DIR = "datasets/Healthy"
OUTPUT_DIR = "datasets/Healthy_aug1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

resize_transform = transforms.Resize(IMAGE_SIZE)
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def add_gaussian_noise(image, mean=0, std=0.1):
    img_tensor = to_tensor(image)
    noise = torch.randn(img_tensor.size()) * std + mean
    noisy_img = img_tensor + noise
    noisy_img = torch.clamp(noisy_img, 0, 1)
    return to_pil(noisy_img)

def apply_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

def apply_affine_transform(image):
    affine_transform = transforms.RandomAffine(
        degrees=(-30, 30),
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=(-10, 10)
    )
    return affine_transform(image)

def apply_rotation(image):
    angle = random.randint(0, 360)
    return image.rotate(angle)

def apply_color_jitter(image):
    jitter = transforms.ColorJitter(
        brightness=random.uniform(0.8, 1.2),
        contrast=random.uniform(0.8, 1.2),
        saturation=random.uniform(0.8, 1.2),
        hue=random.uniform(0, 0.1)
    )
    return jitter(image)

def apply_horizontal_flip(image):
    if random.random() > 0.5:  # 50% chance to flip
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def augment_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = resize_transform(image)

    original_filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(original_filename)
    image.save(os.path.join(OUTPUT_DIR, f"{base_name}_original{ext}"))

    for i in range(AUGMENTED_IMAGES_PER_INPUT):
        aug_image = random.choice([
            add_gaussian_noise(image),
            apply_blur(image),
            apply_affine_transform(image),
            apply_rotation(image),
            apply_color_jitter(image),
            apply_horizontal_flip(image)
        ])
        aug_image.save(os.path.join(OUTPUT_DIR, f"{base_name}_aug_{i}{ext}"))

def process_directory(input_dir):
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Обрабатываем: {filename}")
            augment_image(file_path)

if __name__ == "__main__":
    process_directory(INPUT_DIR)
    print("Аугментация завершена!")