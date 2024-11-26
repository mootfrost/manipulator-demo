import os
import shutil
import random

# Параметры разбиения
TRAIN_RATIO = 0.8
VALIDATE_RATIO = 0.1
TEST_RATIO = 0.1

# Входная и выходная директории
DATASET_DIR = "datasets/Dataset"
OUTPUT_DIR = "datasets/Dataset_Split"

# Создаем выходные папки, если их нет
def create_dirs(output_dir, categories):
    for subset in ['Train', 'Validate', 'Test']:
        for category in categories:
            os.makedirs(os.path.join(output_dir, subset, category), exist_ok=True)

# Функция для разбиения датасета
def split_dataset(input_dir, output_dir, train_ratio, validate_ratio):
    # Получаем список всех категорий (папок)
    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Создаем выходные папки
    create_dirs(output_dir, categories)
    
    # Разбиваем данные по классам
    for category in categories:
        category_path = os.path.join(input_dir, category)
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        
        # Перемешиваем изображения случайным образом
        random.shuffle(images)
        
        # Считаем количество изображений для каждого множества
        num_total = len(images)
        num_train = int(num_total * train_ratio)
        num_validate = int(num_total * validate_ratio)
        num_test = num_total - num_train - num_validate

        # Разбиваем на Train, Validate и Test
        train_images = images[:num_train]
        validate_images = images[num_train:num_train + num_validate]
        test_images = images[num_train + num_validate:]
        
        # Функция для копирования файлов
        def copy_files(file_list, subset):
            for file in file_list:
                src = os.path.join(category_path, file)
                dst = os.path.join(output_dir, subset, category, file)
                shutil.copy2(src, dst)

        # Копируем файлы в соответствующие папки
        copy_files(train_images, 'Train')
        copy_files(validate_images, 'Validate')
        copy_files(test_images, 'Test')
        
        print(f"Категория '{category}' - Train: {len(train_images)}, Validate: {len(validate_images)}, Test: {len(test_images)}")

if __name__ == "__main__":
    split_dataset(DATASET_DIR, OUTPUT_DIR, TRAIN_RATIO, VALIDATE_RATIO)
    print("Разбиение завершено!")
