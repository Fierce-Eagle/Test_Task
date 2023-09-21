import os
import zipfile
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
import numpy as np


class DataLoad:
    def __init__(self, path_from, path_to):
        self.image_dir = path_to + "train/"
        if not os.path.exists(self.image_dir):
            with zipfile.ZipFile(path_from, 'r') as zip_f:
                zip_f.extractall(path_to)

        filenames = os.listdir(self.image_dir)
        labels = [x.split(".")[0] for x in filenames]

        self.data = pd.DataFrame({"filename": filenames, "label": labels})

    def split_data(self):
        labels = self.data['label']
        x_train, x_temp = train_test_split(self.data, test_size=0.2, stratify=labels, random_state=42)

        label_val_test = x_temp['label']
        x_valid, x_test = train_test_split(x_temp, test_size=0.5, stratify=label_val_test, random_state=42)

        return x_train, x_valid, x_test


class CustomDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        """
        Класс обработки датасета

        :param data: датасет вида pd.DataFrame({filename: , label: })
        :param img_dir: путь до директории с изображениями
        :param transform: аугментация
        """
        self.img_dir = img_dir
        self.data = data
        self.transform = transform

    def __len__(self):
        """
        Число всех картинок в датасете
        :return:
        """
        return len(self.data["filename"])

    def __getitem__(self, idx):
        """
        Получение картинки из датасета

        :param idx: позиция картинки в датасете
        :return:
        """
        img_path = self.img_dir + self.data["filename"][idx]
        image = Image.open(img_path)
        label = self.data["label"][idx]
        if self.transform:
            img = self.transform(image=np.asarray(image))
            print(img)
            """
            ПРОБЛЕМА: для аугментаций нужно передавать np.array,
            соответсвенно, возращается он же.
            В аугментациях раньше была функция ToTensor, которая решает эту проблему,
            однако сейчас ее выпилили. 
            Есть вариант вытаскивать 2 элемент из словаря, но сегодня уже не успеваю.
            
            # Есть идеи?
            """
            image = Image.fromarray(img)  # выдает ошибки
        return image, label


# RandomCrop решил не делать
train_augmentation = A.Compose([
    # изменение размеров картинки
    A.Resize(128, 128),

    # применяемые агументации
    A.HorizontalFlip(p=0.2),
    A.Rotate(p=0.4),
    A.RandomBrightnessContrast(),
    A.RandomShadow(),
    A.Blur(blur_limit=3, p=0.1),

    # нормализация
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
], p=0.2)

test_augmentation = A.Compose([
    # изменение размеров картинки
    A.Resize(128, 128),

    # применяемые агументации
    A.Rotate(),
    A.RandomBrightnessContrast(),
    A.RandomShadow(),
    A.Blur(blur_limit=3, p=0.1),

    # нормализация
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
], p=0.15)
