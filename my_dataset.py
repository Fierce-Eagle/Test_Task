import os
import zipfile
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A
from torchvision.utils import save_image


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
    def __init__(self, data, img_dir, transform=None, save_image_path=None):
        """
        Класс обработки датасета

        :param data: датасет вида pd.DataFrame({filename: , label: })
        :param img_dir: путь до директории с изображениями
        :param transform: аугментация
        """
        self.img_dir = img_dir
        self.data = data
        self.transform = transform
        self.save_image_path = save_image_path

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
        assert self.save_image_path is not None

        img_path = self.img_dir + self.data["filename"][idx]
        image = Image.open(img_path)
        label = self.data["label"][idx]
        if self.transform:
            image_dict = self.transform(image=np.array(image))

            # раз конвертации в тензор нет в версии из google colab, значит возьмем напрямую из оффициального репозитория
            # https://github.com/albumentations-team/albumentations/blob/master/albumentations/pytorch/transforms.py
            img = image_dict["image"]
            image = torch.from_numpy(
                np.moveaxis(img / (255.0 if img.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
            new_image_name = label + "." + self.data["filename"][idx].split(".")[1] + "_aug.jpg"
            save_image(image, self.save_image_path + "train/" + new_image_name)

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
], p=0.0001)

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
], p=0.0001)
