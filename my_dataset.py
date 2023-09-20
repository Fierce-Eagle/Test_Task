import os
import zipfile
import random
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A


class CustomDataset:
    def __init__(self, path_from, path_to):
        self.image_dir = path_to + "train/"
        self.is_created_early = True  # во избежание модификации созданного датасета
        if not os.path.exists(self.image_dir):
            self.is_created_early = False
            with zipfile.ZipFile(path_from, 'r') as zip_f:
                zip_f.extractall(path_to)

        filenames = os.listdir(self.image_dir)
        labels = [x.split(".")[0] for x in filenames]

        self.data = pd.DataFrame({"filename": filenames, "label": labels})

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.Rotate(p=0.4),
            A.RandomBrightnessContrast(),
            A.RandomShadow(),
            A.Blur(blur_limit=3, p=0.1)
        ])

    def transformation(self, mutation=0.1):
        """
        Аугментация датасета
        :param mutation: вероятность аугментации для каждого изображения
        :return:
        """
        if not self.is_created_early:
            df = self.data.reset_index()

            for name_index, row in df.iterrows():
                if random.random() < mutation:
                    filename = row["filename"]
                    label = row["label"]
                    new_image_name = label + "." + str(name_index) + "A.jpg"
                    new_image = Image.open(self.image_dir + filename)
                    self.transform(image=np.asarray(new_image))  # конвертация, чтобы не было ошибок
                    new_image.save(self.image_dir + new_image_name)
                    self.data = pd.concat([self.data, pd.DataFrame({"filename": new_image_name, "label": label}, index=[0])])
            

    def split_data(self):
        labels = self.data['label']
        x_train, x_temp = train_test_split(self.data, test_size=0.2, stratify=labels, random_state=42)

        label_val_test = x_temp['label']
        x_valid, x_test = train_test_split(x_temp, test_size=0.5, stratify=label_val_test, random_state=42)

        return x_train, x_valid, x_test
