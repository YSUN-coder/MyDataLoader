from __future__ import print_function, division
import os
import glob
import cv2
from skimage.io import imread
from torch.utils.data import Dataset
import albumentations as A


class BSDDB(Dataset):
    def __init__(self, folder_dir="../img",
                 ext=".png", patch_size=128, fake_legth=1000):
        self.patch_size = patch_size
        self.fake_legth = fake_legth

        img_nms = glob.glob(os.path.join(folder_dir, '*' + ext))
        self.img_paths = []
        for img_nm in img_nms:
            img_path = os.path.join(folder_dir, img_nm)
            self.img_paths.append(img_path)

    def __len__(self):
        return self.fake_legth

    def __getitem__(self, idx):
        image = imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # randomly extract a patch from hard disk
        # random transform through albumentations for image augmentation
        # Declare an augmentation pipeline
        transform = A.Compose([
            A.RandomCrop(width=self.patch_size, height=self.patch_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomBrightness(limit=0.2, always_apply=False, p=0.2),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.2)
        ])

        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        return transformed_image