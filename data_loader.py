from __future__ import print_function, division
import os
import glob
import cv2
import random
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
        # img_patch = self._randomly_crop(img_mat)
        # img_tran_patch = self._randomly_transform(img_patch)
        # Declare an augmentation pipeline
        transform = A.Compose([
            A.RandomCrop(width=self.patch_size, height=self.patch_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        return transformed_image

    def _randomly_crop(self, image_mat):
        (h, w) = image_mat.shape[:2]
        y = random.randint(1, h - self.patch_size)
        x = random.randint(1, w - self.patch_size)
        crop_img = image_mat[(y):(y + self.patch_size), (x):(x + self.patch_size)]
        return crop_img

    def _randomly_transform(self, item):
       trans_item = item
       return trans_item
