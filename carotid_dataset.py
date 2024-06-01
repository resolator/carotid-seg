import cv2
import torch

import numpy as np
import albumentations as albu
import torchvision.transforms.v2 as transforms

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class CarotidDataset(Dataset):
    def __init__(
        self,
        ds_dir,
        aug=False,
        in_memory=False,
        resize_to=256,
        return_names=False
    ):
        assert resize_to % 32 == 0, 'the size should be divisible by 32'

        super().__init__()
        self.aug = aug
        self.in_memory = in_memory
        self.resize_to = resize_to
        self.return_names = return_names

        ds_dir = Path(ds_dir)
        images_dir = ds_dir / 'images'
        masks_dir = ds_dir / 'masks'

        self.images_paths = []
        self.images = []
        self.masks_paths = []
        self.masks = []
        for img_p in images_dir.glob('*.*'):
            mask_p = masks_dir / img_p.name
            if not mask_p.exists():
                print(f'[WARNING]: no mask for {img_p}.')
                continue

            self.images_paths.append(str(img_p))
            self.masks_paths.append(str(mask_p))
            if self.in_memory:
                self.images.append(self.read_img(img_p))
                self.masks.append(self.read_mask(mask_p))

        # define transforms
        img_mean = torch.tensor([0.1142])  # calculated on the train set
        img_std = torch.tensor([0.1621])  # calculated on the train set

        self.img_infer_t = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize(
                self.resize_to,
                max_size=self.resize_to + 1
            ),
            transforms.CenterCrop(self.resize_to),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
        self.img_infer_ut = transforms.Compose([
            transforms.Normalize(mean=[0], std=1 / img_std),
            transforms.Normalize(mean=-img_mean, std=[1]),
            transforms.ToPILImage()
        ])

        self.mask_infer_t = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize(
                self.resize_to,
                max_size=self.resize_to + 1,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.CenterCrop(self.resize_to),
            transforms.ToDtype(torch.float32, scale=True)
        ])

        self.aug_t = albu.Compose([
            albu.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                p=0.5
            ),
            albu.GaussianBlur((3, 5), 2),
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.15),
                contrast_limit=(-0.1, 0.15),
                p=0.8
            ),
            albu.Flip(p=0.5)
        ])

    @staticmethod
    def read_img(img_p):
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)  # not always grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    @staticmethod
    def read_mask(mask_p):
        return cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)  # always binary

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_p, mask_p = self.images_paths[idx], self.masks_paths[idx]
        if not self.in_memory:
            img = self.read_img(img_p)
            mask = self.read_img(mask_p)
        else:
            img = self.images[idx]
            mask = self.masks[idx]

        if self.aug:
            auged = self.aug_t(image=img, mask=mask)
            img = auged['image']
            mask = auged['mask']

        img = self.img_infer_t(img)
        mask = self.mask_infer_t(mask)

        if self.return_names:
            return img, mask, img_p
        else:
            return img, mask

    def visualize(self, img, mask):
        img = np.array(self.img_infer_ut(img))
        mask = np.array(self.mask_infer_ut(mask))

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        masked = img.copy() * 0.75
        masked[:, :, 1] = cv2.add(mask * 0.25, masked[:, :, 1])
        stacked = np.hstack([
            img,
            cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
            masked
        ]).astype(np.uint8)
        return Image.fromarray(stacked)
