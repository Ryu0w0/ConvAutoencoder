import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import transforms as trans
from utils.seed import seed_everything


class ImgTransform:
    def __init__(self, args, is_train):
        self.cifar_10_mean = (0.4914, 0.4822, 0.4465)
        self.cifar_10_std = (0.247, 0.243, 0.261)
        self.transform = self.create_transform(args, is_train)
        self.cnt_seed = 0

    def create_transform(self, args, is_train):
        """
        Convert numpy array into Tensor if dataset is for validation.
        Apply data augmentation method to train dataset while cv or test if args.use_aug is 1.

        is_train: boolean
            flg that dataset is for validation in cv or test
        return: Compose of albumentations
        """
        if is_train and args.use_aug == 1:
            transform = A.Compose([
                trans.Normalize(mean=self.cifar_10_mean, std=self.cifar_10_std, max_pixel_value=1.0),
                trans.HorizontalFlip(p=0.5),
                trans.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=30, p=1),
                trans.CoarseDropout(max_holes=1, min_holes=1, min_width=12, min_height=12,
                                    max_height=12, max_width=12, p=0.5),
                ToTensorV2()]
            )
        else:
            transform = A.Compose([
                trans.Normalize(mean=self.cifar_10_mean, std=self.cifar_10_std, max_pixel_value=1.0),
                ToTensorV2()]
            )
        return transform

    def denormalize(self, img_tensor):
        """
        denormalize pixel ranges of images. Shape of input should be (b, c, w, h).
        This function is supposed to be used for visualization.
        """
        mean_ = torch.Tensor(self.cifar_10_mean).reshape((1, 3, 1, 1))
        std_ = torch.Tensor(self.cifar_10_std).reshape((1, 3, 1, 1))
        denormalized = torch.mul(img_tensor, std_) + mean_
        return denormalized

    def __call__(self, img):
        img = np.asarray(img)
        img = img / 255  # convert value range into [0, 1] from [0, 255]
        seed_everything(local_seed=self.cnt_seed)
        self.cnt_seed += 1
        img = self.transform(image=img)["image"]  # shape is also transformed from (w, h, c) to (c, w, h)
        return img
