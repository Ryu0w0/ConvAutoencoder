import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import transforms as trans


class ImgTransform:
    def __init__(self, args, is_train):
        self.transform = self.create_transform(args, is_train)

    @staticmethod
    def create_transform(args, is_train):
        """
        Convert numpy array into Tensor if dataset is for validation.
        Apply data augmentation method to train dataset while cv or test if args.use_aug is 1.
        :param args: arguments of main.py
        :param is_train: flg that dataset is for validation in cv or test
        :return:
        """
        if is_train and args.use_aug == 1:
            transform = A.Compose([
                trans.HorizontalFlip(p=0.5),
                trans.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=30, p=1),
                # trans.CoarseDropout(max_holes=1, min_holes=1, max_height=6, max_width=6),
                ToTensorV2()]
            )
        else:
            transform = ToTensorV2()
        return transform

    def __call__(self, img):
        img = np.asarray(img)
        img = img / 255  # convert value range into [0, 1] from [0, 255]
        img = self.transform(image=img)["image"]  # shape is also transformed from (w, h, c) to (c, w, h)
        return img
