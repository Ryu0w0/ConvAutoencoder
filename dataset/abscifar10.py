import numpy as np
from torchvision.datasets import CIFAR10 as org_cifar10
from torch.utils.data import Subset
from utils.logger import logger_
from dataset.img_transform import ImgTransform


class AbstractCIFAR10(org_cifar10):
    """
    Abstract class of CIFAR10.
    """
    def __init__(self, root, train, download, args):
        """
        root: str
            location of downloading dataset
        train: boolean
            True for cross-validation, False for testing
        download: boolean
            True for downloading dataset
        """
        super().__init__(root=root, train=train, download=download)
        self.args = args
        self.train_idx_list = None
        self.valid_idx_list = None

    def set_train_transform(self):
        """ Set transform into dataset. Call it every epoch of training"""
        self.transform = ImgTransform(args=self.args, is_train=True)
        self.target_transform = None

    def set_valid_transform(self):
        """ Set transform into dataset. Call it every epoch of validating"""
        self.transform = ImgTransform(args=self.args, is_train=False)
        self.target_transform = None

    def get_train_dataset(self, valid_fold_idx):
        return Subset(self, self.train_idx_list[valid_fold_idx])

    def get_valid_dataset(self, valid_fold_idx):
        return Subset(self, self.valid_idx_list[valid_fold_idx])

    def _show_data_composition(self):
        dataset_type = "TRAIN" if self.train else "TEST"
        logger_.info(f"*** {dataset_type} DATA COMPOSITION in Cross-Validation ***")
        for data_nm, idx_list in zip(["TRAIN", "VALID"], [self.train_idx_list, self.valid_idx_list]):
            for i in range(len(idx_list)):
                res = {}
                for cls_nm, cls_idx in self.class_to_idx.items():
                    cnt = np.count_nonzero(np.array(self.targets)[idx_list[i]] == cls_idx)
                    res[cls_nm] = cnt
                logger_.info(f"[{data_nm}][{i+1}] {res}")

