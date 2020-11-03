import numpy as np
from torchvision.datasets import CIFAR10 as org_cifar10
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
from utils.logger import logger_
from utils.seed import seed_everything
from dataset.img_transform import ImgTransform


class CIFAR10(org_cifar10):
    def __init__(self, root, train, download, args, reg_map=None):
        super().__init__(root=root, train=train, download=download)
        self.args = args
        if reg_map is not None:
            self.__regulate_data_num(reg_map)
        self.__show_data_composition()
        self.train_idx_list, self.valid_idx_list = self.__get_idx_folds()

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

    def __get_idx_folds(self):
        train_idx_list = []
        valid_idx_list = []
        seed_everything(target="random")
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train_idx, valid_idx in skf.split(self.data, self.targets):
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)

        # assertion whether train_idx includes valid idx
        for t_idx, v_idx in zip(train_idx_list, valid_idx_list):
            res = np.intersect1d(np.array(t_idx), np.array(v_idx))
            assert len(res) == 0, f"Images are shared between train and valid dataset: {res}."

        return train_idx_list, valid_idx_list

    def __regulate_data_num(self, reg_map):
        """
        Regulate the number of images.
        reg_map: dict
            {cls_name for the regulation: number of images after the regulation, ....}
        """
        keep_idx = []
        for idx, (class_nm, class_no) in enumerate(self.class_to_idx.items()):
            idx_array = np.where(class_no == np.array(self.targets))[0]
            if self.args.is_local == 1:
                # regulate data of each class into 250 if running pgm at local while development
                seed_everything(target="random")
                idx_array = np.random.choice(idx_array, 250, replace=False)
            elif "all" in reg_map.keys():
                seed_everything(target="random", local_seed=idx)
                idx_array = np.random.choice(idx_array, reg_map["all"], replace=False)
            elif class_nm in reg_map.keys():
                seed_everything(target="random", local_seed=idx)
                idx_array = np.random.choice(idx_array, reg_map[class_nm], replace=False)
            keep_idx.extend(list(idx_array))

        self.targets = list(np.array(self.targets)[keep_idx])
        self.data = self.data[keep_idx, :, :, :]  # shape is (batch, w, h, ch)

    def __show_data_composition(self):
        dataset_type = "TRAIN" if self.train else "TEST"
        logger_.info(f"*** {dataset_type} DATA COMPOSITION ***")
        # show overall info
        logger_.info(self)
        # show specific info
        for class_nm, class_no in self.class_to_idx.items():
            logger_.info(f"{class_nm}: {len(np.where(class_no == np.array(self.targets))[0])}")
