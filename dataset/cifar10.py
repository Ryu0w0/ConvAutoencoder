import random
import numpy as np
from torchvision.datasets import CIFAR10 as org_cifar10
from utils.logger import logger_


class CIFAR10(org_cifar10):
    def __init__(self, root, train, download, reg_map=None):
        super().__init__(root=root, train=train, download=download)
        if reg_map is not None:
            self.regulate_data_num(reg_map)
        self.show_data_composition()

    def regulate_data_num(self, reg_map):
        """
        Regulate the number of images.
        reg_map: dict
            {cls_name for the regulation: number of images after the regulation, ....}
        """
        keep_idx = []
        for class_nm, class_no in self.class_to_idx.items():
            idx_array = np.where(class_no == np.array(self.targets))[0]
            if class_nm in reg_map.keys():
                idx_array = np.random.choice(idx_array, reg_map[class_nm], replace=False)
            keep_idx.extend(list(idx_array))

        self.targets = list(np.array(self.targets)[keep_idx])
        self.data = self.data[keep_idx, :, :, :]  # shape is (batch, w, h, ch)

    def show_data_composition(self):
        dataset_type = "TRAIN" if self.train else "TEST"
        logger_.info(f"*** {dataset_type} DATA COMPOSITION ***")
        # show overall info
        logger_.info(self)
        # show specific info
        for class_nm, class_no in self.class_to_idx.items():
            logger_.info(f"{class_nm}: {len(np.where(class_no == np.array(self.targets))[0])}")
