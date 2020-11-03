import numpy as np
from dataset.abscifar10 import AbstractCIFAR10


class CIFAR10Test(AbstractCIFAR10):
    def __init__(self, root, train, download, args, cifar10_cv):
        super().__init__(root=root, train=train, download=download, args=args)
        # index of train and valid data for test
        num_train = len(cifar10_cv.data)
        num_test = len(self.data)
        self.train_idx_list = [np.arange(num_train)]
        self.valid_idx_list = [np.arange(num_train, num_train + num_test)]
        # set data and targets for test
        self.data = np.concatenate([cifar10_cv.data, self.data])
        self.targets = cifar10_cv.targets + self.targets
        # show data composition
        self._show_data_composition()