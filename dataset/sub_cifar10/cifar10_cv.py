import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dataset.abscifar10 import AbstractCIFAR10
from utils.seed import seed_everything


class CIFAR10CV(AbstractCIFAR10):
    def __init__(self, root, train, download, args, reg_map, expand_map=None):
        super().__init__(root=root, train=train, download=download, args=args)
        if reg_map is not None:
            # overwrite with regulated number of data and labels
            data, targets = self.__regulate_data_num(reg_map)
            self.data = data
            self.targets = targets
        # set a list of train and validation data in n-fold cross-validation
        self.train_idx_list, self.valid_idx_list = self.__get_idx_folds()
        # oversample small number of dataset
        if expand_map is not None:
            self.train_idx_list = self.__expand_data_num(expand_map)
        # show data composition
        self._show_data_composition()

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

        data = self.data[keep_idx, :, :, :]  # shape is (batch, w, h, ch)
        targets = list(np.array(self.targets)[keep_idx])
        return data, targets

    def __expand_data_num(self, expand_map):
        train_idx_list = copy.deepcopy(self.train_idx_list)
        for class_nm, additional_num in expand_map.items():
            class_idx = self.class_to_idx[class_nm]
            for i in range(len(train_idx_list)):
                # indices of train data of 4-folds
                train_indices = train_idx_list[i]
                # labels of train data
                train_labels = np.array(self.targets)[train_indices]
                # indices of the class in train data
                class_indices = train_indices[np.where(train_labels == class_idx)[0]]
                # oversample the specified number of indices
                if self.args.is_local == 1:
                    oversampled_indices = np.random.choice(class_indices, 250, replace=False)
                else:
                    oversampled_indices = np.random.choice(class_indices, additional_num, replace=False)

                # add oversampled indices
                train_idx_list[i] = np.concatenate([train_indices, oversampled_indices])
        return train_idx_list

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
