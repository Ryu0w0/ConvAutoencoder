import torch
from utils.logger import logger_
from utils import global_var as glb
from utils.global_var import TrainType
from dataset.sub_cifar10.cifar10_cv import CIFAR10CV
from models.classifier import Classifier
from models.augoencoder.conv_auto_en import ConvolutionalAutoEncoder
from models.cnn.cnn import CNN


class AbsTrainer:
    def __init__(self, cv_dataset, args, config, device):
        self.cv_dataset = cv_dataset
        self.args = args
        self.config = config
        self.device = device
        self.num_folds = self.args.num_folds if isinstance(cv_dataset, CIFAR10CV) else 1  # 1 fold for test

    def __get_model(self):
        if self.config["use_cae"] and self.config["use_cnn"]:
            return Classifier(self.config)
        elif self.config["use_cnn"]:
            return CNN(self.config)
        elif self.config["use_cae"]:
            return ConvolutionalAutoEncoder(self.config)
        else:
            assert False, "At least one model should be specified."

    @staticmethod
    def __logging_materials_one_time(fold_seq, target_map):
        if fold_seq == 0:
            for name, material in target_map.items():
                logger_.info(f"*** {name} ***")
                logger_.info(material)

    @staticmethod
    def _get_early_stopping():
        pass

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode, es=None):
        pass

    def cross_validation(self):
        """ Performing training and validation. It can be used for cross-validation and testing """
        for i in range(self.num_folds):
            # setup for one pattern of the n-fold cross-validation
            logger_.info(f"** [{i + 1}/{self.num_folds}] {i + 1}-th CROSS-VALIDATION **")
            logger_.info(f"** [{i + 1}/{self.num_folds}] SETUP DATASET and MODEL **")
            # get train dataset consisting of n-1 folds
            train = self.cv_dataset.get_train_dataset(i)
            # get valid dataset consisting of 1 fold
            valid = self.cv_dataset.get_valid_dataset(i)
            # construct model and optimizer
            model = self.__get_model().to(self.device)
            optimizer = model.get_optimizer()
            # define early stopping
            es = self._get_early_stopping()
            self.__logging_materials_one_time(i, {"MODEL": model, "OPTIMIZER": optimizer, "EARLY STOPPING": es})

            for j in range(self.args.num_epoch):
                # train
                self.cv_dataset.set_train_transform()
                self._train_epoch(cur_fold=i + 1,
                                  cur_epoch=j + 1,
                                  num_folds=self.num_folds,
                                  model=model,
                                  optimizer=optimizer,
                                  dataset=train,
                                  mode=TrainType.CV_TRAIN,
                                  es=es)
                # validation
                self.cv_dataset.set_valid_transform()
                with torch.no_grad():
                    self._train_epoch(cur_fold=i + 1,
                                      cur_epoch=j + 1,
                                      num_folds=self.num_folds,
                                      model=model,
                                      optimizer=optimizer,
                                      dataset=valid,
                                      mode=TrainType.CV_VALID,
                                      es=es)
                if es.is_stop:
                    logger_.info("FINISH TRAINING BY EARLY STOPPING")
                    logger_.info("EARLY STOP INFO")
                    logger_.info(es)
                    break
