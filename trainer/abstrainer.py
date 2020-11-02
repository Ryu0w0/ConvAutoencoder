import numpy as np
import torch
from sklearn import metrics
from utils.logger import logger_, writer_
from utils import global_var as glb
from utils.early_stop import EarlyStopping
from models.classifier import Classifier
from models.augoencoder.conv_auto_en import ConvolutionalAutoEncoder
from models.cnn.cnn import CNN


class AbsTrainer:
    def __init__(self, cv_dataset, test_dataset, args, config, device):
        self.cv_dataset = cv_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.config = config
        self.device = device

    def __get_model(self):
        if self.config["use_cae"] and self.config["use_cnn"]:
            return Classifier(self.config)
        elif self.config["use_cnn"]:
            return CNN(self.config).double()
        elif self.config["use_cae"]:
            return ConvolutionalAutoEncoder(self.config).double()
        else:
            assert False, "At least one model should be specified."

    @staticmethod
    def __logging_materials_one_time(fold_seq, target_map):
        if fold_seq == 0:
            for name, material in target_map.items():
                logger_.info(f"*** {name} ***")
                logger_.info(material)

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode, es=None):
        pass

    def cross_validation(self):
        for i in range(self.args.num_folds):
            # setup for one pattern of the n-fold cross-validation
            logger_.info(f"** [{i + 1}/{self.args.num_folds}] {i + 1}-th CROSS-VALIDATION **")
            logger_.info(f"** [{i + 1}/{self.args.num_folds}] SETUP DATASET and MODEL **")
            # get train dataset consisting of n-1 folds
            train = self.cv_dataset.get_train_dataset(i)
            # get valid dataset consisting of 1 fold
            valid = self.cv_dataset.get_valid_dataset(i)
            # construct model and optimizer
            model = self.__get_model().to(self.device)
            optimizer = model.get_optimizer()
            # define early stopping
            es = EarlyStopping(min_delta=0.00001, improve_range=5, score_type="acc")
            self.__logging_materials_one_time(i, {"MODEL": model, "OPTIMIZER": optimizer, "EARLY STOPPING": es})

            for j in range(self.args.num_epoch):
                # train
                self.cv_dataset.set_train_transform()
                self._train_epoch(cur_fold=i + 1,
                                  cur_epoch=j + 1,
                                  num_folds=self.args.num_folds,
                                  model=model,
                                  optimizer=optimizer,
                                  dataset=train,
                                  mode=glb.cv_train,
                                  es=es)
                # validation
                self.cv_dataset.set_valid_transform()
                with torch.no_grad():
                    self._train_epoch(cur_fold=i + 1,
                                      cur_epoch=j + 1,
                                      num_folds=self.args.num_folds,
                                      model=model,
                                      optimizer=optimizer,
                                      dataset=valid,
                                      mode=glb.cv_valid,
                                      es=es)
                if es.is_stop:
                    logger_.info("FINISH TRAINING BY EARLY STOPPING")
                    break

    def test(self, mode):
        pass
