from utils.logger import logger_
from utils import global_var as glb
from models.classifier import Classifier


class AbsTrainer:
    def __init__(self, cv_dataset, test_dataset, args, config, device):
        self.cv_dataset = cv_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.config = config
        self.device = device

    def __setup_early_stop(self):
        pass

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode):
        pass

    def cross_validation(self):
        for i in range(self.args.num_folds):
            # setup for one pattern of the n-fold cross-validation
            logger_.info(f"** [{i + 1}/{self.args.num_folds}] CROSS-VALIDATION **")
            logger_.info(f"** [{i + 1}/{self.args.num_folds}] SETUP DATASET and MODEL **")
            train = self.cv_dataset.get_train_dataset(i)
            valid = self.cv_dataset.get_valid_dataset(i)
            model = Classifier(self.config)
            optimizer = model.get_optimizer()
            # self.__setup_early_stop()

            # train
            for j in range(self.args.num_epoch):
                self.cv_dataset.set_train_transform()
                self._train_epoch(cur_fold=i + 1,
                                  cur_epoch=j + 1,
                                  num_folds=self.args.num_folds,
                                  model=model,
                                  optimizer=optimizer,
                                  dataset=train,
                                  mode=glb.cv_train)
            # validation
                self.cv_dataset.set_valid_transform()
                self._train_epoch(cur_fold=i + 1,
                                  cur_epoch=j + 1,
                                  num_folds=self.args.num_folds,
                                  model=model,
                                  optimizer=optimizer,
                                  dataset=valid,
                                  mode=glb.cv_valid)

    def test(self, mode):
        pass
