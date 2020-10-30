import numpy as np
from sklearn import metrics
from utils.logger import logger_, writer_
from utils import global_var as glb
from utils.early_stop import EarlyStopping
from models.classifier import Classifier


class AbsTrainer:
    def __init__(self, cv_dataset, test_dataset, args, config, device):
        self.cv_dataset = cv_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.config = config
        self.device = device

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode, es=None):
        pass

    def _calc_stat(self, total_loss, preds, labels):
        mean_loss = total_loss / len(preds)
        stats = metrics.classification_report(labels, preds, target_names=self.cv_dataset.classes, output_dict=True)
        return mean_loss, stats

    def _logging_stat(self, mode, cur_epoch, cur_fold, stats, mean_loss):
        # logging overall loss and acc
        for stat_nm, stat in zip(["loss", "acc"], [mean_loss, stats["accuracy"]]):
            writer_.add_scalars(main_tag=f"{mode}/{stat_nm}",
                                tag_scalar_dict={f"fold{cur_fold}": stat},
                                global_step=cur_epoch)
        logger_.info(f"[{cur_fold}/{self.args.num_folds}][{cur_epoch}/{self.args.num_epoch}] "
                     f"{mode} loss: {np.round(mean_loss, 4)}, {mode} acc: {stats['accuracy']}")

        # logging precision, recall and f1-score per class
        for cls_nm, stat in stats.items():
            if cls_nm in self.cv_dataset.classes:
                for stat_type in ["precision", "recall", "f1-score"]:
                    writer_.add_scalars(main_tag=f"{mode}_fold{cur_fold}/{stat_type}",
                                        tag_scalar_dict={f"{cls_nm}": stat[stat_type]},
                                        global_step=cur_epoch)

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
            model = Classifier(self.config).to(self.device)
            optimizer = model.get_optimizer()
            # define early stopping
            es = EarlyStopping(min_delta=0.001, improve_range=5, score_type="acc")

            for j in range(self.args.num_epoch):
                # train
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
                                  mode=glb.cv_valid,
                                  es=es)
                if es.is_stop:
                    logger_.info("STOP BY EARLY STOPPING")
                    break

    def test(self, mode):
        pass
