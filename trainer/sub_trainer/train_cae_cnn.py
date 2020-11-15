import numpy as np
import torch
import torch.nn.functional as F
from utils import global_var as glb
from utils.global_var import TrainType
from utils.seed import seed_everything
from utils.early_stop import EarlyStopping
from torch.utils.data.dataloader import DataLoader
from trainer.abstrainer import AbsTrainer
from trainer.sub_trainer.train_only_cae import TrainOnlyCAE
from trainer.stat_collector import StatCollector


class TrainCAECNN(AbsTrainer):
    """
    Concurrently training CAE and CNN.
    cv_dataset: instance of sub-class of AbstractCIFAR10
    """
    def __init__(self, cv_dataset, args, config, device):
        super().__init__(cv_dataset, args, config, device)
        self.stat_collector = StatCollector(self.cv_dataset.classes, args)

    @staticmethod
    def _get_early_stopping():
        return EarlyStopping(min_delta=0.0001, improve_range=10, score_type="acc")

    def is_only_train_cae(self, cur_epoch):
        """
        Only CAE is trained in the first X epoch.
        This function judges whether only CAE should be trained in the current epoch or not.
        """
        if cur_epoch <= self.config["only_train_cae_until"]:
            return True
        else:
            return False

    @staticmethod
    def __train_epoch_cae_cnn(model, optimizer, dataset, mode, args, device):
        """
        Train CAE and CNN for a single epoch.
        model: instance of Classifier
        """
        seed_everything()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        total_loss_cnn = 0
        total_loss_cae = 0
        preds = []
        gt_labels = []
        if mode == TrainType.CV_TRAIN:
            model.train()
        else:
            model.eval()

        # training iteration
        for id, batch in enumerate(loader):
            images, labels = batch
            images, labels = images.to(device), labels.long().to(device)
            op_cae, op_cnn = model(images)

            # loss
            loss_cae = F.mse_loss(images, op_cae)
            op_cnn = F.log_softmax(op_cnn, dim=1)
            loss_cnn = F.nll_loss(op_cnn, labels)
            loss = loss_cae + loss_cnn

            if mode == TrainType.CV_TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # collect statistics
            total_loss_cnn += loss_cnn.detach().cpu().item()
            _, predicted = torch.max(op_cnn.detach().cpu(), 1)
            preds.extend(predicted.tolist())
            gt_labels.extend(labels.detach().cpu().tolist())
            total_loss_cae += loss_cae.detach().cpu().item()
        return total_loss_cae, total_loss_cnn, preds, gt_labels

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode, es=None):
        if self.is_only_train_cae(cur_epoch):
            total_loss, total_images, images, output = \
                TrainOnlyCAE.train_epoch_cae(model.cae, optimizer.opt_cae, dataset, mode, self.args, self.device)
            if mode == TrainType.CV_VALID:
                # logging statistics
                mean_loss = total_loss / total_images
                self.stat_collector.logging_stat_cae(mode=mode, cur_fold=cur_fold, cur_epoch=cur_epoch,
                                                     mean_loss=mean_loss, num_folds=self.num_folds)
        else:
            total_loss_cae, total_loss_cnn, preds, gt_labels = \
                self.__train_epoch_cae_cnn(model, optimizer, dataset, mode, self.args, self.device)
            if mode == TrainType.CV_VALID:
                # logging statistics
                mean_loss_cnn, stats = self.stat_collector.calc_stat_cnn(total_loss_cnn, np.array(preds), np.array(gt_labels))
                self.stat_collector.logging_stat_cnn(mode, cur_epoch, cur_fold, stats, mean_loss_cnn, self.num_folds)
                mean_loss_cae = total_loss_cae / len(preds)
                self.stat_collector.logging_stat_cae(mode, cur_epoch, cur_fold, mean_loss_cae, self.num_folds)
                # record score for early stopping
                es.set_stop_flg(acc=stats["accuracy"])
