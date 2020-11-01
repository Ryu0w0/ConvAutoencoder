import numpy as np
import torch
import torch.nn.functional as F
from utils import global_var as glb
from torch.utils.data.dataloader import DataLoader
from trainer.abstrainer import AbsTrainer
from trainer.stat_collector import StatCollector


class TrainCAECNN(AbsTrainer):
    def __init__(self, cv_dataset, test_dataset, args, config, device):
        super().__init__(cv_dataset, test_dataset, args, config, device)
        self.stat_collector = StatCollector(self.cv_dataset.classes, args)

    @staticmethod
    def calc_alpha(cur_epoch):
        no_cnn_loss_until = 10
        if cur_epoch <= no_cnn_loss_until:
            return 0
        else:
            cur_epoch -= cur_epoch
            denominator = 30
            numerator = np.min(denominator, cur_epoch)
            return numerator / denominator

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode, es=None):
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        total_loss_cnn = 0
        total_loss_cae = 0
        preds = []
        gt_labels = []
        if mode == glb.cv_train:
            model.train()
        else:
            model.eval()

        # training iteration
        for id, batch in enumerate(loader):
            images, labels = batch
            images, labels = images.to(self.device), labels.long().to(self.device)
            op_cae, op_cnn = model(images)

            # loss
            loss_cae = F.mse_loss(images, op_cae)
            op_cnn = F.log_softmax(op_cnn, dim=1)
            loss_cnn = F.nll_loss(op_cnn, labels)
            loss_cnn = self.calc_alpha(cur_epoch) * loss_cnn
            if mode == glb.cv_train:
                optimizer.zero_grad()
                loss_cnn.backward()
                optimizer.opt_cae.step()
                optimizer.opt_cnn.step()

            # collect statistics
            total_loss_cnn += loss_cnn.detach().cpu().item()
            total_loss_cae += loss_cae.detach().cpu().item()
            _, predicted = torch.max(op_cnn.detach().cpu(), 1)
            preds.extend(predicted.tolist())
            gt_labels.extend(labels.detach().cpu().tolist())

        if mode == glb.cv_valid:
            # logging statistics
            mean_loss_cnn, stats = self.stat_collector.calc_stat_cnn(total_loss_cnn, np.array(preds), np.array(gt_labels))
            self.stat_collector.logging_stat_cnn(mode, cur_epoch, cur_fold, stats, mean_loss_cnn)
            mean_loss_cae = total_loss_cae / len(preds)
            self.stat_collector.logging_stat_cae(mode, cur_epoch, cur_fold, mean_loss_cae)
            # record score for early stopping
            es.set_stop_flg(mean_loss_cnn, stats["accuracy"])
