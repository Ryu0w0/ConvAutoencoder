import numpy as np
import torch
import torch.nn.functional as F
from utils import global_var as glb
from utils.seed import seed_everything
from utils.early_stop import EarlyStopping
from torch.utils.data.dataloader import DataLoader
from trainer.abstrainer import AbsTrainer
from trainer.stat_collector import StatCollector


class TrainOnlyCNN(AbsTrainer):
    def __init__(self, cv_dataset, args, config, device):
        super().__init__(cv_dataset, args, config, device)
        self.stat_collector = StatCollector(self.cv_dataset.classes, args)

    @staticmethod
    def _get_early_stopping():
        return EarlyStopping(min_delta=0.0001, improve_range=10, score_type="acc")

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode, es=None):
        seed_everything()
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        total_loss = 0
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
            output = model(images)  # shape: (data_num, class_num)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, labels)
            if mode == glb.cv_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # collect statistics
            total_loss += loss.detach().cpu().item()
            _, predicted = torch.max(output.detach().cpu(), 1)
            preds.extend(predicted.tolist())
            gt_labels.extend(labels.detach().cpu().tolist())

        if mode == glb.cv_valid:
            # logging statistics
            mean_loss, stats = self.stat_collector.calc_stat_cnn(total_loss, np.array(preds), np.array(gt_labels))
            self.stat_collector.logging_stat_cnn(mode=mode, cur_fold=cur_fold, cur_epoch=cur_epoch,
                                                 mean_loss=mean_loss, stats=stats, num_folds=self.num_folds)
            # record score for early stopping
            es.set_stop_flg(mean_loss, stats["accuracy"])
