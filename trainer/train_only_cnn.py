import torch
import torch.nn.functional as F
from torch import nn
from utils import global_var as glb
from torch.utils.data.dataloader import DataLoader
from trainer.abstrainer import AbsTrainer


class TrainOnlyCNN(AbsTrainer):
    def __init__(self, cv_dataset, test_dataset, args, config, device):
        super().__init__(cv_dataset, test_dataset, args, config, device)
        self.loss_f_cnn = nn.CrossEntropyLoss()

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode):
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        if mode == glb.cv_train:
            model.train()
        else:
            model.eval()

        for id, batch in enumerate(loader):
            images, labels = batch
            images, labels = images.to(self.device), labels.long().to(self.device)
            output = model(images)  # shape: (data_num, class_num)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

