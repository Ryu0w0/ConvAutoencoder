import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import utils as vutils
from utils import global_var as glb
from utils import file_operator as f_op
from trainer.abstrainer import AbsTrainer
from trainer.stat_collector import StatCollector


class TrainOnlyCAE(AbsTrainer):
    def __init__(self, cv_dataset, test_dataset, args, config, device):
        super().__init__(cv_dataset, test_dataset, args, config, device)
        self.stat_collector = StatCollector(self.cv_dataset.classes, args)

    def __save_image_as_grid(self, in_tensor, out_tensor, cur_fold, cur_epoch):
        if cur_epoch % self.args.save_img_per_epoch == 0:
            for img_tensor, file_name in zip([in_tensor, out_tensor], ["org", "reconstructed"]):
                img_tensor = img_tensor.detach()
                save_image_path = f"./files/output/images"
                f_op.create_folder(save_image_path)
                if img_tensor.shape[0] > 8:
                    img_tensor = img_tensor[0:8, :, :, :]
                img_array = vutils.make_grid(img_tensor, padding=2, normalize=False).cpu().numpy()
                img_array = np.transpose(img_array, (1, 2, 0))
                f_op.save_as_jpg(img_array * 255, save_root_path=save_image_path,
                                 save_key=self.args.save_key, file_name=f"{file_name}_{cur_fold}_{cur_epoch}")

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode, es=None):

        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        total_loss = 0
        total_images = 0

        if mode == glb.cv_train:
            model.train()
        else:
            model.eval()

        # training iteration
        for id, batch in enumerate(loader):
            images, _ = batch
            images = images.to(self.device)
            output = model(images)
            loss = F.mse_loss(images, output)
            if mode == glb.cv_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # collect statistics
            total_images += len(images)
            total_loss += loss.detach().cpu().item()

        if mode == glb.cv_valid:
            # logging statistics
            mean_loss = total_loss / total_images
            self.stat_collector.logging_stat_cae(mode=mode, cur_fold=cur_fold, cur_epoch=cur_epoch, mean_loss=mean_loss)
            self.__save_image_as_grid(in_tensor=images, out_tensor=output, cur_fold=cur_fold, cur_epoch=cur_epoch)
            # record score for early stopping
            es.set_stop_flg(mean_loss)
