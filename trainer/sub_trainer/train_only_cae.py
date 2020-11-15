import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import utils as vutils
from utils.global_var import TrainType
from utils import global_var as glb
from utils.early_stop import EarlyStopping
from utils import file_operator as f_op
from utils.seed import seed_everything
from trainer.abstrainer import AbsTrainer
from trainer.stat_collector import StatCollector


class TrainOnlyCAE(AbsTrainer):
    """
    Training Convolutional Autoencoder for a single epoch.
    """
    def __init__(self, cv_dataset, args, config, device):
        super().__init__(cv_dataset, args, config, device)
        self.stat_collector = StatCollector(self.cv_dataset.classes, args)

    @staticmethod
    def _get_early_stopping():
        return EarlyStopping(min_delta=0.0001, improve_range=5, score_type="loss")

    def __save_image_as_grid(self, in_tensor, out_tensor, cur_fold, cur_epoch):
        """ Save a set of input or output images per specified epoch """
        if cur_epoch % self.args.save_img_per_epoch == 0:
            for img_tensor, file_name in zip([in_tensor, out_tensor], ["org", "reconstructed"]):
                # convert normalized images back to denormalized images for visualization
                img_tensor = self.cv_dataset.transform.denormalize(img_tensor.detach().cpu())
                save_image_path = "./files/output/images"
                f_op.create_folder(save_image_path)
                # pick up images at most 8 images
                if img_tensor.shape[0] > 8:
                    img_tensor = img_tensor[0:8, :, :, :]
                # line up a set of images as a grid image
                img_array = vutils.make_grid(img_tensor, padding=2, normalize=False).cpu().numpy()
                img_array = np.transpose(img_array, (1, 2, 0))
                # save
                f_op.save_as_jpg(img_array * 255, save_root_path=save_image_path,
                                 save_key=self.args.save_key, file_name=f"{file_name}_{cur_fold}_{cur_epoch}")

    @staticmethod
    def train_epoch_cae(model, optimizer, dataset, mode, args, device):
        seed_everything()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        total_loss = 0
        total_images = 0

        if mode == TrainType.CV_TRAIN:
            model.train()
        else:
            model.eval()

        # training iteration
        for id, batch in enumerate(loader):
            images, _ = batch
            images = images.to(device)
            _, output = model(images)
            loss = F.mse_loss(images, output)
            if mode == TrainType.CV_TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # collect statistics
            total_images += len(images)
            total_loss += loss.detach().cpu().item()
        return total_loss, total_images, images, output

    def _train_epoch(self, cur_fold, cur_epoch, num_folds, model, optimizer, dataset, mode, es=None):
        """
        Training a model for a single epoch.

        model: instance of CAE
        dataset: instance of sub-class of AbstractCIFAR10
        mode: glb.cv_train or glb.cv_valid
        es: instance of EarlyStopping
        """
        # train for a single epoch
        total_loss, total_images, images, output = \
            self.train_epoch_cae(model, optimizer, dataset, mode, self.args, self.device)
        # store results
        if mode == TrainType.CV_VALID:
            # logging statistics
            mean_loss = total_loss / total_images
            self.stat_collector.logging_stat_cae(mode=mode, cur_fold=cur_fold, cur_epoch=cur_epoch, mean_loss=mean_loss, num_folds=self.num_folds)
            self.__save_image_as_grid(in_tensor=images, out_tensor=output, cur_fold=cur_fold, cur_epoch=cur_epoch)
            # record score for early stopping
            es.set_stop_flg(mean_loss)

