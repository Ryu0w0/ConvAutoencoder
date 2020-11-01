import numpy as np
from sklearn import metrics
from utils.logger import logger_, writer_


class StatCollector:
    def __init__(self, class_names, args):
        self.class_names = class_names
        self.args = args

    # CAE
    def logging_stat_cae(self, mode, cur_epoch, cur_fold, mean_loss):
        logger_.info(f"[{cur_fold}/{self.args.num_folds}][{cur_epoch}/{self.args.num_epoch}] "
                     f"{mode} loss: {np.round(mean_loss, 6)}")
        writer_.add_scalars(main_tag=f"{mode}/loss",
                            tag_scalar_dict={f"fold{cur_fold}": mean_loss},
                            global_step=cur_epoch)

    # CNN
    def calc_stat_cnn(self, total_loss, preds, labels):
        mean_loss = total_loss / len(preds)
        stats = metrics.classification_report(labels, preds, target_names=self.class_names, output_dict=True)
        return mean_loss, stats

    def logging_stat_cnn(self, mode, cur_epoch, cur_fold, stats, mean_loss):
        # logging overall loss and acc
        for stat_nm, stat in zip(["loss", "acc"], [mean_loss, stats["accuracy"]]):
            writer_.add_scalars(main_tag=f"{mode}/{stat_nm}",
                                tag_scalar_dict={f"fold{cur_fold}": stat},
                                global_step=cur_epoch)
        logger_.info(f"[{cur_fold}/{self.args.num_folds}][{cur_epoch}/{self.args.num_epoch}] "
                     f"{mode} loss: {np.round(mean_loss, 4)}, {mode} acc: {np.round(stats['accuracy'], 4)}")

        # logging precision, recall and f1-score per class
        for cls_nm, stat in stats.items():
            if cls_nm in self.class_names:
                for stat_type in ["precision", "recall", "f1-score"]:
                    writer_.add_scalars(main_tag=f"{mode}_fold{cur_fold}/{stat_type}",
                                        tag_scalar_dict={f"{cls_nm}": stat[stat_type]},
                                        global_step=cur_epoch)
