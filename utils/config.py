import environ
from utils import logger


@environ.config()
class Config:
    # BASIC PARAMETERS
    save_key: str = environ.var(name="SAVE_KEY", converter=str, default="default",
                                help="Used as a file name of dataset and log files")
    log_level: int = environ.var(name="LOG_LEVEL", converter=logger.get_log_level_from_name,
                                 default="INFO")
    use_gpu: int = environ.var(name="USE_GPU", converter=bool, default=True,
                               help="1 if use GPU otherwise 0")
    is_seed: bool = environ.var(name="IS_SEED", converter=bool, default=False,
                                help="1 if feed seeds to all random procedures")
    is_local: bool = environ.var(name="IS_LOCAL", converter=bool, default=False,
                                 help="1 if reduce training data for running with CPU otherwise 0")
    do_cv: bool = environ.var(name="DO_CV", converter=bool, default=True,
                              help="1 if do cross-validation otherwise 0")
    do_test: bool = environ.var(name="DO_TEST", converter=bool, default=False,
                                help="1 if do testing otherwise 0")
    # MODEL
    model_config_key: str = environ.var(name="MODEL_CONFIG_KEY", converter=str, default="cae_cnn_lr1e-05_oversampled",
                                        help="Name of config file specifying a model architecture.")
    # TRAINING
    use_aug: bool = environ.var(name="USE_AUG", converter=bool, default=False,
                                help="1 if augment train data otherwise 0")
    num_folds: bool = environ.var(name="NUM_FOLDS", converter=int, default=5,
                                  help="Specify n for n-folds cross-validation")
    num_epoch: bool = environ.var(name="NUM_EPOCH", converter=int, default=10)
    batch_size: bool = environ.var(name="BATCH_SIZE", converter=int, default=64)
    num_workers: bool = environ.var(name="NUM_WORKERS", converter=int, default=1)
    save_img_per_epoch: bool = environ.var(name="SAVE_IMG_PER_EPOCH", converter=int, default=5,
                                           help="Save org and reconstructed images per specified epoch")
