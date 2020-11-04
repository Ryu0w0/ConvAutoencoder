def initialization():
    """
    Initializing arguments, logger, tensorboard recorder and json files.
    """
    import argparse
    from torch.utils.tensorboard import SummaryWriter
    from utils import file_operator as f_op
    from utils import logger as log_util
    from utils import seed

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_key", type=str, default="auto_en",
                        help="Used as a file name of dataset and log files")
    parser.add_argument("-log_level", type=str, default="INFO")
    parser.add_argument("-use_gpu", type=int, default=0)
    parser.add_argument("-is_reproducible", type=int, default=1)
    parser.add_argument("-is_local", type=int, default=0)
    # MODEL
    parser.add_argument("-model_config_key", type=str, default="cnn_lr1e-05_oversampled",
                        help="Name of config file specifying a model architecture.")
    # TRAINING
    parser.add_argument("-use_aug", type=int, default=1, help="1 if augment train data otherwise 0")
    parser.add_argument("-num_folds", type=int, default=5)
    parser.add_argument("-num_epoch", type=int, default=10)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-num_workers", type=int, default=1)
    parser.add_argument("-save_img_per_epoch", type=int, default=5,
                        help="Save org and reconstructed images per specified epoch")
    # TEST
    parser.add_argument("-do_test", type=int, default=1, help="1 if do testing otherwise 0")
    args = parser.parse_args()

    # create logger
    args.log_level = log_util.get_log_level_from_name(args.log_level)
    logger_ = log_util.create_logger("main", "./files/output/logs", args.save_key, args.log_level)
    log_util.logger_ = logger_

    # show specified arguments
    logger_.info("*** ARGUMENTS ***")
    for k, v in args.__dict__.items():
        logger_.info(f"{k}: {v}")

    # create TensorBoard writer
    board_root_dir = f"./files/output/board/{args.save_key}"
    f_op.create_folder(board_root_dir)
    log_util.writer_ = SummaryWriter(board_root_dir)

    # load model config
    logger_.info("** LOAD MODEL CONFIG **")
    config = f_op.load_json("./files/input/models/configs", args.model_config_key)
    logger_.info(config)

    # set flg of using seeds
    if args.is_reproducible:
        seed.feed_seed = True

    return args, config


def main():
    args, config = initialization()
    from utils.logger import logger_
    from dataset.sub_cifar10.cifar10_cv import CIFAR10CV
    from dataset.sub_cifar10.cifar10_test import CIFAR10Test
    from trainer.sub_trainer.train_cae_cnn import TrainCAECNN
    from trainer.sub_trainer.train_only_cnn import TrainOnlyCNN
    from trainer.sub_trainer.train_only_cae import TrainOnlyCAE

    logger_.info("*** SET DEVICE ***")
    device = "cpu" if args.use_gpu == 0 else "cuda"
    logger_.info(f"Device is {device}")

    logger_.info("*** CREATE DATASET ***")
    trainset = CIFAR10CV(root='./files/input/dataset', train=True, download=True, args=args,
                         reg_map=config["train_data_regulation"],
                         expand_map=config["train_data_expansion"])
    testset = CIFAR10Test(root='./files/input/dataset', train=False, download=True, args=args, cifar10_cv=trainset)

    logger_.info("*** PREPARE TRAINING ***")
    if config["use_cae"] and config["use_cnn"]:
        trainer_cv = TrainCAECNN(trainset, args, config, device)
        trainer_test = TrainCAECNN(testset, args, config, device)
    elif config["use_cnn"]:
        trainer_cv = TrainOnlyCNN(trainset, args, config, device)
        trainer_test = TrainOnlyCNN(testset, args, config, device)
    elif config["use_cae"]:
        trainer_cv = TrainOnlyCAE(trainset, args, config, device)
        trainer_test = TrainOnlyCAE(testset, args, config, device)
    else:
        assert False, "At least one model should be specified."
        
    if args.do_cv:
        logger_.info("*** CROSS-VALIDATION ***")
        trainer_cv.cross_validation()
    if args.do_test:
        logger_.info("*** TEST ***")
        trainer_test.cross_validation()

    exit(0)


if __name__ == '__main__':
    main()


