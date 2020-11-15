"""
This script solely train Convolutional Autoencoder or CNN, or jointly train them according to
the setting specified in ./files/input/models/configs/*.json
The location of the Json file is specified in the argument of model_config_key
"""


def initialization():
    """
    Initializing arguments, logger, tensorboard recorder and json files.
    """
    import argparse
    from torch.utils.tensorboard import SummaryWriter
    from utils import file_operator as f_op
    from utils import logger as log_util
    from utils import seed
    from utils.config import Config
    import environ

    # create config object from arguments
    args = environ.to_config(Config)

    # create logger
    logger_ = log_util.create_logger("main", "./files/output/logs", args.save_key, args.log_level)
    log_util.logger_ = logger_

    # show specified arguments
    logger_.info("*** ARGUMENTS ***")
    logger_.info(args)

    # create TensorBoard writer
    board_root_dir = f"./files/output/board/{args.save_key}"
    f_op.create_folder(board_root_dir)
    log_util.writer_ = SummaryWriter(board_root_dir)

    # load model config
    logger_.info("** LOAD MODEL CONFIG **")
    config = f_op.load_json("./files/input/models/configs", args.model_config_key)
    logger_.info(config)

    # set flg of using seeds
    if args.is_seed:
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
                         expand_map=config["train_data_expansion"] if "train_data_expansion" in config.keys() else None)
    testset = CIFAR10Test(root='./files/input/dataset', train=False, download=True, args=args, cifar10_cv=trainset)

    logger_.info("*** DEFINE TRAINER ***")
    # train both CAE and CNN
    if config["use_cae"] and config["use_cnn"]:
        trainer_cv = TrainCAECNN(trainset, args, config, device)
        trainer_test = TrainCAECNN(testset, args, config, device)
    # only train CNN
    elif config["use_cnn"]:
        trainer_cv = TrainOnlyCNN(trainset, args, config, device)
        trainer_test = TrainOnlyCNN(testset, args, config, device)
    # only train CAE
    elif config["use_cae"]:
        trainer_cv = TrainOnlyCAE(trainset, args, config, device)
        trainer_test = TrainOnlyCAE(testset, args, config, device)
    else:
        assert False, "At least one model should be specified."

    # training and validation
    if args.do_cv:
        logger_.info("*** CROSS-VALIDATION ***")
        trainer_cv.cross_validation()
    if args.do_test:
        logger_.info("*** TEST ***")
        trainer_test.cross_validation()

    exit(0)


if __name__ == '__main__':
    main()


