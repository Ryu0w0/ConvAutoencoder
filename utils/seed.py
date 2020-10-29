import os
import random
import numpy as np
import torch

# GLOBAL VARIABLE
cur_seed = 1  # it links with cur_epoch, and epoch begins from 1, hence 1 is specified here
feed_seed = False


def seed_everything(target="all", local_seed=None, force_apply=False):
    """
    target: specify which are seeded
        Seeding torch functions is slower than others such as numpy.
        "random" should be specified if target is not related to torch
        to avoid unnecessary seeding.
    """
    if feed_seed or force_apply:
        seed = local_seed if local_seed is not None else cur_seed
        if target in ["torch", "all"]:
            torch.manual_seed(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        if target in ["random", "all"]:
            random.seed(seed)
            np.random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)

