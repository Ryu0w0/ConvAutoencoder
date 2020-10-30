from torch import nn, optim
from utils.logger import logger_


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        )

    def get_optimizer(self):
        logger_.info("*** CNN OPTIMIZER ***")
        lr = self.config["opt"]["lr"]
        beta1, beta2 = self.config["opt"]["betas"]
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                               betas=(beta1, beta2), eps=1e-8)
        logger_.info(self.optimizer)
        return optimizer

    def forward(self, x):
        return x
