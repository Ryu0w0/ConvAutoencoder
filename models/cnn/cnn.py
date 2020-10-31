from torch import nn, optim
from utils.logger import logger_
from utils.seed import seed_everything


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config["cnn"]
        self.main = nn.Sequential(
            # 1st block, output 16x16
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=32),
            # 2nd block, output 8x8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=64),
            # 3rd block, output 4x4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=128),
            # classifying
            nn.Flatten(),
            # nn.Dropout(),
            nn.Linear(in_features=4*4*128, out_features=10)
        )
        self.__initialize_weight()

    def __initialize_weight(self):
        for idx, m in enumerate(self.modules()):
            seed_everything(local_seed=idx)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                seed_everything(local_seed=idx)
                nn.init.constant_(m.bias.data, 0)

    def get_optimizer(self):
        lr = self.config["opt"]["lr"]
        beta1, beta2 = self.config["opt"]["betas"]
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                               betas=(beta1, beta2), eps=1e-8)
        return optimizer

    def forward(self, x):
        return self.main(x)

