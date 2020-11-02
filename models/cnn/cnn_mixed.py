import torch
from torch import nn, optim
import torch.nn.functional as F
from utils.seed import seed_everything


class MixedCNN(nn.Module):
    def __init__(self, config, num_class=10):
        super().__init__()
        self.config = config["cnn"]
        # feature extractor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.linear = nn.Linear(in_features=128 * (4 ** 2), out_features=num_class)
        # init weight
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

    def forward(self, x, code):
        # 32 to 16
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x_code = torch.cat((x, code), dim=1)
        x_code = F.leaky_relu(x_code, 0.2)
        x_code = self.bn1(x_code)
        # 16 to 8
        x_code = self.conv2(x_code)
        x_code = F.max_pool2d(x_code, kernel_size=2, stride=2)
        x_code = F.leaky_relu(x_code, 0.2)
        x_code = self.bn2(x_code)
        # 8 to 4
        x_code = self.conv3(x_code)
        x_code = F.max_pool2d(x_code, kernel_size=2, stride=2)
        x_code = F.leaky_relu(x_code, 0.2)
        x_code = self.bn3(x_code)
        # classifying
        x_code = self.linear(torch.flatten(x_code, 1))

        return x_code
