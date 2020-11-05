from torch import nn, optim
from utils.seed import seed_everything


class CNN(nn.Module):
    def __init__(self, config, num_class=10):
        super().__init__()
        self.config = config["cnn"]
        self.main = self.main = nn.Sequential(*self.__build_structure(num_class))
        self.__initialize_weight()

    def __build_structure(self, num_class):
        input_resolution = self.config["input_resolution"]
        components = []
        # feature extractor
        for from_, to_, use_pool in self.config["block_sizes"]:
            components.append(nn.Conv2d(in_channels=from_, out_channels=to_, kernel_size=3, stride=1, padding=1))
            if use_pool:
                components.append(nn.MaxPool2d(kernel_size=2, stride=2))
            components.append(nn.LeakyReLU(0.2))
            components.append(nn.BatchNorm2d(num_features=to_))
        # classifying
        components.append(nn.Flatten())
        num_pooling = len([comp for comp in components if isinstance(comp, nn.MaxPool2d)])
        last_resolution = int((input_resolution / (2 ** num_pooling)))
        last_depth = self.config["block_sizes"][-1][1]
        components.append(nn.Linear(in_features=last_depth * last_resolution ** 2, out_features=num_class))
        return components

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

