from torch import nn, optim


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            # 1st block, 32x32 to 16x16
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=32),
            # HIDDEN, 16x16 to 8x8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=64)
        )
        self.decoder = nn.Sequential(
            # 1st block, 8x8 to 16x16
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2),
            # 2nd block, 16x16 to 32x32
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.LeakyReLU(0.2)
        )

    def get_optimizer(self):
        lr = self.config["opt"]["lr"]
        beta1, beta2 = self.config["opt"]["betas"]
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                               betas=(beta1, beta2), eps=1e-8)
        return optimizer

    def forward(self, x):
        code = self.encoder(x)
        reconstruct = self.decoder(code)
        return code, reconstruct


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
