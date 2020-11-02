from torch import nn, optim


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, config):
        self.config = config["cae"]
        super().__init__()
        enc_components = []
        dec_components = []
        for from_, to_ in self.config["enc_block_sizes"]:
            enc_components.extend([
                nn.Conv2d(in_channels=from_, out_channels=to_, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(num_features=to_),
            ])
        self.encoder = nn.Sequential(*enc_components)
        for from_, to_ in self.config["dec_block_sizes"]:
            dec_components.extend([
                nn.ConvTranspose2d(in_channels=from_, out_channels=to_, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=to_),
                nn.LeakyReLU(0.2)
            ])
        self.decoder = nn.Sequential(*dec_components)

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
