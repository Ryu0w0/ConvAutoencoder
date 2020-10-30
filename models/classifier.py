from torch import nn
from models.augoencoder.conv_auto_en import ConvolutionalAutoEncoder
from models.cnn.cnn import CNN
from utils.multiple_optimizer import MultipleOptimizer


class Classifier(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.conv_auto_en = ConvolutionalAutoEncoder(config).to(device) if self.config["use_cae"] else None
        self.cnn = CNN(config).double().to(device)

    def get_optimizer(self):
        # create optimizer for autoencoder and cnn respectively as needed
        if self.config["use_cae"]:
            return MultipleOptimizer([self.conv_auto_en.get_optimizer(),
                                      self.cnn.get_optimizer()])
        else:
            return self.cnn.get_optimizer()

    def forward(self, x):
        if self.config["use_cae"]:
            x_hidden, x_auto_en = self.conv_auto_en(x)
            x_cnn = self.cnn(x_hidden)
            return x_auto_en, x_cnn
        else:
            x = self.cnn(x)
            return x
