from torch import nn
from models.augoencoder.conv_auto_en import ConvolutionalAutoEncoder
from models.cnn.cnn import CNN
from utils.multiple_optimizer import MultipleOptimizer


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cae = ConvolutionalAutoEncoder(config).double() if self.config["use_cae"] else None
        self.cnn = CNN(config).double() if self.config["use_cnn"] else None

    def get_optimizer(self):
        # create optimizer for autoencoder and cnn respectively as needed
        if self.config["use_cae"] and self.config["use_cnn"]:
            return MultipleOptimizer([self.cae.get_optimizer(),
                                      self.cnn.get_optimizer()])
        elif self.config["use_cnn"]:
            return self.cnn.get_optimizer()
        elif self.config["use_cae"]:
            return self.cae.get_optimizer()
        else:
            assert False, "At least one model should be specified."

    def forward(self, x):
        if self.config["use_cae"] and self.config["use_cnn"]:
            x_hidden, x_auto_en = self.cae(x)
            x_cnn = self.cnn(x_hidden)
            return x_auto_en, x_cnn
        elif self.config["use_cnn"]:
            x = self.cnn(x)
            return x
        elif self.config["use_cae"]:
            _, x_auto_en = self.cae(x)
            return x_auto_en
        else:
            assert False, "At least one model should be specified."
