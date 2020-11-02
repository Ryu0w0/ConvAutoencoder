from torch import nn
from models.augoencoder.conv_auto_en import ConvolutionalAutoEncoder
from models.cnn.cnn import CNN
from utils.multiple_optimizer import MultipleOptimizer


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cae = ConvolutionalAutoEncoder(config).double()
        self.cnn = CNN(config).double()

    def get_optimizer(self):
        return MultipleOptimizer(opt_cae=self.cae.get_optimizer(), opt_cnn=self.cnn.get_optimizer())

    def forward(self, x):
        x_hidden, x_auto_en = self.cae(x)
        x_cnn = self.cnn(x_hidden)
        return x_auto_en, x_cnn
