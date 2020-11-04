from torch import nn
from models.augoencoder.conv_auto_en import ConvolutionalAutoEncoder
from models.cnn.cnn import CNN
from models.cnn.cnn_mixed import MixedCNN
from utils.multiple_optimizer import MultipleOptimizer


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cae = ConvolutionalAutoEncoder(config)
        if config["cnn"]["use_mixed_input"]:
            self.cnn = MixedCNN(config)
        else:
            self.cnn = CNN(config)

    def get_optimizer(self):
        return MultipleOptimizer(opt_cae=self.cae.get_optimizer(), opt_cnn=self.cnn.get_optimizer())

    def forward(self, x):
        x_hidden, x_auto_en = self.cae(x)
        if isinstance(self.cnn, MixedCNN):
            x_cnn = self.cnn(x, x_hidden)
        else:
            x_cnn = self.cnn(x_hidden)
        return x_auto_en, x_cnn
