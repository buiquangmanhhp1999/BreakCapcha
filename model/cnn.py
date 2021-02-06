from torch import nn
from model.resnet18 import resnet_18


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = resnet_18()

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
