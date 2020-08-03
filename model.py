import torch
import random
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as Fvgg
import numpy as np
import torchvision

def vgg():
    vgg = torchvision.models.vgg11()
    vgg.features[0].in_channels = 1


def resnet():
    ResNet = torchvision.models.resnet101(num_classes=2)
    ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # ResNet.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    return ResNet

if __name__ == '__main__':
    print(resnet())
    for child in list(resnet().children())[:-1]:
        print(child)