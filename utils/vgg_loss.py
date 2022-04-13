import math
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
import torch.nn.functional as F


class Vgg16FeatureExtractor(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16FeatureExtractor, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        # h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        vgg_outputs = namedtuple("VggOutputs", ["relu2_2", "relu3_3", "relu4_3"])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        out = vgg_outputs(h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def vgg_loss(X, Y, device):
    loss_net = Vgg16FeatureExtractor()
    loss_net = loss_net.to(device)
    feat_X = loss_net(X)
    feat_Y = loss_net(Y)

    loss = F.mse_loss(feat_X.relu2_2, feat_Y.relu2_2)
    loss = loss + F.mse_loss(feat_X.relu4_3, feat_Y.relu4_3)

    return loss