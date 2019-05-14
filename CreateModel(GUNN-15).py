import torch 
import torch.nn as nn
import math
from math import sqrt
from gunn_layer import GUNN_layer
import torch.nn.functional as F
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

   
class GUNN15(nn.Module):
    expansion = 1
    def CreateModel(self, opt):
        super(GUNN15, self).__init__() 

        cfg = [240, 300, 360]
        stg = [20, 25, 30]
        self.opt = opt
        self.layers = []
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layers += [self.conv1, nn.BatchNorm2d(64), nn.ReLU(inplace=True)]

        self.conv2 = nn.Conv2d(64, cfg[1], kernel_size=1, stride=1, padding=0)
        self.layers += [self.conv2, nn.BatchNorm2d(cfg[1]), nn.ReLU(inplace=True), GUNN_layer(cfg[1], stg[1])]
    
        self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, stride=1, padding=0)
        self.layers += [self.conv3, nn.BatchNorm2d(cfg[2]), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=1, padding=0), GUNN_layer(cfg[2], stg[2])]
    
        self.conv4 = nn.Conv2d(cfg[2], cfg[3], kernel_size=1, stride=1, padding=0)
        self.layers += [self.conv4, nn.BatchNorm2d(360), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=1, padding=0), GUNN_layer(cfg[3], stg[3])]
    
        self.conv5 = nn.Conv2d(cfg[3], cfg[3], kernel_size=1, stride=1, padding=0)
        self.layers += [self.conv5, nn.BatchNorm2d(360), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=8, stride=1, padding=0), Reshape(cfg[3])]
        self.linear = nn.Linear(cfg[3]*self.expansion, 10)
        self.model = nn.Sequential(*self.layers, self.linear)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out += self.model(x) 
        out = self.linear(out)
        return out
        
        modelParam, np = self.model.parameters(), 0

        print(self.model.format('| number of parameters: %d', np))
        return self.model
        

    
        
