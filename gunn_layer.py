from __future__ import print_function
import torch 
import torch.nn as nn 
from torch.nn import init 
from data import get_data 
import copy


"""GUNN layer as was described in the paper"""

class GUNN_layer(nn.Module):
    expansion = 1

    def __init__ (self, nChannels, nSegments):

        super(GUNN_layer, self).__init__()              
        self.nChannels = nChannels
        self.nSegments = nSegments 
        assert (nChannels%nSegments == 0)
        oChannels = nChannels / nSegments
        self.oChannels = oChannels
        self.modules = []
        self.shortcutModules = []
        self.inputTable = {}
        self.outputTable = {}
        self.InputContigous = []
        self.gradInputContigous = []
        self.sharedGradInput = []
        self.model = []

        self.convLayer = nn.Sequential(
            nn.Conv2d (nChannels, oChannels*2, kernel_size =1, stride =1, padding =0),
            nn.BatchNorm2d (oChannels*2),
            nn.ReLU (inplace=True),
            nn.Conv2d (oChannels*2, oChannels*2, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm2d (oChannels*2),
            nn.ReLU (inplace=True),
            nn.Conv2d (oChannels*2, self.expansion * oChannels, kernel_size =1, stride =1, padding =0),
            nn.BatchNorm2d (self.expansion * oChannels))
        self.modules.append(self.convLayer)

        self.shortcut = nn.Sequential(
            nn.Conv2d (nChannels, self.expansion * oChannels, kernel_size =1, stride =1, padding =0),
            nn.BatchNorm2d (self.expansion*oChannels))
        self.shortcutModules.append(self.shortcut)

        self.model.append(self.modules, self.shortcutModules)



    """Function to implement Forward update
        in Neural Networks 
    """
    def updateOutput (self, x):
        nSegments = self.nSegments
        self.inputTable = []
        self.outputTable = []
        self.InputContigous = copy.copy(x)
        self.gradInputContigous = []
        self.sharedGradInput = []
        self.output =[]

        for i in range(nSegments):
            self.inputTable.append(self.InputContigous.narrow(x, 2, i*self.oChannels, self.expansion * self.oChannels))
            self.sharedGradInput = copy.copy(self.inputTable)

        for i in range(nSegments):
            layer = self.modules[i]
            inputTable = {}
            for j in range(1, i):
                inputTable[j] = self.outputTable[j]
        
            for j in range(i, nSegments):
                inputTable[j] = self.inputTable[j]
            
            self.outputTable[i] = layer.forward(self.InputContigous, self.sharedGradInput)      
            self.output += [self.outputTable[i], self.inputTable[i]]    
        return self.output  
    
    def backward (self, x, output):
        nSegments = self.nSegments
        self.InputContigous = copy.copy(x)
        self.sharedGradInput = []
        self.gradInputContiguous = copy.copy(output)
        self.output_layer = []
        for i in (nSegments):
        
            layer = self.modules[i]
            inputTable = []
            for j in range(1, i):
                inputTable[j] = self.outputTable[j]
        
            for j in range(i, nSegments):
                inputTable[j] = self.inputTable[j]
        
            
            self.inputTable.append(self.gradInputContiguous.narrow(x, 2, (1 + ((i - 1) * self.oChannels)), self.expansion * self.oChannels))
            
            self.sharedGradInput = copy.copy(self.inputTable)
            self.netGradInput[i] = layer.backward(self.InputContiguous, self.sharedGradInput)
            self.output_layer += [self.gradInputContiguous[i], self.netGradInput[i]]
    
        return self.output_layer



        