import torch
from torch import nn
from torch.nn import functional as F

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU())
        
    def forward(self, x):
        x = self.conv(x)
        return x



class CNNEncorder(nn.Module):
    def __init__(self, inc, start_fm):
        super(CNNEncorder, self).__init__()
        self.double_conv1 = double_conv(inc, start_fm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.double_conv2 = double_conv(start_fm, start_fm * 2)
        #Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        #Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4)
        #Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        #Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.double_conv1(x)
        x = self.maxpool1(x)

        x = self.double_conv2(x)
        x = self.maxpool2(x)

        x = self.double_conv3(x)
        x = self.maxpool3(x)

        x = self.double_conv4(x)
        x = self.maxpool4(x) # torch.Size([1, 512, 14, 14])
        return self.avgpool(x) # torch.Size([1, 512, 1, 1])



