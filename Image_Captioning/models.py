## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # maxpool that uses a square window of kernel_size=2, stride=2

        self.conv_1 = nn.Conv2d(1, 32, 5)
        
        self.conv_2 = nn.Conv2d(32, 64, 5)
        
        self.conv_3 = nn.Conv2d(64, 128, 5)
        
        self.conv_4 = nn.Conv2d(128, 256, 5)
        
        self.conv_5 = nn.Conv2d(256, 512, 5)
        

        self.pool_1 = nn.MaxPool2d(2, 2)
        
        self.pool_2 = nn.MaxPool2d(2, 2)
        
        self.pool_3 = nn.MaxPool2d(2, 2)
        
        self.pool_4 = nn.MaxPool2d(2, 2)
        
        self.pool_5 = nn.MaxPool2d(2, 2)
        
        
        

        self.dropout_1 = nn.Dropout(p=0.15)
        
        self.dropout_2 = nn.Dropout(p=0.15)
             
        self.dropout_3 = nn.Dropout(p=0.20)
       
        self.dropout_4 = nn.Dropout(p=0.25)

        self.dropout_5 = nn.Dropout(p=0.30)
        
        self.dropout_6 = nn.Dropout(p=0.25)
        
        self.dropout_7 = nn.Dropout(p=0.35)
        
        self.dropout_8 = nn.Dropout(p=0.25)
        
        

        self.fcnn_1 = nn.Linear(4608, 2304)
        
        self.fcnn_2 = nn.Linear(2304, 1152)        

        self.fcnn_3 = nn.Linear(1152, 576)
        
        self.fcnn_4 = nn.Linear(576, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))   (self.conv1_bn

        x = self.dropout_1(self.pool_1(F.relu(self.conv_1(x))))

        x = self.dropout_2(self.pool_2(F.relu(self.conv_2(x))))

        x = self.dropout_3(self.pool_3(F.relu(self.conv_3(x))))

        x = self.dropout_4(self.pool_4(F.relu(self.conv_4(x))))

        x = self.dropout_5(self.pool_5(F.relu(self.conv_5(x))))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)

        # one linear layer
        x = self.dropout_6(F.relu(self.fcnn_1(x)))

        x = self.dropout_7(F.relu(self.fcnn_2(x)))

        x = self.dropout_8(F.relu(self.fcnn_3(x)))

        x = self.fcnn_4(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

