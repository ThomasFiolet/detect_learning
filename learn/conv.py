
import time
import math as m

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv2

torch.set_default_device('cuda')

class Conv(nn.Module):

    def __init__(self, activation):
        super(Conv, self).__init__()

        #self.criterion = nn.MSELoss()
        torch.cuda.manual_seed_all(time.time())

        self.conv1 = nn.Conv2d(1, 3, 5)
        #nn.init.uniform_(self.conv1.weight, -1.0, 1.0)
        #nn.init.normal_(self.conv1.weight, mean=0.0, std=m.sqrt(2/(1*127*127*5*5)))
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.activation1 = activation

        self.conv2 = nn.Conv2d(3, 5, 5)
        #nn.init.uniform_(self.conv2.weight, -1.0, 1.0)
        #nn.init.normal_(self.conv2.weight, mean=0.0, std=m.sqrt(2/(3*62*62*5*5)))
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.activation2 = activation
        
        #self.conv3 = nn.Conv2d(5, 7, 5)
        #nn.init.uniform_(self.conv3.weight, -1.0, 1.0)
        #nn.init.normal_(self.conv3.weight, mean=0.0, std=m.sqrt(2/(5*29*29*5*5)))
        #self.activation3 = activation

        # self.conv4 = nn.Conv2d(7, 9, 5)
        # nn.init.normal_(self.conv4.weight, mean=0.0, std=m.sqrt(2/(7*12*12*5*5)))
        # self.activation4 = activation
    
    def forward(self, input):

        tensor = input.cpu().numpy() # make sure tensor is on cpu
        cv2.imwrite("input.png", tensor[0]*255)

        x = self.conv1(input)           #124 #508
        x = self.activation1(x)
        x = F.max_pool2d(x, (2,2))      #62  #254

        tensor = x.detach().cpu().numpy() # make sure tensor is on cpu
        cv2.imwrite("conv_1_1.png", tensor[0]*255)
        cv2.imwrite("conv_1_2.png", tensor[1]*255)
        cv2.imwrite("conv_1_3.png", tensor[2]*255)

        x = self.conv2(x)               #58  #250
        x = self.activation2(x)
        x = F.max_pool2d(x, (2,2))      #29  #125

        tensor = x.detach().cpu().numpy() # make sure tensor is on cpu
        cv2.imwrite("conv_2_1.png", tensor[0]*255)
        cv2.imwrite("conv_2_2.png", tensor[1]*255)
        cv2.imwrite("conv_2_3.png", tensor[2]*255)
        cv2.imwrite("conv_2_4.png", tensor[3]*255)
        cv2.imwrite("conv_2_5.png", tensor[4]*255)

        # x = self.conv3(x)               #25  #121
        # x = self.activation3(x)
        # x = F.max_pool2d(x, (2,2))      #12  #60

        # tensor = x.detach().cpu().numpy() # make sure tensor is on cpu
        # cv2.imwrite("conv_3_1.png", tensor[0]*255)
        # cv2.imwrite("conv_3_2.png", tensor[1]*255)
        # cv2.imwrite("conv_3_3.png", tensor[2]*255)
        # cv2.imwrite("conv_3_4.png", tensor[3]*255)
        # cv2.imwrite("conv_3_5.png", tensor[4]*255)
        # cv2.imwrite("conv_3_6.png", tensor[5]*255)
        # cv2.imwrite("conv_3_7.png", tensor[6]*255)

        # x = self.conv4(x)             #8
        # x = self.activation4(x)
        # x = F.max_pool2d(x, (2,2))    #4
        x = x.reshape([1, 5 * 29 * 29])

        return x
