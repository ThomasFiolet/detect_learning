import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math as m

torch.set_default_device('cuda')

class Conv(nn.Module):

    def __init__(self):
        super(Conv, self).__init__()

        #self.criterion = nn.MSELoss()
        torch.cuda.manual_seed_all(time.time())

        self.conv1 = nn.Conv2d(1, 3, 5)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=m.sqrt(2/(1*127*127*5*5)))
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(3, 5, 5)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=m.sqrt(2/(3*62*62*5*5)))
        self.activation2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(5, 7, 5)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=m.sqrt(2/(5*29*29*5*5)))
        self.activation3 = nn.ReLU()
    
    def forward(self, input):

        x = self.conv1(input)           #124
        x = self.activation1(x)
        x = F.max_pool2d(x, (2,2))   #62
        x = self.conv2(x)               #58
        x = self.activation2(x)
        x = F.max_pool2d(x, (2,2))   #29
        x = self.conv3(x)               #25
        x = self.activation3(x)
        x = F.max_pool2d(x, (2,2))   #12
        x = x.reshape([1, 7 * 12 * 12])

        return x
