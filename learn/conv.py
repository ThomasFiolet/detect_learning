import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_device('cuda')

class Conv(nn.Module):

    def __init__(self):
        super(Conv, self).__init__()

        self.criterion = nn.MSELoss()

        self.conv1 = nn.Conv2d(1, 4, 5)
        nn.init.uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(4, 8, 5)
        nn.init.uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(8, 16, 5)
        nn.init.uniform_(self.conv2.weight)
    
    def forward(self, input):
        #512
        c1 = F.softmax(self.conv1(input))
        #508
        s1 = F.max_pool2d(c1, (2,2))
        #254
        c2 = F.softmax(self.conv2(s1))
        #250
        s2 = F.max_pool2d(c2, (2,2))
        #125
        c3 = F.softmax(self.conv3(s2))
        #121
        c_im = c3.reshape([1, 16 * 121 * 121])
        #c_im = c3
        
        #print("Output :")
        #print(output)
        return c_im
