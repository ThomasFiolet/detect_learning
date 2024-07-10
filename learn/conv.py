import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.set_default_device('cuda')

class Conv(nn.Module):

    def __init__(self):
        super(Conv, self).__init__()

        #self.criterion = nn.MSELoss()
        torch.cuda.manual_seed_all(time.time())

        self.conv1 = nn.Conv2d(1, 1, 5)
        nn.init.normal_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(1, 1, 5)
        nn.init.normal_(self.conv2.weight)
        
        self.conv3 = nn.Conv2d(1, 1, 5)
        nn.init.normal_(self.conv3.weight)
        
        #self.conv4 = nn.Conv2d(1, 1, 5)
        #nn.init.constant_(self.conv4.weight, 0)
    
    def forward(self, input):

        m = nn.Sigmoid()
        
        #128
        c1 = m(self.conv1(input))
        #124
        s1 = F.max_pool2d(c1, (2,2))
        #62
        c2 = m(self.conv2(s1))
        #58
        s2 = F.max_pool2d(c2, (2,2))
        #29
        c3 = m(self.conv3(s2))
        #25
        #c4 = m(self.conv4(c3))
        #21
        c_im = c3.reshape([1, 1 * 25 * 25])


        # #514
        # c1 = F.relu(self.conv1(input))
        # #508
        # s1 = F.max_pool2d(c1, (2,2))
        # #254
        # c2 = F.relu(self.conv2(s1))
        # #248
        # s2 = F.max_pool2d(c2, (2,2))
        # #124
        # c3 = F.relu(self.conv3(s2))
        # #118
        # c_im = c3.reshape([1, 1 * 118 * 118])
        
        #print("Output :")
        #print(output)
        return c_im
