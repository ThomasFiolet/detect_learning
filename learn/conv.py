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
        #nn.init.constant_(self.conv1.weight, 0)
        #nn.init.uniform_(self.conv1.weight, a=-1.0, b=1.0)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=m.sqrt(2/(1*128*128*5*5)))
        self.activation1 = nn.Softplus()
        
        self.conv2 = nn.Conv2d(3, 5, 5)
        #nn.init.constant_(self.conv2.weight, 0)
        #nn.init.uniform_(self.conv2.weight, a=-1.0, b=1.0)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=m.sqrt(2/(3*62*62*5*5)))
        self.activation2 = nn.Softplus()
    
    def forward(self, input):

        x = self.conv1(input)           #124
        #print("x : " + str(x))
        x = self.activation1(x)
        #print("x : " + str(x))
        x = F.max_pool2d(x, (2,2))   #62
        #print("x : " + str(x))
        x = self.conv2(x)               #58
        #print("x : " + str(x))
        x = self.activation2(x)
        #print("x : " + str(x))
        x = F.max_pool2d(x, (2,2))   #29
        #print("x : " + str(x))
        x = x.reshape([1, 5 * 29 * 29])
        #print("x : " + str(x))
        
        #128
        #c1 = m(self.conv1(input))
        #124
        #s1 = F.max_pool2d(c1, (2,2))
        #62
        #c2 = m(self.conv2(s1))
        #58
        #s2 = F.max_pool2d(c2, (2,2))
        #29
        #c3 = m(self.conv3(s2))
        #25
        #c4 = m(self.conv4(c3))
        #21
        #c_im = c3.reshape([1, 1 * 25 * 25])

        #f1 = m(self.fc1(c_im))
        #f2 = m(self.fc2(f1))
        #f3 = m(self.fc3(f2))
        #f4 = m(self.fc4(f3))

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
        return x
