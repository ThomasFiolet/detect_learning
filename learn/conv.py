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
        nn.init.constant_(self.conv1.weight, 0)
        
        self.conv2 = nn.Conv2d(1, 1, 5)
        nn.init.constant_(self.conv2.weight, 0)
        
        self.conv3 = nn.Conv2d(1, 1, 5)
        nn.init.constant_(self.conv3.weight, 0)

        self.fc1 = nn.Linear(1 * 25 * 25, 30)
        nn.init.constant_(self.fc1.weight, 0)

        self.fc2 = nn.Linear(30, 30)
        nn.init.constant_(self.fc2.weight, 0)

        self.fc3 = nn.Linear(30, 30)
        nn.init.constant_(self.fc3.weight, 0)

        self.fc4 = nn.Linear(30, 30)
        nn.init.constant_(self.fc4.weight, 0)
        
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

        f1 = m(self.fc1(c_im))
        f2 = m(self.fc2(f1))
        f3 = m(self.fc3(f2))
        f4 = m(self.fc4(f3))

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
        return f4
