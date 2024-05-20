import torch
import torch.nn as nn
import torch.nn.functional as F

class QSwitch(nn.Module):

    def __init__(self, n_outputs):
        super(QSwitch, self).__init__()

        self.criterion = nn.MSELoss()

        self.conv1 = nn.Conv2d(1, 4, 5)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)

        self.fc1 = nn.Linear(16 * 61 * 61, 200)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, n_outputs)
    
    def forward(self, input):
        #Input image size : 516*516
        c1 = F.relu(self.conv1(input))

        s2 = F.max_pool2d(c1, (2, 2))
        c2 = F.relu(self.conv2(s2))

        s3 = F.max_pool2d(c2, (2,2))
        c3 = F.relu(self.conv3(s3))

        s4 = F.max_pool2d(c3, (2,2))
        s4 = torch.flatten(s4, 1)
        f1 = F.relu(self.fc1(s4))

        f2 = F.relu(self.fc2(f1))

        f3 = F.relu(self.fc3(f2))

        output = f3
        return output
