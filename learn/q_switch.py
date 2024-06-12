import torch
import torch.nn as nn
import torch.nn.functional as F

class QSwitch(nn.Module):

    def __init__(self, n_outputs):
        super(QSwitch, self).__init__()

        self.criterion = nn.MSELoss()

        self.fc1 = nn.Linear(16 * 121 * 121, 200)
        nn.init.uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(200, 50)
        nn.init.uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(50, n_outputs)
        nn.init.uniform_(self.fc3.weight)

        self.last_prediction = None

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))

        output = f3
        self.last_prediction = output
        return output
