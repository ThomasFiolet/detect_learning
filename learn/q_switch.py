import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

torch.set_default_device('cuda')

class QSwitch(nn.Module):

    def __init__(self, n_outputs):
        super(QSwitch, self).__init__()

        self.criterion = nn.MSELoss()

        self.fc1 = nn.Linear(1 * 21 * 21, 50)
        nn.init.uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(50, 50)
        nn.init.uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(50, 50)
        nn.init.uniform_(self.fc3.weight)

        self.fc4 = nn.Linear(50, 50)
        nn.init.uniform_(self.fc4.weight)

        self.fc5 = nn.Linear(50, 50)
        nn.init.uniform_(self.fc5.weight)

        self.fc6 = nn.Linear(50, n_outputs)
        nn.init.uniform_(self.fc6.weight)

        self.last_prediction = torch.full((1, n_outputs), 1/n_outputs)

        #self.FORWARDED = 0

    def forward(self, input):

        m = nn.Sigmoid()

        f1 = m(self.fc1(input))
        f2 = m(self.fc2(f1))
        f3 = m(self.fc3(f2))
        f4 = m(self.fc4(f3))
        f5 = m(self.fc5(f4))
        f6 = 13*m(self.fc6(f5))

        output = f6
        self.last_prediction = output[0]
        #print("Last Prediction : ")
        #print(self.last_prediction)
        #self.FORWARDED = 1
        return output
