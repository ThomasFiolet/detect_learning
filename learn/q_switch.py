import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math as m
torch.autograd.set_detect_anomaly(True)

torch.set_default_device('cuda')

class QSwitch(nn.Module):

    def __init__(self, n_inputs, n_outputs, activation_function):
        super(QSwitch, self).__init__()
        torch.cuda.manual_seed_all(time.time())

        #self.linear1 = nn.Linear(n_inputs, 50)
        self.linear1 = nn.Linear(n_inputs, 100)
        #nn.init.uniform_(self.linear1.weight, a=0.0, b=1.0)
        nn.init.normal_(self.linear1.weight, mean=0.0, std=m.sqrt(2/(5*29*29)))
        #nn.init.constant_(self.linear1.weight, 0)
        self.activation1 = activation_function

        self.linear2 = nn.Linear(100, n_outputs)
        #nn.init.uniform_(self.linear2.weight, a=0.0, b=1.0)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=m.sqrt(2/(50)))
        #nn.init.constant_(self.linear2.weight, 0)
        self.activation2 = activation_function

        self.last_prediction = torch.full((1, n_outputs), 1/n_outputs)

    def forward(self, input):

        x = self.linear1(input)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)

        self.last_prediction = x

        return x
