import torch
import torch.nn as nn
import torch.nn.functional as F
import time
torch.autograd.set_detect_anomaly(True)

torch.set_default_device('cuda')

class QSwitch(nn.Module):

    def __init__(self, n_inputs, n_outputs, isConvNet):
        super(QSwitch, self).__init__()

        #self.criterion = nn.MSELoss()
        torch.cuda.manual_seed_all(time.time())

        self.fc4 = nn.Linear(n_inputs, 10)
        #nn.init.constant_(self.fc4.weight, 0)
        nn.init.uniform_(self.fc4.weight, a=0, b=1.0)

        self.fc5 = nn.Linear(10, n_outputs)
        #nn.init.constant_(self.fc5.weight, 0)
        nn.init.uniform_(self.fc5.weight, a=0, b=1.0)

        #self.layer = nn.Linear(1 * 25 * 25, n_outputs)
        #nn.init.constant_(self.layer.weight, 0)

        self.last_prediction = torch.full((1, n_outputs), 1/n_outputs)

        #self.FORWARDED = 0

    def forward(self, input):

        #m = nn.Sigmoid()
        m = nn.Softplus()

        f4 = m(self.fc4(input))
        f5 = m(self.fc5(f4))

        output = f5
        self.last_prediction = output
        #print("Last Prediction : ")
        #print(self.last_prediction)
        #self.FORWARDED = 1
        return output
