import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_device('cuda')

class Learner:
    def __init__(self, c_parameters, q_parameters):
        self.criterion = nn.CrossEntropyLoss()
        if c_parameters is None:
            parameters = list(q_parameters)
        else:
            parameters = list(c_parameters) + list(q_parameters)
        self.optimizer = optim.SGD(parameters, lr=0.001, momentum=0.0)
        #self.optimizer = optim.Adam(parameters, lr=0.0001)
        #self.optimizer = optim.Rprop(parameters)

        self.choosen_idx = -1

    def train(self, output, target):
        # print('--------------------------')
        self.optimizer.zero_grad()
        loss = self.criterion(output, target).detach() #Avoid inplace error while training
        
        loss.requires_grad = True
        loss.backward()
        #print('Loss : ')
        #print(loss.item())
        self.optimizer.step()

        return loss.item()