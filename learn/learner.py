import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_device('cuda')

class Learner:
    def __init__(self, c_parameters, q_parameters):
        self.criterion = nn.CrossEntropyLoss()

        if c_parameters is None: parameters = list(q_parameters)
        else: parameters = list(c_parameters) + list(q_parameters)

        self.optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)

        self.choosen_idx = -1

    def train(self, output, target):
        self.optimizer.zero_grad()

        loss = self.criterion(output, target).detach() #Avoid inplace error while training
        
        loss.requires_grad = True
        loss.backward()
        self.optimizer.step()

        return loss.item()