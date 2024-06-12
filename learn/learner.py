import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Learner:
    def __init__(self, c_parameters, q_parameters):
        self.criterion = nn.MSELoss()
        parameters = list(q_parameters) + list(c_parameters)
        self.optimizer = optim.SGD(parameters, lr=0.9)

    def train(self, output, reward):
        target = output
        idx = torch.argmax(target)
        target[idx] = reward
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()