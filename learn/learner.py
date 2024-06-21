import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_device('cuda')

class Learner:
    def __init__(self, c_parameters, q_parameters):
        self.criterion = nn.MSELoss()
        parameters = list(c_parameters) + list(q_parameters)
        self.optimizer = optim.SGD(parameters, lr=0.9)

    def train(self, output, reward):
        target = torch.clone(output)
        idx = torch.argmax(target)
        # print("Target :")
        # print(target.shape)
        # print(target)
        target[:][idx] = reward
        
        # print("Target reward:")
        # print(target.shape)
        # print(target)
        # out = torch.clone(output)
        # tar = torch.clone(target)
        loss = self.criterion(output, target).detach() #Avoid inplace error while training
        loss.requires_grad = True
        loss.backward(retain_graph=True)
        self.optimizer.step()