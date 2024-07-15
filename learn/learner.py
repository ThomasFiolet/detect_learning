import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_device('cuda')

class Learner:
    def __init__(self, c_parameters, q_parameters):
        self.criterion = nn.CrossEntropyLoss()
        parameters = list(c_parameters) + list(q_parameters)
        self.optimizer = optim.SGD(parameters, lr=0.9, momentum=0.0)
        self.choosen_idx = -1

    def train(self, output, reward):
        target = torch.clone(output)
        # for k, t in enumerate(target):
        #     target[:][k] = max(0, 13 - reward)
        #     if k is self.choosen_idx:
        #         target[:][k] = reward
        if self.choosen_idx >= 0 and self.choosen_idx < len(target):
            target[:][self.choosen_idx] = reward
        else:
            idx = torch.argmin(target)
            target[:][idx] = reward
        
        # print(output)
        # print(target)
        # print('--------------------------')
        #self.optimizer.zero_grad()
        loss = self.criterion(output, target).detach() #Avoid inplace error while training
        
        loss.requires_grad = True
        loss.backward(retain_graph=True)
        #print('Loss : ' + str(loss.item()))
        self.optimizer.step()