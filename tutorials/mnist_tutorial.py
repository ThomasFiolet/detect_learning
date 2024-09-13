from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import math
from IPython.core.debugger import set_trace

#Classes
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        #self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        #self.bias = nn.Parameter(torch.zeros(10))
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        #return xb @ self.weights + self.bias
        return self.lin(xb)

#Loss function
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

#Loss function
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

#Forward
# def model(xb):
#     #return log_softmax(xb @ weights + bias)
#     return xb @ weights + bias

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

#Training
lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            #         set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            #with torch.no_grad():
                # weights -= weights.grad * lr
                # bias -= bias.grad * lr
                # weights.grad.zero_()
                # bias.grad.zero_()
                #for p in model.parameters(): p -= p.grad * lr
                #model.zero_grad()
            opt.step()
            opt.zero_grad()

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape

#Neural Network from scratch
# weights = torch.randn(784, 10) / math.sqrt(784)
# weights.requires_grad_()
# bias = torch.zeros(10, requires_grad=True)

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape

#loss_func = nll
loss_func = F.cross_entropy

yb = y_train[0:bs]

#model = Mnist_Logistic()

fit()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
