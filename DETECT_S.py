#LIBRARIES
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import cv2 as cv2
cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import tesserocr
import zxingcpp

from agent import Agent
from utils import read_files
from utils import read_join_dataset
from utils import read_functions
from utils import sort_training_test
from utils import sort_no_training
from utils import zxing
from utils import tesser
from utils import zbar
from utils import iter_extract
from utils import indx_extract

#PARAMETERS
dataset_size = 105
training_size = 60
testing_size = 45
criterion = nn.CrossEntropyLoss()
EPOCH = 10

#VARIABLES
Agent = Agent()

#PREPARING DATA
dataset_list = ['real']
images, ground_truth, len_files = read_join_dataset(dataset_list)
images, ground_truth = sort_no_training(images, ground_truth)
images_check = [0] * len(images)
index = 0


#MAIN LOOP
while index < 1:
#while index < len(images):
    print("INDEX : " + str(index))
    im = images[index]
    gt = ground_truth[index]

    #Data Gathering
    Agent.read_data(im, gt, index)

    #State Estimation
    Agent.estimate_state()

    #Ethical Evaluation
    Agent.evaluate_score()

    #Training
    if Agent.TRAINING is True:
        print("|-----TRAINING")

        #EXTERNAL TRAINING
        output = Agent.action_output
        target = torch.clone(Agent.action_output)

        succ = iter(Agent.action_list)
        last_action = Agent.next_action
        oidx = indx_extract(succ, last_action)
        target[0][oidx] = Agent.score
        
        #for i in range(0, EPOCH):
        parameters = list(Agent.conv_net.parameters()) + list(Agent.action_decider.parameters())
        optimizer = optim.SGD(parameters, lr=0.001, momentum=0.0)
        loss_action = criterion(output, target)
        loss_action.backward()
        optimizer.step()
        optimizer.zero_grad()

        #INTERNAL TRAINING
        output = Agent.reader_output
        target = torch.clone(Agent.reader_output)

        succ = iter(Agent.barrecode_reader_list)
        last_reader = Agent.next_reader
        oidx = indx_extract(succ, last_reader)
        target[0][oidx] = Agent.score
        
        #for i in range(0, EPOCH):
        parameters = list(Agent.conv_net.parameters()) + list(Agent.reader_decider.parameters())
        optimizer = optim.SGD(parameters, lr=0.001, momentum=0.0)
        loss_reader = criterion(output, target)
        loss_reader.backward()
        optimizer.step()
        optimizer.zero_grad()

    #Path Planning
    Agent.plan_path()

    #Action
    images[index], index = Agent.do_actions()