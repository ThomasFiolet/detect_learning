from os import listdir
from os.path import join, isfile
import random
import math

import cv2 as cv2
import numpy as np
import torch
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import read_functions
from utils import iter_extract
from metrics import reward

def dectect_learning(training_set, training_label, spl, conv_net):

    PIPE = 1
    SOURCE = 0
    SINK = 2

    pipeline_list = []

    #Generate Random Pipelines
    while len(pipeline_list) < len(training_set):
        ppl = Pipeline()
        ppl.zero_data()
        spl.current_node = "im = im_g"
        ppl.append(spl.current_node)

        while spl.graph.nodes[spl.current_node]['subset']  != SINK :
            idx = random.randrange(0, sum(1 for _ in spl.graph.successors(spl.current_node)))
            succ = spl.graph.successors(spl.current_node)
            spl.current_node = iter_extract(succ, idx)
            ppl.append(spl.current_node)

        if list(set(pipeline_list) & set([ppl])) is []:
            pipeline_list.append(ppl)

