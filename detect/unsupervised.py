from os import listdir
from os.path import join, isfile
import random
import math

import cv2 as cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_device('cuda')
import torchvision.transforms as transforms
import networkx as nx

cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import PIL
from PIL import Image, ImageOps
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'
import tesserocr
import zxingcpp

from utils import zxing
from utils import tesser
from utils import zbar

from graph import Sample
from graph import Pipeline
from learn import Conv
from utils import read_functions
from utils import iter_extract
from metrics import reward
from metrics import compute_image_metrics

def detect_unsupervised(im_g, spl, conv_net):

    PIPE = 1
    SOURCE = 0
    SINK = 2

    pipeline = Pipeline()

    down_width = 128
    down_height = 128
    down_points = (down_width, down_height)

    im = im_g

    spl.current_node = "im = im_g"
    spl.graph.nodes[spl.current_node]['nuse'] += 1
    pipeline.append(spl.current_node)
    while spl.graph.nodes[spl.current_node]['subset']  != SINK :
        exec(spl.current_node)
        im_p = im
        im_s = cv2.resize(im_p, down_points, interpolation= cv2.INTER_LINEAR)
        im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        c_im = conv_net.forward(im_t)
        idx = torch.argmin(spl.graph.nodes[spl.current_node]['QTable'].forward(c_im))
        idx = idx.item()
        succ = spl.graph.successors(spl.current_node)
        spl.current_node = iter_extract(succ, idx)
        pipeline.append(spl.current_node)
        spl.graph.nodes[spl.current_node]['nuse'] += 1

    pipeline.browse(im_g)
    pipeline.score(None)
    barre_code = pipeline.barre_code

    return barre_code