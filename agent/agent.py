#LIBRARIES
import numpy as np
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
import PIL
from PIL import Image, ImageOps
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'
import tesserocr
import zxingcpp

from learn import Conv
from learn import QSwitch
from metrics import reward
from utils import iter_extract
from utils import indx_extract
from utils import zxing
from utils import tesser
from utils import zbar

#PARAMETERS
#activation_function = nn.Softplus()
activation_function = nn.ReLU()
down_width = 128
down_height = 128
down_points = (down_width, down_height)

#CLASS
class Agent():
    #Main Functions
    def __init__(self):
        print("Initializing Agent")

        self.TRAINING = False

        #Data
        self.image = None
        self.ground_truth = None
        self.index = 0

        #State
        self.barrecode_reader_list = ["self.barre_code = zxing(im, zxingcpp.BarcodeFormat.EAN13)",
                                      "self.barre_code = tesser(im)",
                                      "retval, self.barre_code, decoded_type = cv_barcode_detector.detectAndDecode((im*255).astype(np.uint8))"]
        self.barrecode_reader = self.barrecode_reader_list[0]
        self.conv_net = Conv(activation_function)

        self.image_vector = torch.zeros([1, 5 * 29 * 29])
        self.barrecode = ""

        #Ethics
        self.score = 0

        #Path
        self.action_list = ["im = im",
                            "sucess, im = saliency.computeSaliency(im)",
                            "im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))",
                            "im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))",
                            "im = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8))",
                            "im = cv2.morphologyEx(im, cv2.MORPH_TOPHAT, np.ones((5,5),np.uint8))",
                            "im = cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, np.ones((5,5),np.uint8))",
                            "im = cv2.erode(im,np.ones((5,5),np.uint8),iterations = 1)",
                            "im = cv2.dilate(im,np.ones((5,5),np.uint8),iterations = 1)",
                            "im = cv2.Sobel(im, -1, 0, 1, ksize=9)",
                            "im = cv2.Canny((im*255).astype(np.uint8),100,200)",
                            "im = cv2.Laplacian((im*255).astype(np.uint8),cv2.CV_8U)"]
        
        self.action_decider = QSwitch(self.image_vector.size(dim=1), len(self.action_list), activation_function)
        self.reader_decider = QSwitch(self.image_vector.size(dim=1), len(self.barrecode_reader_list), activation_function)
        self.action_output = torch.zeros(len(self.action_list))
        self.reader_output = torch.zeros(len(self.barrecode_reader_list))
        self.next_action = ""
        self.next_reader = ""

    #------------------------------------------------------------------------------------------------------------------------------
    def waking(self, im, gt, index):
        print("|-----WAKING AGENT")
        self.image = im
        self.ground_truth = gt
        self.index = index

        im = self.image

        im_g = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        im_b = cv2.rotate(im_g, cv2.ROTATE_180)
        im_s = cv2.resize(im_b, down_points, interpolation= cv2.INTER_LINEAR)
        im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.image_vector = self.conv_net.forward(im_t)

        output = self.action_decider.forward(self.image_vector)
        idx = torch.argmax(output[0])
        self.next_action = self.action_list[idx]

        output = self.reader_decider.forward(self.image_vector)
        idx = torch.argmax(output[0])
        self.next_reader = self.barrecode_reader_list[idx]

        self.barrecode_reader = self.next_reader

    def read_data(self, im, gt, index):
        print("|-----READING DATA")

        self.image = im
        self.ground_truth = gt
        self.index = index

    def estimate_state(self):
        print("|-----ESTIMATING STATE")

        im = self.image
        exec(self.barrecode_reader)

        self.image_vector = torch.zeros([1, 5 * 29 * 29])

        im_g = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        im_b = cv2.rotate(im_g, cv2.ROTATE_180)
        im_s = cv2.resize(im_b, down_points, interpolation= cv2.INTER_LINEAR)
        im_t = transforms.ToTensor()(im_s).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        #print(im_t)
        self.image_vector = self.conv_net.forward(im_t)
        
        print("    Barrecode : " + str(self.barrecode))

    def evaluate_score(self):
        print("|-----COMPUTING SCORE")

        self.score = reward(self.barrecode, self.ground_truth)
        print("    Score : " + str(self.score))

    def plan_path(self):
        print("|-----PATH PLANNING")

        #self.index += 1
        if self.score == 1:
            self.index += 1

        else:
            self.action_output = self.action_decider.forward(self.image_vector)
            self.reader_output = self.reader_decider.forward(self.image_vector)
            idx = torch.argmax(self.action_output[0])
            self.next_action = self.action_list[idx]
            print("    Action : " + str(self.next_action))

            idx = torch.argmax(self.reader_output[0])
            self.next_reader = self.barrecode_reader_list[idx]
            print("    Reader : " + str(self.next_reader))

    def do_actions(self):
        print("|-----DOING ACTIONS")

        self.TRAINING = True
        im = self.image
        if self.score != 1:
            self.barrecode_reader = self.next_reader
            exec(self.next_action)
        else :
            print("Barrecode : " + str(self.barrecode))

        return im, self.index