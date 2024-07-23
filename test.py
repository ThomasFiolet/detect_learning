import pyzbar
from pyzbar.pyzbar import decode
from PIL import Image, ImageOps
import numpy as np
from processing_py import *
import networkx as nx
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from pyxdameraulevenshtein import damerau_levenshtein_distance
import cv2 as cv2
cv_barcode_detector = cv2.barcode.BarcodeDetector()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'
import tesserocr
import zxingcpp
import csv

from utils import zxing
from utils import tesser
from utils import zbar
from metrics import reward
from utils import read_files
from utils import sort_no_training
from utils import conditionnal

#---GET DATA---#
# suffix = 'colors'
# images, ground_truth, len_files = read_files(suffix)
# set, label = sort_no_training(images, ground_truth)

# # im = clahe.apply((im*255).astype(np.uint8))
# # im = cv2.equalizeHist((im*255).astype(np.uint8))
# # th, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
# # th, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY_INV)
# # th, im = cv2.threshold(im, 128, 255, cv2.THRESH_TRUNC)
# # th, im = cv2.threshold(im, 128, 255, cv2.THRESH_TOZERO)
# # th, im = cv2.threshold(im, 128, 255, cv2.THRESH_TOZERO_INV)
# # im = cv2.adaptiveThreshold((im*255).astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# # im = cv2.adaptiveThreshold((im*255).astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# # th, im = cv2.threshold((im*255).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # sucess, im = saliency.computeSaliency(im)
# # im = cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
# # im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
# # im = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8))
# # im = cv2.morphologyEx(im, cv2.MORPH_TOPHAT, np.ones((5,5),np.uint8))
# # im = cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, np.ones((5,5),np.uint8))
# # im = cv2.erode(im,np.ones((5,5),np.uint8),iterations = 1)
# # im = cv2.dilate(im,np.ones((5,5),np.uint8),iterations = 1)
# # im = cv2.Sobel(im, -1, 0, 1, ksize=9)
# # im = cv2.Canny((im*255).astype(np.uint8),100,200)
# # im = cv2.Laplacian((im*255).astype(np.uint8),cv2.CV_8U)

# i = 0
# false_cases = []
# for k in range(len(set)):
#     im_g = cv2.cvtColor(set[k], cv2.COLOR_BGR2GRAY)
#     im = im_g
#     #im = cv2.equalizeHist((im*255).astype(np.uint8))
#     barre_code = conditionnal(im)
#     print(barre_code)
#     if barre_code is not None:
#         i += 1
#     else:
#         false_cases.append(set[k])

# print("Results : " + str(i))

# for im in false_cases:
#     cv2.imshow('False Image', cv2.resize(im, (512, 512), interpolation= cv2.INTER_LINEAR))
#     cv2.waitKey(0)

import csv
with open('data/lists/result.csv', "rU") as f_input:
    csv_input = csv.reader(f_input)
    header = next(csv_input)
    data = sorted(csv_input, key=lambda x: x[0])
    print(data)