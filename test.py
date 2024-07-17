import pyzbar
from pyzbar.pyzbar import decode
from PIL import Image
import cv2 as cv2
import numpy as np

from utils import zbar
from utils import check_EAN_13
from utils import read_files

images, ground_truth, len_files = read_files('colors')

for barre_code in ground_truth:
    print(check_EAN_13(barre_code))