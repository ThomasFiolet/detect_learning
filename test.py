import pyzbar
from pyzbar.pyzbar import decode
from PIL import Image
import cv2 as cv2

from utils import zbar

im = cv2.imread('data/tests/img_1674223597_8252652.jpg')
# im_pil = Image.fromarray(im)
# decoded_list = decode(im_pil)
# print(decoded_list)
print(zbar(im))
