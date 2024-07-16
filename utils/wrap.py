import numpy as np
import PIL
from PIL import Image, ImageOps
import zxingcpp
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/home/thomasfiolet/miniconda3/envs/py39/bin/pytesseract'
tessdata_dir_config = r'--tessdata-dir "./eng.traineddata"'
import tesserocr
from pyzbar.pyzbar import decode
import cv2 as cv2

def zxing(image, bc_format):
    try: barre_code = zxingcpp.read_barcodes(image, bc_format)[0].text
    except: barre_code = '0'
    return barre_code

def tesser(image):
    barre_code = ''.join(c for c in ''.join(tesserocr.image_to_text(Image.fromarray((image*255).astype(np.uint8))).splitlines()) if c.isdecimal())
    return barre_code

def zbar(image):
    im_pil = Image.fromarray(image)
    decoded_list = decode(im_pil)
    if not decoded_list:
        return None
    else:
        return decoded_list[0].data.decode()