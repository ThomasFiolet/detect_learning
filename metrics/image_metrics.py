import cv2 as cv2
saliency = cv2.saliency.StaticSaliencyFineGrained_create()

def compute_image_metrics(im):
    height, width = im.shape
    brightness = im.mean()
    contrast = im.std()
    sucess, im_s = saliency.computeSaliency(im)
    sal = im_s.mean()
    remarkability = im_s.mean()
    im_c = cv2.Canny(im,100,200)
    sharpness = im_c.mean()
    bluriness = im_c.mean()
    maximum = im.max()
    minimum = im.min()


    return (brightness, contrast, sal, remarkability, sharpness, bluriness, maximum, minimum)