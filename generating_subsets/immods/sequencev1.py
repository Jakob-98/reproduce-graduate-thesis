import cv2
import numpy as np
import json
import shutil
import os
import glob
from matplotlib import pyplot as plt
from scipy import ndimage
import cv2 as cv
import PIL

def getSequenceBGSub(seq_images):
    bgs = []
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    for im in seq_images:
        backgroundsubbed = fgbg.apply(im)
        backgroundsubbed = cv2.erode(backgroundsubbed, kernel)
        backgroundsubbed = cv2.dilate(backgroundsubbed, kernel2)
        backgroundsubbed = cv2.erode(backgroundsubbed, kernel)
        backgroundsubbed = cv2.dilate(backgroundsubbed, kernel2)
        bgs.append(backgroundsubbed)
    # also sub the first image
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    fgbg.apply(seq_images[-1])
    backgroundsubbed = fgbg.apply(seq_images[0])
    backgroundsubbed = cv2.erode(backgroundsubbed, kernel)
    backgroundsubbed = cv2.dilate(backgroundsubbed, kernel2)
    backgroundsubbed = cv2.erode(backgroundsubbed, kernel)
    backgroundsubbed = cv2.dilate(backgroundsubbed, kernel2)    
    bgs[0] = backgroundsubbed
    return bgs

def getBox(src):
    # https://stackoverflow.com/questions/60646384/python-opencv-background-subtraction-and-bounding-box
    src_gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)

    # adjust brightness
    src_bright = cv.convertScaleAbs(src_gray, alpha = 255.0/src.max(), beta = 0)
    # apply threshold
    threshold = 50
    _, img_thresh = cv.threshold(src_bright, threshold, 255, 0)
    # apply erode
    erosion_size = 7
    erosion_type = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    img_erosion = cv.erode(img_thresh, element)
    # apply dilate
    dilatation_size = 17
    dilatation_type = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    img_dilate = cv.dilate(img_erosion, element)

    # apply canny and find contours
    threshold = 50
    canny_output = cv.Canny(img_dilate, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    xmax = ymax = 1
    xmin = ymin = 99999
    for c in contours:
        for cs in c:
            x, y = cs[0]
            if x > xmax: xmax = x
            if x < xmin: xmin = x
            if y > ymax: ymax = y
            if y < ymin: ymin = y
    return xmin, xmax, ymin, ymax


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)


    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def generate_boxed_by_sequence(seq_paths: list, size: int):
    # seq_images = [cv2.imread(img) for img in seq_paths]
    # seq_images = [np.array(PIL.Image.open(img)) for img in seq_paths]
    # print('OK... Seq images are pre-loaded before passing to generate_boxed...')
    seq_images = seq_paths 
    bgs = getSequenceBGSub(seq_images)
    imgs = []
    preds = [] # for testing the 'accuracy' of generating bounding boxes
    for i, bg in enumerate(bgs):
        width, height = bg.shape
        boolbackground = bg != 0
        img = seq_images[i]
        img_booled = img * boolbackground[..., np.newaxis]
        xmin, xmax, ymin, ymax = getBox(img_booled)
        if (xmax-xmin) < 10 or (ymax - ymin) < 10:
            preds.append(0)
            img = letterbox(img, (size, size), auto=False)[0]
        else:
            preds.append(1)
            xmin, xmax = max(xmin-15, 1), min(xmax+15, width)
            ymin, ymax = max(ymin-15, 1), min(ymax+15, height)
            if abs(ymin - ymax) < size: # pad if size too low, but never pad beyond borders.
                difference = (abs(ymin-ymax) - size)//2 + 1
                maxpadheight = height - ymax
                if ymin > difference and maxpadheight > difference:
                    ymin -= difference
                    ymax += difference
                elif ymin > difference and maxpadheight <= difference:
                    toppad = maxpadheight - 1
                    bottompad = difference + maxpadheight
                    ymin -= bottompad
                    ymax += toppad
                else: 
                    toppad = difference + ymin
                    bottompad = ymin - 1
                    ymin += bottompad
                    ymax += toppad
            if abs(xmin - xmax) < size: # pad if size too low, but never pad beyond borders.
                difference = (abs(xmin-xmax) - size)//2 + 1
                maxpadheight = width - xmax
                if xmin > difference and maxpadheight > difference:
                    xmin -= difference
                    xmax += difference
                elif xmin > difference and maxpadheight <= difference:
                    toppad = maxpadheight - 1
                    bottompad = difference + maxpadheight
                    xmin -= bottompad
                    xmax += toppad
                else: 
                    toppad = difference + xmin
                    bottompad = xmin - 1
                    xmin += bottompad
                    xmax += toppad
            reshaped_img = img[ymin:ymax, xmin:xmax, :]
            if 0 in reshaped_img.shape or reshaped_img is None: 
                reshaped_img = img
            try:
                img = letterbox(reshaped_img, (size, size), auto=False)[0]
            except cv2.error:
                print("Error resizing:", img.shape)
                img = letterbox(img, (size, size), auto=False)[0]
        imgs.append(img)
    return imgs, None, None    
