import cv2
import os
import glob
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import json
import mahotas
from mahotas.thresholding import soft_threshold

# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (dw, dh)

# def truncate(n, decimals=0):
#     multiplier = 10 ** decimals
#     return int(n * multiplier) / multiplier

# def saveBbox(im, bbox, cat_id, width, height):
#     dw = 1. / width
#     dh = 1. / height
#     with open(save_loc_labels + im.split('.jpg')[0] + '.txt', "a") as myfile:
#             xmin = bbox[0]
#             ymin = bbox[1]
#             xmax = bbox[2] + bbox[0]
#             ymax = bbox[3] + bbox[1]
            
#             x = (xmin + xmax)/2
#             y = (ymin + ymax)/2
            
#             w = xmax - xmin
#             h = ymax-ymin
            
#             x = x * dw
#             w = w * dw
#             y = y * dh
#             h = h * dh
#             mystring = str(str(cat_id) + " " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
#             myfile.write(mystring)
#             myfile.write("\n")

def waveletCompress(img):
    t = mahotas.daubechies(img,'D14')
    t /= 10
    t = t.astype(np.int8)
    r = mahotas.idaubechies(t, 'D14')
    r *= 8
    return r

def converGrayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# width, height = 256, 256
# for im in images:
#     img = cv2.imread(ena_local + im)
#     f = img
#     # f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     reshaped, ratio, pad = letterbox(f, (width, height), auto=False)
#     bbox, cat_id = bboxbyid.get(im.split('.jpg')[0])
#     bbox = np.array(bbox) * ratio[0]
#     bbox[1] = bbox[1] + pad[1]
#     if cat_id == 22: #BEAR
#         cat_id = 8
#     saveBbox(im, bbox, cat_id, width, height)
#     if config.applyWavelet:
#         r = waveletCompress(reshaped)
#     else: 
#         r = reshaped # REMOVE 
#     cv2.imwrite(save_loc + im, r, [cv2.IMWRITE_JPEG_QUALITY, 75]) 

def saveCompressed(filepath, image, quality):
    cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, quality])