
from ast import Str
import os

def generateSeqLabel(label):
    return str(str(label) + " 0.5 0.5 1 1") 


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def generateSingleLabel(cat, dimensions, bbox):
    dw = 1. / dimensions[0]
    dh = 1. / dimensions[1]

    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2] + bbox[0]
    ymax = bbox[3] + bbox[1]
    
    x = (xmin + xmax)/2
    y = (ymin + ymax)/2
    
    w = xmax - xmin
    h = ymax-ymin
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    mystring = str(str(cat) + " " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
    return mystring