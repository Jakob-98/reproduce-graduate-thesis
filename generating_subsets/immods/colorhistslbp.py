from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import mahotas
from tqdm import tqdm
from six.moves import cPickle as pickle #for performance
from functools import partial
import cv2
import time

def getLpb(imageArray, singlechannel=False):
        a = time.time()
        gray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
        # lb, lg, lr =  (mahotas.features.lbp(imageArray[:, :, i], 1, 8) for i in (0, 1, 2))
        lg = mahotas.features.lbp(gray, 1, 8)
        chb, cbg, cbr = (np.squeeze(cv2.calcHist(imageArray, [i], None, [256], [0,256])) for i in (0,1,2))
        result = np.concatenate((lg, chb, cbg, cbr))
            # result = str(lb) + '\n' + str(lg) + '\n' + str(lr) + '\n' + str(chb) + '\n' + str(cbg) + '\n' + str(cbr)
        return result 