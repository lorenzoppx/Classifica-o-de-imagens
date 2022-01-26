import cv2
import numpy as np

from imageio import imread
from matplotlib import pyplot as plt

from skimage import data, io, segmentation, color, morphology

#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

mypath="exemplos_granito_2/28-05-2021_14-43-44_B69480_only_left"
img = imread(mypath+"/1_500_left.png")
img = cv2.resize(img, (500,500), interpolation = cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#kernal sensitive to horizontal lines
kernel = np.array([[-1.5, -1.5, -1.5, -1.5,-1.5], 
                   [-1.5, 3.0, 5.0,3.0, -1.5],
                  [-1.5, -1.5, -1.5, -1.5,-1.5]])
kernel = kernel 
kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

#filter the source image
img_rst = cv2.filter2D(img,-1,kernel)

cv2.imshow("Final",img_rst)
cv2.waitKey(0)