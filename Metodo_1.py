import cv2
import numpy as np

from imageio import imread
from matplotlib import pyplot

from skimage import data, io, segmentation, color, morphology

mypath="exemplos_granito_2/28-05-2021_14-43-44_B69480_only_left"
im = imread(mypath+"/1_0_left.png")
im_cinza = color.rgb2gray(im)
im_cinza = cv2.blur(im_cinza,(5,5))
im_cinza = cv2.Laplacian(im_cinza,cv2.CV_64F)
im_cinza[im_cinza < 0] = 0
im_cinza = 0 + 100*np.log(im_cinza + 1)

im_cinza = im_cinza.astype(np.uint8)
im_cinza = cv2.equalizeHist(im_cinza)
pyplot.figure(figsize = (30,10))
pyplot.imshow(im_cinza,cmap='gray')
pyplot.show()
