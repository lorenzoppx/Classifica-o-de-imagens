
import cv2
import numpy as np

from imageio import imread
from matplotlib import pyplot

from skimage import data, io, segmentation, color, morphology

#exemplos_granito_2/28-05-2021_14-43-44_B69480_only_left

mypath="exemplos_granito_2/28-05-2021_14-43-44_B69480_only_left"
im = imread(mypath+"/1_427_left.png")
#mypath="exemplos_granito/28-05-2021_10-13-28_B69428_only_left"
#im = imread(mypath+"/1_147_left.png")
im = cv2.resize(im, (500,500), interpolation = cv2.INTER_CUBIC)
im = cv2.GaussianBlur(im,(5,5),5)
im_cinza = color.rgb2gray(im)

sobelx_filter = cv2.Sobel(im_cinza, ddepth=-1, dx=2, dy=0, ksize=3, scale=1, borderType=cv2.BORDER_CONSTANT)
sobelx_filter = cv2.dilate(sobelx_filter,(5,5),iterations=2)
cv2.imshow("Sobel X Filter", sobelx_filter)
sobely_filter = cv2.Sobel(im_cinza, ddepth=-1, dx=0, dy=2, ksize=3, scale=1, borderType=cv2.BORDER_CONSTANT)
sobely_filter = cv2.dilate(sobely_filter,(5,5),iterations=20)
cv2.imshow("Sobel Y Filter", sobely_filter)
sobel_filter = sobelx_filter + sobely_filter
cv2.imshow("Sobel Filter", sobel_filter)
sobel_filter = cv2.threshold(sobel_filter,127,255,cv2.THRESH_BINARY)
#segment = segmentation.slic(sobel_filter, n_segments=3, compactness=10.0)
"""
pyplot.figure(figsize = (30,5))
pyplot.imshow(segment,cmap='gray')
pyplot.show()

scharrx_filter = cv2.Scharr(sobel_filter, ddepth=-1, dx=1, dy=0,scale=1, borderType=cv2.BORDER_DEFAULT)
cv2.imshow("Scharr X Filter", scharrx_filter)

scharry_filter = cv2.Scharr(sobel_filter, ddepth=-1, dx=0, dy=1, scale=1, borderType=cv2.BORDER_DEFAULT)
cv2.imshow("Scharr Y Filter", scharry_filter)

scharr_filter = scharrx_filter + scharry_filter
cv2.imshow("Scharr Filter", scharr_filter)
"""
cv2.waitKey(0)
cv2.destroyAllWindows()