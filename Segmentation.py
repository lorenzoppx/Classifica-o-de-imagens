import cv2
import numpy as np

mypath="exemplos_granito_2/28-05-2021_14-43-44_B69480_only_left/1_0_left.png"
imagem = cv2.imread(mypath)
suave = cv2.medianBlur(imagem, 1)
gray = cv2.cvtColor(suave, cv2.COLOR_BGR2GRAY)
#gray = cv2.equalizeHist(gray)
(T, bin) = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
edge = cv2.Canny(suave,160,255)
bin = cv2.erode(bin,kernel,iterations = 2)
adaptative = cv2.adaptiveThreshold(bin,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,5)
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
result = np.vstack([gray, lap])




while(1):
    cv2.imshow("Edge", edge)
    cv2.imshow("Filtro Laplaciano", result)
    cv2.imshow("Adaptative threshold", adaptative)
    cv2.imshow("Threshold", bin)
    cv2.imshow("Gray", gray)
    cv2.imshow("Imagem modificada", imagem)
    tecla=cv2.waitKey(20)
    if tecla & 0xFF == 27:
        break
