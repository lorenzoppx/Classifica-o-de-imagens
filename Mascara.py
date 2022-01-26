# A mascára funciona com pixels brancos e pretos, 
# como uma espécie de seleção dessa imagem
import cv2
import numpy as np
image = cv2.imread("simpsons.jpg")
#image.shape[:2] -> (altura,largura)
#uint8 -> 0 a 256
mascara = np.zeros(image.shape[:2],dtype = "uint8")
cv2.circle(mascara,(10,10), 250, 255, -1)
# Círculo centrado em (10,10), raio de 250, cor branca, preenchido
cv2.circle(mascara,(310,310), 250, 255, -1)
image = cv2.bitwise_and(image,image,mask=mascara)
cv2.imshow("Janela",image)
cv2.waitKey(0)
