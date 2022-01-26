import numpy as np
import cv2

image = cv2.imread("All_left/1_0_left.png")
cv2.imshow("Original",image)
largura = image.shape[1]
altura = image.shape[0]
proporcao = float(altura/largura)
largura_nova = 400
altura_nova = int(proporcao*largura_nova)
tamanho_novo = (largura_nova,altura_nova)
image = cv2.resize(image,tamanho_novo,interpolation=cv2.INTER_AREA)
cv2.imshow("Modificate",image)
cv2.waitKey(0)
