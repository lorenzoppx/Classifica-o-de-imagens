import cv2

image = cv2.imread("simpsons.jpg")
image = cv2.flip(image,1)
# 1 -> flip horizontal
# 0 -> flip vertical
# -1 -> flip horizontal e vertical
(altura, largura) = image.shape[:2]
centro = (int(altura/2),int(largura/2))
M = cv2.getRotationMatrix2D(centro,45,1.5)
image = cv2.warpAffine(image,M,(largura,altura))
cv2.imshow("Janela", image)
cv2.waitKey(1000)



