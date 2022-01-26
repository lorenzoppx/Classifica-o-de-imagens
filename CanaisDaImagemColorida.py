import cv2
img = cv2.imread("simpsons.jpg")
(channelBlue,channelRed,channelGreen)= cv2.split(img)
cv2.imshow("Janela",channelRed)
cv2.imshow("Original",img)
cv2.waitKey(0)
