# Além do já conhecido BGR adotado pelo openCV, temos outros sistemas de cores
# ,como por exemplo, tons de cinza, hsv, L*a*b* , black&white
import cv2
img = cv2.imread('simpsons.jpg')
cv2.imshow("Original", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
cv2.waitKey(0)

