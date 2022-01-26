import cv2

path = "All_left/1_0_left.png"
image = cv2.imread(path)
print(image.shape)
''' 
[0] -> altura da imagem
[1] -> largura da imagem
[2] -> quantidade de canais
'''
# Técnica de "slicing" para alterar vários pixeis da imagem de uma única vez
#image[5,0:image.shape[1]] = (0,0,255)
#image[0:50, 0:50] = (255, 0, 255)
# Comandos pré-definidos da lib openCV para criar formas geométricas
#cv2.line(image,(0,0),(100,100),(0,0,255),50)
#cv2.rectangle(image,(100,100),(150,150),(255,0,255),5,-1)
# Texto
fonte = cv2.FONT_HERSHEY_TRIPLEX
cv2.putText(image, "Hey mundo!@!",(10,100),fonte,1,(0,235,0),2,cv2.LINE_AA)
cv2.imshow("Nome da janela",image)
print(image[0,0])
# (b,g,r) = image[0,0] captura a cor de cada canal, lembrando que a ordem é bgr e não rgb
cv2.waitKey(1000)
# espera pressionar tecla (delay em milisegundos)
cv2.imwrite("imagem_alterada.jpg", image)
#imprimi a imagem em questão