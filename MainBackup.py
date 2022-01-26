#Teste abertura de imagem pelos indices
import cv2
import numpy as np
pos = [[0,0],[0,0],[0,0],[0,0]]
contador = [0,0]
def print_quad(string,position):
    f = open("Quadrilatero.txt","a")
    #reset the txt
    f.writelines("@"+"1_"+ str(index) +"_left.png"+"@"+str(position)+"@"+"\n")
    f.close()
    return index
def prox_file(var):
    f = open("text.txt","r")
    index = int(f.read())
    #reset the txt
    open('text.txt', 'w').close()
    f = open("text.txt","a")
    if var==1:
        index = index + 1
    f.write(str(index))
    f.close()
    return index
# mouse callback function
def position2d(event,x,y,flags,contador):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if contador[0]<=3:
            pos[contador[0]]=(x,y)
            contador[0] = contador[0] + 1
            cv2.circle(img,(x,y),10,(0,255,0),-1)
            if contador[0]==4:
                contador[1]= 1
                print("Coordenates full")
        
# Create a black image, a window and bind the function to window
index = prox_file(0)
path = ("All_left/"+"1_"+ str(index) +"_left.png")
print(path)
img = cv2.imread(path)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('img',position2d,contador)
while(1):
    cv2.moveWindow('img', 0, 0)
    cv2.imshow('img',img)
    cv2.resizeWindow('img', 1000, 1000)
    print(pos)
    print(contador[1])
    print("index:")
    print(index)
    if contador[1]==1:
        for i in [0,1,2,3]:
            if i==3:
                cv2.line(img,pos[i-3],pos[i],(0,255,0),25)
            else:
                cv2.line(img,pos[i],pos[i+1],(0,255,0),25)
        path = ("All_left_mark/"+"1_"+ str(index) +"_left.png")
        print(path)
        cv2.imwrite(path,img)
        print_quad(("1_"+ str(index) +"_left.png"),pos)
        pos = [[0,0],[0,0],[0,0],[0,0]]
        contador[1] = 0
        contador[0] = 0
    tecla=cv2.waitKey(20)
    if tecla & 0xFF == 27:
        break
    elif tecla==ord('r'):
        index = prox_file(1)
        path = ("All_left/"+"1_"+ str(index) +"_left.png")
        print(path)
        img = cv2.imread(path)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('img',position2d,contador)

cv2.destroyAllWindows()
