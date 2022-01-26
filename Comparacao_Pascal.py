import cv2
import numpy as np
from Aquisicao_coordenadas import *
from imageio import imread
from matplotlib import pyplot

from skimage import data, io, segmentation, color, morphology

def indice_segmentacao(im1,mypath,index):
    # Imagem fonte
    #mypath = "exemplos_granito/28-05-2021_10-13-28_B69428_only_left"
    #im1 = cv2.imread(mypath+"/1_41_left.png")
    #im1 = cv2.resize(im1,(500,500),cv2.INTER_CUBIC)
    #im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    #ret, im1 = cv2.threshold(im1,127,255,cv2.THRESH_BINARY)

    # Imagem marcada
    mypath_mark = mypath + "_mark"
    im2 = cv2.imread(mypath + "/1_" + str(index) + "_left.png")
    img = np.zeros((im2.shape[0],im2.shape[1]), dtype="uint8")
    mypath_mark = mypath_mark + "/Quadrilatero.txt"
    string1 = "1_" + str(index) + "_left.png"
    pos = search(mypath_mark,string1)
    print(pos)
    #nh = np.array([[1824, 612], [1814, 153], [322, 1723], [310, 520]])
    #cv2.fillPoly(img,[nh] , color=255)
    #cv2.drawContours(img,np.array(pos),0,255,3)
    for i in [0,1,2,3]:
        if i==3:
            cv2.line(img,pos[i-3],pos[i],255,10)
        else:
            cv2.line(img,pos[i],pos[i+1],255,10)

    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,contours,0,255,-1)
    print(np.array(pos))
    print(contours)
    #img = cv2.convexHull(pos)
    img = cv2.resize(img,(500,500),cv2.INTER_CUBIC)
    """
    # Draw a rectangle/ Região base
    rectangle = np.zeros((500, 500), dtype="uint8")
    cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
    cv2.imshow("Rectangle", rectangle)
    """
    """
    # Draw a rectangle/ Região teste
    rectangle2 = np.zeros((500, 500), dtype="uint8")
    cv2.rectangle(rectangle2, (25, 25), (150, 275), 255, -1)
    cv2.imshow("Rectangle2", rectangle2)
    """

    bitwiseAnd = cv2.bitwise_and(im1,img)
    intersecao_areas = cv2.countNonZero(bitwiseAnd)
    bitwiseOr = cv2.bitwise_or(im1,img)
    uniao_areas = cv2.countNonZero(bitwiseOr)
    indice = (intersecao_areas/uniao_areas)*100
    print("Intersecao das areas: ", intersecao_areas)
    print("Indice de qualidade de segmentacao(%): ", indice )

    """
    cv2.imshow("OR", bitwiseOr)
    cv2.imshow("AND", bitwiseAnd)
    cv2.imshow("Imagem segmentada",im1)
    cv2.imshow("Imagem marcada",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    return indice,bitwiseAnd,bitwiseOr,img
