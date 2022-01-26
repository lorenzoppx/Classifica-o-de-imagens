import cv2
import numpy as np
import io as lib

from imageio import imread
from matplotlib import pyplot
from Comparacao_Pascal import *
import PIL
from skimage import data, io, segmentation, color, morphology, img_as_ubyte

#Pega os nomes dos arquivos e manda para uma lista
from os import listdir
from os.path import isfile, join


mypath_all = ["exemplos_granito/28-05-2021_10-34-24_B68715", \
         "exemplos_granito/28-05-2021_11-42-34_B68703", \
         "exemplos_granito/28-05-2021_14-43-44_B69480", \
         "exemplos_granito/28-05-2021_14-51-12_B58905", \
         "exemplos_granito/28-05-2021_14-56-01_B58940", \
         "exemplos_granito/28-05-2021_15-05-12_B58609" ]
indice_all = 0
numero_fotos = 0
media_anterior = 0
constante_log = 110
numero_fotos = 0
media_all = 0
indice_all = 0
for i in range(0,6,1):
    mypath = mypath_all[i]
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for index in range(0,len(onlyfiles),1):
        im = imread(mypath + "/1_" + str(index) + "_left.png")
        im = cv2.resize(im, (500,500), interpolation = cv2.INTER_CUBIC)
        im = cv2.GaussianBlur(im,(5,5),5)
        print(onlyfiles)
        print("size: ",len(onlyfiles))

        """
        mypath="exemplos_granito/28-05-2021_10-13-28_B69428_only_left"
        im = imread(mypath+"/1_40_left.png")
        mypath="exemplos_granito/28-05-2021_10-13-28_B69428_only_left"
        im = imread(mypath+"/1_390_left.png")
        """
        """
        #kernal sensitive to horizontal lines
        kernel = np.array([[-1.5, -1.5, 3, -1.5,-1.5], 
                            [-1.5, 3.0, -18.0,3.0, -1.5],
                            [-1.5, -1.5, 3, -1.5,-1.5]])
        kernel = kernel 
        kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

        #filter the source image
        im = cv2.filter2D(im,-1,kernel)
        """
        im_cinza = color.rgb2gray(im)
        #im_cinza = 1*(im_cinza)**(5)
        im_cinza[im_cinza >= 255] = 255
        im_cinza = cv2.Laplacian(im_cinza,cv2.CV_64F,5,borderType=cv2.BORDER_CONSTANT )
        im_cinza[im_cinza < 0] = 0
        #constante de ativação log about
        # incio constante_log = 100 -> 200
        im_cinza = 0 + 110*np.log(im_cinza + 1)

        im_cinza = im_cinza.astype(np.uint8)
        im_cinza = cv2.equalizeHist(im_cinza)

        windows = np.array([15,15])
        print(windows)
        half_win = np.floor(windows/2) # função escada, joga um valor quebrado para um inteiro
        half_win = half_win.astype(int) # transforma essa data em tipo inteiro
        width = im_cinza.shape[0] # Largura
        length = im_cinza.shape[1] # Altura
        print("half_win[0]: ",half_win[0])
        print("half_win[1]: ",half_win[1])
        segment = np.zeros((width-2*half_win[0]+1,length-2*half_win[1]+1), dtype=np.uint8)
        media_array = np.zeros((600,1),dtype=np.uint8)
        desvio_array = np.zeros((600,1),dtype=np.uint8)
        print(segment.shape) # (511,509)

        for i in range(half_win[0],width-half_win[0]+1):
            for j in range(half_win[1],length-half_win[1]+1):
                media = np.mean(im_cinza[i-half_win[0]:i+half_win[0],j-half_win[1]:j+half_win[1]])
                desvio = np.std(im_cinza[i-half_win[0]:i+half_win[0],j-half_win[1]:j+half_win[1]])
                #print("i: ",i)
                media_array[i] = media
                desvio_array[i] = desvio

        # extracao de valores para limiar 
        #pyplot.plot(media_array)
        media_array_modf = media_array[media_array != 0]
        media_total= np.mean(media_array_modf)
        #media_limiar= int(media_total)
        media_limiar= int(media_total*0.8)
        print("Media total: ", media_total)
        #pyplot.plot(desvio_array)
        desvio_array_modf = desvio_array[desvio_array != 0]
        desvio_total = np.mean(desvio_array_modf)
        desvio_limiar = int(desvio_total*0.8)
        #desvio_limiar = int(desvio_total)
        print("Desvio total: ", desvio_total)

        for i in range(half_win[0],width-half_win[0]+1):
            for j in range(half_win[1],length-half_win[1]+1):
                media = np.mean(im_cinza[i-half_win[0]:i+half_win[0],j-half_win[1]:j+half_win[1]])
                desvio = np.std(im_cinza[i-half_win[0]:i+half_win[0],j-half_win[1]:j+half_win[1]])
                #print("i: ",i)
                if (media > media_limiar) and (desvio > desvio_limiar):
                    segment[i-half_win[0],j-half_win[1]] = 255

        values_l = np.zeros(segment.shape[1], float) 
        for j in range(segment.shape[1]):
            values_l[j] = np.sum(segment[:,j])
            values_l[j] = values_l[j]/(255)
        media = np.mean(values_l)
        print("media_values_l:",np.mean(values_l))
        for j in range(segment.shape[1]):
            if (values_l[j] < int(media)):
                #print("--->")
                segment[:,j] = np.zeros(segment.shape[0],dtype = np.uint8)

        values_w = np.zeros(segment.shape[0], float) 
        for i in range(segment.shape[0]):
            values_w[i] = np.sum(segment[i,:])
            values_w[j] = values_w[j]/255
        media = np.mean(values_w)
        print("media_values_w:",np.mean(values_w))
        for i in range(segment.shape[0]):
            if (values_w[i] < int(media)):
                #print("-------->")
                segment[i,:] = np.zeros(segment.shape[1],dtype = np.uint8)

        #ordem de cores
        #azul,laranja,verde,vermelho
        #pyplot.plot(values_l) #como slider vertical
        values_l = cv2.GaussianBlur(values_l,(21,21),10)
        div = np.zeros((values_l.shape[0],1))
        h=1/5
        for i in range(0,values_l.shape[0],1):
            if i!=values_l.shape[0]-1:
                div[i] = int((values_l[i+1]-values_l[i])/h)
                print(div[i])

        min_value = np.min(div)
        result = np.where(div == min_value)
        index_minimo = result[0][0]
        print("min index: ", index_minimo)
        
        for j in range(0,segment.shape[0],1):
            segment[j][index_minimo]=255
        
        max_value = np.max(div)
        result = np.where(div == max_value)
        index_maximo = result[0][0]
        print("max index: ",index_maximo)
        
        for j in range(0,segment.shape[0],1):
            segment[j][index_maximo]=255
        
        div = cv2.GaussianBlur(div,(21,21),15)
        pyplot.plot(values_l/1000) #como slider vertical
        pyplot.plot(div) #como slider vertical

        values_w = cv2.GaussianBlur(values_w,(21,21),10)
        div = np.zeros((values_w.shape[0],1))

        h=1/5
        for i in range(0,values_w.shape[0],1):
            if i!=values_w.shape[0]-1:
                div[i] = int((values_w[i+1]-values_w[i])/h)
                print(div[i])

        min_value = np.min(div)
        result = np.where(div == min_value)
        index_minimo = result[0][0]
        print("min index: ", index_minimo)
        
        for i in range(0,segment.shape[1],1):
            segment[index_minimo][i]=255
        
        max_value = np.max(div)
        result = np.where(div == max_value)
        index_maximo = result[0][0]
        print("max index: ",index_maximo)
        
        for i in range(0,segment.shape[1],1):
            segment[index_maximo][i]=255
        
        
        # slider horizontal
        faixa = 0
        anterior = 0
        largura_minima = 10
        for j in range(0,segment.shape[0],1):
            for i in range(0,segment.shape[1],1):
                if segment[j][i]==255:
                    faixa = faixa + 1
                    anterior = 1
                elif segment[j][i]==0 and anterior==1:
                    # faixa fora da largura minima
                    if(faixa <= largura_minima):
                        for x in range(1,faixa+1,1):
                            segment[j][i-x]=0  
                if segment[j][i]==0:
                    faixa = 0
                    anterior = 0

        # slider vertical
        faixa = 0
        anterior = 0
        altura_minima = 10
        for i in range(0,segment.shape[1],1):
            for j in range(0,segment.shape[0],1):
                if segment[j][i]==255:
                    faixa = faixa + 1
                    anterior = 1
                elif segment[j][i]==0 and anterior==1:
                    # faixa fora da largura minima
                    if(faixa <= altura_minima):
                        for x in range(1,faixa+1,1):
                            segment[j-x][i]=0  
                if segment[j][i]==0:
                    faixa = 0
                    anterior = 0
            
        
        div = cv2.GaussianBlur(div,(21,21),15)
        pyplot.plot(values_w) #como slider vertical
        pyplot.plot(div) #como slider vertical

        # Configs to plot in only one page
        fig = pyplot.figure(figsize = (30,30))
        rows = 2
        columns = 4

        fig.add_subplot(rows, columns, 1)
        pyplot.imshow(im_cinza,cmap='gray')
        pyplot.title("Imagem tratada")

        fig.add_subplot(rows, columns, 2)
        pyplot.imshow(segment, cmap='gray')
        pyplot.title("Imagem segmentada")
        
        print("finish")

        footprint = morphology.rectangle(11,11)
        #segment2 = cv2.erode(segment,(15,15),iterations=5)
        segment2 = morphology.binary_closing(segment, footprint)
        
        fig.add_subplot(rows, columns, 3)
        pyplot.imshow(segment2,cmap='gray')
        pyplot.title("Imagem segmentada técnica de fechamento")
        
        segment3 = morphology.convex_hull_image(segment2 == 1)
        
        fig.add_subplot(rows, columns, 4)
        pyplot.imshow(segment3,cmap='gray')
        pyplot.title("Imagem segmentada pós-processada")
        
        print(segment3)
        segment3 = segment3.astype(np.uint8)  #convert to an unsigned byte
        segment3*=255
        #PIL_image = PIL.Image.fromarray(np.uint8(segment3))
        #segment3 = img_as_ubyte(segment3)
        #PIL_image = PIL.Image.fromarray(numpy_image.astype('uint8'), 'RGB')
        #segment3 = PIL.Image.fromarray(segment3, "RGB")
        print(segment3)
        segment3 = cv2.resize(segment3,(500,500),cv2.INTER_AREA)

        indice, bitwiseAND, bitwiseOR, img = indice_segmentacao(segment3,mypath,index)
        
        fig.add_subplot(rows, columns, 5)
        pyplot.imshow(img,cmap='gray')
        pyplot.title("Imagem marcada manualmente")

        fig.add_subplot(rows, columns, 6)
        pyplot.imshow(bitwiseAND,cmap='gray')
        pyplot.title("Interseção de áreas")

        fig.add_subplot(rows, columns, 7)
        pyplot.imshow(bitwiseOR,cmap='gray')
        pyplot.title("União de áreas")

        fig.add_subplot(rows, columns, 8)
        pyplot.axis([0, 10, 0, 10])
        pyplot.text(2, 5, "Indice: " + str(indice) + "\n" \
                            "Constante log: " + "110" + "\n" \
                            "Gaussian Blur kernel: " + "(5,5)" + "\n" \
                            "Média limiar: " + "80% da média" + "\n" \
                            "Desvio padrão limiar: " + "80% da média" + "\n" \
                            "Windows size: " + "(15,15)" + "\n" \
                            "Filtro tamanho de faixa: " + "Sm" + "\n" \
                            "Tamanho de faixa vert. e hor.: " + "20" + "\n" \
                            "Binary closing footprint: " + "(11,11)" + "\n" \
            , style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 15})
        
        pyplot.savefig( "Analise_" + mypath + "/" + "Analise_1_" + str(index) + "_left.png")
        pyplot.show()
        pyplot.close()

        indice_all = indice_all + indice
    numero_fotos = numero_fotos + len(onlyfiles)
    print("Numero: ",numero_fotos)
    print("Indice_all: ",indice_all)
    print("Constante Log:", constante_log)

    
    media_all = indice_all/numero_fotos
    print("media_all: ",media_all)
    #f = open("Analise_exemplos_granito.txt","a")
    #reset the txt
    #f.writelines("@" + "D_fix e M_fix: " + str(m_per) + " " + str(d_per)  + "@" + str(media_all) + "@" + "\n")
    #f.close()