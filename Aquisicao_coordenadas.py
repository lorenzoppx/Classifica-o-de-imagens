"""
import cv2
import numpy as np

from imageio import imread
from matplotlib import pyplot

from skimage import data, io, segmentation, color, morphology

mypath_mark="exemplos_granito/28-05-2021_10-13-28_B69428_only_left_mark/Quadrilatero.txt"
string1 = "1_290_left.png"
"""

def search(mypath_mark,string1):
    # Using readlines()
    file1 = open(mypath_mark, 'r')

    
    # read file content
    readfile = file1.read()
    count = 0
    encontro = 0
    raw_data=""
    with open(mypath_mark, 'r') as reader:
        # Read and print the entire file line by line
        line = reader.readline()
        while line != '':  # The EOF char is an empty string
            print(line, end='')
            if line.find(string1)!=-1:
                print("------------>",line.find(string1))
                for i in range(0,len(line),1):
                    if encontro == 1 and line[i]=='@':
                        break
                    elif encontro == 1 and line[i]!='@':
                        raw_data = raw_data + line[i]
                    elif line[i]=='@':
                        count = count + 1
                        if count==2:
                            encontro = 1
                #raw_data = raw_data.split()
                raw_data = raw_data.replace("[","")
                raw_data = raw_data.replace("]","")
                raw_data = raw_data.replace(",","")
                raw_data = raw_data.replace("(","")
                raw_data = raw_data.replace(")","")
                raw_data = raw_data.split()
                print(raw_data)
                for num in range(0,len(raw_data),1):
                    raw_data[num] = int(raw_data[num])
                final_data = [(raw_data[0],raw_data[1]),(raw_data[2],raw_data[3]),\
                    (raw_data[4],raw_data[5]),(raw_data[6],raw_data[7])]
                print("***: ",final_data)
                return final_data
            line = reader.readline()

