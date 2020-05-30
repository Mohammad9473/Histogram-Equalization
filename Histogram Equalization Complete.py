# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:43:41 2019

@author: Mohammadreza
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ImageFile',0)

plt.imshow(img,cmap='gray')
plt.figure()

#print(img.shape)

dimension = img.shape

height = img.shape[1]
width = img.shape[0]



ret,thresh = cv2.threshold(img,127,1,cv2.THRESH_BINARY_INV)

def image_Histogram(image):
    


    #G = np.max(img) + 1

    H = np.zeros(255)
    #print(H)
    for row in image:
        for i in row:
            H[i] = H[i] + 1 
            
    return(H)
        
def cummulative_image_hist(image):

    H = image_Histogram(image)
    #plt.plot(H)
    #plt.show()
    for j in range(1,len(H)):
        H[j] = H[j-1] + H[j]
    return(H)
    
def Hist_equalization(image):

    H = cummulative_image_hist(image)    
    G = len(H)
    #print(G)
    #plt.plot(H)
    #plt.show()
    for x in range(height):
        for y in range(width):
            p = img[y][x]
            f = round((G-1)/(height*width)*(H[p]),0)
            #print((f))
            img[y][x] = f
    return(image)
    
    
img = Hist_equalization(img)
plt.imshow(img,cmap = 'gray')
plt.figure()

#H = cummulative_image_hist(img)
#plt.plot(H)
#plt.show()


#print(H)
