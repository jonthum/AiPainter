#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:59:06 2020

@author: jonthum
"""

#CROP ANALYSIS FOR WIKIART DATABASE (REQUIRES DATABASE DOWNLOAD)
#by JON THUM


import json
import os.path
import cv2
import numpy as np
import time

start = time.process_time()
print('Processing ....')


#PROCESS DATASET (USING SUPPLIED CLASSIFICATION FILE)   
with open("data/wikiart/class_data.json", "r") as read_file:
    data = json.load(read_file)

datafile = []
count = 0
idx = 0

DATA_DIR = 'data/wikiart/'
MIN_SIZE = 250
RES = 1024
HALF_RES = int(RES/2)

#FACE DETECTOR
faceCascade = cv2.CascadeClassifier('../imageanalysis/haarcascade_frontalface_default.xml')


for r in data:
    
    if (os.path.isfile(DATA_DIR + r[0])):  
                
        #OPEN IMAGE 
        image = cv2.imread(DATA_DIR + r[0])
        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #DETECT FACES
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, \
                                             minSize=(MIN_SIZE, MIN_SIZE))
        #print("Found {0} faces!".format(len(faces)))      
        
        #FIND BIGGEST FACE
        x = y = w = h = 0
        for (X, Y, W, H) in faces:    
            #print('Face size: {}x{}'.format(W, H))
            if(W>w and H>h):
                x=X 
                y=Y 
                h=H
                w=W           
        #print(x,y,h,w)
        
        #IF FACE IS BIG CROP CENTRED TO FACE
        if(w>MIN_SIZE and h>MIN_SIZE):
            print('Face size: {}x{}'.format(w, h))
        
            X = int(x + w/2) - HALF_RES
            Y = int(y + h/2) - HALF_RES
            
            if(X<0):
                offX = -X
            elif(X+RES>width):
                offX = width-X-RES
            else:
                offX = 0
                
            if(Y<0):
                offY = -Y
            elif(Y+RES>height):
                offY = width-Y-RES
            else:
                offY = 0
            
        #ELSE CROP CENTRED TO IMAGE
        else:
            X = int((width-RES)/2)
            Y = int((height-RES)/2)
            offX = offY = 0
           
        info = [X+offX, X+RES+offX, Y+offY, Y+RES+offY, idx]    
        datafile.append(info)

        idx += 1
        
        if(idx%1000 == 0):
            print(time.process_time() - start, int(idx/1000))
            
        
    count += 1


print('TOTAL', count)

#SAVE DATA
datafile = np.array(datafile, dtype=np.int)
np.save('wikiart_crop_analysis.npy', datafile)
print(datafile.shape)


print(time.process_time() - start)
