

#IMAGE ANALYSIS FOR WIKIART DATABASE (REQUIRES DATABASE DOWNLOAD)
#by JON THUM



import json
import os.path
import numpy as np
from PIL import Image, ImageFilter, ImageStat, ImageChops, ImageMath
import statistics as st
import time

start = time.process_time()
print('Processing ....')


#HSV COLOUR HUES  
colour = {'red': 255, 'yellow': 42, 'green': 85, 'cyan': 127, 'blue':170, 'magenta': 212}
centre = 0

#KEY HUE VALUES
def hue_key(x):
    global centre
    radius = 43     #max 127
    
    #Shift everything to origin at 128 to avoid negative numbers    
    y = (x-centre+128)%255       
    key = int(3*(radius - abs(y-128)))
    key = max(key, 0)   
    
    return key


#MEAN VALUE FOR SPECIFIED COLOUR = HUE KEY x SAT x VAL
def colour_stat(hue, sat, val):
    key = Image.eval(hue, hue_key)
    col = ImageMath.eval("convert(a*b/255, 'L')", a=key, b=sat)
    colval = ImageMath.eval("convert(a*b/255, 'L')", a=col, b=val)
    stat = ImageStat.Stat(colval)
    
    return stat.mean[0]



#PROCESS DATASET (USING SUPPLIED CLASSIFICATION FILE)   
with open("data/wikiart/class_data.json", "r") as read_file:
    data = json.load(read_file)
    
#OPEN PRE-PROCESSED CROP DATA
crop = np.load('wikiart_crop_analysis.npy')


datafile = []
count = 0
idx = 0

DATA_DIR = 'data/wikiart/'
RES = 1024

for r in data:
    
    if (os.path.isfile(DATA_DIR + r[0])):  
              
        image = Image.open(DATA_DIR + r[0])
        
        #CROP CENTRE
        left = crop[idx][0]
        right = crop[idx][1]
        bottom = crop[idx][2]
        top = crop[idx][3]

        image = image.crop((left, bottom, right, top))        

        #DETAIL ANALYSIS
        grey = image.convert('L')
        blur = grey.filter(ImageFilter.GaussianBlur(10))
        diff = ImageChops.difference(grey, blur)
        
        D = 4   #Normalisation constant
        stat = ImageStat.Stat(diff)
        detail = int(D*stat.mean[0])
                
        #DOWNSIZE TO SAVE COMPUTATION
        image = image.resize((128,128))
             
        #HSV ANALYSIS
        hsv = image.convert(mode="HSV")
        
        hue = hsv.getchannel('H')
        sat = hsv.getchannel('S')
        val = hsv.getchannel('V')
     
        stat = ImageStat.Stat(hsv)
        brightness = int(stat.mean[2]/2)
        contrast = int(stat.stddev[2])
        saturation = int(stat.mean[1]/2)        
    
        #INDIVIDUAL COLOUR ANALYSIS
        C = 8   #Normalisation constant
        
        centre = colour['red']
        red_mean = int(C*colour_stat(hue, sat, val))
        
        centre = colour['yellow']
        yellow_mean = int(C*colour_stat(hue, sat, val))
        
        centre = colour['green']
        green_mean = int(C*colour_stat(hue, sat, val))
        
        centre = colour['cyan']
        cyan_mean = int(C*colour_stat(hue, sat, val))
        
        centre = colour['blue']
        blue_mean = int(C*colour_stat(hue, sat, val))
        
        centre = colour['magenta']
        magenta_mean = int(C*colour_stat(hue, sat, val))    
    
        #COLOUR VARIETY
        T = 60   #Max value for completeness
        
        col_means = [red_mean, yellow_mean, green_mean, cyan_mean, blue_mean, magenta_mean]
        mean = st.mean(col_means)
        stdev = st.stdev(col_means)
        total = min(red_mean, T) + min(yellow_mean, T) + min(green_mean, T) + \
                min(cyan_mean, T) + min(blue_mean, T) + min(magenta_mean, T)
        #variety = total*min(mean, 20)/stdev
        variety = int(total*mean/(stdev+1)/2)    
                
         #APPEND DATA
        info = [detail, brightness, contrast, saturation, variety, 
                red_mean, yellow_mean, green_mean, 
                cyan_mean, blue_mean, magenta_mean, idx]
        #print(info)
        datafile.append(info)        
        
        idx += 1
        
        if(idx%1000 == 0):
            print(time.process_time() - start, int(idx/1000))
            
        
    count += 1


print('TOTAL', count)
#81444

#SAVE DATA
datafile = np.array(datafile, dtype=np.int)
np.save('wikiart_image_analysis.npy', datafile)
print(datafile.shape)

print(time.process_time() - start)


