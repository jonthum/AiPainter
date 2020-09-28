

#CUSTOM DATABASE CREATION - BAROQUE (REQUIRES WIKIART DATABASE DOWNLOAD)
#by JON THUM


import json
import os.path
import numpy as np
from PIL import Image
import re
import unidecode



#OPEN PRE-PROCESSED DATAFILES
analysis = np.load('wikiart_image_analysis.npy')


#PROCESS DATASET (USING SUPPLIED CLASSIFICATION FILE)   
with open("data/wikiart/class_data.json", "r") as read_file:
    data = json.load(read_file)

filenames = []
count_all = count_genre = count_sel = 0
 

STYLE = [144, 151, 162, 163]
GENRE = [129, 130, 131, 133, 134, 135, 138, 139]
    

DATA_DIR = 'data/wikiart/'
RES = 1024
OUT_DIR = 'data/baro_style/'    
idx = 0

for r in data:
        
    #REMOVE ANY SPECIAL CHARACTERS FROM FILENAME
    filename = str(os.path.basename(DATA_DIR + r[0]))
    newname = re.sub(r'[^\w\-\_\.]', '', filename, re.A)
    newname = unidecode.unidecode(newname)
    
    if (os.path.isfile(DATA_DIR + r[0]) \
        and r[1][1] in GENRE \
        and r[1][2] in STYLE \
        and not os.path.isfile(OUT_DIR + newname) \
        and filename[0:21] != 'ferdinand-georg-waldm'):  
             
        image = Image.open(DATA_DIR + r[0])
        width, height  = image.size
        res= min(width, height)
        
        if (res>RES):     
            #CROP CENTRE
            aspect = width/height
            
            if(aspect>1):
                new_width = int(RES*aspect)
                image = image.resize((new_width, RES))
                #CROP CENTRE
                left = int((new_width - RES)/2)
                right = left + RES
                bottom = 0
                top = RES              
            else:
                new_height = int(RES/aspect)
                image = image.resize((RES, new_height))
                #CROP FOR PORTRAIT
                left = 0
                right = RES
                bottom = int((new_height - RES)/4)
                top = bottom + RES
                
            image = image.crop((left, bottom, right, top))            

            #DETAIL ANALYSIS
            detail = analysis[idx][0]   
            var = analysis[idx][4]
            
            if (detail>20 and var>20):
                #SAVE IMAGE + DATA
                image.save(OUT_DIR + newname) 
                filenames.append(OUT_DIR + newname)
             
                count_sel += 1
          
        count_genre += 1
        
    idx += 1 
    
    if(idx%1000 == 0):
            print(int(idx/1000))
            
    count_all += 1


print('TOTAL', count_all)
print('GENRE', count_genre)
print('SELECTED', count_sel)


#SAVE DATAFILES
json.dump(filenames, open("filenames_baro_style.json", "w"))





'''
GENRE
129 => 'abstract_painting',            5.0k           
130 => 'cityscape',                    4.6k        
131 => 'genre_painting',              10.9k
132 => 'illustration',                 1.9k  
133 => 'landscape',                   13.4k 
134 => 'nude_painting',                1.9k 
135 => 'portrait',                    14.1k 
136 => 'religious_painting',           6.5k  
137 => 'sketch_and_study',             3.9k
138 => 'still_life',                   2.8k 
139 => 'Unknown Genre',               16.5k 
STYLE
140 => 'Abstract_Expressionism',      2782
141 => 'Action_painting',               98
142 => 'Analytical_Cubism'             110
143 => 'Art_Nouveau',                 4334
144 => 'Baroque',                     4240
145 => 'Color_Field_Painting',        1615
146 => 'Contemporary_Realism',         481
147 => 'Cubism',                      2235
148 => 'Early_Renaissance',           1391
149 => 'Expressionism',               6736
150 => 'Fauvism',                      934
151 => 'High_Renaissance',            1343
152 => 'Impressionism',              13060
153 => 'Mannerism_Late_Renaissance',  1279
154 => 'Minimalism',                  1337
155 => 'Naive_Art_Primitivism',       2405
156 => 'New_Realism',                  314
157 => 'Northern_Renaissance',        2552
158 => 'Pointillism',                  513
159 => 'Pop_Art',                     1483
160 => 'Post_Impressionism',          6450
161 => 'Realism',                    10734 
162 => 'Rococo',                      2089
163 => 'Romanticism',                 7019
164 => 'Symbolism',                   4528
165 => 'Synthetic_Cubism',             216
166 => 'Ukiyo_e'                      1167
'''


