

#CUSTOM DATABASE CREATION - ALL PART1 (REQUIRES WIKIART DATABASE DOWNLOAD)
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
classes = []
count_all = count_genre = count_sel = 0


STYLE1 = [140, 141, 142, 147, 149, 150, 165, 144, 151, 162, 163]
STYLE2 = [152, 158, 160]
GENRE = [129, 130, 131, 133, 134, 135, 138, 139]


DATA_DIR = 'data/wikiart/'
RES = 1024
OUT_DIR = 'data/all_part1/'    
idx = 0

for r in data:
    
    #PICK ONLY HALF OF IMPRESSIONISTS
    if (idx%2 == 0):
        SELECT = True
    else:
        SELECT = False
        
    #REMOVE ANY SPECIAL CHARACTERS FROM FILENAME
    filename = str(os.path.basename(DATA_DIR + r[0]))
    newname = re.sub(r'[^\w\-\_\.]', '', filename, re.A)
    newname = unidecode.unidecode(newname)
    
    if (os.path.isfile(DATA_DIR + r[0]) \
        and r[1][1] in GENRE \
        and (r[1][2] in STYLE1 or (r[1][2] in STYLE2 and SELECT)) \
        and not os.path.isfile(OUT_DIR + newname) \
        and filename[0:21] != 'ferdinand-georg-waldm'):  
           
        image = Image.open(DATA_DIR + r[0])
        width, height  = image.size
        res= min(width, height)
        
        if (res>RES):     
         
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
            
            if (detail>20 and var>24):
                #SAVE IMAGE + DATA     
                image.save(OUT_DIR + newname) 
                filenames.append(OUT_DIR + newname)
                labels = np.zeros(167,dtype=np.uint8)
                for l in r[1]:
                    labels[l] = 1        
                classes.append(labels)
                
                count_sel += 1
          
        count_genre += 1
        
    idx += 1 
    
    if(idx%1000 == 0):
            print(int(idx/1000))
            
    count_all += 1


print('TOTAL', count_all)
print('GENRE', count_genre)
print('SELECTED', count_sel)

classes = np.array(classes)
print(classes.shape)

#SAVE DATAFILES
json.dump(filenames, open("filenames_all_part1.json", "w"))
np.save("classes_all_part1.npy",classes)




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



    
'''
UNKNOWN
167 - unknown artist (STYLE????)
168 - unknown genre
ARTISTS
0 => 'Unknown Artist',
1 => 'boris-kustodiev',
2 => 'camille-pissarro',
3 => 'childe-hassam',
4 => 'claude-monet',
5 => 'edgar-degas',
6 => 'eugene-boudin',
7 => 'gustave-dore',
8 => 'ilya-repin',
9 => 'ivan-aivazovsky',
10 => 'ivan-shishkin',
11 => 'john-singer-sargent',
12 => 'marc-chagall',
13 => 'martiros-saryan',
14 => 'nicholas-roerich',
15 => 'pablo-picasso',
16 => 'paul-cezanne',
17 => 'pierre-auguste-renoir',
18 => 'pyotr-konchalovsky',
19 => 'raphael-kirchner',
20 => 'rembrandt',
21 => 'salvador-dali',
22 => 'vincent-van-gogh',
23 => 'hieronymus-bosch',
24 => 'leonardo-da-vinci',
25 => 'albrecht-durer',
26 => 'edouard-cortes',
27 => 'sam-francis',
28 => 'juan-gris',
29 => 'lucas-cranach-the-elder',
30 => 'paul-gauguin',
31 => 'konstantin-makovsky',
32 => 'egon-schiele',
33 => 'thomas-eakins',
34 => 'gustave-moreau',
35 => 'francisco-goya',
36 => 'edvard-munch',
37 => 'henri-matisse',
38 => 'fra-angelico',
39 => 'maxime-maufra',
40 => 'jan-matejko',
41 => 'mstislav-dobuzhinsky',
42 => 'alfred-sisley',
43 => 'mary-cassatt',
44 => 'gustave-loiseau',
45 => 'fernando-botero',
46 => 'zinaida-serebriakova',
47 => 'georges-seurat',
48 => 'isaac-levitan',
49 => 'joaquÃ£Â­n-sorolla',
50 => 'jacek-malczewski',
51 => 'berthe-morisot',
52 => 'andy-warhol',
53 => 'arkhip-kuindzhi',
54 => 'niko-pirosmani',
55 => 'james-tissot',
56 => 'vasily-polenov',
57 => 'valentin-serov',
58 => 'pietro-perugino',
59 => 'pierre-bonnard',
60 => 'ferdinand-hodler',
61 => 'bartolome-esteban-murillo',
62 => 'giovanni-boldini',
63 => 'henri-martin',
64 => 'gustav-klimt',
65 => 'vasily-perov',
66 => 'odilon-redon',
67 => 'tintoretto',
68 => 'gene-davis',
69 => 'raphael',
70 => 'john-henry-twachtman',
71 => 'henri-de-toulouse-lautrec',
72 => 'antoine-blanchard',
73 => 'david-burliuk',
74 => 'camille-corot',
75 => 'konstantin-korovin',
76 => 'ivan-bilibin',
77 => 'titian',
78 => 'maurice-prendergast',
79 => 'edouard-manet',
80 => 'peter-paul-rubens',
81 => 'aubrey-beardsley',
82 => 'paolo-veronese',
83 => 'joshua-reynolds',
84 => 'kuzma-petrov-vodkin',
85 => 'gustave-caillebotte',
86 => 'lucian-freud',
87 => 'michelangelo',
88 => 'dante-gabriel-rossetti',
89 => 'felix-vallotton',
90 => 'nikolay-bogdanov-belsky',
91 => 'georges-braque',
92 => 'vasily-surikov',
93 => 'fernand-leger',
94 => 'konstantin-somov',
95 => 'katsushika-hokusai',
96 => 'sir-lawrence-alma-tadema',
97 => 'vasily-vereshchagin',
98 => 'ernst-ludwig-kirchner',
99 => 'mikhail-vrubel',
100 => 'orest-kiprensky',
101 => 'william-merritt-chase',
102 => 'aleksey-savrasov',
103 => 'hans-memling',
104 => 'amedeo-modigliani',
105 => 'ivan-kramskoy',
106 => 'utagawa-kuniyoshi',
107 => 'gustave-courbet',
108 => 'william-turner',
109 => 'theo-van-rysselberghe',
110 => 'joseph-wright',
111 => 'edward-burne-jones',
112 => 'koloman-moser',
113 => 'viktor-vasnetsov',
114 => 'anthony-van-dyck',
115 => 'raoul-dufy',
116 => 'frans-hals',
117 => 'hans-holbein-the-younger',
118 => 'ilya-mashkov',
119 => 'henri-fantin-latour',
120 => 'm.c.-escher',
121 => 'el-greco',
122 => 'mikalojus-ciurlionis',
123 => 'james-mcneill-whistler',
124 => 'karl-bryullov',
125 => 'jacob-jordaens',
126 => 'thomas-gainsborough',
127 => 'eugene-delacroix',
128 => 'canaletto',
'''