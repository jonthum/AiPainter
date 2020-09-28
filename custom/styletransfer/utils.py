#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:51:05 2020

@author: jonthum
"""

#STYLE TRANSFER IMAGE UTILITY FUNCTIONS
#by JON THUM


from __future__ import print_function
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from skimage.exposure import match_histograms, equalize_hist, equalize_adapthist
from skimage.exposure import rescale_intensity, adjust_gamma
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import io
import dlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LOAD FACE DETECTORS
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#FEEDBACK IF REQUIRED
VERBOSE = False


#GET DESIRED ASPECT RATIO FROM CONTENT IMAGE

def aspect(image, size):
    
  aspect = image.size[0]/image.size[1]
  if(aspect>1):
      im_size = (size, int(size/aspect))
  else:
      im_size = (int(size*aspect), size)
  #print(im_size)

  if(VERBOSE):
      print("Content image size:", image.size)
        
  return(im_size)


#TILING FOR STYLE IMAGE

def image_tile(image, scale, size, keep_aspect):

    width = image.size[0]
    height = image.size[1]  
    W = size[0]
    H = size[1]    
        
    #Resize value to fit style into format
    fit = min(H/height, W/width)

    #Fit to height or fit to width
    if(H/height<W/width):
        aspectx = W/H
        aspecty = 1
    else:
        aspectx = 1
        aspecty = H/W    
        
    #Resize factor for aspect
    if (keep_aspect):
        sx = aspectx*scale
        sy = aspecty*scale
    else:
        sx = scale
        sy = scale

    #Tile size
    x = int(width*fit*aspectx/sx)
    y = int(height*fit*aspecty/sy)    
   
    #Calculate step sizes for (stepx x stepy) grid
    stepx = int(sx+0.999) 
    stepx = stepx - stepx%2 + 1
    stepy = int(sy+0.999) 
    stepy = stepy - stepy%2 + 1    
    
    #Calculate offsets
    offx = int((stepx-sx)*x/2)
    offy = int((stepy-sy)*y/2)
     
    #Make bigger image (PIL does not allow negative value pasting)
    new = Image.new('RGB', [W+offx, H+offy])
 
    #Use lanczos filter for upsizing
    if(x>height or y>width):
        im = image.resize([x, y], Image.LANCZOS)
        if(VERBOSE):
            print('Tile size: {} filter=Lanczos'.format((x,y)))
    #Use bicubic for downsizing
    else:
        im = image.resize([x, y], Image.BICUBIC)
        if(VERBOSE):
            print('Tile size: {} filter=Bicubic'.format((x,y)))

    #Flipflop if necessary to maintain smoothness
    if(int(((stepy+1)/2)%2)==0):
        im = im.transpose(Image.ROTATE_180)
    
    #Make tiled grid
    py = 0
    for j in range(stepy):
        px = 0
        imx = im
        for i in range(stepx):
            new.paste (imx, [px, py])
            imx = imx.transpose(Image.FLIP_LEFT_RIGHT)
            px += x
        py += y  
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

    #Crop back to original size
    new = new.crop((offx, offy, W+offx, H+offy))

    return new
    

#ADD NOISE TO INPUT IMAGE IF REQUIRED

def noise_mix(image, mask, noise_size):
    
    #Generate noise
    noise = np.random.rand(image.size[1], image.size[0], 3)
    noise = 256*((noise-0.5)*noise_size[2] + 0.5)
    noise = noise.astype(int)
    noise = Image.fromarray(np.uint8(noise)) 

    #Composite noise using mask (amounts FG_NOISE/BG_NOISE) 
    bg_mix = Image.blend(image, noise, noise_size[0])  
    fg_mix = Image.blend(image, noise, noise_size[1])
    comp = Image.composite(fg_mix, bg_mix, mask) 

    return comp


#DISPLAY LANDMARKS ON IMAGE

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.title('Landmarks')
    plt.savefig('./landmarks.jpg', dpi=300)
    plt.pause(0.001)  
    plt.show()
    


#FACE MASK VERTICES

def face_mask(landmarks):
    polygon = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
            landmarks[10], landmarks[12], landmarks[14], landmarks[16], landmarks[26],
             landmarks[24], landmarks[20], landmarks[18]])
    polygon = polygon.flatten().tolist()  
    return(polygon)


#EYE MASK VERTICES

def eyes_mask(landmarks):    
    polygon1 = np.array([landmarks[36], landmarks[37], landmarks[38], landmarks[39], 
                         landmarks[40], landmarks[41]])
    polygon1 = polygon1.flatten().tolist()   
    polygon2 = np.array([landmarks[42], landmarks[43], landmarks[44], landmarks[45], 
                         landmarks[46],landmarks[47]])
    polygon2 = polygon2.flatten().tolist()    
    return(polygon1, polygon2)    
    
    
#NOSE MOUTH MASK VERTICES

def features_mask(landmarks):    
    polygon3 = np.array([landmarks[27], landmarks[31], landmarks[33], landmarks[35]])
    polygon3 = polygon3.flatten().tolist()   
    polygon4 = np.array([landmarks[48], landmarks[50], landmarks[52], landmarks[54], 
                         landmarks[56], landmarks[57], landmarks[58]])
    polygon4 = polygon4.flatten().tolist()    
    return(polygon3, polygon4) 
    
    
#DRAW MASK

def draw_mask(landmarks, width, height, feature):
    mask = Image.new('L', (width, height), 0) 
    
    #BLUR FACTOR NEEDED TO MAKE SOFTER MASKS FOR BIGGER IMAGES
    blur_factor = (width+height)/1000
    
    if (feature=='Face'):
        polygon = face_mask(landmarks)
        #DRAW POLYGON
        ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)
        #LARGE BLUR
        mask = mask.filter(ImageFilter.GaussianBlur(10*blur_factor))
    elif (feature=='Eyes' or feature=='Features'):
        polygon1, polygon2 = eyes_mask(landmarks)
        #DRAW POLYGONS
        ImageDraw.Draw(mask).polygon(polygon1, outline=255, fill=255)
        ImageDraw.Draw(mask).polygon(polygon2, outline=255, fill=255)
        if (feature=='Features'):
            polygon3, polygon4 = features_mask(landmarks)
            ImageDraw.Draw(mask).polygon(polygon3, outline=255, fill=255)
            ImageDraw.Draw(mask).polygon(polygon4, outline=255, fill=255)
        #SMALL BLUR
        mask_blur = mask.filter(ImageFilter.GaussianBlur(5*blur_factor))
        #ENLARGE
        enhance = ImageEnhance.Brightness(mask_blur)
        mask = enhance.enhance(3*blur_factor)
        mask = mask.filter(ImageFilter.GaussianBlur(4*blur_factor))
    else:
        if(VERBOSE):
            print('Warning: Feature must be either Face or Eyes ')
    return mask
    
    
#GET FACIAL LANDMARKS

def get_landmarks(image):
    
    landmarks = []
    gray = np.array(image.convert('L'))
    
    #FACE DETECTOR
    faces = detector(gray)
    if(VERBOSE):
        print('Faces detected: {}'.format(len(faces)))
        
    #LOOP THROUGH FACES
    for face in faces:
        landmark = []
        
        #LANDMARK DETECTOR
        points = predictor(gray, face)
       
        for n in range(0,68):
            x=points.part(n).x
            y=points.part(n).y
            landmark.append([x, y])
        landmarks.append(landmark)
        
    return (np.array(landmarks), len(faces))
    
    
#EXTRACT MASK

def mask_extract(image, feature):

    #If image is'RGB' create landmarks; if none found create black (empty) mask
    if(len(image.getbands())==3):
        
        landmarks, num_faces = get_landmarks(image)
        
        if(num_faces==0):
            if(VERBOSE):
                print("No alpha channel or landmarks found: required for added functionality")
            mask = Image.new('L', image.size)
        else:
            mask = draw_mask(landmarks[0], image.size[0], image.size[1], feature)
            if(VERBOSE):
                show_landmarks(image, landmarks[0])
                plt.imshow(mask, cmap='gray')
                plt.title('Mask')
                plt.savefig('./mask.jpg', dpi=300)
                plt.show()
                

    #If image is 'RGBA' extract alpha channel                
    elif(len(image.getbands())==4):
        mask = image.getchannel('A')
    else:
        if(VERBOSE):
            print("Input image needs to be either RGB or RGBA")

    return mask


#IMAGE NORMALISATION

def image_normalise(style, content, gamma=1, sharpen=1, S_TYPE=None, C_TYPE=None):
    
    style= np.array(style)
    content = np.array(content)
    
    #DIFFERENT TYPES OF NORMALISATION FOR STYLE IMAGE
    if(S_TYPE=='MATCH'):
        style = match_histograms(style, content, multichannel=True)
    elif(S_TYPE=='CONTRAST'):
        p2, p98 = np.percentile(style, (2, 98))
        style = rescale_intensity(style, in_range=(p2, p98))
    elif(S_TYPE=='HISTO'):
        style = equalize_hist(style)
        style = (style*255).astype(np.uint8)
    elif(S_TYPE=='ADAPT'):
        style = equalize_adapthist(style, clip_limit=0.01)
        style = (style*255).astype(np.uint8)

    #DIFFERENT TYPES OF NORMALISATION FOR CONTENT IMAGE
    if(C_TYPE=='MATCH'):
        content = match_histograms(content, style, multichannel=True)
    elif(C_TYPE=='CONTRAST'):
        p2, p98 = np.percentile(content, (2, 98))
        content = rescale_intensity(content, in_range=(p2, p98))
    elif(C_TYPE=='HISTO'):
        content = equalize_hist(content)
        content = (content*255).astype(np.uint8)
    elif(C_TYPE=='ADAPT'):
        content = equalize_adapthist(content, clip_limit=0.01)
        content = (content*255).astype(np.uint8)
    
    #GAMMA ADJUSTMENT
    if(gamma != 1):
        style = adjust_gamma(style, gamma)
        content = adjust_gamma(content, gamma)
    
    style = Image.fromarray(style)
    content = Image.fromarray(content)
    
    #SHARPENING
    if(sharpen != 1):
        style = ImageEnhance.Sharpness(style).enhance(sharpen)
        content = ImageEnhance.Sharpness(content).enhance(sharpen)
    
    if(VERBOSE):
        plt.title('Style: Norm={}, {}, {}'.format(S_TYPE, gamma, sharpen))
        plt.imshow(style)
        plt.show()    
        
        plt.title('Content: Norm={}, {}, {}'.format(C_TYPE, gamma, sharpen))
        plt.imshow(content)
        plt.show()    
        
    return style, content
    

#IMAGE LOADING

def input_loader(image, mask, noise_size, im_size):
    image = image.convert('RGB')          #Strip out alpha channel

    #IF NOISE AMOUNT IS NON ZERO
    if(noise_size[0] or noise_size[1]):
        image = noise_mix(image, mask, noise_size)    #Add noise

    image = image.resize(im_size)         #Set processing size
    loader = transforms.ToTensor()        #Transform to tensors
    image = loader(image).unsqueeze(0)
    
    return image.to(device, torch.float)


def content_loader(image, mask, net, im_size):
    image = image.convert('RGB')          #Strip out alpha channel
    
    #SET PROCESSING SIZE
    if(image.size[0]<im_size[0]):
        image = image.resize(im_size, Image.LANCZOS) 
        if(VERBOSE):
            print('Processing size: {} filter=Lanczos'.format(im_size))
    else:
        image = image.resize(im_size, Image.BICUBIC)
        if(VERBOSE):
            print('Processing size: {} filter=Bicubic'.format(im_size))
        
    #MASK FOR CONTENT LAYER
    mask = mask.resize(im_size)
    level = int(net['CONTENT_LAYERS'][0][4]) #Get level of content layer 
    for i in range(level-1):              #CLEVER BIT - reduce by half for (level-1) pooling levels
        width = int(mask.size[0]/2)
        height = int(mask.size[1]/2)
        mask = mask.resize([width,height])  #Resized mask to fit content loss convolution layer
    if(VERBOSE):
        print('Mask size:', mask.size)        
        
    loader = transforms.ToTensor()        #Transform to tensors
    image = loader(image).unsqueeze(0)
    mask = loader(mask).unsqueeze(0)

    return image.to(device, torch.float), mask.to(device, torch.float)


def style_loader(image, tile, im_size, keep_aspect=True):
    image = image.convert('RGB')          #Strip out alpha channel
    
    image = image_tile(image, tile, im_size, keep_aspect)     #Tile style image 
    loader = transforms.ToTensor()        #Transform to tensors
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)  
    

#IMAGE CONVERSION

def PIL_to_bytes(image):
    array = io.BytesIO()
    image.save(array, format='JPEG')
    image_bytes = array.getvalue()
    return image_bytes

def image_to_bytes(image):
    image = Image.fromarray(image)
    array = io.BytesIO()
    image.save(array, format='JPEG')
    image_bytes = array.getvalue()
    return image_bytes

def tensor_to_PIL(tensor):
    result = tensor.cpu().clone()  
    result = result.squeeze(0)     
    unloader = transforms.ToPILImage()  #Reconvert into PIL image
    result = unloader(result)
    return result
    
    
#MAKE IMAGE QUARTERS AND RECOMBINE FOR MEMORY EFFICIENT UPSCALING

def make_quarters(array, border):
    quarters = []
    x = array.shape[0]; y = array.shape[1]
    halfx = int(x/2); halfy = int(y/2)
    quarters.append(array[:halfx+border, :halfy+border, :])
    quarters.append(array[:halfx+border, halfy-border:, :])
    quarters.append(array[halfx-border:, halfy-border:, :])
    quarters.append(array[halfx-border:, :halfy+border, :])
    return quarters, x, y, halfx, halfy

def combine_quarters(quarters, x, y, halfx, halfy, border):
    array = np.ones((x, y, 3))
    array[:halfx, :halfy, :] = quarters[0][:halfx, :halfy, :]
    array[:halfx, halfy:, :] = quarters[1][:halfx, border:, :]
    array[halfx:, halfy:, :] = quarters[2][border:, border:, :]
    array[halfx:, :halfy, :] = quarters[3][border:, :halfy, :]
    return array


#COLOUR SWAP - RESTORE ORIGINAL CONTENT COLOUR

def colour_swap(content, result, mix):
    content_hsv = content.convert(mode="HSV")
    result_hsv = result.convert(mode="HSV")
    
    h = content_hsv.getchannel('H')
    s = content_hsv.getchannel('S')
    v = result_hsv.getchannel('V')
    
    swap = Image.merge("HSV", (h, s, v))
    swap = swap.convert("RGB")

    if(mix != 1):
        swap = Image.blend(result, swap, mix)
    
    return swap

    
#SAVE LOSS PLOTS
    
def gen_plots(out_dir, id, net, style_plots, content_plots, save=True):
    
    plt.ioff()

    #PLOT NAMES AND DIRECTORIES
    style_loss_name = out_dir + 'Style_loss_' + id + '.jpg' 
    content_loss_name = out_dir + 'Content_loss_' + id + '.jpg' 

    #SAVE CONTENT LOSS PLOT
    fig = plt.figure()
    plt.plot(content_plots, 'red')
    plt.legend( ['Content loss'], loc = 'upper right')
    plt.xlabel('Epochs')
    plt.title('CONTENT LOSSES ' + id)
    plt.axis([None, None, 0, None])
    if(save):
        plt.savefig(content_loss_name)
    plt.close(fig)
    #plt.show()

    #SAVE STYLE LOSS PLOTS
    fig = plt.figure()
    style_loss_plot = np.array(style_plots)*net['GLOBAL_STYLE_WEIGHT']
    plt.plot(style_loss_plot[:,0:1], 'blue')
    plt.plot(style_loss_plot[:,1:2], 'red')
    plt.plot(style_loss_plot[:,2:3], 'green')
    plt.plot(style_loss_plot[:,3:4], 'gray')
    plt.plot(style_loss_plot[:,4:5], 'orange')
    plt.legend( ['Style loss 1', 'Style loss 2', 'Style loss 3', 'Style loss 4', 'Style loss 5'], loc = 'upper right')
    plt.xlabel('Epochs')
    plt.title('STYLE LOSSES ' + id)
    max_y = 5e-5*net['GLOBAL_STYLE_WEIGHT']
    plt.axis([None, None, 0, max_y])
    if(save):
        plt.savefig(style_loss_name)
    plt.close(fig)
    #plt.show()    
    