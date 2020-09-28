#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:51:52 2020

@author: jonthum
"""

#STYLE TRANSFER NETWORK LIBRARY FUNCTIONS
#by JON THUM


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#FEEDBACK
VERBOSE = False


#CONTENT LOSS

class ContentLoss(nn.Module):

    def __init__(self, target, mask, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight

        #ADJUST CONTENT WEIGHTS FOR FOREGROUND THROUGH MASK
        self.mask = mask*(self.weight-1) + 1
        
    def forward(self, input):
        loss = F.mse_loss(input, self.target, reduction='none')
        loss = torch.mul(loss, self.mask)     #CLEVER BIT - multiply pixel losses by mask
        self.loss = torch.mean(loss) 
    
        return input


#GRAM MATRIX   

def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d) 

    G = torch.mm(features, features.t())  
    return G.div(a * b * c * d)


#STYLE LOSS

class StyleLoss(nn.Module):

    def __init__(self, target_feature, loss_function):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss_function = loss_function

    def forward(self, input):
        G = gram_matrix(input)
        #SMOOTH_L1 
        if(self.loss_function=='SL1'):
          self.loss = F.smooth_l1_loss(G, self.target)
        #MEAN SQUARED ERROR  
        elif(self.loss_function=='MSE'):
          self.loss = F.mse_loss(G, self.target)
        else:
          if(VERBOSE):
              print('Style Loss undefined')
        return input


#NORMALISE INPUT IMAGE
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
  
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


#BUILD MODEL WITH ACCESS TO LAYERS FOR STYLE AND CONTENT LOSS

def get_style_model_and_losses(net, cnn, normalization_mean, normalization_std,
                               style_img, content_img, mask):
    cnn = copy.deepcopy(cnn)

    #NORMALISATION MODULE
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    #SEQUENTIAL MODEL
    model = nn.Sequential(normalization)

    i = 0  #Increment for each conv layer
    j = 1  #Increment for each level
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv{}_{}'.format(j, i)

        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(j, i)         
            layer = nn.ReLU(inplace=False)

        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool{}_{}'.format(j, i)
            j += 1
            i = 0

        #OPTIONAL AVERAGE POOLING
            if (net['AVE_POOLING']):
                layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}_{}'.format(j, i)
  
        model.add_module(name, layer)

        #DIFFERENT LOSS FUNCTIONS
        if name in net['CONTENT_LAYERS']:
            #CONTENT LOSS = MSE:
            target = model(content_img).detach()
            content_loss = ContentLoss(target, mask, net['FG_CONTENT_WEIGHT'])
            model.add_module("content_loss_MSE{}_{}".format(j, i), content_loss)
            content_losses.append(content_loss)
            #print(name, 'Content MSE loss')

        if name in net['STYLE_LAYERS_SL1']:
            #STYLE LOSS = SL1:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, 'SL1')
            model.add_module("style_loss_SL1{}_{}".format(j, i), style_loss)
            style_losses.append(style_loss)
            #print(name, 'Style SL1 loss')

        if name in net['STYLE_LAYERS_MSE']:
            #STYLE LOSS = MSE:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, 'MSE')
            model.add_module("style_loss_MSE{}_{}".format(j, i), style_loss)
            style_losses.append(style_loss)
            #print(name, 'Style MSE loss')

    #TRIM REMAINING LAYERS
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    #print(model)

    return model, style_losses, content_losses


#GRADIENT DESCENT

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()], max_iter=10, history_size=10)

    return optimizer


#NEURAL TRANSFER

def run_style_transfer(net, cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, mask):
  
    if(VERBOSE):
        print('Building style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(net, cnn,
        normalization_mean, normalization_std, style_img, content_img, mask)
    optimizer = get_input_optimizer(input_img)

    #LOSS PLOTS
    style_plots = []
    content_plots = []
    
    #CONVERGENCE VARIABLES
    converged = [False]
    converge_epoch = [0]
    converge = [0, 0, 0]

    if(VERBOSE):
        print('Optimizing..')
    epoch = [0]
    
    #LOOP UNTIL MAX EPOCHS REACHED OR CONVERGED
    while (epoch[0]<net['EPOCHS'] and not converged[0]):

        #EVALUATE MODEL AND RETURN LOSS
        def eval():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0
            style_plot = []
            content_plot = []

            #MULTIPLY LOSSES BY RELATIVE WEIGHTS
            for i, sl in enumerate(style_losses):
                loss = sl.loss*net['STYLE_WEIGHTS'][i]
                style_score += loss
                style_plot.append(float(loss))
            style_plots.append(style_plot)
            
            for i, cl in enumerate(content_losses):
                loss = cl.loss*net['CONTENT_WEIGHTS'][i]
                content_score += loss
                content_plot.append(float(loss))
            content_plots.append(content_plot)

            #MULTIPLY LOSSESS BY GLOBAL WEIGHTS
            style_score *= net['GLOBAL_STYLE_WEIGHT']
            content_score *= net['GLOBAL_CONTENT_WEIGHT']
  
            #TOTAL LOSS
            loss = style_score + content_score 
            loss.backward()

            #CONVERGENCE TEST
            converge[2]=converge[1]; converge[1]=converge[0]; converge[0]=style_score.item()
            if (epoch[0]>4 and converge[0]>net['CONV']*converge[1] and converge[1]>net['CONV']*converge[2]):
                converged[0] = True
                if(VERBOSE):
                    print('Converged at {} epochs'.format(epoch))
                
            epoch[0] += 1
            
            #FEEDBACK
            if epoch[0] % 10 == 0:
                if(converged[0]):
                    converge_epoch[0] = epoch[0]
                    #print('Converged at {} epochs'.format(epoch))
                if(VERBOSE):
                    print('Epoch {}: Style Loss = {:4f} Content Loss = {:4f}'.format(
                    epoch, style_score.item(), content_score.item()))
                    #imshow(input_img)

            return style_score + content_score #+ tv_score

        optimizer.step(eval)

    #MAKE SURE VALUES STAY IN RANGE
    input_img.data.clamp_(0, 1)

    return input_img, style_plots, content_plots, converge_epoch[0]
