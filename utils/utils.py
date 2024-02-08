"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from PIL import Image
import numpy as np
from torch import nn
import math
from torch.backends import cudnn
import random

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x



def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = acc * 100

    return acc


def oht_to_scalar(y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    return y_pred_tags

def oht_to_scalar_binary(y_pred):
    y_pred_tags=[]
    for i in range(len(y_pred)):
        for pixel in y_pred[i]: 
            if(pixel>=0.5):
                y_pred_tags.append(1)
            else:
                y_pred_tags.append(0)
    y_pred_tags=torch.FloatTensor(y_pred_tags)
    return y_pred_tags

def oht_to_scalar_regression(y_pred):
    y_pred_tags=(y_pred*255).type(torch.int)
    return y_pred_tags

def latent_to_image(g_all, latents, stylegan_version, use_style_latents=False,
                    style_latents=None, process_out=True, return_stylegan_latent=False, dim=512):
    '''Given a input latent code, generate corresponding image and concatenated feature maps'''

    device = torch.device('cuda')
    if use_style_latents:
        style_latents=latents
    else :
       if stylegan_version==1 :
           style_latents = g_all.truncation(g_all.g_mapping(latents))
           style_latents = style_latents.clone()
    
       elif stylegan_version==2 :
           label = torch.zeros([1, 0 ],device=device)
           style_latents=g_all.g_mapping(latents,label,truncation_psi=0.7)

    if return_stylegan_latent:
        return  style_latents
    

    img_list, affine_layers = g_all.g_synthesis(style_latents)


    number_feautre = 0

    affine_layers_upsamples=[]
    for item in affine_layers:
        the_item = item.detach().cpu().numpy()
        affine_layers_upsamples.append(the_item)
    
    if process_out:
        img_list = img_list.cpu().detach().numpy()
        img_list = process_image(img_list)
        img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)

    return img_list, affine_layers_upsamples

def in_size(value,imsize=1024):
    if value>imsize-129:
        value =imsize-129
    if value<0:
        value=0
    return value

def process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    images = images.astype(int)
    images[images > 255] = 255
    images[images < 0] = 0

    return images.astype(int)

def colorize_mask(mask, palette):
    # mask: numpy array of the mask

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


def get_label_stas(data_loader):
    count_dict = {}
    for i in range(data_loader.__len__()):
        x, y = data_loader.__getitem__(i)
        if int(y.item()) not in count_dict:
            count_dict[int(y.item())] = 1
        else:
            count_dict[int(y.item())] += 1

    return count_dict

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def set_seed(seed):
    cudnn.benchmark     = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)