
"""
Residual based image compression using simple cnn

usage: tested for PNG images. uses convert_png() to convert image to png
       Pads images for correct crop but padding not removed

Requires: Pytorch, skimage, PIL, patchify, cv2

use:
import test_utils as utils

utils.compress_img(name.png) -> saves a .npz , thumbnail file in the same directory
utils.decompress_img(name'.npz) -> saves a png image





"""


from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from skimage.measure import compare_psnr, compare_mse, compare_ssim
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import cv2
import torch.nn as nn
from torch.autograd import Variable
from patchify import unpatchify
import os
from patchify import patchify as cp


# ## Test utils


def read_image(name_):
    img = Image.open(name_)
    img = np.array(img)
    img = img.astype('int16')
    if img.shape[-1] == 4:
        img = img[:,:,:3]
    return img



def pad_image(img, crop = 128):
    crop_size = crop
    H, W, D = img.shape
    if H % crop_size == 0 and W % crop_size == 0:
        return img
    

    h_pad = 0 
    w_pad = 0
    if H % crop_size != 0:
        q = H // crop_size
        new_H = (q + 1) * crop_size
        h_pad = new_H - H
    
    if W % crop_size != 0:
        q = W // crop_size
        new_W = (q + 1) * crop_size
        w_pad = new_W - W

    d_type = img.dtype
    
    
    new_img = np.zeros((H + h_pad, W + w_pad, D),dtype = d_type)

    
    h_pad_pos = int(h_pad / 2)
    w_pad_pos = int(w_pad / 2)
    
    
    new_img[h_pad_pos:h_pad_pos + H, w_pad_pos: w_pad_pos + W, :] = img
    
    return new_img


def convert_png(fnames,png_dir):
    for row,i in zip(fnames,range(len(fnames))):
        img = read_image(row)
        name_ = row.split('/')[-1]
        name_ = name_.split('.')[0]
        plt.imsave(png_dir + "/{}.png".format(name_),img)
        print("Converted {} of {} to PNG".format(i+1,len(fnames)))

def create_crops(name_,thumb_name_,crop=128):

    img = read_image(name_)
    img = pad_image(img)
    
    H, W, D = img.shape
    upscale_img = read_image(thumb_name_)
    upscale_img = cv2.resize(upscale_img,(W,H))
    
    res_img = img - upscale_img
    
    patches_1 = cp(res_img[:,:,0],(128,128),128)
    patches_2 = cp(res_img[:,:,1],(128,128),128)
    patches_3 = cp(res_img[:,:,2],(128,128),128)
    
    pshape0 = patches_1.shape[0]
    pshape1 = patches_1.shape[1]
    
    patches = np.stack((patches_1,patches_2,patches_3),axis=-1)
    
    patches_array= np.reshape(patches, (pshape0 * pshape1, 128, 128, 3), order='F')


    return patches_array, pshape0,pshape1



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 128, 128]
        # Output size: [batch, 3, 128, 128]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 64, 64]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 32, 32]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 16, 16]
            nn.ReLU(),
			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 8, 8]
            nn.ReLU(),
            nn.Conv2d(96,192,4,stride=2,padding=1),              # [batch,192, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),  # [batch, 96, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 16, 16]
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 32, 32]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3,128, 128]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x).clamp(min=0.0,max=20.0)
        decoded = self.decoder(encoded)
        return encoded, decoded


def get_encoding(input_array,name_,p1,p2,q_val=100.0):

    
    if input_array.max() > 1.0:
        input_array = input_array.astype('float32') / 255.
        input_array += 1.0
        input_array /= 2.0
    img = read_image(name_)
    img = pad_image(img)
    H,W,D = img.shape
    if img.shape[-1] == 4:
        img = img[:,:,:3]

    input_array = np.transpose(input_array,(0,3,1,2))
    input_array = torch.from_numpy(input_array)
    input_array = Variable(input_array).cuda()
    model = Autoencoder().cuda()
    model.load_state_dict(torch.load("comp_med_101.pt"))
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(input_array).clamp(min=0.0,max=20.0)
  
    encoded = np.array(encoded.cpu().data)
    encoded /= 20.0
    encoded *= q_val
    encoded = encoded.round()
    encoded = encoded.astype('uint8')
    
    name_ = name_.split('/')[-1]
    name_ = name_.split('.')[0]
    
    f_name ='{}_{}_{}_{}_{}.npz'.format(name_,p1,p2,H,W)
    np.savez_compressed(f_name,vals=encoded)
    return f_name
    

def reconstruct_residual(crops,H,W,D,pshape0,pshape1,crop_size):


    patchify_patches= np.reshape(crops, (pshape0,pshape1,crop_size,crop_size,3), order='F')
    
    c1 = patchify_patches[:,:,:,:,0]
    c2 = patchify_patches[:,:,:,:,1]
    c3 = patchify_patches[:,:,:,:,2]
    
    dec1 = unpatchify(c1,(H,W))
    dec2 = unpatchify(c2,(H,W))
    dec3 = unpatchify(c3,(H,W))
    
    img = np.stack((dec1,dec2,dec3),axis=-1)
    img = img.astype('int16')

    
    return img



def get_decoding(name_,p1,p2,q_val = 100.0):
    
    encoded = np.load(name_)
    encoded = encoded['vals']
    encoded = encoded.astype('float32')
    encoded /= q_val
    encoded *= 20.0
    
    encoded = torch.from_numpy(encoded)
    encoded = Variable(encoded).cuda()
    
    model = Autoencoder().cuda()
    model.load_state_dict(torch.load("comp_med_101.pt"))
    model.eval()
    with torch.no_grad():
        decoded = model.decoder(encoded)
    
    decoded = np.array(decoded.cpu().data)
    name_ = name_.split('/')[-1]
    name_ = name_.split('.')[0]
    n, p1,p2,H, W = name_.split('_')
    decoded = np.transpose(decoded,(0,2,3,1))
    decoded *= 2.0
    decoded -= 1.0
    decoded *= 255.
    decoded = decoded.astype('int16')
    img = reconstruct_residual(decoded,int(H),int(W),3,int(p1),int(p2),128)
    return img
    
    



def get_reconstructed_img(name_,q_val=100.0):
    
    name__ = name_.split('/')[-1]
    name__ = name__.split('.')[0]
    n, p1,p2,H, W = name__.split('_')
    W = W.split('.')[0]
    
     
    residual = get_decoding(name_,p1,p2,q_val)
    
    thumb_name = name_.split('_')[-5]
    
    
    thumb_img = read_image(thumb_name+"_thumb.jpeg")
    up_img = cv2.resize(thumb_img,dsize=(int(W),int(H)))
    

#     print(residual.dtype)
    reconstructed_img = residual + up_img
#     print(reconstructed_img.dtype)
    reconstructed_img = np.clip(reconstructed_img,a_min=0,a_max=255)
    return reconstructed_img
    


# In[ ]:


def create_thumbnail(name_):
    img = read_image(name_)
    name_ = name_.split('/')[-1]
    name_ = name_.split('.')[0]
    H,W,D = img.shape
    thumb_img = cv2.resize(img,(int(W/8),int(H/8)),cv2.INTER_AREA)
    thumb_img = thumb_img.astype('uint8')
    img = Image.fromarray(thumb_img)
    img.save("{}_thumb.jpeg".format(name_),qualtiy=80)


# In[ ]:


def get_bpp(f_name,t_name,H,W):
    s1 = os.path.getsize(f_name) * 8.0
    s2 = os.path.getsize(t_name) * 8.0
    bpp = (s1 + s2) / (H * W)
    
    return bpp
    



def compress_img(name):
    
    temp = name.split('.')[-1]
    org_img = read_image(name)
    create_thumbnail(name)
    tn = name.split('/')[-1]
    tn = tn.split('.')[0]
    tn = tn + "_thumb.jpeg"
    crops,p1,p2 = create_crops(name,tn)
    encoded_file = get_encoding(crops,name,p1,p2)
    print("File saved as {}".format(encoded_file))


def decompress_img(name):
    recon = get_reconstructed_img(name)
    name = name.split('/')[-1]
    name = name.split('_')[0]
    plt.imsave('{}_org.png'.format(name),recon)

