import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import torchvision
import sys
import pdb

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def np_to_tensor(x):
    return torch.tensor(x).unsqueeze(0)

def tensor_to_np(x):
    return x.detach().cpu().squeeze().numpy()

def plot(x, block=True, title=None):
    if torch.is_tensor(x):
        x = tensor_to_np(x)
    C = x.shape[0]
    plt.figure()
    for i in range(C):
        plt.subplot(1,C,i+1); plt.imshow(x[i,:,:].squeeze()); plt.colorbar(); plt.title("Channel {}".format(i+1))
    if title is not None:
        plt.suptitle(title)
    if block:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    return

def plot_volume(x, block=True, title=None):
    plt.figure()
    plt.subplot(2,2,1); plt.title('XY MIP'); plt.imshow(np.max(x,axis=2)); plt.colorbar()
    plt.subplot(2,2,2); plt.title('YZ MIP'); plt.imshow(np.max(x,axis=1)); plt.colorbar()
    plt.subplot(2,2,3); plt.title('XZ MIP'); plt.imshow(np.max(x, axis=0).T); plt.colorbar()
    if title is not None:
        plt.suptitle(title)
    if block:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    return

class Hessian2DNorm():
    def __init__(self):
        pass
    def __call__(self, img):
        # Compute Individual derivatives
        fxx = img[..., 1:-1, :-2] + img[..., 1:-1, 2:] - 2*img[..., 1:-1, 1:-1]
        fyy = img[..., :-2, 1:-1] + img[..., 2:, 1:-1] - 2*img[..., 1:-1, 1:-1]
        fxy = img[..., :-1, :-1] + img[..., 1:, 1:] - \
              img[..., 1:, :-1] - img[..., :-1, 1:]
          
        return torch.sqrt(fxx.abs().pow(2) +\
                          2*fxy[..., :-1, :-1].abs().pow(2) +\
                          fyy.abs().pow(2)).sum()

class Hessian3DNorm():
    def __init__(self):
        pass
    def __call__(self, img):
        # Compute Individual derivatives
        fxx = img[...,1:-1, 1:-1, :-2] + img[...,1:-1, 1:-1, 2:] - 2*img[...,1:-1, 1:-1, 1:-1]
        fyy = img[...,1:-1, :-2, 1:-1] + img[...,1:-1, 2:, 1:-1] - 2*img[...,1:-1, 1:-1, 1:-1]
        fxy = img[...,1:-1, :-1, :-1] + img[...,1:-1, 1:, 1:] - \
                img[...,1:-1, 1:, :-1] - img[...,1:-1, :-1, 1:]
        fzz = img[...,:-2, 1:-1, 1:-1] + img[...,2:, 1:-1, 1:-1] - 2*img[...,1:-1, 1:-1, 1:-1]
        fxz = img[...,:-1, 1:-1, :-1] + img[...,1:, 1:-1, 1:] - \
                img[...,1:, 1:-1, :-1] - img[...,:-1, 1:-1, 1:]
        fyz = img[...,:-1, :-1, 1:-1] + img[...,1:, 1:, 1:-1] - \
                img[...,1:, :-1, 1:-1] - img[...,:-1, 1:, 1:-1]
          
        return torch.sqrt(fxx.abs().pow(2) +\
                          2*fxy[..., :-1, :-1].abs().pow(2) +\
                          fyy.abs().pow(2) + fzz.abs().pow(2) +\
                          2*fxz[...,:-1, :, :-1].abs().pow(2) + 2*fyz[...,:-1,:-1,:].abs().pow(2) ).sum()

class TV2DNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, img):
        grad_x = img[..., 1:, 1:] - img[..., 1:, :-1]
        grad_y = img[..., 1:, 1:] - img[..., :-1, 1:]
        
        if self.mode == 'isotropic':
            #return torch.sqrt(grad_x.abs().pow(2) + grad_y.abs().pow(2)).mean()
            return torch.sqrt(grad_x**2 + grad_y**2).sum()
        elif self.mode == 'l1':
            return abs(grad_x).sum() + abs(grad_y).sum()
        elif self.mode == 'hessian':
            return Hessian2DNorm()(img)
        else:
            return (grad_x.pow(2) + grad_y.pow(2)).sum()     
       
class TV3DNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, img):
        grad_x = img[...,1:, 1:, 1:] - img[...,1:, 1:, :-1]
        grad_y = img[...,1:, 1:, 1:] - img[...,1:, :-1, 1:]
        grad_z = img[...,1:, 1:, 1:] - img[...,:-1, 1:, 1:]
        
        if self.mode == 'isotropic':
            #return torch.sqrt(grad_x.abs().pow(2) + grad_y.abs().pow(2)).mean()
            return torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2).sum()
        elif self.mode == 'l1':
            return abs(grad_x).sum() + abs(grad_y).sum() + abs(grad_z).sum() 
        elif self.mode == 'hessian':
            return Hessian3DNorm()(img)
        else:
            return (grad_x.pow(2) + grad_y.pow(2) + grad_z.pow(2)).sum()     

class Convolve3DFFT(nn.Module):
    def __init__(self, psf, cuda_device):
        super(Convolve3DFFT, self).__init__()

        self.cuda_device = cuda_device
        self.c_psf = psf.shape[3]
        self.d_psf = psf.shape[2]
        self.h_psf = psf.shape[0]
        self.w_psf = psf.shape[1]
        self.pad_h = [ int(self.h_psf//2), int(self.w_psf//2) ]
        psf = np.transpose(psf, axes=(3,2,0,1))
        self.h_var = torch.nn.Parameter(torch.tensor(psf, dtype=torch.float32, 
                                            device=self.cuda_device), requires_grad=False)
        print("Created conv3d obj for PSF of size {:3d}x{:3d}x{:3d}x{:3d}".format(self.h_psf,self.w_psf,self.d_psf,self.c_psf))                                    

    def forward(self, x):
        _, d, h, w = x.shape
        pad_x = [ int(h//2), int(w//2) ]
        H = F.pad(self.h_var, (pad_x[1],pad_x[1],pad_x[0],pad_x[0]), 'constant', 0)
        H = fft.fft2(fft.ifftshift(H, (2,3)))
        x = F.pad(x, (self.pad_h[1],self.pad_h[1],self.pad_h[0],self.pad_h[0]), 'constant', 0)
        x = fft.fft2(x)
        x = H*x
        x = torch.real(fft.ifft2(x))
        x = torch.sum(x, 1, keepdim=True)
        x = x[:, :, self.pad_h[0]:self.pad_h[0]+h, self.pad_h[1]:self.pad_h[1]+w]
        return x

class NN_constraint(nn.Module):
    def __init__(self):
        super(NN_constraint, self).__init__()

    def forward(self, x):
        return F.relu(x)
        # return torch.clamp(x, min=0)
