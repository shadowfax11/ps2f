import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import torchvision
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def np_to_tensor(np):
    return torch.tensor(np).unsqueeze(0)

def tensor_to_np(ts):
    return ts.detach().cpu().squeeze().numpy()

def plot(groundtruth, recons):
    recons = tensor_to_np(recons)
    recons /= np.max(recons)
    plt.figure()
    plt.subplot(121)
    plt.title('Groundtruth')
    plt.axis('off')
    plt.imshow(groundtruth)
    plt.subplot(122)
    plt.title('Reconstruction')
    plt.axis('off')
    plt.imshow((recons))
    plt.show()

def create_3dvol(scene_gt, dmap_gt, z_vals, add_poisson_noise=False):
    if add_poisson_noise:
        scene_gt = 1e-5*np.random.poisson(1e5*scene_gt)
    V_gt = np.zeros((scene_gt.shape[0], scene_gt.shape[1], len(z_vals)))
    for i in range(V_gt.shape[0]):
        for j in range(V_gt.shape[1]):
            d = dmap_gt[i,j]
            idx = np.argmin(np.abs(d-z_vals))
            V_gt[i,j,idx] = scene_gt[i,j]
    return V_gt

class Convolve3DFFT(nn.Module):
    def __init__(self, psf, cuda_device=0):
        super(Convolve3DFFT, self).__init__()

        self.cuda_device = cuda_device
        if len(psf.shape)==3:
            self.d_psf = psf.shape[2]
            self.h_psf = psf.shape[0]
            self.w_psf = psf.shape[1]
            self.pad_h = [ int(self.h_psf//2), int(self.w_psf//2)]
            self.is_4d = False

            psf = np.transpose(psf, axes=(2,0,1))
            self.h_var = torch.nn.Parameter(torch.tensor(psf, dtype=torch.float32, device=self.cuda_device),
                                                requires_grad=False)
            print("Created conv3d obj for PSF of size {:3d}x{:3d}x{:3d}".format(self.h_psf,self.w_psf,self.d_psf))
        if len(psf.shape)==4:
            self.c_psf = psf.shape[3]
            self.d_psf = psf.shape[2]
            self.h_psf = psf.shape[0]
            self.w_psf = psf.shape[1]
            self.pad_h = [ int(self.h_psf//2), int(self.w_psf//2) ]
            self.is_4d = True

            psf = np.transpose(psf, axes=(3,2,0,1))
            self.h_var = torch.nn.Parameter(torch.tensor(psf, dtype=torch.float32, device=self.cuda_device),
                                                requires_grad=False)
            print("Created conv3d obj for PSF of size {:3d}x{:3d}x{:3d}x{:3d}".format(self.h_psf,self.w_psf,self.d_psf,self.c_psf))

    def forward(self, x):
        _, d, h, w = x.shape
        pad_x = [ int(h//2), int(w//2)]
        H = F.pad(self.h_var, (pad_x[1],pad_x[1],pad_x[0],pad_x[0]), 'constant', 0)
        if self.is_4d:
            H = fft.rfft2(fft.ifftshift(H,(2,3)))
        else:
            H = fft.rfft2(fft.ifftshift(H,(1,2)))
        x = F.pad(x, (self.pad_h[1],self.pad_h[1],self.pad_h[0],self.pad_h[0]), 'constant', 0)
        x = fft.rfft2(x)
        x = H*x
        # x = torch.real(fft.ifft2(x))
        x = fft.irfft2(x)
        x = torch.sum(x, 1).unsqueeze(0)
        x = x[:, :, self.pad_h[0]:self.pad_h[0]+h, self.pad_h[1]:self.pad_h[1]+w]
        return x
