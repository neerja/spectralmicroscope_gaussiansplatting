import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

"""
General functions for diffuserscope image processing
"""



def fftshift(x):
    return torch.fft.fftshift(x, dim=(0, 1))

def ifftshift(x):
    return torch.fft.ifftshift(x, dim=(0, 1))

def convolve(x, h):
    # Assumes x, h have same size
    x_padded = pad(x)
    h_padded = pad(h)
    
    # Center the data
    x_centered = torch.fft.fftshift(x_padded, dim=(0, 1))
    h_centered = torch.fft.fftshift(h_padded, dim=(0, 1))
    
    # Compute FFT
    x_fft = torch.fft.fftn(x_centered, dim=(0, 1))
    h_fft = torch.fft.fftn(h_centered, dim=(0, 1))
    
    # Convolve in frequency domain
    result_fft = x_fft * h_fft
    
    # Inverse FFT and shift back
    result = ifftshift(torch.fft.ifftn(result_fft, dim=(0, 1)))
    
    # Crop the result back to the original size
    return crop(result.real)
    