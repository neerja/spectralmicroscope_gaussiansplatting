# gausssplat.py - library of functions for gaussian splatting for spectral diffuserscope
# Neerja Aggarwal
# Date created: Jan 2nd, 2024 
# Last updated: Jan 2nd, 2024
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussObject:
    def __init__(self, muy = 0.0, mux = 0.0, mul = 0.0, sigy =1.0, sigx = 1.0, sigl = 1.0, amp = 1.0): #TODO: add more parameters
        self.initParams(muy,mux,mul,sigy,sigx,sigl, amp)

    def initParams(self,muy,mux,mul,sigy,sigx,sigl, amp): # only used at initialization
        self.mux = torch.tensor(mux,requires_grad = True)
        self.muy = torch.tensor(muy,requires_grad = True)
        self.mul = torch.tensor(mul,requires_grad = True)
        self.sigl = torch.tensor(sigl,requires_grad = True)
        self.covariancematrix = torch.tensor([[sigy**2, 0.0], [0.0, sigx**2]], requires_grad=True)
        self.amplitude = torch.tensor(amp,requires_grad = True)

    def __str__(self):
        return f"gaussObject(mu_x = {self.mux}, mu_y = {self.muy}, mu_l = {self.mul}), cov = {self.covariancematrix}"
    
    def computeValues(self, coordinates,ny,nx):
        mvn = MultivariateNormal(torch.Tensor([self.muy,self.mux]), self.covariancematrix)
        pdf_values = mvn.log_prob(coordinates).exp() * self.amplitude
        pdf_values = pdf_values.view(ny, nx) 
        return pdf_values

    def plot(self,coordinates, ny, nx):
        pdf_values = self.computeValues(coordinates, ny, nx)
        plt.figure()
        plt.imshow(pdf_values.detach().numpy())
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y')

    def gradStep(self,learningrate):
        self.mux.data = self.mux.data - learningrate[0]*self.mux.grad.data
        self.muy.data = self.mux.data - learningrate[1]*self.muy.grad.data
        self.mul.data = self.mux.data - learningrate[2]*self.mul.grad.data
        self.covariancematrix.data = self.covariancematrix.data - learningrate[3]*self.covariancematrix.grad.data
        self.sigl.data = self.mux.data - learningrate[4]*self.sigl.grad.data
        self.amplitude.data = self.amplitude.data - learningrate[5]*self.amplitude.grad.data

    def zeroGrad(self):
        self.mux.grad.data.zero_()
        self.muy.grad.data.zero_()
        self.mul.grad.data.zero_()
        self.sigl.grad.data.zero_()
        self.amplitude.grad.data.zero_()
        self.covariancematrix.grad.data.zero_()
    

def createMeshGrid(nx,ny):  
    # Create a 2D grid
    y, x = torch.meshgrid(torch.linspace(-ny/2, ny/2, ny), torch.linspace(-nx/2, nx/2, nx), indexing="ij" )  #[y,x]
    x.requires_grad_(False)
    y.requires_grad_(False)
    # Flatten the grid to obtain coordinates for evaluation
    coordinates = torch.stack([y.flatten(), x.flatten()], dim=1)  # mvn is y then x also
    return [x,y, coordinates]


def createGaussFilter(covariance_matrix, coordinates,nx,ny, amplitude, sf = 1):
    # sf = empirical scaling factor to make filter roloff in the right place
    # compute inverse of gauss object covariance matrix to use as the filter variance
    scaleFactor = torch.tensor([[sf*ny**2, 0.0],[  0.0, sf*nx**2]]) 
    sigx = covariance_matrix[1,1]
    sigy = covariance_matrix[0,0]
    covinv = torch.tensor([[1/sigy**2, 0.0],[0.0, 1/sigx**2]])
    filterVar = torch.matmul(scaleFactor,covinv)
    filterVar = (filterVar + filterVar.t()) / 2.0  # ensure that it's positive-definite
    # create a multivariate normal distribution with the filter variance centered at the origin
    mean = torch.tensor([0.0, 0.0])
    mvn = MultivariateNormal(mean, filterVar)
    # Evaluate the PDF at each point in the grid
    pdf_values = mvn.log_prob(coordinates).to(torch.complex64).exp() * amplitude  
    pdf_values = pdf_values.view(ny, nx)
    pdf_values = pdf_values/torch.amax(torch.abs(pdf_values))
    return pdf_values



def createPhasor(x,y, xshift,yshift):
    freq_x = xshift # Adjust this value to control the frequency of the phase ramp
    freq_y = yshift

    phase_ramp =  2.0 * torch.pi * (-1*(freq_x * x) - (freq_y * y))
    phasor = torch.exp(1j*phase_ramp)
    return (phasor, phase_ramp)


def createWVFilt(lam, mul, sigl,m):
    gaus_lam = torch.exp(-(lam-mul)**2/(2*sigl**2))
    mout = torch.sum(torch.mul(m,gaus_lam), dim=2)
    return mout


def computeMeas(Hfft,pdf_values,phasor, mout):
    bfft = torch.mul(Hfft, pdf_values)
    bfft2 = torch.mul(bfft,phasor)
    bout = torch.fft.ifft2(torch.fft.fftshift(bfft2))
    b = torch.mul(torch.abs(bout),mout) # TODO: figure out why the abs is needed. i.e. why it's becoming negative with certain phaseramps. Maybe aliasing?
    return b


def forwardSingleGauss(g, coordinates, nx, ny, lam, Hfft, x, y, m, sf = 1):
    pdf_values = createGaussFilter(g.covariancematrix, coordinates, nx, ny, g.amplitude, sf)
    (phasor,phase_ramp) = createPhasor(x,y,g.mux,g.muy)
    mout = createWVFilt(lam,g.mul,g.sigl,m)
    b = computeMeas(Hfft,pdf_values,phasor,mout)
    return b
