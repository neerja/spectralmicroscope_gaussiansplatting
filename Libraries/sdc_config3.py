#configuration file

import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch.nn as torchnn
import torch
import torchvision.transforms
import csv

xglobal = 0
losslist = 0
l2losslist = 0

gpu = 2
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")

def sumFilterArray(filterstack,wv,wvmin,wvmax,wvstep):
    #filterstack is ndarray.  wvmin, wvmax, wvstep are scalers
    #returns new ndarray msum with dimensions yx same as filterstack, and lambda depending on length of wvnew
    wvnew = np.arange(wvmin,wvmax+wvstep,wvstep)
    #resample the filterarray
    #find where in wvindex meets wvnew
    j0 = np.where(wvnew[0]==wv)[0][0]
    (dim0,dim1,dim2) = filterstack.shape
    dim2 = len(range(len(wvnew)))
    msum = np.zeros((dim0,dim1,dim2))

    for k in range(dim2):
        #get index according to wv
        #sum and add
        if k<dim2-1:
            j1 = np.where(wvnew[k+1]==wv)[0][0]
        else:
            j0 = np.where(wvmax==wv)[0][0] #handle the last index
            j1 = np.where(wvmax+wvstep==wv)[0][0]
        msum[:,:,k] = np.sum(filterstack[:,:,j0:j1],axis = 2)
        j0=j1
    return msum

#import specific tiff file from directory
#inputs - datafolder (string): directory, fname (string): specific file
#outputs - imarray (numpy array) containing tiff file as image
def importTiff(datafolder,fname):

    im = Image.open(os.path.join(datafolder, fname))
    imarray = np.array(im)
    return imarray

#import all tiff files from a directory
#inputs - path (string): directory
#outputs - imageStack (torch array): stack of tiff images along dim 2
def tif_loader(path):
    fnames = [fname for fname in os.listdir(path) if fname.endswith('.tif')]
    for ii in range(len(fnames)):
        file = fnames[ii]
        im = Image.open(os.path.join(path,file))
        im = torchvision.transforms.ToTensor()(im)
        if ii==0:
            imageStack = torch.zeros((im.shape[1],im.shape[2],len(fnames)))
        imageStack[:,:,ii] = im  
    return imageStack

def bksub(im,bk,desireddtype='float32'):
    #make into signed integers
    desireddtype='float32'
    a = im.astype(desireddtype)-bk.astype(desireddtype)
    return a

def cropci(im,ci):
    return im[ci[1]:ci[3],ci[0]:ci[2]]

def resample(psf,oldpix=1.67,newpix=5.3):
    zoom = oldpix/newpix
    s = psf.shape
    newsize = (int(s[1]*zoom),int(s[0]*zoom)) #flip to x,y
    pilpsf = Image.fromarray(psf)
    pilpsfzoom = pilpsf.resize(newsize)
    psf0 = np.array(pilpsfzoom)
    return psf0

def importFilterStack(datafolder,fname):
    annots = loadmat(os.path.join(datafolder, fname))
    wv = np.squeeze(annots['wv'])
    ci = annots['ci'][0]
    #add one to avoid divide by 0 error
    filterstack = annots['filterstack']+1 
    return (wv,ci,filterstack)

def psfcrop(psf,st,size):
    psf = psf[st[0]:st[0]+size[0],st[1]:st[1]+size[1]]
    return psf

def setupGPU(device_no):
    #device_no = 2 #CHANGE THIS TO WHICHEVER GPU YOU WANT TO USE
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    torch.cuda.set_device(device_no)
    device = torch.device("cuda:"+str(device_no) if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(device_no))
    # device = 'cpu'  #ACTIVATE TO SWITCH TO cPU
    if device != 'cpu':
        print('using gpu')
    return device

def forwardmodel3d(xpad,hfftpad,m):
    # use fft to do convolution between x and h.
    # torch does fft on LAST two dimensions
    y = torch.fft.fftshift(torch.fft.ifft2(torch.multiply(fft3d(xpad),hfftpad),dim=(0,1)),dim=(0,1))
    y = crop(y)
    y = torch.multiply(y,m)
    y = torch.abs(torch.sum(y, dim=2))
    return y
    
def inversemodel(y,hfftpad,m):    
    #use fft to do deconvolution between y and h
    dims = m.shape
    y = y.repeat(dims[2],1,1).permute(1,2,0) #cascade it along the third dimension
    y = torch.divide(y,m) #divide by m
    y = pad(y) #pad y
    y = torch.fft.fft2(y,dim=(0,1)) #fft
    x = torch.fft.fftshift(torch.fft.ifft2(torch.divide(y,hfftpad),dim=(0,1)),dim=(0,1)) #deconv
    return torch.abs(x)

def adjointoperation(y,hfftpad,m):    
    #use fft to do deconvolution between y and h
    dims = m.shape
    y = y.repeat(dims[2],1,1).permute(1,2,0) #cascade it along the third dimension
    y = torch.multiply(y,m) #still do multiplyication for adjoint
    y = pad(y) #pad y
    y = fft3d(y) #fft
    x = torch.fft.fftshift(torch.fft.ifft2(torch.multiply(y,torch.conj(hfftpad)),dim=(0,1)),dim=(0,1)) #deconv
    return torch.real(x)

def softthresh(x,thresh):
    #for all values less than thresh, set to 0, else subtract thresh.  #maintain pos/neg signage
    xout = torch.maximum(torch.abs(x)-thresh,torch.tensor(0))
    return torch.multiply(torch.sign(x),xout) #need to implement complex softthresh. 

def nonneg(x,thresh=0): 
    #set negative values to zero
    return torch.maximum(x,torch.tensor(0)) 

def computel2loss(resid):
    l2loss = torch.linalg.norm(resid)
    return l2loss.cpu().detach().numpy()

def computel1loss(x):
    l1loss = torch.sum(torch.abs(x))
    return l1loss.cpu().detach().numpy()

def pad(x):
    dims = x.shape
    if len(dims) == 3:
        return torch.nn.functional.pad(x, (0, 0, int(dims[1]/2), int(dims[1]/2), int(dims[0]/2), int(dims[0]/2)), mode='constant', value=0)
    return torch.nn.functional.pad(x, (int(dims[1]/2), int(dims[1]/2), int(dims[0]/2), int(dims[0]/2)), mode='constant', value=0)
    
def crop(x):
    dims = x.shape
    st0 = int(dims[0]/4)
    st1 = int(dims[1]/4)
    end0 = int(3*dims[0]/4)
    end1 = int(3*dims[1]/4)
    
    if len(dims) == 3:
        return x[st0:end0, st1:end1, :]
    return x[st0:end0, st1:end1]

def flatten(x):
    return torch.sum(x,dim=2)
    
def make3d(x,l):
    return x.repeat(l,1,1).permute(1,2,0)

def fft3d(x):
    return torch.fft.fft2(x,dim=(0,1))

def makedark(datafolder,pixind=(100,100)):
    # gets all tiff files in datafolder: filepath and averages them
    # returns tuple: (dark avg: numpy 2D array, dark pixel at index pixind over time: numpy 1D array)
    fnames = [fname for fname in os.listdir(datafolder) if fname.endswith('.tif')]
    imshape = (1024,1280,len(fnames))
    imstack = np.empty(imshape)
    #get all files in this folder
    for k in range(len(fnames)):
        fname = fnames[k]
        im = importTiff(datafolder,fname) 
        imstack[:,:,k] = im
    darkavg = np.sum(imstack,2)/len(fnames)
    darkpix = imstack[pixind[0],pixind[1],:]
    return (darkavg,darkpix)

def avgFrames(datafolder,imgind = 0,maxFrame = 1000):
    # gets all tiff files in datafolder: filepath and averages them
    # returns tuple: (imgavg: numpy 2D array, imgframe: single image also numpy 2D array)
    # stops at maxFrame or number of tiff files in folder, whichever is smaller
    fnames = [fname for fname in os.listdir(datafolder) if fname.endswith('.tif')]
    imshape = (1024,1280,len(fnames))
    imstack = np.empty(imshape)
    #get all files in this folder
    for k in range(len(fnames)):
        fname = fnames[k]
        im = importTiff(datafolder,fname) 
        imstack[:,:,k] = im
        if k == maxFrame: 
            continue
    imgavg = np.sum(imstack,2)/len(fnames)
    imgframe = imstack[:,:,imgind]
    return (imgavg,imgframe)

def fistaloop3dGPU (xk,h,m,ytrue,specs):
    try:  #put everything inside a try/except loop to help free up memory in case of KeyboardInterrupt
        alpha = 0.1
        tau = 0.08
        kmax = specs['iterations']
        step_size = specs['step_size']
        tau1 = specs['tau1']
        thresh = tau1*step_size # threshold is equal to tau
        kprint = specs['print_every']
        # TDOO: add an if statement for total variation
        if specs['prior'] == 'soft-threshold':
            prox = lambda x, tmax: nonneg(softthresh(x,tmax))
            computeloss = lambda r,x: computel2loss(r)**2 + tau1*computel1loss(x) #weighted sum, remember to square the l2-loss!
        if specs['prior'] == 'non-negativity':
            prox = nonneg
            computeloss = lambda r,x: computel2loss(r)**2 #remember to square the l2-loss!
        if specs['prior'] == '3dtv':
            prox = tv3dApproxHaar
            computeloss = lambda r,x: computel2loss(r)**2
        else: #do nothing and just return x
            prox = lambda x, tmax : x
            computeloss = lambda r,x: computel2loss(r)**2 #remember to square the l2-loss!
        xkm1 = xk
        kcheck = specs['listevery']

        # store into global variables in case function doesn't run to completion
        global xglobal
        global losslist
        global l2losslist
        losslist = np.zeros(0) #reinitialize 
        l2losslist = np.zeros(0) #reinitialize 
        xglobal = torch.zeros_like(xk).to('cpu')  #keep off the gpu to save memory

        tk = 1 #fista variable
        vk = xk #fista variable

        # gradient descent loop
        for k in range(kmax):
            if np.mod(k,kcheck)==0 or k==kmax-1:
                print(k)
                xglobal = xk.cpu() # create a copy on cpu
            #compute yest
            yest = forwardmodel3d(xk,h,m)
            #compute residual
            resid = yest-ytrue #unpadded
            #gradient update
            gradupdate = adjointoperation(resid,h,m) #remember to use adjoint instead of inverse!
            vk = vk - step_size*gradupdate
            #proximal update
# uncomment the line below to use priors other than non-negativity
#             xk = prox(vk,thresh)
            xk = torch.maximum(vk,torch.tensor(0))
            #fista code - from Beck et al paper
            tkp1 = (1 + np.sqrt(1+4*np.square(tk)))/2
            vkp1 = xk + (tk - 1)/(tkp1)*(xk - xkm1)
            #update fista variables
            vk = vkp1
            tk = tkp1
            xkm1 = xk

            #compute loss
            l2loss = computel2loss(resid)
            totloss = computeloss(resid,xk) # depends on prior
            losslist = np.append(losslist,totloss)
            l2losslist = np.append(l2losslist,l2loss)
            
#             print(k)

            # plot every once in a while
            if np.mod(k,kprint)==0 or k==kmax-1:
                plt.figure(figsize = (12,4))
                plt.subplot(1,3,1)
                xcrop = crop(flatten(xk)).cpu().detach().numpy() #zoom to centerish
                plt.imshow(xcrop)
                plt.title('X Estimate (Zoomed)')
                plt.subplot(1,3,2)
                ycrop = crop(yest).cpu().detach().numpy() #zoom to centerish
                plt.imshow(ycrop)
                plt.title('Y Estimate (Zoomed)')
                plt.subplot(1,3,3)
                #plt.plot(losslist,'b')
                plt.plot(l2losslist,'r')
                plt.xlabel('Iteration')
                plt.ylabel('Cost Function')
                plt.title('L2 Loss Only')
                plt.tight_layout()
                plt.show()

        return (xk,gradupdate,vk)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        return
        


def loadspectrum(path):
    file = open(path)
    csvreader = csv.reader(file)
    header = next(csvreader)
    rows = []
    for row in csvreader:
    #     print(row)
        rows.append(row)
    file.close()
    spec = rows[32:-1]
    wavelength = []
    intensity = []
    for ii in spec:
        vals = ii[0].split(';')
        wavelength.append(float(vals[0]))
        intensity.append(float(vals[1]))
    return (wavelength,intensity)

# make RGB False Color representation of hyperspectral datacube
# inputs: reflArray (numpy or torch array): hyperspectral data cube with wavelength along dim 2
# output: stackedRGB (numpy array): RGB image along dim 2
def stack_rgb_opt(reflArray, lams, scaling = [1,1,2.5]):

    red = gauss_red(lams)
    green = gauss_green(lams)
    blue = gauss_blue(lams)
    
    red = red/np.max(red)
    green = green/np.max(green)
    blue = blue/np.max(blue)
    
    reflArray = reflArray/np.max(reflArray)
    
    red_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    green_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    blue_channel = np.zeros((reflArray.shape[0], reflArray.shape[1]))
    
    for i in range(0,len(lams)):
        red_channel = red_channel + reflArray[:,:,i]*red[i]*scaling[0]
        green_channel = green_channel + reflArray[:,:,i]*green[i]*scaling[1]
        blue_channel = blue_channel + reflArray[:,:,i]*blue[i]*scaling[2]
        
    red_channel = red_channel/len(lams)
    green_channel = green_channel/len(lams)
    blue_channel = blue_channel/len(lams)

    stackedRGB = np.stack((red_channel,green_channel,blue_channel),axis=2)

    return stackedRGB

#helper functions
def gauss_green(x):
    std = 42
    mean = 550
    return 1/(std*np.sqrt(6.28))*np.exp(-(x-mean)**2/(2*std**2))

def gauss_red(x):
    std = 50
    mean = 625
    return 1/(std*np.sqrt(6.28))*np.exp(-(x-mean)**2/(2*std**2))

def gauss_blue(x):
    std = 42
    mean = 450
    return 1/(std*np.sqrt(6.28))*np.exp(-(x-mean)**2/(2*std**2))


# ERIC'S CODE


import torch.nn as nn

def A_op(obj,H,padded_filt):
    padded_obj = padPow2_3D(obj)
    OBJ = torch.fft.fftn(padded_obj,dim=(0,1))
    meas = torch.sum(padded_filt*torch.fft.ifftshift(torch.fft.ifftn(OBJ*H,dim=(0,1)),dim=(0,1)),2)
    meas = crop(obj.shape[0:2],meas)
    return meas.real

def Aadj_op(b, padded_psf, filt):
    Atb = []
    padded_b = padPow2_3D(b.unsqueeze(-1).repeat(1,1,filt.shape[2])*filt)
    B = torch.fft.fftn(padded_b,dim=(0,1))
    x = torch.fft.ifftshift(torch.fft.ifftn(B*torch.conj(padded_psf),dim=(0,1)),dim=(0,1))
    return crop(psf.shape,x.real)

def nonneg_soft_thresh(gamma, signal):
    binary = (signal-gamma) > 0
    sign = torch.sign(signal)
    return (signal.abs()-gamma) * binary * sign

def soft_py(x, tau):
    threshed = torch.maximum(torch.abs(x)-tau, torch.tensor(0))
    threshed = threshed*torch.sign(x)
    return threshed


# total variation functions
def ht3(x, ax, shift, thresh):
    C = 1./np.sqrt(2.)
    
    if shift == True:
        x = torch.roll(x, -1, dims = ax)
    if ax == 0:
        w1 = C*(x[1::2,:,:] + x[0::2, :, :])
        w2 = soft_py(C*(x[1::2,:,:] - x[0::2, :, :]), thresh)
    elif ax == 1:
        w1 = C*(x[:, 1::2,:] + x[:, 0::2, :])
        w2 = soft_py(C*(x[:,1::2,:] - x[:,0::2, :]), thresh)
    elif ax == 2:
        w1 = C*(x[:,:,1::2] + x[:,:, 0::2])
        w2 = soft_py(C*(x[:,:,1::2] - x[:,:,0::2]), thresh)
    return w1, w2

def iht3(w1, w2, ax, shift, shape):
    
    C = 1./np.sqrt(2.)
    y = torch.zeros(shape,device=device)

    x1 = C*(w1 - w2); x2 = C*(w1 + w2); 
    if ax == 0:
        y[0::2, :, :] = x1
        y[1::2, :, :] = x2
     
    if ax == 1:
        y[:, 0::2, :] = x1
        y[:, 1::2, :] = x2
    if ax == 2:
        y[:, :, 0::2] = x1
        y[:, :, 1::2] = x2
        
    
    if shift == True:
        y = torch.roll(y, 1, dims = ax)
    return y

def tv3dApproxHaar(x, tau, alpha):
    D = 3
    fact = np.sqrt(2)*2

    thresh = D*fact
    
    y = torch.zeros_like(x,device=device)
    for ax in range(0,len(x.shape)):
        if ax ==2:
            t_scale = alpha
        else:
            t_scale = tau;

        w0, w1 = ht3(x, ax, False, thresh*t_scale)
        w2, w3 = ht3(x, ax, True, thresh*t_scale)
        
        t1 = iht3(w0, w1, ax, False, x.shape)
        t2 = iht3(w2, w3, ax, True, x.shape)
        y += t1 + t2
        
    y = y/(2*D)
    return y
