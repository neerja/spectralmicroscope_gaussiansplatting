# gausssplat.py - library of functions for gaussian splatting for spectral diffuserscope
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import sdc_config3 as sdc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussObject:

    def __init__(self, muy=0.0, mux=0.0, mul=0.0, sigy=1.0, sigx=1.0, sigl=1.0, amp=1.0, learningrate=0.01):
        self.initParams(muy, mux, mul, sigy, sigx, sigl, amp, learningrate)

    def initParams(self,muy,mux,mul,sigy,sigx,sigl,amp,learningrate): 
        self.mux = torch.tensor(mux, requires_grad=True, device=device)
        self.muy = torch.tensor(muy, requires_grad=True, device=device)
        self.mul = torch.tensor(mul, requires_grad=True, device=device)
        self.sigx = torch.tensor(sigx, requires_grad=True, device=device)
        self.sigy = torch.tensor(sigy, requires_grad=True, device=device)
        self.sigl = torch.tensor(sigl, requires_grad=True, device=device)
        self.amplitude = torch.tensor(amp, requires_grad=True, device=device)
        self.learningrate = learningrate
#         self.optimizer = torch.optim.Adam([self.mux, self.muy, self.mul, self.sigx, self.sigy, self.sigl], lr=self.learningrate)
        self.optimizer = torch.optim.Adam([self.mux, self.muy], lr=self.learningrate)

    def __str__(self):
        return f"gaussObject(mu_x = {self.mux}, mu_y = {self.muy}, mu_l = {self.mul}), sig_x = {self.sigx}, sig_y = {self.sigy}, sig_l = {self.sigl}"
    
    def computeValues(self, coordinates, ny, nx):
        """Compute the values of the Gaussian object at the given coordinates."""
        mean = torch.tensor([self.muy, self.mux], device=device)
        covariance_matrix = torch.tensor([[self.sigy**2, 0.0], [0.0, self.sigx**2]])
        multivariate_normal = MultivariateNormal(mean, covariance_matrix)
        pdf_values = multivariate_normal.log_prob(coordinates).exp() * self.amplitude
        return pdf_values.view(ny, nx)
    
    def plot(self, coordinates, ny, nx):
        """
        Plot the values of the Gaussian object.
        """
        # Compute the values of the Gaussian object at the given coordinates
        pdf_values = self.computeValues(coordinates, ny, nx)

        # Plot the values
        plt.figure()
        plt.imshow(pdf_values.detach().cpu().numpy())  # Move to CPU for plotting
        plt.title("2D Gaussian Object")
        plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y')

    def gradStep(self):
        """
        Take a gradient step for the optimizer.
        """
        self.optimizer.step()

    def zeroGrad(self):
        """
        Zero the gradients of the parameters in the optimizer.
        """
        self.optimizer.zero_grad()
        
    def createPixelGauss(self, ny, nx, nl):
        """
        Create 3D pixel representation of Gaussian
        """
        mean = torch.tensor([self.muy, self.mux, self.mul], device=device)
        scale_factors = torch.tensor([self.sigy**2, self.sigx**2, self.sigl**2], device=device)
        covariance_matrix = torch.diag(scale_factors)  # No covariance, diagonal matrix

        coords = [torch.arange(-s // 2, s // 2, device=device, dtype=torch.float32) for s in [ny,nx,nl]]
        grid = torch.meshgrid(coords, indexing='ij')
        coordinates = torch.stack([g.flatten() for g in grid], dim=1)

        mvn = MultivariateNormal(mean, covariance_matrix)
        pdf_values = mvn.log_prob(coordinates).exp() * self.amplitude
        return pdf_values.view(*[ny,nx,nl])
    
    def plot3D(self, ny, nx, nl):
        g3d = self.createPixelGauss(ny,nx,nl)
        g3d_proj = g3d.sum(dim=2).detach()

        plt.figure()
        plt.imshow(g3d_proj.numpy())
        plt.colorbar()
        plt.title('2D Projection of 3D Gaussian')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


def createMeshGrid(nx, ny):
    """
    Create a 2D grid with the given dimensions.
    Moved to device for GPU compatibility.
    """
    # Create a 2D grid
    y, x = torch.meshgrid(
        # Create a 1D grid along the y direction
        torch.linspace(-ny/2, ny/2, ny, device=device),
        # Create a 1D grid along the x direction
        torch.linspace(-nx/2, nx/2, nx, device=device),
        # Use the indexing convention from MATLAB
        indexing="ij"
    )
    # Flatten and combine to create coordinate pairs
    coordinates = torch.stack([y.flatten(), x.flatten()], dim=1)
    return [x, y, coordinates]

def createGaussFilter(sigx, sigy, nx, ny, amplitude):
    [x,y,coordinates] = createMeshGrid(nx, ny)
    scale_x = nx / (2 * np.pi)
    scale_y = ny / (2 * np.pi)

    coords_x, coords_y = coordinates[:, 1], coordinates[:, 0]
    gauss_filt = torch.exp(-0.5 * ((coords_x * sigx / scale_x) ** 2 + (coords_y * sigy / scale_y) ** 2)) * amplitude
    gauss_filt = gauss_filt.reshape(ny, nx)
    gauss_filt = gauss_filt / torch.amax(gauss_filt)

    return gauss_filt

def createGaussFilterPadded(sigx, sigy, nx, ny, amplitude):
    '''
    Create the magnitude of the Fourier transform of the Gauss object, at twice the resolution
    '''
    [x_padded,y_padded,coordinates_padded] = createMeshGrid(nx*2, ny*2)
    gauss_filt_padded = createGaussFilter(sigx, sigy, coordinates_padded, nx*2, ny*2, amplitude)
    return gauss_filt_padded

def createPhasor(nx, ny, mux, muy):
    '''
    Create the phase of the Fourier transform of the Gauss object
    '''
    [x,y,coordinates] = createMeshGrid(nx, ny)
    phase_ramp = 2.0 * torch.pi * (- (mux * x / x.shape[1]) - (muy * y / y.shape[0]))
    phasor = torch.cos(phase_ramp) + 1j * torch.sin(phase_ramp)
    return phasor, phase_ramp


def createPhasorPadded(nx, ny, mux, muy):
    '''
    Create the phase of the Fourier transform of the Gauss object, at twice the resolution
    '''
    [x_padded,y_padded,coordinates_padded] = createMeshGrid(nx*2, ny*2)
    return createPhasor(x_padded, y_padded, mux, muy)

def createWVFilt(mul, sigl, nl, m):
    coords = torch.arange(-nl // 2, nl // 2)
    gaus_lam = torch.exp(-0.5 * ((coords - mul) / sigl)**2)
    gaus_lam = gaus_lam / gaus_lam.sum()
    mout = torch.sum(m * gaus_lam, dim=2)
    return gaus_lam, mout

def computeMeas(Hfft,pdf_values,phasor, mout):
    bfft = torch.mul(Hfft, pdf_values)
    bfft2 = torch.mul(bfft,phasor)
    bout = torch.fft.ifft2(torch.fft.fftshift(bfft2))
    b = torch.mul(torch.abs(bout),mout) # TODO: figure out why the abs is needed. i.e. why it's becoming negative with certain phaseramps. Maybe aliasing?
    b_norm = b / torch.norm(b)
    return b

# this step calculates the measurement values for a single gaussian object

def computeMeas(Hfft,pdf_values,phasor, mout):
    bfft = torch.mul(Hfft, pdf_values)
    bfft2 = torch.mul(bfft,phasor)
    bout = torch.fft.ifft2(torch.fft.fftshift(bfft2))
    b = torch.mul(torch.abs(bout),mout) # TODO: figure out why the abs is needed. i.e. why it's becoming negative with certain phaseramps. Maybe aliasing?
    return b

def forwardSingleGauss(g, nx, ny, nl, h_expanded, m):
    gauss_fm_padded = createGaussFilter(g.sigx, g.sigy, nx, ny, g.amplitude)
    gauss_fp_padded, _ = createPhasor(nx, ny, g.mux, g.muy)
    gaus_lam, mout = createWVFilt(g.mul, g.sigl, nl, m)
    hf = torch.fft.fftshift(torch.fft.fft2(h_expanded))
    b = computeMeas(hf, gauss_fm_padded, gauss_fp_padded, mout)
    return b