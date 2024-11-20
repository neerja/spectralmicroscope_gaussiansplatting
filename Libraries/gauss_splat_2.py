import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device("cpu") # TODO: add GPU later

class GaussObject:
    def __init__(self, muy=0.0, mux=0.0, mul=0.0, sigy=1.0, sigx=1.0, sigl=1.0, amp=1.0, learningrate=0.01):
        self.initParams(muy, mux, mul, sigy, sigx, sigl, amp)
        self.optimizer = torch.optim.Adam([self.mux, self.muy], lr=learningrate)

    def initParams(self,muy,mux,mul,sigy,sigx,sigl, amp): 
        self.mux = torch.tensor(mux, requires_grad=True, device=device)
        self.muy = torch.tensor(muy, requires_grad=True, device=device)
        self.mul = torch.tensor(mul, requires_grad=True, device=device)
        self.sigx = torch.tensor(sigx, requires_grad=True, device=device)
        self.sigy = torch.tensor(sigy, requires_grad=True, device=device)
        self.sigl = torch.tensor(sigl, requires_grad=True, device=device)
        self.covariancematrix = torch.tensor([[sigy**2, 0.0], [0.0, sigx**2]], requires_grad=True, device=device)
        self.amplitude = torch.tensor(amp, requires_grad=True, device=device)

    def __str__(self):
        return f"gaussObject(mu_x = {self.mux}, mu_y = {self.muy}, mu_l = {self.mul}), cov = {self.covariancematrix}"
    
    def computeValues(self, coordinates, ny, nx):
        """Compute the values of the Gaussian object in the given frame."""
        mean = torch.tensor([self.muy, self.mux], device=device)
        covariance_matrix = self.covariancematrix
        multivariate_normal = MultivariateNormal(mean, covariance_matrix)
        gauss_values = multivariate_normal.log_prob(coordinates).exp() * self.amplitude
        return gauss_values.view(ny, nx)

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
        mean = torch.tensor([self.muy, self.mux, self.mul], device=device, requires_grad=True)
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
        
    def createMeshGrid(self, nx, ny):
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
    
    def computeFFTMagValues(self, coordinates, ny, nx):
        """
        Take Fourier Transform of Gaussian object and compute its values, without FFT
        """
        mean = torch.zeros(2, device=device) # only computing magnitude of F(G)
        cov_inv = torch.inverse(self.covariance_matrix)
        cov_inv = 1 / (2 * torch.pi) ** 2 * cov_inv # get covariance matrix of F(G)
        cov_inv = (cov_inv + cov_inv.T)/2.0 # make it positive definite
        
        mvn = MultivariateNormal(mean, cov_inv)
        gauss_f_values = mvn.log_prob(coordinates).exp() * self.amplitude
        return gauss_f_values.view(ny, nx)

def createGaussFilter(covariance_matrix, coordinates, nx, ny, amplitude, sf=1e-8):
    """
    Create a Gaussian filter, which is a 2D Gaussian distribution in the spatial domain.
    """
    mean = torch.zeros(2, device=device)
    # Scale the covariance matrix
    scaleFactor = torch.diag(torch.tensor([sf * nx**2, sf * ny**2], device=device))
    filterVar = torch.matmul(scaleFactor, covariance_matrix)
    # Ensure the covariance matrix is positive-definite
    filterVar = (filterVar + filterVar.T) / 2.0
    # Create a multivariate normal distribution
    mvn = MultivariateNormal(mean, filterVar)
    # Compute the probability density values of the Gaussian filter
    pdf_values = mvn.log_prob(coordinates).exp() * amplitude
    return pdf_values.view(ny, nx)



def createPhasor(x, y, xshift, yshift):
    phase_ramp = 2.0 * torch.pi * (-1 * (xshift * x) - (yshift * y))
    phasor = torch.exp(1j * phase_ramp)
    return phasor, phase_ramp


def createWVFilt(lam, mul, sigl, m):
    gaus_lam = torch.exp(-(lam - mul)**2 / (2 * sigl**2))
    mout = torch.sum(m * gaus_lam, dim=2)
    return mout


# this step calculates the measurement values
def computeMeas(Hfft, pdf_values, phasor, mout):
    # multiply by the Fourier transform of the point spread function
    bfft = Hfft * pdf_values
    bfft2 = bfft * phasor
    bout = torch.fft.ifft2(torch.fft.fftshift(bfft2)) # torch.fft.ifft2(torch.fft.fftshift(bfft2))
     # multiply by the weighted gaussian filter
    # can be seen as amplitude modulation where the output values are scaled by the modulation function.
    b = (torch.abs(bout) * mout)
    return b

# this step calculates the measurement values for a single gaussian object

def forwardSingleGauss(g, coordinates, nx, ny, lam, Hfft, x, y, m):
    pdf_values = createGaussFilter(g.covariancematrix, coordinates, nx, ny, g.amplitude)
    phasor, _ = createPhasor(x, y, g.mux, g.muy)
    mout = createWVFilt(lam, g.mul, g.sigl, m)
    return computeMeas(Hfft, pdf_values, phasor, mout)
