import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.utils import spectral_norm
import torchvision
import numpy as np

# -----------------------------------------------------------------------------
# Generator (unchanged except no change to support WGAN)
# -----------------------------------------------------------------------------
class Generator256(nn.Module):
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # 1→4
            nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(),

            # 4→8
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(),

            # 8→16
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(),

            # 16→32
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(),

            # 32→64
            nn.ConvTranspose2d(ngf,    ngf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//2),    nn.ReLU(),

            # 64→128  ← new
            nn.ConvTranspose2d(ngf//2, ngf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//4),    nn.ReLU(),

            # 128→256 ← new
            nn.ConvTranspose2d(ngf//4,    1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# -----------------------------------------------------------------------------
# Discriminator with spectral norm and no sigmoid
# -----------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(1, ndf,    4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(ndf*8, 1,    4, 1, 0, bias=False)),
            # outputs a single-channel feature‐map
        )
    def forward(self, x):
        x = self.net(x)                         # [B,1,H,W]
        x = F.adaptive_avg_pool2d(x, (1,1))     # [B,1,1,1]
        return x.view(-1)                       # [B]  <-- raw scores

# -----------------------------------------------------------------------------
# Gradient penalty for WGAN-GP
# -----------------------------------------------------------------------------
def gradient_penalty(D, real, fake, λ=10.0):
    B, C, H, W = real.shape
    α = torch.rand(B, 1, 1, 1, device=real.device)
    interp = (α * real + (1 - α) * fake).requires_grad_(True)
    d_interp = D(interp)
    grads = grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_norm = grads.view(B, -1).norm(2, dim=1)
    return λ * ((grad_norm - 1) ** 2).mean()
