"""model.py"""
import sys
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import json
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))
from model.nn import MLP as MLP_ilcm


def reparametrize(mu, logvar):
    std = logvar.div(2).exp() + 1e-6
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def compute_kl(z_1, z_2, logvar_1, logvar_2):
    var_1 = logvar_1.exp() + 1e-6
    var_2 = logvar_2.exp() + 1e-6
    return var_1/var_2 + ((z_2-z_1)**2)/var_2 - 1 + logvar_2 - logvar_1


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3, pcl=False, architecture="standard_conv", image_shape=None):
        super(BetaVAE_H, self).__init__()
        self.pcl = pcl
        self.z_dim = z_dim
        self.nc = nc
        self.architecture = architecture
        self.image_shape = image_shape
        if self.architecture == "standard_conv":
            assert self.image_shape == (nc, 64, 64)
            self.encoder = nn.Sequential(
                nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
                nn.ReLU(True),
                View((-1, 256*1*1)),                 # B, 256
                nn.Linear(256, z_dim if pcl else z_dim*2),             # B, z_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 256),               # B, 256
                View((-1, 256, 1, 1)),               # B, 256,  1,  1
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
            )

            self.weight_init()
        elif self.architecture == "ilcm_tabular":
            assert len(self.image_shape) == 1
            self.encoder = MLP_ilcm(image_shape[0], z_dim if pcl else 2 * z_dim, 512, 6, spectral_norm=False, batch_norm=False)
            self.decoder = MLP_ilcm(z_dim, image_shape[0], 512, 6, spectral_norm=False, batch_norm=False)
            self.x_logsigma = torch.nn.Parameter(-5 * torch.ones((1,)))

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=False):
        distributions = self._encode(x)
        if self.pcl:
            return None, distributions, None
        else:
            mu = distributions[:, :self.z_dim]
            logvar = distributions[:, self.z_dim:]
            z = reparametrize(mu, logvar)
            x_recon = self._decode(z)

            if len(self.image_shape) == 1:
                x_recon = (x_recon, torch.nn.functional.softplus(self.x_logsigma) + 1e-6)

            if return_z:
                return x_recon, mu, logvar, z
            else:
                return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()