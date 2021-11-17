import math

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm as sn


class MLP(torch.nn.Module):
    def __init__(self, ni, no, nhidden, nlayers, spectral_norm=False, batch_norm=True):
        super().__init__()
        self.nlayers = nlayers
        self.batch_norm = batch_norm
        for i in range(nlayers):
            if i == 0:
                if spectral_norm:
                    setattr(self, "linear%d" % i, sn(torch.nn.Linear(ni, nhidden, bias=(not batch_norm))))
                else:
                    setattr(self, "linear%d" % i, torch.nn.Linear(ni, nhidden, bias=(not batch_norm)))

            else:
                if spectral_norm:
                    setattr(self, "linear%d" % i, sn(torch.nn.Linear(nhidden, nhidden, bias=(not batch_norm))))
                else:
                    setattr(self, "linear%d" % i, torch.nn.Linear(nhidden, nhidden, bias=(not batch_norm)))
            if batch_norm:
                setattr(self, "bn%d" % i, torch.nn.BatchNorm1d(nhidden))
        if nlayers == 0:
            nhidden = ni
        self.linear_out = torch.nn.Linear(nhidden, no)

    def forward(self, x):
        for i in range(self.nlayers):
            linear = getattr(self, "linear%d" % i)
            x = linear(x)
            if self.batch_norm:
                bn = getattr(self, "bn%d" % i)
                x = bn(x)
            x = F.leaky_relu(x, 0.2, True)
        return self.linear_out(x)


class ParallelMLP(torch.nn.Module):
    def __init__(self, ni, no, nhidden, nlayers, nMLPs, bn=True):
        super().__init__()
        self.nlayers = nlayers
        self.nMLPs = nMLPs
        self.bn = bn
        for i in range(nlayers):
            if i == 0:
                setattr(self, "linear%d" % i, ParallelLinear(ni, nhidden, nMLPs, bias=False))
            else:
                setattr(self, "linear%d" % i, ParallelLinear(nhidden, nhidden, nMLPs, bias=False))
            if self.bn:
                setattr(self, "bn%d" % i, torch.nn.BatchNorm1d(nhidden * nMLPs))
        if nlayers == 0:
            nhidden = ni
        self.linear_out = ParallelLinear(nhidden, no, nMLPs)

    def forward(self, x):
        assert self.nMLPs == x.shape[1]
        bs, nMLPs, ni = x.shape
        for i in range(self.nlayers):
            linear = getattr(self, "linear%d" % i)
            x = linear(x)
            # this `reshape` instead of `view` call is necessary since x is not contiguous.
            if self.bn:
                bn = getattr(self, "bn%d" % i)
                x = bn(x.reshape(bs, -1)).view(bs, nMLPs, -1)  # TODO: should I worry about the copy made by reshape?
            x = F.leaky_relu(x, 0.2, True)
        return self.linear_out(x)


class ParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, num_linears, bias=True):
        super(ParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_linears = num_linears
        self.weight = torch.nn.Parameter(torch.Tensor(num_linears, out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(num_linears, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for linear in range(self.num_linears):
            torch.nn.init.kaiming_uniform_(self.weight[linear], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[linear])
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # input shape: (bs, num_linears, in_features)
        x = torch.einsum("bli,lji->blj", input, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return  x  # (bs, num_linears, out_features)

