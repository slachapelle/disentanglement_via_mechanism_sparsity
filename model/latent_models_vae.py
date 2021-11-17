import math

import numpy as np
import torch
import torch.nn as nn

from .gumbel_masks import GumbelSigmoid
from .nn import MLP, ParallelMLP, ParallelLinear

class LatentModel(nn.Module):
    def __init__(self, z_max_dim, cont_c_dim, disc_c_dim, disc_c_n_values, network_arch, n_lag=1, freeze_m=False,
                 freeze_g=False, freeze_gc=False, freeze_dummies=False, gumbel_threshold=None, no_gumbel=False,
                 no_drawhard=False, dim_suff_stat=None, output_delta=False, delta_factor=1.,
                 n_layers=0., hid_dim=10, bn=False, gumbel_temperature=1.):
        """
        Base class for transitions model in the latent space.
        y = (z, noise) with z called the semantical latent variable.
        The model captures the sequence of observations {y_t}, including the latent noise variable
        :param z_max_dim: dimensionality of z is learned, this is the maximal dimension
        :param cont_c_dim: continuous context vector
        :param disc_c_dim: discrete context vector
        :param disc_c_n_values: list containing the number of possible values for each entry of disc_c
        :param network_arch: string describing the architecture of `network` function.
        :param n_lag: number of time lags (1 implies markov property)
        :param freeze_m: Will freeze the m parameter during training
        :param freeze_g: Will freeze the g parameter during training
        :param dim_suff_stat: number of scalars required per z_i.
        """
        super(LatentModel, self).__init__()
        assert disc_c_dim == len(disc_c_n_values)

        self.z_max_dim = z_max_dim  # dimension of z is learnable.
        self.z_block_size = 1  # dimension of each block in z
        assert self.z_max_dim % self.z_block_size == 0, "z_max_dim must be a multiple of z_block_size"
        self.num_z_blocks = self.z_max_dim // self.z_block_size
        self.cont_c_dim = cont_c_dim
        self.disc_c_dim = disc_c_dim  # number of component in "c" which are discrete
        self.disc_c_n_values = disc_c_n_values  # list giving the number of values for each entry of disc_c
        self.n_lag = n_lag
        self.freeze_m = freeze_m
        self.freeze_g = freeze_g
        self.freeze_gc = freeze_gc
        self.dim_suff_stat = dim_suff_stat
        self.output_delta = output_delta
        self.delta_factor = delta_factor
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.bn = bn

        # This parameter is only used to know the device of the module
        self.dummy_param = nn.Parameter(torch.empty(0))

        # mask for semantical latent variables
        self.m = GumbelSigmoid((self.num_z_blocks,), freeze=freeze_m, drawhard=(not no_drawhard), tau=gumbel_temperature)
        if freeze_m:
            self.m.log_alpha.requires_grad = False

        # mask for connections between z^t and z^{<t}
        if self.n_lag > 0:
            self.g = GumbelSigmoid((self.num_z_blocks, self.num_z_blocks), freeze=freeze_g, drawhard=(not no_drawhard), tau=gumbel_temperature)  # g_ij = 1 iff i <- j
            if freeze_g:
                self.g.log_alpha.requires_grad = False

            # dummy parameter which replaces masked out z_i's in the transition network input
            self.gamma = torch.nn.Parameter(torch.zeros((1, self.num_z_blocks, self.n_lag, self.num_z_blocks, self.z_block_size)))
            if freeze_dummies:
                self.gamma.requires_grad = False

        # mask for connections between c^{t-1} and z^t
        if self.cont_c_dim > 0:
            self.gc = GumbelSigmoid((self.num_z_blocks, self.cont_c_dim), freeze=freeze_gc, drawhard=(not no_drawhard), tau=gumbel_temperature)  # gc_ij = 1 iff i <- j
            if freeze_gc:
                self.gc.log_alpha.requires_grad = False

            # dummy parameter which replaces masked out z_i's in the transition network input
            self.gamma_gc = torch.nn.Parameter(torch.zeros((1, self.num_z_blocks, self.cont_c_dim)))
            if freeze_dummies:
                self.gamma_gc.requires_grad = False

        if self.disc_c_dim != 0:
            assert self.disc_c_dim == 1, "The masking mechanism works only when there is only one action integer"
            self.one_hot_dim = np.prod(disc_c_n_values)
            # cumprod will be useful to convert discrete vector into an integer
            self.register_buffer("cumprod",
                                 torch.Tensor(np.cumprod([1] + self.disc_c_n_values))[:-1].unsqueeze(0).long(),
                                 persistent=False)  # do not save the this buffer
            if no_gumbel:
                self.gc_disc = MaskSigmoid((self.num_z_blocks, self.disc_c_dim), freeze=freeze_gc, drawhard=(not no_drawhard), tau=gumbel_temperature)  # gc_ij = 1 iff i <- j
            elif gumbel_threshold is None:
                self.gc_disc = GumbelSigmoid((self.num_z_blocks, self.disc_c_dim), freeze=freeze_gc, drawhard=(not no_drawhard), tau=gumbel_temperature)  # gc_ij = 1 iff i <- j

            # dummy parameter which replaces masked out gc_i's in the transition network input
            self.gamma_gc_disc = torch.nn.Parameter(torch.zeros((1, self.num_z_blocks, self.one_hot_dim)))
            if freeze_dummies:
                self.gamma_gc_disc.requires_grad = False

        # Initialize the network layers
        if network_arch == "MLP":
            # the number of parameters outputted by each networks
            output_dim = self.dim_suff_stat * self.z_block_size

            # ---- First Layer ---- #
            next_dim = output_dim if n_layers == 0 else hid_dim  # next layer's dimension

            if self.n_lag != 0:
                self.linear_z_lag = ParallelLinear(self.n_lag * self.z_max_dim, next_dim, self.num_z_blocks, bias=False)
            else:
                self.linear_z_lag = None

            self.input_bias = nn.Parameter(torch.zeros((self.num_z_blocks, next_dim,)))

            if self.cont_c_dim != 0:
                self.linear_cont_c = ParallelLinear(cont_c_dim, next_dim, self.num_z_blocks, bias=False)

            if self.disc_c_dim != 0:
                ## the following takes integer in {0, ..., prod(disc_c_n_values)} and outputs embedding of dimension (next_dim,)
                #self.input_embedding_disc_c = nn.Embedding(np.prod(disc_c_n_values), self.num_z_blocks * next_dim)

                # we can't use nn.Embedding here since we want a masking mechanism for disc_c
                self.linear_disc_c = ParallelLinear(self.one_hot_dim, next_dim, self.num_z_blocks, bias=False)

            # ---- Next layers ---- #
            if self.n_layers > 0:
                self.parallel_mlp = ParallelMLP(hid_dim, output_dim, hid_dim, self.n_layers - 1, self.num_z_blocks, bn=self.bn)
        else:
            raise NotImplementedError

    def freeze_masks(self):
        if hasattr(self, "g"):
            self.g.freeze = True
        if hasattr(self, "gc"):
            self.gc.freeze = True
        if hasattr(self, "gc_disc"):
            self.gc_disc.freeze = True
        if hasattr(self, "m"):
            self.m.freeze = True

    def unfreeze_masks(self):
        if hasattr(self, "g") and not self.freeze_g:
            self.g.freeze = False
        if hasattr(self, "gc") and not self.freeze_gc:
            self.gc.freeze = False
        if hasattr(self, "gc_disc") and not self.freeze_gc:
            self.gc_disc.freeze = False
        if hasattr(self, "m") and not self.freeze_m:
            self.m.freeze = False

    def m_regularizer(self):
        return torch.sum(self.m.get_proba())

    def g_regularizer(self):
        if self.n_lag > 0:
            return torch.sum(self.g.get_proba())
        else:
            return torch.zeros((1,))

    def gc_regularizer(self):
        rval = torch.tensor(0.).to(self.dummy_param.device)
        if self.cont_c_dim > 0:
            rval = rval + torch.sum(self.gc.get_proba())
        if self.disc_c_dim > 0:
            rval = rval + torch.sum(self.gc_disc.get_proba())
        return rval

    def rollout(self, z_lag, num_steps, cont_cs=None, disc_cs=None, sample=True):
        bs = z_lag.size(0)

        assert z_lag.size(1) == self.n_lag
        if cont_cs is not None:
            assert cont_cs.size(1) == num_steps
        if disc_cs is not None:
            assert disc_cs.size(1) == num_steps
        z_out = z_lag.view(bs, self.n_lag, self.num_z_blocks, self.z_block_size)

        for t in range(num_steps):
            cont_c_t = cont_cs[:, t] if cont_cs is not None else None
            disc_c_t = disc_cs[:, t] if disc_cs is not None else None

            z_tp1 = self.get_next_latent(z_out[:, -self.n_lag:], cont_c_t, disc_c_t, sample=sample)  # (batch_size, 1, z_dim)
            z_out = torch.cat([z_out, z_tp1.unsqueeze(1)], dim=1)

        return z_out[:, self.n_lag:].view(bs, -1, self.z_max_dim)  # (batch_size, num_steps, z_dim)

    def _mask_z_lag(self, z_lag, m, g):
        bs, t, _ = z_lag.shape
        z_lag = z_lag.view(bs, 1, t, self.num_z_blocks, self.z_block_size)
        m_g = m.view(bs, 1, 1, self.num_z_blocks, 1) * g.view(bs, self.num_z_blocks, 1, self.num_z_blocks, 1)
        masked_z_lag = m_g * z_lag.view(bs, 1, t, self.num_z_blocks, self.z_block_size) + (1 - m_g) * self.gamma  # (bs, num_blocks, n_lag, num_blocks, block_size)
        masked_z_lag = masked_z_lag.view(bs, self.num_z_blocks, t, self.z_max_dim)
        return masked_z_lag  # (bs, num_blocks, n_lag, z_dim)

    def _mask_cont_c(self, cont_c, gc):
        bs = cont_c.shape[0]
        cont_c = cont_c.view(bs, 1, self.cont_c_dim)
        masked_cont_c = gc * cont_c + (1 - gc) * self.gamma_gc
        return masked_cont_c  # (bs, num_blocks, cont_c_dim)

    def _mask_disc_c_one_hot(self, disc_c, gc_disc):
        bs = disc_c.shape[0]
        idx = self.disc_c_vec2int(disc_c)  # (batch_size,)
        disc_c_one_hot = torch.nn.functional.one_hot(idx, num_classes=self.one_hot_dim)  # (batch_size, one_hot_dim)
        disc_c_one_hot = disc_c_one_hot.view(bs, 1, self.one_hot_dim)
        # TODO: in the following, we assume gc_disc.shape == (bs, num_blocks, 1)
        masked_disc_c_one_hot = gc_disc * disc_c_one_hot + (1 - gc_disc) * self.gamma_gc_disc
        return masked_disc_c_one_hot  # (bs, num_blocks, one_hot_dim)

    def get_next_latent(self, z_lag, cont_c, disc_c, sample=True):
        bs, t, _, _ = z_lag.size()
        if sample:
            m = self.m(bs)  # (batch_size, num_z_blocks)
            if self.n_lag > 0:
                g = self.g(bs)  # (batch_size, num_z_blocks, num_z_blocks)
            if self.cont_c_dim > 0:
                gc = self.gc(bs)  # (batch_size, num_z_blocks, cont_c_dim)
            if self.disc_c_dim > 0:
                gc_disc = self.gc_disc(bs)  # (batch_size, num_z_blocks, disc_c_dim) TODO: disc_c_dim == 1 for now...
        else:
            m = (self.m.get_proba() > 0.5).type(z_lag.type()).unsqueeze(0).expand(bs, -1)
            if self.n_lag > 0:
                g = (self.g.get_proba() > 0.5).type(z_lag.type()).unsqueeze(0).expand(bs, -1, -1)
            if self.cont_c_dim > 0:
                gc = (self.gc.get_proba() > 0.5).type(z_lag.type()).unsqueeze(0).expand(bs, -1, -1)
            if self.disc_c_dim > 0:
                gc_disc = (self.gc_disc.get_proba() > 0.5).type(z_lag.type()).unsqueeze(0).expand(bs, -1, -1)

        # compute masked_z_lag
        if self.n_lag > 0:
            masked_z_lag = self._mask_z_lag(z_lag.view(bs, t, -1), m, g)  # (bs, num_blocks, n_lag, z_dim)
        else:
            masked_z_lag = None

        # compute masked_cont_c
        if self.cont_c_dim > 0:
            masked_cont_c = self._mask_cont_c(cont_c, gc)  # (bs, num_blocks, cont_c_dim)
        else:
            masked_cont_c = None

        # compute masked_disc_c_one_hot
        if self.disc_c_dim > 0:
            masked_disc_c_one_hot = self._mask_disc_c_one_hot(disc_c, gc_disc)  # (bs, num_blocks, cont_c_dim)
        else:
            masked_disc_c_one_hot = None

        # feed in transition net
        p_params = self.network(masked_z_lag, masked_cont_c, masked_disc_c_one_hot)  # (b, num_blocks, block_size * dim_suff_stat)
        if self.n_lag > 0:
            z_tm1 = z_lag[:, -1].view(bs, self.num_z_blocks, self.z_block_size)
        else:
            z_tm1 = None
        p_params = self.transform_p_params(p_params, z_tm1)

        if sample:
            z_tp1 = self.sample(p_params)  # (bs, num_blocks, block_size)
        else:
            z_tp1 = self.mean(p_params)  # (bs, num_blocks, block_size)

        return z_tp1

    def network(self, masked_z_lag, masked_cont_c, masked_disc_c_one_hot):
        """Outputs parameters (before the transform_p_params) of the distribution given previous z's"""

        # ---- First Layer ---- #
        x = self.input_bias
        if self.linear_z_lag is not None:
            b = masked_z_lag.size(0)
            masked_z_lag = masked_z_lag.view(b, self.num_z_blocks, -1)
            x = x + self.linear_z_lag(masked_z_lag)
        if self.cont_c_dim > 0:
            x = x + self.linear_cont_c(masked_cont_c)  # (batch_size, num_blocks, out)
        if self.disc_c_dim > 0:
            x = x + self.linear_disc_c(masked_disc_c_one_hot)
            #b = disc_c.size(0)
            # x = x + self.input_embedding_disc_c(idx).view(b, self.num_z_blocks, -1)  # (batch_size, num_blocks, out)

        if self.n_layers > 0:
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
            return self.parallel_mlp(x)  # (b, num_blocks, block_size * dim_suff_stat)
        else:
            return x

    def compute_kl(self, p_params, q_params, z_tm1):
        """

        :param p_params: parameters of the transition model (i.e. the prior) before transform_p_params
        :param q_params: parameters of the encoder model (i.e. q(z|x)) before transform_q_params
        :param z_tm1:
        :return:
        """
        raise NotImplementedError

    def transform_p_params(self, p_params, z_tm1):
        """
        Returns the transform parameters (e.g.  var = exp(log_var))
        """
        raise NotImplementedError

    def transform_q_params(self, q_params):
        """
        Returns the transformed parameters (e.g.  var = exp(log_var))
        """
        raise NotImplementedError

    def sample(self, transformed_params):
        """Return samples from the distribution"""
        raise NotImplementedError

    def reparameterize(self, q_params):
        raise NotImplementedError

    def disc_c_vec2int(self, disc_c):
        """"Maps a batch of disc_c in vector form to an integer"""
        # disc_c: (batch_size, disc_c_dim)
        assert self.disc_c_dim > 0
        return torch.sum(self.cumprod * disc_c, dim=1)


class FCGaussianLatentModel(LatentModel):
    def __init__(self, z_max_dim, cont_c_dim, disc_c_dim, disc_c_n_values, network_arch, n_lag=1, n_layers=0,
                 hid_dim=5, freeze_m=False, freeze_g=False, freeze_gc=False, freeze_dummies=False, no_drawhard=False,
                 output_delta=True, delta_factor=1., var_p_mode="dependent", bn=False, gumbel_temperature=1.0):
        """
        Gaussian model with mean and std given by fully connected neural network.
        :param z_max_dim: dimensionality of z is learned, this is the maximal dimension
        :param cont_c_dim: continuous context vector
        :param disc_c_dim: discrete context vector
        :param disc_c_n_values: list containing the number of possible values for each entry of disc_c
        :param n_lag: number of time lags (1 implies markov property)
        :param n_layers: number of hidden layers (0 implies linear model)
        :param hid_dim: number of hidden units per layers
        """
        super(FCGaussianLatentModel, self).__init__(z_max_dim, cont_c_dim, disc_c_dim, disc_c_n_values, network_arch, n_lag=n_lag,
                                                    freeze_m=freeze_m, freeze_g=freeze_g, freeze_gc=freeze_gc, freeze_dummies=freeze_dummies,
                                                    no_drawhard=no_drawhard, output_delta=output_delta, delta_factor=delta_factor,
                                                    dim_suff_stat=2, n_layers=n_layers, hid_dim=hid_dim, bn=bn,
                                                    gumbel_temperature=gumbel_temperature)
        self.var_p_mode = var_p_mode

        assert self.n_lag > 0 or self.cont_c_dim > 0 or self.disc_c_dim > 0, "Should use another model..."
        assert disc_c_dim < 2, "Please revise function disc_c_vec2int to make sure it works as intended."

        self.cst = math.log(math.e**1e-4 - 1.)  # to initialize the variance to 1e-4.
        if self.var_p_mode == "independent":
            self.logvar = torch.nn.Parameter(self.cst * torch.ones((1, self.num_z_blocks, self.z_block_size)))
        elif self.var_p_mode == "fixed":
            self.register_buffer("logvar", self.cst * torch.ones((1, self.num_z_blocks, self.z_block_size)), persistent=False)

        self.register_buffer("logvar_shift", torch.ones((1,)))
        self.logvar_shift_init = False

        # mean and variance initialized to 0 and 1.0
        self.init_p_params = torch.nn.Parameter(torch.cat([torch.zeros((1, self.num_z_blocks, self.z_block_size)),
                                                           math.log(math.e - 1.) * torch.ones((1, self.num_z_blocks, self.z_block_size))], 2))  # before transform_p_params_init

    def transform_p_params(self, p_params, z_tm1):
        # p_params shape: (b, num_blocks, block_size * dim_suff_stat)
        # z_tm1 shape: (b, num_blocks, block_size)
        bs = p_params.shape[0]
        p_params = p_params.view(bs, self.num_z_blocks, self.z_block_size * self.dim_suff_stat)
        mean, logvar = p_params.chunk(2, 2)

        # This ensures the learned variance is initialized to a small value.
        if not self.logvar_shift_init :
            mean_logvar = logvar.mean().item()
            self.logvar_shift[0] = self.cst - mean_logvar
            self.logvar_shift_init = True

        logvar = logvar + self.logvar_shift

        if self.var_p_mode in ["independent", "fixed"]:
            logvar = self.logvar.expand(bs, -1, -1)

        if self.output_delta:
            mean = self.delta_factor * mean + z_tm1

        var = nn.functional.softplus(logvar) + 1e-6
        return mean, var  # each of shape (bs, num_blocks, block_size)

    def transform_p_params_init(self, p_params):
        # p_params shape: (b, num_blocks, block_size * dim_suff_stat)
        # z_tm1 shape: (b, num_blocks, block_size)
        bs = p_params.shape[0]
        p_params = p_params.view(bs, self.num_z_blocks, self.z_block_size * self.dim_suff_stat)
        mean, logvar = p_params.chunk(2, 2)

        var = nn.functional.softplus(logvar) + 1e-6
        return mean, var  # each of shape (bs, num_blocks, block_size)

    def transform_q_params(self, q_params):
        bs = q_params.shape[0]
        q_params = q_params.view(bs, self.num_z_blocks, self.z_block_size * self.dim_suff_stat)
        mean, logvar = q_params.chunk(2, 2)
        var = nn.functional.softplus(logvar) + 1e-4
        return mean, var  # each of shape (bs, num_blocks, block_size)

    def reparameterize(self, q_params):
        # q_params shape: (bs, z_max_dim * dim_suff_stat)
        mean, var = self.transform_q_params(q_params)
        eps = torch.randn_like(var)
        z = mean + eps * torch.sqrt(var)
        return z.view(z.shape[0], self.z_max_dim)

    def sample(self, transformed_params):
        mean, var = transformed_params  # each of shape (bs, num_blocks, block_size)
        return torch.distributions.Normal(mean, torch.sqrt(var)).sample()  # (bs, num_blocks, block_size)

    def compute_kl(self, p_params, q_params, z_tm1, init=False):
        # returns kl(q, p)
        if init:
            mean_p, var_p = self.transform_p_params_init(p_params)  # each with shape (bs, num_blocks, block_size)
        else:
            mean_p, var_p = self.transform_p_params(p_params, z_tm1)  # each with shape (bs, num_blocks, block_size)
        mean_q, var_q = self.transform_q_params(q_params)  # each with shape (bs, num_blocks, block_size)

        dmu = mean_q - mean_p

        return torch.sum(0.5 * (torch.log(var_p) - torch.log(var_q)) + 0.5 * (var_q + dmu ** 2) / var_p - 0.5, 2)

    def mean(self, transformed_params):
        return transformed_params[0]

    def mean_var(self, transformed_params):
        return transformed_params

    def log_likelihood(self, p_params, z_t, z_tm1):
        mean_p, var_p = self.transform_p_params(p_params, z_tm1)  # each with shape (bs, num_blocks, block_size)

        ll = torch.distributions.normal.Normal(mean_p, torch.sqrt(var_p)).log_prob(z_t.view(-1, self.num_z_blocks, self.z_block_size))
        return ll.view(ll.shape[0], -1).sum(1)

