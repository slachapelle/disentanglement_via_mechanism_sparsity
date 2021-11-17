import math
import torch
import torch.nn.functional as F
import numpy as np

from .nn import MLP


class ILCM_VAE(torch.nn.Module):
    def __init__(self, latent_model, image_shape, cont_c_dim, disc_c_dim, disc_n_values, opt):
        super().__init__()
        self.latent_model = latent_model
        self.z_dim = opt.z_max_dim
        self.n_lag = opt.n_lag
        self.cont_c_dim = cont_c_dim
        self.disc_c_dim = disc_c_dim
        self.disc_c_n_values = disc_n_values
        self.beta = opt.beta
        self.learn_var = opt.learn_decoder_var
        self.opt = opt

        if self.opt.full_seq:
            assert self.n_lag == 1, "--full_seq is supported only for --n_lag 1"

        # choosing encoder
        if opt.encoder == "tabular":
            assert len(image_shape) == 1, f"The encoder {opt.encoder} works only on tabular data."
            self.encoder = MLP(image_shape[0], 2 * self.z_dim, 512, 3 * opt.encoder_depth_multiplier, spectral_norm=False, batch_norm=opt.bn_enc_dec)
        else:
            raise NotImplementedError(f"The encoder {opt.encoder} is not implemented.")

        # choosing decoder
        if opt.decoder == "tabular":
            assert len(image_shape) == 1, f"The decoder {opt.decoder} works only on tabular data."
            self.decoder = MLP(self.z_dim, image_shape[0], 512, 3 * opt.decoder_depth_multiplier, spectral_norm=False, batch_norm=opt.bn_enc_dec)
        else:
            raise NotImplementedError(f"The decoder {opt.decoder} is not implemented.")

        # dummy parameter which replaces masked out z_i's in the decoder input
        self.eta = torch.nn.Parameter(torch.zeros((self.z_dim,)))
        if opt.freeze_dummies:
            self.eta.requires_grad = False

        if len(image_shape) == 3:
            self.image_shape = image_shape[2:3] + image_shape[0:2]
        else:
            self.image_shape = image_shape

        # that's a bit messy, keep it like this to keep previous default behavior.
        if self.learn_var:
            if self.opt.init_decoder_var is None:
                self.decoder_logvar =  torch.nn.Parameter(-10 * torch.ones((1,)))
            else:
                self.decoder_logvar = torch.nn.Parameter(math.log(opt.init_decoder_var) * torch.ones((1,)))
        else:
            if self.opt.init_decoder_var is None:
                self.decoder_logvar =  torch.nn.Parameter(torch.zeros((1,)))
            else:
                self.decoder_logvar = torch.nn.Parameter(math.log(opt.init_decoder_var) * torch.ones((1,)))
            self.decoder_logvar.requires_grad = False

    def encode(self, x):
        b = x.size(0)
        z_params = self.encoder(x)
        return z_params

    def decode(self, input, logit=False):
        if len(self.image_shape) == 1:
            return self.decoder(input)
        elif len(self.image_shape) == 3:
            if logit:
                return self.decoder(input)
            else:
                return torch.sigmoid(self.decoder(input))

    def _mask_z_before_decoding(self, z, m):
        # z shape: (b, t, z_max_dim)
        b, t = z.shape[0:2]
        num_blocks = m.shape[1]
        m = m.view(b, 1, num_blocks, 1)
        z = z.view(b,t, num_blocks, -1)
        eta = self.eta.view(1, 1, num_blocks, -1)
        z_masked_eta = m * z + (1 - m) * eta
        return z_masked_eta.view(b * t, -1)

    def elbo(self, obs, cont_c, disc_c):
        b, t = obs.shape[0:2]

        q_params = self.encode(obs.view((b * t,) + self.image_shape))
        z = self.latent_model.reparameterize(q_params)
        m = self.latent_model.m(b) # sample a mask

        ## --- Reconstruction --- ##
        z_masked_eta = self._mask_z_before_decoding(z.view(b, t, -1), m)
        # including the reconstruction term not only for x_t, but also for x_t-1, ..., x_t-k.

        reconstructions = self.decode(z_masked_eta)
        std = torch.exp(0.5 * self.decoder_logvar) + 1e-4
        # SL: This choice of reduction is picked to keep the relative importance of each terms in line with the original ELBO
        rec_loss = - torch.distributions.normal.Normal(reconstructions.view(b, t, -1), std).log_prob(obs.view(b, t, -1)).mean(dim=(1, 2))

        ## --- KL divergence --- #
        # mask z and c
        if self.latent_model.n_lag > 0:
            g = self.latent_model.g(b)
            z_lag = z.view(b, t, -1)[:, :-1]
            z_masked_gamma_lag = self.latent_model._mask_z_lag(z_lag, m, g)  # (bs, num_blocks, n_lag, z_dim)
            z_tm1 = z_lag[:, -1].view(b, -1, self.latent_model.z_block_size)
        else:
            z_masked_gamma_lag = None
            z_tm1 = None

        if self.latent_model.cont_c_dim > 0:
            gc = self.latent_model.gc(b)
            masked_cont_c = self.latent_model._mask_cont_c(cont_c, gc)
        else:
            masked_cont_c = None

        if self.disc_c_dim > 0:
            gc_disc = self.latent_model.gc_disc(b)  # (batch_size, num_z_blocks, disc_c_dim) TODO: disc_c_dim == 1 for now...
            masked_disc_c_one_hot = self.latent_model._mask_disc_c_one_hot(disc_c, gc_disc)  # (bs, num_blocks, cont_c_dim)
        else:
            masked_disc_c_one_hot = None

        p_params = self.latent_model.network(z_masked_gamma_lag, masked_cont_c, masked_disc_c_one_hot)
        q_params = q_params.view(b, t, -1)
        kl = self.latent_model.compute_kl(p_params, q_params[:, -1], z_tm1)

        if self.opt.full_seq:
            # we divide by two to preserve the relative importance between rec_loss and kl in line with original elbo
            init_params = self.latent_model.init_p_params.expand(b, -1, -1)
            kl = 0.5 * (kl + self.latent_model.compute_kl(init_params, q_params[:, 0], None, init=True))

        kl_reduced = torch.sum(kl, 1) / int(np.product(self.image_shape))

        elbo = -rec_loss - self.beta * kl_reduced
        return elbo, rec_loss.mean().item(), kl_reduced.mean().item(), kl

    def log_likelihood(self, gt_z, cont_c, disc_c):
        b, t = gt_z.shape[0:2]

        m = self.latent_model.m(b)  # sample a mask

        # mask z and c
        if self.latent_model.n_lag > 0:
            g = self.latent_model.g(b)
            z_lag = gt_z[:, :-1]
            z_masked_gamma_lag = self.latent_model._mask_z_lag(z_lag, m, g)  # (bs, num_blocks, n_lag, z_dim)
            z_tm1 = z_lag[:, -1].view(b, -1, self.latent_model.z_block_size)
        else:
            z_masked_gamma_lag = None
            z_tm1 = None

        if self.latent_model.cont_c_dim > 0:
            gc = self.latent_model.gc(b)
            masked_cont_c = self.latent_model._mask_cont_c(cont_c, gc)
        else:
            masked_cont_c = None

        p_params = self.latent_model.network(z_masked_gamma_lag, masked_cont_c, disc_c)
        ll = self.latent_model.log_likelihood(p_params, gt_z[:, -1], z_tm1)

        return ll

