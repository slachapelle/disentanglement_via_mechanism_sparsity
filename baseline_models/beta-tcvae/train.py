import os
import sys
import pathlib
import time
import math
from numbers import Number
import argparse
import json


try:
    from comet_ml import Experiment
    COMET_AVAIL = True
except:
    COMET_AVAIL = False
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow

#from elbo_decomposition import elbo_decomposition
#from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from train import get_dataset, get_loader
from universal_logger.logger import UniversalLogger
from metrics import mean_corr_coef, get_linear_score
from model.nn import MLP as MLP_ilcm

class View(torch.nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class VAE(nn.Module):
    def __init__(self, z_dim, image_shape, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, mss=False, architecture="mlp"):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.mss = mss
        self.image_shape = image_shape
        self.architecture = architecture
        if len(self.image_shape) == 3:
            self.x_dist = dist.Bernoulli()
        else:
            self.x_dist = dist.Normal()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if architecture == "conv":
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        elif architecture == "mlp":
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)
        elif architecture == "ilcm_tabular":
            assert len(self.image_shape) == 1
            self.encoder = MLP_ilcm(image_shape[0], 2 * z_dim, 512, 6, spectral_norm=False, batch_norm=False)
            self.decoder = MLP_ilcm(z_dim, image_shape[0], 512, 6, spectral_norm=False, batch_norm=False)
            self.x_logsigma = torch.nn.Parameter(-5 * torch.ones((1,)))
        elif architecture == "standard_conv":
            assert image_shape[:2] == (64, 64)
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(image_shape[2], 32, 4, 2, 1),  # B,  32, 32, 32
                torch.nn.ReLU(True),
                torch.nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                torch.nn.ReLU(True),
                torch.nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
                torch.nn.ReLU(True),
                View((-1, 256 * 1 * 1)),  # B, 256
                torch.nn.Linear(256, z_dim * 2),  # B, z_dim*2
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(z_dim, 256),  # B, 256
                View((-1, 256, 1, 1)),  # B, 256,  1,  1
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(32, image_shape[2], 4, 2, 1),  # B, nc, 64, 64
            )
        else:
            raise NotImplementedError(f"the architecture {architecture} is not implemented for TC-VAE and beta-VAE.")

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        # x_params = self.decoder.forward(zs)
        x_params = self.decoder(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        if self.architecture in ["conv", "mlp", "standard_conv"]:
            x = x.view(x.size(0), self.image_shape[2], self.image_shape[0], self.image_shape[1])
            # use the encoder to get the parameters used to define q(z|x)
            #z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
            z_params = self.encoder(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        elif self.architecture == "ilcm_tabular":
            z_params = self.encoder(x).view(x.size(0), self.z_dim, self.q_dist.nparams)

        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        if self.architecture in ["conv", "mlp", "standard_conv"]:
            #x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
            x_params = self.decoder(z).view(z.size(0), self.image_shape[2], self.image_shape[0], self.image_shape[1])
        elif self.architecture == "ilcm_tabular":
            x_mean = self.decoder(z).view(z.size(0), -1, 1)
            x_params = torch.cat([x_mean, self.x_logsigma.expand(x_mean.shape[0], x_mean.shape[1]).unsqueeze(2)], 2)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        #x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat([
        test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)
    win_test_reco = vis.images(
        list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
        opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        #xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs_walk = model.decoder(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main(args):
    torch.cuda.set_device(0)
    device = torch.device('cuda:0' if args.cuda else 'cpu')

    ## ---- Save hparams ---- ##
    kwargs = vars(args)
    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    # ---- Data ---- #
    args.no_norm = False
    args.n_lag = 0 # get dataset expects this argument, but no effect.
    image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, train_dataset, valid_dataset, test_dataset = get_dataset(args)
    train_loader, valid_loader, test_loader = get_loader(args, train_dataset, valid_dataset, test_dataset)

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()
    vae = VAE(z_dim=train_dataset.z_dim, image_shape=image_shape, use_cuda=args.cuda, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, mss=args.mss, architecture=args.architecture)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    print([n for n, p in vae.named_parameters()])

    # ---- Logger ---- ##
    # setup visdom for visualization
    #if args.visdom:
    #    vis = visdom.Visdom(env=args.save, port=4500)
    if COMET_AVAIL and args.comet_key is not None:
        comet_exp = Experiment(api_key=args.comet_key, project_name=args.comet_project_name,
                               workspace=args.comet_workspace, auto_metric_logging=False, auto_param_logging=False)
        comet_exp.log_parameters(vars(args))
        if args.comet_tag is not None:
            comet_exp.add_tag(args.comet_tag)
    else:
        comet_exp = None
    logger = UniversalLogger(comet=comet_exp,
                             stdout=(not args.no_print),
                             json=args.output_dir, throttle=None)

    train_elbo = []

    # timer
    if args.time_limit is not None:
        time_limit = args.time_limit * 60 * 60  # convert to seconds
        t0 = time.time()
    else:
        time_limit = np.inf
        t0 = 0

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = args.max_iter
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations and time.time() - t0 <= time_limit:
        for i, x in enumerate(train_loader):
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()

            x = x[0][:, -1]  # Extract x_t only, dropping x_t-1, ... x_t-k
            # transfer to GPU
            x = x.to(device)
            # wrap the mini-batch in a PyTorch Variable
            #x = Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                logger.log_metrics(step=iteration, metrics={"beta": vae.beta, "lambda": vae.lamb, "loss_train": -elbo_running_mean.val})

    # save model
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.output_dir, 0)

    ## ---- Evaluate performance ---- ##
    # compute MCC and save representation
    if args.tcvae:
        args.mode = "tcvae"
    else:
        args.mode = "betavae"
    mcc, cc_program_perm, assignments, z, z_hat = mean_corr_coef(vae, test_loader, device, opt=args)
    linear_score = get_linear_score(z_hat, z)

    ## ---- Save ---- ##
    # save scores
    logger.log_metrics(step=0, metrics={"mcc": mcc, "linear_score": linear_score})

    # save both ground_truth and learned latents
    np.save(os.path.join(args.output_dir, "z_hat.npy"), z_hat)
    np.save(os.path.join(args.output_dir, "z_gt.npy"), z)


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to output logs and model checkpoints")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Type of the dataset to be used 'toy-MANIFOLD/TRANSITION_MODEL'")
    parser.add_argument("--dataroot", type=str, default="./",
                        help="path to dataset")
    parser.add_argument("--gt_z_dim", type=int, default=10,
                        help="ground truth dimensionality of z (for TRANSITION_MODEL == 'linear_system')")
    parser.add_argument("--gt_x_dim", type=int, default=20,
                        help="ground truth dimensionality of x (for MANIFOLD == 'nn')")
    parser.add_argument("--num_samples", type=float, default=int(1e6),
                        help="number of trajectories in toy dataset")
    parser.add_argument("--architecture", type=str, default='ilcm_tabular', choices=['ilcm_tabular', "standard_conv"],
                        help="VAE encoder/decoder architecture.")
    parser.add_argument("--train_prop", type=float, default=None,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--valid_prop", type=float, default=0.10,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--test_prop", type=float, default=0.10,
                        help="proportion of all samples used in test set")
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--time_limit", type=float, default=None,
                        help="After this amount of time, terminate training. (in hours)")
    parser.add_argument("--max_iter", type=int, default=int(1e6),
                        help="Maximal amount of iterations")
    parser.add_argument("--seed", type=int, default=0,
                        help="manual seed")
    parser.add_argument('--no_print', action="store_true",
                        help='do not print')
    parser.add_argument('--comet_key', type=str, default=None,
                        help="comet api-key")
    parser.add_argument('--comet_tag', type=str, default=None,
                        help="comet tag, to ease comparison")
    parser.add_argument('--comet_workspace', type=str, default=None,
                        help="comet workspace")
    parser.add_argument('--comet_project_name', type=str, default=None,
                        help="comet project_name")
    parser.add_argument("--no_cuda", action="store_false", dest="cuda",
                        help="Disables cuda")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="batch size used during training")
    parser.add_argument("--eval_batch_size", type=int, default=1024,
                        help="batch size used during evaluation")
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')
    #parser.add_argument('--gpu', type=int, default=0)
    #parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--log_freq', default=100, type=int, help='num iterations per log')
    args = parser.parse_args()

    model = main(args)
