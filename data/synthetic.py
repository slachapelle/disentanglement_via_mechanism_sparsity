import os
from os.path import join
import gzip
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.linalg import block_diag, norm


def get_decoder(manifold, x_dim, z_dim, rng_data_gen):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if manifold == "nn":
        # NOTE: injectivity requires z_dim <= h_dim <= x_dim
        h_dim = x_dim
        neg_slope = 0.2

        # sampling NN weight matrices
        W1 = rng_data_gen.normal(size=(z_dim, h_dim))
        W1 = np.linalg.qr(W1.T)[0].T
        # print("distance to identity:", np.max(np.abs(np.matmul(W1, W1.T) - np.eye(self.z_dim))))
        W1 *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (z_dim + h_dim))
        W1 = torch.Tensor(W1).to(device)
        W1.requires_grad = False

        W2 = rng_data_gen.normal(size=(h_dim, h_dim))
        W2 = np.linalg.qr(W2.T)[0].T
        # print("distance to identity:", np.max(np.abs(np.matmul(W2, W2.T) - np.eye(h_dim))))
        W2 *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (2 * h_dim))
        W2 = torch.Tensor(W2).to(device)
        W2.requires_grad = False

        W3 = rng_data_gen.normal(size=(h_dim, h_dim))
        W3 = np.linalg.qr(W3.T)[0].T
        # print("distance to identity:", np.max(np.abs(np.matmul(W3, W3.T) - np.eye(h_dim))))
        W3 *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (2 * h_dim))
        W3 = torch.Tensor(W3).to(device)
        W3.requires_grad = False

        W4 = rng_data_gen.normal(size=(h_dim, x_dim))
        W4 = np.linalg.qr(W4.T)[0].T
        # print("distance to identity:", np.max(np.abs(np.matmul(W4, W4.T) - np.eye(h_dim))))
        W4 *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (x_dim + h_dim))
        W4 = torch.Tensor(W4).to(device)
        W4.requires_grad = False

        # note that this decoder is almost surely invertible WHEN dim <= h_dim <= x_dim
        # since Wx is injective
        # when columns are linearly indep, which happens almost surely,
        # plus, composition of injective functions is injective.
        def decoder(z):
            with torch.no_grad():
                z = torch.Tensor(z).to(device)
                h1 = torch.matmul(z, W1)
                h1 = torch.maximum(neg_slope * h1, h1)  # leaky relu
                h2 = torch.matmul(h1, W2)
                h2 = torch.maximum(neg_slope * h2, h2)  # leaky relu
                h3 = torch.matmul(h2, W3)
                h3 = torch.maximum(neg_slope * h3, h3)  # leaky relu
                out = torch.matmul(h3, W4)
            return out.cpu().numpy()

        noise_std = 0.01
    elif manifold == "linear":
        W = rng_data_gen.normal(size=(x_dim, z_dim))
        W = torch.Tensor(W).to(device)
        bias = rng_data_gen.normal(size=(x_dim,))
        bias = torch.Tensor(bias).to(device)

        def decoder(z):
            with torch.no_grad():
                z = torch.Tensor(z).to(device)
                out = torch.matmul(z, W.T) + bias
            return out.cpu().numpy()

        noise_std = 0.01
    else:
        raise NotImplementedError(f"The manifold {self.manifold} is not implemented.")

    return decoder, noise_std


class ActionToyManifoldDataset(torch.utils.data.Dataset):
    def __init__(self, manifold, transition_model, num_samples, seed, x_dim, z_dim, no_norm=False, gt_gc=None, c_dim=0):
        super(ActionToyManifoldDataset, self).__init__()
        self.manifold = manifold
        self.transition_model = transition_model
        self.rng = np.random.default_rng(seed)  # use for dataset sampling
        self.rng_data_gen = np.random.default_rng(265542)  # use for sampling actual data generating process.
        self.x_dim = x_dim
        self.z_dim = z_dim
        if c_dim == 0:
            self.c_dim = self.z_dim
        else:
            self.c_dim = c_dim
        self.num_samples = num_samples
        self.no_norm = no_norm
        if gt_gc is not None:
            assert self.transition_model.startswith("action_sparsity_non_trivial")

        if self.transition_model == "action_sparsity_trivial":
            assert self.c_dim == self.z_dim
            def get_mean_var(c, var_fac=0.0001):
                mu_tp1 = np.sin(c)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.eye(self.z_dim)
        elif self.transition_model == "action_sparsity_non_trivial":
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.c_dim, 1)
            if gt_gc is None:
                assert self.c_dim == self.z_dim
                gt_gc = np.concatenate([np.eye(self.z_dim), np.eye(self.z_dim)[:, 0:1]], 1)[:, 1:] + np.eye(self.z_dim)
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.c_dim, 1)

            def get_mean_var(c, var_fac=0.0001):  #var_fac=0.1**2):  #var_fac=0.5**2):
                mu_tp1 = np.sum(gt_gc * np.sin(c[:, None, :] * mat_range + shift), 2)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)

        elif self.transition_model == "action_sparsity_non_trivial_k=2":
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.c_dim, 1)
            if gt_gc is None:
                assert self.c_dim == self.z_dim
                gt_gc = np.concatenate([np.eye(self.z_dim), np.eye(self.z_dim)[:, 0:1]], 1)[:, 1:] + np.eye(self.z_dim)
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.c_dim, 1)

            def get_mean_var(c):
                mu_tp1 = np.sum(gt_gc * np.sin(c[:, None, :] * mat_range + shift), 2)
                var_tp1 = 0.1 / c.shape[1] * np.exp(np.sum(gt_gc * np.cos(c[:, None, :] * mat_range + shift), 2))  # max at 0.1 * np.euler
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)

        elif self.transition_model == "action_sparsity_non_trivial_sigmoid":
            if gt_gc is None:
                assert self.c_dim == self.z_dim
                gt_gc = np.concatenate([np.eye(self.z_dim), np.eye(self.z_dim)[:, 0:1]], 1)[:, 1:] + np.eye(self.z_dim)
            shift = np.linspace(-2, 2, self.z_dim)

            def get_mean_var(c,  var_fac=0.0001):  #var_fac=0.1**2):  #var_fac=0.5**2):
                mu_tp1 = np.sum(gt_gc * (1 / (1 + np.exp(-4 * (c[:, None, :] - shift[:, None])))), 2)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)

        elif self.transition_model == "action_sparsity_non_trivial_norm":
            if gt_gc is None:
                assert self.c_dim == self.z_dim
                gt_gc = np.concatenate([np.eye(self.z_dim), np.eye(self.z_dim)[:, 0:1]], 1)[:, 1:] + np.eye(self.z_dim)
            shift = self.rng_data_gen.normal(size=(self.z_dim , self.c_dim))

            def get_mean_var(c,  var_fac=0.0001):  #var_fac=0.1**2):  #var_fac=0.5**2):
                mu_tp1 = norm(gt_gc * (c[:, None, :] - shift), axis=2)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)

        elif self.transition_model == "action_sparsity_non_trivial_no_suff_var":
            assert self.c_dim == self.z_dim
            gt_gc = np.concatenate([np.eye(self.z_dim), np.eye(self.z_dim)[:, 0:1]], 1)[:, 1:] + np.eye(self.z_dim)
            A = self.rng_data_gen.normal(size=(self.z_dim, self.z_dim)) * gt_gc

            def get_mean_var(c, var_fac=0.0001):
                mu_tp1 = np.matmul(c, A.T)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)

        elif self.transition_model == "action_sparsity_non_trivial_no_graph_crit":
            assert self.z_dim % 2 == 0
            assert self.c_dim == self.z_dim
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.z_dim, 1)
            gt_gc = block_diag(*[np.ones((2, 2)) for _ in range(int(self.z_dim / 2))])
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.z_dim, 1)

            def get_mean_var(c, var_fac=0.0001):
                mu_tp1 = np.sum(gt_gc * np.sin(c[:, None, :] * mat_range + shift), 2)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)
        elif self.transition_model == "action_sparsity_2d":
            assert self.z_dim == 2
            assert self.c_dim == 2

            def get_mean_var(c, var_fac=0.0001):  #var_fac=0.5**2):
                z_1 = 0.5 * c[:, 0:1] ** 2
                z_2 = 0.5 * c[:, 0:1] ** 3 + 0.5 * c[:, 1:2] ** 2
                mu_tp1 = np.concatenate([z_1, z_2], 1)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(np.array([[1, 0],
                                                [1, 1]]))
        else:
            raise NotImplementedError(f"The transition model {self.transition_model} is not implemented.")

        self.decoder, self.noise_std = get_decoder(self.manifold, self.x_dim, self.z_dim, self.rng_data_gen)
        self.get_mean_var = get_mean_var
        self.create_data()

    def __len__(self):
        return self.num_samples

    def sample_z_given_c(self, c):
        mu_tp1, var_tp1 = self.get_mean_var(c)
        return self.rng.normal(mu_tp1, np.sqrt(var_tp1))

    def create_data(self):
        c = self.rng_data_gen.uniform(-2, 2, size=(self.num_samples, self.c_dim))
        z = self.sample_z_given_c(c)
        x = self.decoder(z)

        # normalize
        if not self.no_norm:
            x = (x - x.mean(0)) / x.std(0)

        x = x + self.noise_std * self.rng.normal(0, 1, size=(self.num_samples, self.x_dim))

        self.x = torch.Tensor(x)
        self.z = torch.Tensor(z)
        self.c = torch.Tensor(c)

    def __getitem__(self, item):
        obs = self.x[item: item + 1]  # must have a dimension for time (of size 1 since no temporal dependencies)
        cont_c = self.c[item]
        disc_c = torch.Tensor(np.array([0.])).long()
        valid = True
        other = self.z[item: item + 1]  # must have a dimension for time (of size 1 since no temporal dependencies)
        return obs, cont_c, disc_c, valid, other


class TemporalToyManifoldDataset(torch.utils.data.Dataset):
    def __init__(self, manifold, transition_model, num_samples, seed, x_dim, z_dim, no_norm=False, gt_g=None):
        super(TemporalToyManifoldDataset, self).__init__()
        self.manifold = manifold
        self.transition_model = transition_model
        self.rng = np.random.default_rng(seed)  # use for dataset sampling
        self.rng_data_gen = np.random.default_rng(265542)   # use for sampling actual data generating process.
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.num_samples = num_samples
        self.no_norm = no_norm
        if gt_g is not None:
            assert self.transition_model.startswith("temporal_sparsity_non_trivial")
            assert (np.diag(gt_g) == 1).all()

        if self.transition_model == "temporal_sparsity_trivial":
            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                mu_tp1 =  z_t + lr * np.sin(z_t)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.eye(self.z_dim)

        elif self.transition_model == "temporal_sparsity_non_trivial":
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.z_dim, 1)
            if gt_g is None:
                gt_g = np.tril(np.ones((self.z_dim, self.z_dim)))
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.z_dim, 1)

            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                delta = np.sum(gt_g * np.sin(z_t[:, None, :] * mat_range + shift), 2)
                mu_tp1 = z_t + lr * delta
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)

        elif self.transition_model == "temporal_sparsity_non_trivial_k=2":
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.z_dim, 1)
            if gt_g is None:
                gt_g = np.tril(np.ones((self.z_dim, self.z_dim)))
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.z_dim, 1)

            def get_mean_var(z_t, lr=0.5):
                delta = np.sum(gt_g * np.sin(z_t[:, None, :] * mat_range + shift), 2)
                mu_tp1 = z_t + lr * delta
                var_tp1 = 0.1 / z_t.shape[1] * np.exp(np.sum(gt_g * np.cos(z_t[:, None, :] * mat_range + shift), 2))  # max at 0.1 * np.euler
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)

        elif self.transition_model == "temporal_sparsity_non_trivial_sigmoid":
            if gt_g is None:
                gt_g = np.tril(np.ones((self.z_dim, self.z_dim)))
            shift = np.linspace(-2, 2, self.z_dim)

            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                delta = np.sum(gt_g * (1 / (1 + np.exp(-4 * (z_t[:, None, :] - shift[:, None])))), 2)
                mu_tp1 = z_t + lr * delta
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)

        elif self.transition_model == "temporal_sparsity_non_trivial_norm":
            if gt_g is None:
                gt_g = np.tril(np.ones((self.z_dim, self.z_dim)))
            shift = self.rng_data_gen.normal(size=(self.z_dim , self.z_dim))

            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):  #var_fac=0.1**2):  #var_fac=0.5**2):
                delta = norm(gt_g * (z_t[:, None, :] - shift), axis=2)
                mu_tp1 = z_t + lr * delta
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)

        elif self.transition_model == "temporal_sparsity_non_trivial_no_graph_crit":
            assert self.z_dim % 2 == 0
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.z_dim, 1)
            gt_g = block_diag(np.ones((int(self.z_dim / 2),int(self.z_dim / 2))), np.ones((int(self.z_dim / 2),int(self.z_dim / 2))))
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.z_dim, 1)

            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                delta = np.sum(gt_g * np.sin(z_t[:, None, :] * mat_range + shift), 2)
                mu_tp1 = z_t + lr * delta
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)

        elif self.transition_model == "temporal_sparsity_non_trivial_no_suff_var":
            gt_g = np.tril(np.ones((self.z_dim, self.z_dim)))
            A = self.rng_data_gen.normal(size=(self.z_dim, self.z_dim)) * gt_g

            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                delta = np.matmul(z_t, A.T)
                mu_tp1 = z_t + lr * delta
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)
        else:
            raise NotImplementedError(f"The transition model {self.transition_model} is not implemented.")

        self.decoder, self.noise_std = get_decoder(self.manifold, self.x_dim, self.z_dim, self.rng_data_gen)
        self.get_mean_var = get_mean_var
        self.create_data()

    def __len__(self):
        return self.num_samples

    def next_z(self, z_t):
        mu_tp1, var_tp1 = self.get_mean_var(z_t)
        if not self.transition_model.startswith("laplacian"):
            return self.rng.normal(mu_tp1, np.sqrt(var_tp1))
        else:
            return self.rng.laplace(mu_tp1, np.sqrt(0.5 * var_tp1))

    def rollout(self,):
        z_init = self.rng.normal(0, 1, size=(self.num_samples, self.z_dim))

        zs = np.zeros((self.num_samples, 2, self.z_dim))
        zs[:, 0, :] = z_init
        zs[:, 1, :] = self.next_z(zs[:, 0])

        return zs

    def create_data(self):
        # rollout in latent space
        z = self.rollout()

        # decode
        x = self.decoder(z.reshape(2 * self.num_samples, self.z_dim))

        # normalize
        if not self.no_norm:
            x = (x - x.mean(0)) / x.std(0)

        x = x + self.noise_std * self.rng.normal(0, 1, size=(2 * self.num_samples, self.x_dim))

        self.x = torch.Tensor(x.reshape(self.num_samples, 2, self.x_dim))
        self.z = torch.Tensor(z)

    def __getitem__(self, item):
        obs = self.x[item]
        cont_c = torch.Tensor(np.array([0.]))
        disc_c = torch.Tensor(np.array([0.])).long()
        valid = True
        other = self.z[item]

        return obs, cont_c, disc_c, valid, other


def get_ToyManifoldDatasets(manifold, transition_model, split=(0.7, 0.15, 0.15), z_dim=2, x_dim=10, num_samples=1e6,
                            no_norm=False, rand_g_density=None, gt_graph_name=None, seed=0):
    c_dim = z_dim
    if rand_g_density is not None:
        assert 0 <= rand_g_density <= 1
        if transition_model.startswith("temporal_sparsity"):
            graph_proba = np.minimum(1, np.eye(z_dim) + rand_g_density)  # forces to have all self-loops (diagonal)
        else:
            graph_proba = np.zeros((z_dim, z_dim)) + rand_g_density
        rng = np.random.default_rng(45321 + seed)  # same graph is going to be used for a given seed
        gt_graph = rng.binomial(1, graph_proba)
        print(gt_graph)
    elif gt_graph_name == "graph_action_1":
        assert z_dim == 10
        gt_graph = np.array([[0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                             [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]).astype("float")
        c_dim = z_dim
    elif gt_graph_name == "graph_action_1.1":
        assert z_dim == 10
        gt_graph = np.array([[0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                             [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]).astype("float")
        c_dim = z_dim
    elif gt_graph_name == "graph_action_2":
        assert z_dim % 2 == 0
        c_dim = z_dim
        gt_graph = block_diag(*[np.ones((2, 2)) for _ in range(int(z_dim / 2))]).astype("float")
    elif gt_graph_name == "graph_action_3_easy":
        assert z_dim == 10
        c_dim = 5
        gt_graph = np.array([[1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1]])
    elif gt_graph_name == "graph_action_3_hard":
        assert z_dim == 10
        c_dim = 5
        gt_graph = np.array([[1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1]])
    elif gt_graph_name == "graph_action_3.1_hard":
        assert z_dim == 10
        c_dim = 5
        gt_graph = np.array([[1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 0, 1, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1]])
    elif gt_graph_name == "graph_action_3.2_hard":
        assert z_dim == 10
        c_dim = 5
        gt_graph = np.array([[1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 1, 1, 0, 0],
                             [0, 1, 1, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0],
                             [1, 0, 0, 0, 1],
                             [1, 0, 0, 0, 1]])
    elif gt_graph_name == "graph_temporal_1":
        assert z_dim == 10
        gt_graph = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                             [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype("float")
    elif gt_graph_name == "graph_temporal_2":
        assert z_dim % 2 == 0
        gt_graph = block_diag(np.ones((int(z_dim / 2),int(z_dim / 2))), np.ones((int(z_dim / 2),int(z_dim / 2)))).astype("float")
    elif gt_graph_name == "graph_temporal_3_easy":
        gt_graph = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    elif gt_graph_name == "graph_temporal_3_hard":
        gt_graph = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]])
    elif gt_graph_name == "graph_temporal_3.1_hard":
        gt_graph = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                             [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    else:
        gt_graph = None

    if transition_model.startswith("action_sparsity"):
        cont_c_dim = c_dim
        disc_c_dim = 0
        disc_c_n_values = []
        train_dataset = ActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[0]), seed=1,
                                                 x_dim=x_dim, z_dim=z_dim, no_norm=no_norm, gt_gc=gt_graph, c_dim=c_dim)
        valid_dataset = ActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[1]), seed=2,
                                                 x_dim=x_dim, z_dim=z_dim, no_norm=no_norm, gt_gc=gt_graph, c_dim=c_dim)
        test_dataset = ActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[2]), seed=3,
                                                x_dim=x_dim, z_dim=z_dim, no_norm=no_norm, gt_gc=gt_graph, c_dim=c_dim)

    elif transition_model.startswith("temporal_sparsity"):
        cont_c_dim = 0
        disc_c_dim = 0
        disc_c_n_values = []
        train_dataset = TemporalToyManifoldDataset(manifold, transition_model, int(num_samples * split[0]), seed=1,
                                                 x_dim=x_dim, z_dim=z_dim, no_norm=no_norm, gt_g=gt_graph)
        valid_dataset = TemporalToyManifoldDataset(manifold, transition_model, int(num_samples * split[1]), seed=2,
                                                 x_dim=x_dim, z_dim=z_dim, no_norm=no_norm, gt_g=gt_graph)
        test_dataset = TemporalToyManifoldDataset(manifold, transition_model, int(num_samples * split[2]), seed=3,
                                                x_dim=x_dim, z_dim=z_dim, no_norm=no_norm, gt_g=gt_graph)

    image_shape = (x_dim,)

    return image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, train_dataset, valid_dataset, test_dataset

