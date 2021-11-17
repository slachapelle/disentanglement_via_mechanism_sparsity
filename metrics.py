import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

""" 
From the doc v4.3:

By default, Metrics are epoch-wise, it means

reset() is triggered every EPOCH_STARTED (See Events).

update() is triggered every ITERATION_COMPLETED.

compute() is triggered every EPOCH_COMPLETED.
"""

class MyMetrics(Metric):

    def __init__(self, include_invalid=False, suffix="", output_transform=lambda x: x):
        self._include_invalid = include_invalid
        self._suffix = suffix

        # accumulators
        self._sum_nll = None
        self._sum_nll_square = None
        self._sum_rec_loss = None
        self._sum_kl = None
        self._sum_kl_per_dim = None
        self._num_examples = None
        self._num_examples_all = None  # includes invalid
        super(MyMetrics, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._sum_nll = 0.
        self._sum_nll_square = 0.
        self._sum_rec_loss = 0.
        self._sum_kl = 0.
        self._sum_kl_per_dim = 0.
        self._num_examples = 0.
        self._num_examples_all = 0.
        super(MyMetrics, self).reset()

    @reinit__is_reduced
    def update(self, output):
        log_likelihood, valid, rec_loss, kl, kl_per_dim  = output
        valid = valid.type(log_likelihood.type())

        if self._include_invalid:
            self._sum_nll += - torch.sum(log_likelihood).item()
            self._sum_nll_square += torch.sum(log_likelihood ** 2).item()
            self._sum_kl_per_dim = self._sum_kl_per_dim + kl_per_dim.detach().sum(0)
            self._num_examples += log_likelihood.size(0)
        else:
            self._sum_nll += - torch.dot(log_likelihood, valid).item()
            self._sum_nll_square += torch.dot(log_likelihood ** 2, valid).item()
            self._sum_kl_per_dim = self._sum_kl_per_dim + torch.einsum("bj,b->j", kl_per_dim.detach(), valid)
            self._num_examples += torch.sum(valid).item()

        # bypassing valid stuff because lazy AND coherent with evaluation on training set.
        self._sum_rec_loss += log_likelihood.size(0) * rec_loss
        self._sum_kl += log_likelihood.size(0) * kl
        self._num_examples_all += log_likelihood.size(0)

    @sync_all_reduce("_num_examples", "_sum_nll", "_sum_nll_square", "_reg")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('NLL must have at least one example before it can be computed.')
        nll_mean = self._sum_nll / self._num_examples
        nll_var = self._sum_nll_square / self._num_examples - nll_mean ** 2
        rec_loss = self._sum_rec_loss / self._num_examples_all
        kl = self._sum_kl / self._num_examples_all
        rval = {"nll_" + self._suffix: nll_mean,
                "nll_var_" + self._suffix: nll_var,
                "reconstruction_loss_" + self._suffix: rec_loss,
                "kl_" + self._suffix: kl,
                "num_examples_" + self._suffix: self._num_examples}

        # compute kl_per_dim and split up in separate metrics.
        kl_per_dim = self._sum_kl_per_dim.cpu().numpy() / self._num_examples
        for i in range(kl_per_dim.shape[0]):
            rval["kl_{}_".format(i + 1) + self._suffix] = float(kl_per_dim[i])

        return rval

def get_linear_score(x, y):
    reg = LinearRegression().fit(x, y)
    return reg.score(x, y)


def linear_regression_metric(model, data_loader, device, num_samples=int(1e5), indices=None, opt=None):
    with torch.no_grad():
        if model.latent_model.z_block_size != 1:
            raise NotImplementedError("This function is implemented only for z_block_size == 1")
        model.eval()
        z_list = []
        z_hat_list = []
        sample_counter = 0
        for batch in data_loader:
            obs, _, _, _, z = batch
            obs, z = obs[:, -1].to(device), z[:, -1].to(device)
            if opt.mode in ["vae", "random_vae", "supervised_vae"]:
                z_hat = model.latent_model.mean(model.latent_model.transform_q_params(model.encode(obs)))
                z_hat = z_hat.view(z_hat.shape[0], -1)
            elif opt.mode in ["infonce", "random_infonce", "supervised_infonce"]:
                z_hat = model.latent_model.transform_q_params(model.encode(obs))[0]
                z_hat = z_hat.view(z_hat.shape[0], -1)
            else:
                raise NotImplementedError(f"function linear_regression_metric is not implemented for --mode {opt.mode}")

            z_list.append(z)
            z_hat_list.append(z_hat)
            sample_counter += obs.shape[0]
            if sample_counter >= num_samples:
                break

        z = torch.cat(z_list, 0)[:int(num_samples)]
        z_hat = torch.cat(z_hat_list, 0)[:int(num_samples)]

        z, z_hat = z.cpu().numpy(), z_hat.cpu().numpy()

        score = get_linear_score(z_hat, z)

        # masking z_hat
        # TODO: this does not take into account case where z_block_size > 1
        if indices is not None:
            z_hat_m = z_hat[:, indices[-z.shape[0]:]]
            score_m = get_linear_score(z_hat_m, z)
        else:
            score_m = 0

        return score, score_m


def mean_corr_coef_np(x, y, method='pearson', indices=None):
    """
    Source: https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py

    A numpy implementation of the mean correlation coefficient metric.
    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))

    cc = np.abs(cc)
    if indices is not None:
        cc_program = cc[:, indices[-d:]]
    else:
        cc_program = cc

    assignments = linear_sum_assignment(-1 * cc_program)
    score = cc_program[assignments].mean()

    perm_mat = np.zeros((d, d))
    perm_mat[assignments] = 1
    # cc_program_perm = np.matmul(perm_mat.transpose(), cc_program)
    cc_program_perm = np.matmul(cc_program, perm_mat.transpose())  # permute the learned latents

    return score, cc_program_perm, assignments


def mean_corr_coef(model, data_loader, device, num_samples=int(1e5), method='pearson', indices=None, opt=None):
    """Source: https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py"""
    with torch.no_grad():
        model.eval()
        z_list = []
        z_hat_list = []
        sample_counter = 0
        for batch in data_loader:
            obs, cont_c, disc_c, _, z = batch
            obs, z = obs[:, -1].to(device), z[:, -1].to(device)

            if opt.mode in ["vae", "random_vae", "supervised_vae"]:
                if model.latent_model.z_block_size != 1:
                    raise NotImplementedError("This function is implemented only for z_block_size == 1")
                z_hat = model.latent_model.mean(model.latent_model.transform_q_params(model.encode(obs)))
                z_hat = z_hat.view(z_hat.shape[0], -1)
            elif opt.mode in ["infonce", "random_infonce", "supervised_infonce"]:
                if model.latent_model.z_block_size != 1:
                    raise NotImplementedError("This function is implemented only for z_block_size == 1")
                z_hat = model.latent_model.transform_q_params(model.encode(obs))[0]
                z_hat = z_hat.view(z_hat.shape[0], -1)
            elif opt.mode == "ivae":
                # WARNING: not using disc_c (discrete auxiliary information).
                _, encoder_params, _, _ = model(obs, cont_c)
                z_hat = encoder_params[0]  # extract mean
            elif opt.mode in ["tcvae", "betavae"]:
                z_hat = model.encode(obs)[1].select(-1, 0)
            elif opt.mode in ["slowvae", "pcl"]:
                z_hat = model._encode(obs)[:, :model.z_dim]
            else:
                raise NotImplementedError(f"function mean_corr_coef is not implemented for --mode {opt.mode}")
            z_list.append(z)
            z_hat_list.append(z_hat)
            sample_counter += obs.shape[0]
            if sample_counter >= num_samples:
                break
        # if num_samples is greater than number of examples in dataset
        if sample_counter < num_samples:
            num_samples = sample_counter

        z = torch.cat(z_list, 0)[:int(num_samples)]
        z_hat = torch.cat(z_hat_list, 0)[:int(num_samples)]

        z, z_hat = z.cpu().numpy(), z_hat.cpu().numpy()
        score, cc_program_perm, assignments = mean_corr_coef_np(z, z_hat, method, indices)
        return score, cc_program_perm, assignments, z, z_hat


def edge_errors(target, pred):
    diff = target - pred

    fn = (diff == 1).sum()
    fp = (diff == -1).sum()

    return float(fn), float(fp)


def shd(target, pred):
    return sum(edge_errors(target, pred))




