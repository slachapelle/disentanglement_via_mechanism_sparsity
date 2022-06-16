import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn
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


def get_z_z_hat(model, data_loader, device, num_samples=int(1e5), opt=None):
    with torch.no_grad():
        model.eval()
        z_list = []
        z_hat_list = []
        sample_counter = 0
        for batch in data_loader:
            obs, cont_c, disc_c, _, z = batch
            b, t = obs.shape[0:2]
            #obs, z = obs[:, -1].to(device), z[:, -1].to(device)
            #obs, z = obs[:, 0].to(device), z[:, 0].to(device)  # using first sample instead of last one.
            obs, z = obs.reshape((b * t,) + obs.shape[2:]).to(device), z.reshape((b * t,) + z.shape[2:]).to(device)  # using all steps.

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
                raise NotImplementedError(f"function get_z_z_hat is not implemented for --mode {opt.mode}")
            z_list.append(z)
            z_hat_list.append(z_hat)
            sample_counter += obs.shape[0]
            assert z.shape[0] == z_hat.shape[0] and z.shape[0] == obs.shape[0]
            if sample_counter >= num_samples:
                break
        # if num_samples is greater than number of examples in dataset
        if sample_counter < num_samples:
            num_samples = sample_counter

        z = torch.cat(z_list, 0)[:int(num_samples)]
        z_hat = torch.cat(z_hat_list, 0)[:int(num_samples)]

        return z.cpu().numpy(), z_hat.cpu().numpy()


def mean_corr_coef_np(z, z_hat, method='pearson', indices=None):
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
    x, y = z, z_hat
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]  # z x z_hat
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


def test_consistency(z, z_hat, assignments, dataset):
    z_dim = z.shape[1]
    perm_mat = np.zeros((z_dim, z_dim))
    perm_mat[assignments] = 1 # perm_mat is P^T in paper

    # Compute G-consistency pattern
    C = np.ones_like(perm_mat)
    if hasattr(dataset, "gt_g"):
        gt_g = dataset.gt_g.cpu().numpy()
        g_pattern = np.maximum(0, 1 - np.matmul(gt_g, 1 - gt_g.T)) * np.maximum(0, 1 - np.matmul(gt_g.T, 1 - gt_g))
        C *= g_pattern
    if hasattr(dataset, "gt_gc"):
        gt_gc = dataset.gt_gc.cpu().numpy()
        gc_pattern = np.maximum(0, 1 - np.matmul(gt_gc, 1 - gt_gc.T))
        C *= gc_pattern

    L_pattern = np.matmul(C, perm_mat)

    consistent_r = 0
    for i in range(z.shape[1]):
        masked_z_hat = z_hat * L_pattern[i, :]
        reg = LinearRegression(fit_intercept=True).fit(masked_z_hat, z[:, i])
        consistent_r += np.sqrt(reg.score(masked_z_hat, z[:, i]))  # the sqrt of R^2 is called the "coefficient of multiple correlation".
    consistent_r /= z.shape[1]

    # same as above, but with transposed C
    L_pattern_ = np.matmul(C.T, perm_mat)

    transposed_consistent_r = 0
    for i in range(z.shape[1]):
        masked_z_hat = z_hat * L_pattern_[i, :]
        reg = LinearRegression(fit_intercept=True).fit(masked_z_hat, z[:, i])
        transposed_consistent_r += np.sqrt(reg.score(masked_z_hat, z[:, i]))  # the sqrt of R^2 is called the "coefficient of multiple correlation".
    transposed_consistent_r /= z.shape[1]

    return consistent_r, transposed_consistent_r, C


def get_linear_score(x, y):
    reg = LinearRegression(fit_intercept=True).fit(x, y)
    y_pred = reg.predict(x)
    r2s = sklearn.metrics.r2_score(y, y_pred, multioutput='raw_values')
    r = np.mean(np.sqrt(r2s))  # To be comparable to MCC (this is the average of R = coefficient of multiple correlation)
    return r, reg.coef_


def linear_regression_metric(z, z_hat, indices=None):
    # standardize z and z_hat
    z = (z - np.mean(z, 0)) / np.std(z, 0)
    z_hat = (z_hat - np.mean(z_hat, 0)) / np.std(z_hat, 0)

    score, L_hat = get_linear_score(z_hat, z)

    # masking z_hat
    # TODO: this does not take into account case where z_block_size > 1
    if indices is not None:
        z_hat_m = z_hat[:, indices[-z.shape[0]:]]
        score_m, _ = get_linear_score(z_hat_m, z)
    else:
        score_m = 0

    return score, score_m, L_hat


def edge_errors(target, pred):
    diff = target - pred

    fn = (diff == 1).sum()
    fp = (diff == -1).sum()

    return float(fn), float(fp)


def shd(target, pred):
    return sum(edge_errors(target, pred))


def evaluate_disentanglement(model, data_loader, device, opt):
    z, z_hat = get_z_z_hat(model, data_loader, device, opt=opt)
    mcc, cc, assignments = mean_corr_coef_np(z, z_hat)
    consistent_r, transposed_consistent_r, C_pattern = test_consistency(z, z_hat, assignments, data_loader.dataset)
    r, _, L_hat = linear_regression_metric(z, z_hat)

    perm_mat = np.zeros_like(L_hat)
    perm_mat[assignments] = 1  # perm_mat is P^T in paper
    C_hat = np.matmul(L_hat, perm_mat.T)

    #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    #print(np.corrcoef(z, rowvar=False))
    return mcc, consistent_r, r, cc, C_hat, C_pattern, perm_mat, z, z_hat, transposed_consistent_r

