import argparse
import os
import sys
import pathlib
import pickle
import json

try:
    from comet_ml import Experiment
    COMET_AVAIL = True
except:
    COMET_AVAIL = False
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from baseline_models.icebeem.models.ivae.ivae_wrapper import IVAE_wrapper
from baseline_models.icebeem.models.icebeem_wrapper import ICEBEEM_wrapper
from train import get_dataset, get_loader
from universal_logger.logger import UniversalLogger
from metrics import evaluate_disentanglement

def parse_sim():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--output_dir", required=True,
                        help="Directory to output logs and model checkpoints")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Type of the dataset to be used 'toy-MANIFOLD/TRANSITION_MODEL'")
    parser.add_argument('--method', type=str, default='icebeem', choices=['icebeem', 'ivae'],
                        help='Method to employ. Should be TCL, iVAE or ICE-BeeM')
    parser.add_argument("--dataroot", type=str, default="./",
                        help="path to dataset")
    parser.add_argument("--gt_z_dim", type=int, default=5,
                        help="ground truth dimensionality of z (for TRANSITION_MODEL == 'linear_system')")
    parser.add_argument("--gt_x_dim", type=int, default=10,
                        help="ground truth dimensionality of x (for MANIFOLD == 'nn')")
    parser.add_argument("--num_samples", type=float, default=int(1e6),
                        help="Number of samples")
    parser.add_argument("--rand_g_density", type=float, default=None,
                        help="Probability of sampling an edge. When None, the graph is set to a default (or to gt_graph_name).")
    parser.add_argument("--gt_graph_name", type=str, default=None,
                        help="Name of the ground-truth graph to use in synthetic data.")
    parser.add_argument("--architecture", type=str, default='ilcm_tabular', choices=['ilcm_tabular'],
                        help="encoder/decoder architecture.")
    parser.add_argument("--learn_decoder_var", action='store_true',
                        help="Whether to learn the variance of the output decoder")
    parser.add_argument("--train_prop", type=float, default=None,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--valid_prop", type=float, default=0.10,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--test_prop", type=float, default=0.10,
                        help="proportion of all samples used in test set")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="batch size used during training")
    parser.add_argument("--eval_batch_size", type=int, default=1024,
                        help="batch size used during evaluation")
    parser.add_argument("--time_limit", type=float, default=None,
                        help="After this amount of time, terminate training. (in hours)")
    parser.add_argument("--max_iter", type=int, default=int(1e6),
                        help="Maximal amount of iterations")
    parser.add_argument("--ivae_lr", type=float, default=1e-4,
                        help="After this amount of time, terminate training. (in hours)")
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

    #parser.add_argument('--dataset', type=str, default='TCL', help='Dataset to run experiments. Should be TCL or IMCA')

    #parser.add_argument('--config', type=str, default='imca.yaml', help='Path to the config file')
    #parser.add_argument('--run', type=str, default='run/', help='Path for saving running related data.')
    #parser.add_argument('--nSims', type=int, default=10, help='Number of simulations to run')

    #parser.add_argument('--test', action='store_true', help='Whether to evaluate the models from checkpoints')
    #parser.add_argument('--plot', action='store_true')

    return parser.parse_args()

def main(args):
    print('WARNING: this code do not support discrete auxiliary variable. See warning in mean_corr_coef function in metrics.py')
    print('Running {} experiments using {}'.format(args.dataset, args.method))

    device = torch.device('cuda:0' if args.cuda else 'cpu')

    if args.method.lower() == 'ivae':
        args.mode = "ivae"
    else:
        raise ValueError('Unsupported method {}'.format(args.method))

    ## ---- Save hparams ---- ##
    kwargs = vars(args)
    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    ## ---- Data ---- ##
    args.n_lag = 0
    args.no_norm = False
    args.n_workers = 0  # can't put it to 4 since we get weird error msg...
    image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, train_dataset, valid_dataset, test_dataset = get_dataset(args)
    _, _, test_loader = get_loader(args, train_dataset, valid_dataset, test_dataset)
    x, y, s = train_dataset.x.cpu().numpy(), train_dataset.c.cpu().numpy(), train_dataset.z.cpu().numpy()

    ## ---- Logger ---- ##
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

    ## ---- Running ---- ##
    # the argument `architecture="ilcm_tabular"` will choose the same encoder/decoder as in ilcm for synthetic experiments.
    model = IVAE_wrapper(x, y, args.gt_z_dim, ckpt_folder=args.output_dir, batch_size=args.batch_size, max_iter=args.max_iter, #max_iter=100,
                        seed=args.seed, n_layers=5, hidden_dim=512, lr=args.ivae_lr,
                        architecture=args.architecture, logger=logger, time_limit=args.time_limit, learn_decoder_var=args.learn_decoder_var)

    ## ---- Evaluate performance ---- ##
    mcc, consistent_r, r, cc, C_hat, C_pattern, perm_mat, z, z_hat, transposed_consistent_r = evaluate_disentanglement(model, test_loader, device, args)

    ## ---- Save ---- ##
    # save scores
    metrics = {"mean_corr_coef_final": mcc,
               "consistent_r_final": consistent_r,
               "r_final": r,
               "transposed_consistent_r_final": transposed_consistent_r}
    logger.log_metrics(step=0, metrics=metrics)

    # save both ground_truth and learned latents
    np.save(os.path.join(args.output_dir, "z_hat_final.npy"), z_hat)
    np.save(os.path.join(args.output_dir, "z_gt_final.npy"), z)

if __name__ == '__main__':
    args = parse_sim()
    main(args)
