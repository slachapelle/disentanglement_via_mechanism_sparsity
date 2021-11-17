import argparse
import shutil
import os, json, sys, traceback, time
import pathlib

try:
    from comet_ml import Experiment
    COMET_AVAIL = True
except:
    COMET_AVAIL = False
import numpy as np
import torch
import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from scripts.solver import Solver

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from train import get_dataset, get_loader
from universal_logger.logger import UniversalLogger
from metrics import mean_corr_coef, get_linear_score

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args, writer=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if args.cuda else 'cpu')

    ## ---- Data ---- ##
    args.no_norm = False
    args.n_lag = 0  # get dataset expects this argument, but no effect.
    args.num_workers = args.n_workers
    image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, train_dataset, valid_dataset, test_dataset, = get_dataset(args)
    train_loader, valid_loader, test_loader = get_loader(args, train_dataset, valid_dataset, test_dataset)
    data_loader = train_loader
    if len(image_shape) == 3:
        args.num_channel = image_shape[-1]
    else:
        args.num_channel = None

    ## ---- Logging ---- ##
    if COMET_AVAIL and args.comet_key is not None and args.comet_workspace is not None and args.comet_project_name is not None:
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

    t0 = time.time()

    # saving hp
    ## ---- Save hparams ---- ##
    if args.pcl:
        args.mode = "pcl"
    else:
        args.mode = 'slowvae'
    kwargs = vars(args)
    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)
    with open(os.path.join(args.output_dir, "args"), "w") as f:
        json.dump(args.__dict__, f)

    net = Solver(args, image_shape, data_loader=data_loader, logger=logger, z_dim=train_dataset.z_dim)
    failure = net.train(writer)
    if failure:
        print('failed in %.2fs' % (time.time() - t0))
        #shutil.rmtree(args.output_dir)
    else:
        print('done in %.2fs' % (time.time() - t0))

        ## ---- Evaluate performance ---- ##
        # compute MCC and save representation
        mcc, cc_program_perm, assignments, z, z_hat = mean_corr_coef(net.net, test_loader, device, opt=args)
        linear_score = get_linear_score(z_hat, z)

        ## ---- Save ---- ##
        # save scores
        logger.log_metrics(step=0, metrics={"mcc": mcc, "linear_score": linear_score})

        # save both ground_truth and learned latents
        np.save(os.path.join(args.output_dir, "z_hat.npy"), z_hat)
        np.save(os.path.join(args.output_dir, "z_gt.npy"), z)


### For Random Search ###
def randint(low, high):
    return np.int(np.random.randint(low, high, 1)[0])

def uniform(low, high):
    return np.random.uniform(low, high, 1)[0]

def loguniform(low, high):
    return np.exp(np.random.uniform(np.log(low), np.log(high), 1))[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='slowVAE')
    parser.add_argument('--pcl', action='store_true')
    parser.add_argument('--r_func', type=str, default='default', choices=['default', 'mlp'],
                        help='Type of regression function used for PCL')
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
    parser.add_argument("--architecture", type=str, default='ilcm_tabular', choices=['ilcm_tabular', 'standard_conv'],
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
    parser.add_argument("--add_noise", type=float, default=0.0,
                        help="Add normal noise sigma = add_noise on images (only training data)")
    parser.add_argument("--no_cuda", action="store_false", dest="cuda",
                        help="Disables cuda")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="batch size used during training")
    parser.add_argument("--eval_batch_size", type=int, default=1024,
                        help="batch size used during evaluation")
    parser.add_argument('--beta', default=1, type=float,
                        help='weight for kl to normal')
    parser.add_argument('--gamma', default=10, type=float,
                        help='weight for kl to laplace')
    parser.add_argument('--rate_prior', default=6, type=float,
                        help='rate (or inverse scale) for prior laplace (larger -> sparser).')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Adam optimizer beta2')
    parser.add_argument('--ckpt-name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--log_step', default=100, type=int,
                        help='numer of iterations after which data is logged')
    parser.add_argument('--save_step', default=10000, type=int,
                        help='number of iterations after which a checkpoint is saved')
    args = parser.parse_args()

    args = main(args)
