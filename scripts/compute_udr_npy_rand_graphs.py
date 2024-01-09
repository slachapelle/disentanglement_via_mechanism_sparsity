import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time
import pathlib

import numpy as np
import pandas as pd

#sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath('..'))
from metrics import mean_corr_coef_np

def load_hparams(folder_path):
    with open(os.path.join(folder_path, 'hparams.json'), 'r') as infile:
        opt = json.load(infile)

    class Bunch:
        def __init__(self, opt):
            self.__dict__.update(opt)
    return Bunch(opt)

def compute_udr_mcc_from_npy(inferred_model_reps):
  """Computes the UDR score using scikit-learn.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_functions: functions that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: numpy random state used for randomness.
    batch_size: Number of datapoints to compute in a single batch. Useful for
      reducing memory overhead for larger models.
    num_data_points: total number of representation datapoints to generate for
      computing the correlation matrix.
    correlation_matrix: Type of correlation matrix to generate. Can be either
      "lasso" or "spearman".
    filter_low_kl: If True, filter out elements of the representation vector
      which have low computed KL divergence.
    include_raw_correlations: Whether or not to include the raw correlation
      matrices in the results.
    kl_filter_threshold: Threshold which latents with average KL divergence
      lower than the threshold will be ignored when computing disentanglement.

  Returns:
    scores_dict: a dictionary of the scores computed for UDR with the following
    keys:
      raw_correlations: (num_models, num_models, latent_dim, latent_dim) -  The
        raw computed correlation matrices for all models. The pair of models is
        indexed by axis 0 and 1 and the matrix represents the computed
        correlation matrix between latents in axis 2 and 3.
      pairwise_disentanglement_scores: (num_models, num_models, 1) - The
        computed disentanglement scores representing the similarity of
        representation between pairs of models.
      model_scores: (num_models) - List of aggregated model scores corresponding
        to the median of the pairwise disentanglement scores for each model.
  """

  num_models = len(inferred_model_reps)
  mcc_all = np.zeros((num_models, num_models))

  for i in range(num_models):
    for j in range(num_models):
      if i == j:
        continue

      mcc = mean_corr_coef_np(inferred_model_reps[i],
                        inferred_model_reps[j],
                        method='pearson', indices=None)[0]

      mcc_all[i, j] = mcc
  off_diag = mcc_all[~np.eye(mcc_all.shape[0], dtype=bool)]
  return {'median': np.median(off_diag), 'mean': np.mean(off_diag)}

def create_mode_entry(all_logs_pd):
    # for tcvae
    all_logs_pd.loc[all_logs_pd["tcvae"] == True, 'mode'] = "tcvae"

    # TODO: create for other methods if necessary.

    return all_logs_pd

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_logs_file", type=str,
                        help="Absolute path to all_logs.npy files")

    GRAPH_DENSITIES = [0.25, 0.5, 0.75]
    DATASET_NAMES = ["toy-nn/action_sparsity_non_trivial", "toy-nn/temporal_sparsity_non_trivial"]
    MODES = ['vae'] #, 'pcl', 'slowvae', 'tcvae', 'ivae', 'random_vae', 'supervised_vae']
    HPARAM_NAMES = {"vae": ['gc_reg_coeff', 'g_reg_coeff']}
                    #"random_vae": [],
                    #"supervised_vae": [],
                    #"tcvae": ["beta"],
                    #"ivae": [],
                    #"pcl": [],
                    #"slowvae": ['gamma', 'rate_prior']}

    opt = parser.parse_args()

    all_logs = np.load(opt.all_logs_file, allow_pickle=True).tolist()
    all_logs_pd = pd.DataFrame(all_logs)
    #all_logs_pd = create_mode_entry(all_logs_pd)

    # to be filled with udr values
    all_logs_pd_udr = all_logs_pd.copy()

    for dataset in DATASET_NAMES:
        for graph_density in GRAPH_DENSITIES:
            for mode in MODES:
                print("########## dataset:", dataset, "graph_density", graph_density, "mode:", mode, "##########")
                condition_data_mode = (all_logs_pd['dataset'] == dataset) & (all_logs_pd['rand_g_density'] == graph_density) & (all_logs_pd['mode'] == mode)
                #condition = (all_logs_pd['dataset'] == gt_graph_name) & (all_logs_pd['mode'] == mode)  # only for debugging
                logs = all_logs_pd[condition_data_mode]

                if len(logs) == 0:
                    print("No logs found.")
                    continue

                if len(HPARAM_NAMES[mode]) != 0:
                    hparams_values = logs[HPARAM_NAMES[mode]].drop_duplicates()

                    for i in range(len(hparams_values)):
                        # selecting only the runs with this specific hparam value
                        condition_hp = (hparams_values.iloc[i] == logs[HPARAM_NAMES[mode]]).all(axis=1)
                        logs_specific_hp = logs[condition_hp]

                        # logging number of seeds used to compute UDR
                        all_logs_pd_udr.loc[condition_data_mode & condition_hp, "num_seeds"] = len(logs_specific_hp)

                        # cannot compute UDR when there is only one seed
                        if len(logs_specific_hp) == 1:
                            print(f"Not computing UDR for {hparams_values.iloc[i].to_dict()} since only one seed.")
                            all_logs_pd_udr.loc[condition_data_mode & condition_hp, "udr_mean"] = -1
                            all_logs_pd_udr.loc[condition_data_mode & condition_hp, "udr_median"] = -1
                            continue

                        # load their z_hat_final.npy
                        z_hat_list = []
                        for output_dir in logs_specific_hp["output_dir"]:
                            z_hat_list.append(np.load(os.path.join(output_dir, "z_hat_final.npy")))

                        print(f"Computing UDR for {hparams_values.iloc[i].to_dict()} on {len(logs_specific_hp)} seeds.")
                        udr = compute_udr_mcc_from_npy(z_hat_list)

                        # Add udr values to table
                        all_logs_pd_udr.loc[condition_data_mode & condition_hp, "udr_mean"] = udr["mean"]
                        all_logs_pd_udr.loc[condition_data_mode & condition_hp, "udr_median"] = udr["median"]

                else:
                    print(f"No hyperparameter search. Total of {len(logs)} seeds.")
                    # number of sucessful seeds
                    all_logs_pd_udr.loc[condition_data_mode, "num_seeds"] = len(logs)
                    # if the method has no hparameter, just set UDR score to -1
                    all_logs_pd_udr.loc[condition_data_mode, "udr_mean"] = -1
                    all_logs_pd_udr.loc[condition_data_mode, "udr_median"] = -1


    np.save(opt.all_logs_file.replace(".npy", "_udr.npy"), all_logs_pd_udr.to_dict('records'))


if __name__ == "__main__":
    main()
