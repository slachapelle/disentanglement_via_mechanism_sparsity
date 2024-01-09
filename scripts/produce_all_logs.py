import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
# sys.path.insert(0, os.path.abspath('..'))
from copy import deepcopy

import numpy as np

import time

def load_hparams(folder_path):
    with open(os.path.join(folder_path, 'hparams.json'), 'r') as infile:
        opt = json.load(infile)
    return opt
    # class Bunch:
    #     def __init__(self, opt):
    #         self.__dict__.update(opt)
    #return Bunch(opt)

def load_metrics(folder_path):
    with open(os.path.join(folder_path, 'log.ndjson'), 'r') as infile:
        metrics = json.loads(infile.readlines()[-1])  # reading last line
    return metrics

def main(args=None):
    ROOT = '/network/scratch/l/lachaseb/identifiable_latent_causal_model/exp/'
    #META_EXPS = {"temporal_ilcm": "e0b63788-0eeb-11ee-8f41-424954050100",
    #             "temporal_random": "0b4e90b8-0f81-11ee-b8a0-424954050300",
    #             "temporal_supervised": "e3baec86-0f80-11ee-9b91-424954050300",
    #             "temporal_tcvae": "25e9291a-0f95-11ee-9cf4-424954050100",
    #             "temporal_pcl": "81168a28-106f-11ee-bb87-424954050200",
    #             "temporal_slowvae": "ee3dd866-1071-11ee-ab0d-424954050200",
    #             "action_ilcm": "4472ac22-1124-11ee-80d1-48df37d42c20",
    #             "action_random": "f668798e-1124-11ee-86e1-48df37d42c20",
    #             "action_supervised": "f81f68f4-1125-11ee-9e63-48df37d42c20",
    #             "action_tcvae": "fc4313ee-1504-11ee-bca8-d8c497b83240",
    #             "action_ivae": "5d35b610-1119-11ee-bc8f-424954050100",
    #             }
    #META_EXPS = {"temporal_ilcm": "71fee648-4e81-11ee-9556-424954050200",
    #             "action_ilcm": "3645f1aa-4e81-11ee-b87d-424954050200",
    #             }
    #META_EXPS = {"temporal_ilcm": "462a70b4-5318-11ee-8099-424954050100",
    #             "action_ilcm": "66143ee0-5332-11ee-b033-424954050100",
    #             }
    #META_EXPS = {"temporal_ilcm": "95b2eb58-53ee-11ee-929b-424954050300",
    #             "action_ilcm": "3d959e64-5404-11ee-8e54-424954050300",
    #            }
    #META_EXPS = {"temporal_ilcm": "4d5d4c0a-54b0-11ee-be29-424954050100",
    #             "action_ilcm": "f854193c-54af-11ee-be2a-424954050100",
    #            }
    META_EXPS = {"temporal_ilcm": "2cdcae28-58b9-11ee-befb-424954050100",
                 "action_ilcm": "77382470-58b9-11ee-a8ff-424954050100",
                }



    all_logs = []
    for name, meta_exp in META_EXPS.items():
        meta_exp_path = os.path.join(ROOT, meta_exp)
        num_exps = 0
        for exp in os.listdir(meta_exp_path):
            exp_path = os.path.join(meta_exp_path, exp)
            if not os.path.isdir(exp_path):
                continue

            # verify if run is completed
            if not os.path.exists(os.path.join(exp_path, "z_hat_final.npy")):
                print(f"{name}: Run {meta_exp}/{exp} is not completed thus excluded from all_logs file")
                #log = load_hparams(exp_path)
                #print("beta", log["beta"])
                continue

            num_exps += 1

            log = load_hparams(exp_path)
            metrics = load_metrics(exp_path)
            log.update(metrics)

            all_logs.append(log)
        print(f"{name}: Done with {num_exps} experiments.")

    save_path = os.path.join(ROOT, "all_logs_23sep2023_JMLR_const_rand_g.npy")
    np.save(save_path, all_logs)
    print("Saved to:", save_path)

if __name__ == "__main__":
    main()
