# Disentanglement via Mechanism Sparsity Regularization: A New Principle for Nonlinear ICA.

This repository contains the code used to run the experiments in the paper "Disentanglement via Mechanism Sparsity Regularization: A New Principle for Nonlinear ICA".

### Environment:

Tested on python 3.7.

See `requirements.txt`.

### Time-sparsity experiment
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR --mode vae --dataset toy-nn/temporal_sparsity_non_trivial --freeze_g --freeze_gc --z_dim 10 --gt_z_dim 10 --gt_x_dim 20 --n_lag 1 --full_seq --time_limit 3
```

### Action-sparsity experiment
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
DATASET=<name of dataset>
python disentanglement_via_mechanism_sparsity/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR --mode vae --dataset toy-nn/action_sparsity_non_trivial --freeze_g --freeze_gc --z_dim 10 --gt_z_dim 10 --gt_x_dim 20 --n_lag 0 --time_limit 3
```

### Regularization
In the minimal commands provided above, all regularizations are deactivated (via the `--freeze_g` and `--freeze_gc` flags). 
To activate the regularization for say the temporal mask G^z, replace `--freeze_g` by `--g_reg_coeff COEFF_VALUE`. 
Same syntax works also for the action mask G^a (named `gc` in the code). Here's the correspondence between the mask names in the code (left) and in the paper (right):

`g` = Mask G^z (Time sparsity)

`gc` = Mask G^a (Action sparsity)

### Synthetic datasets
For synthetic data (--dataset toy-*), the data is generated before training, so no need to download anything. Here are the datasets used in the paper:
- toy-nn/temporal_sparsity_trivial
- toy-nn/temporal_sparsity_non_trivial
- toy-nn/temporal_sparsity_non_trivial_no_graph_crit
- toy-nn/temporal_sparsity_non_trivial_no_suff_var
- toy-nn/action_sparsity_trivial
- toy-nn/action_sparsity_non_trivial
- toy-nn/action_sparsity_non_trivial_no_graph_crit
- toy-nn/action_sparsity_non_trivial_no_suff_var

#### Baselines
##### TCVAE
Code adapted from: https://github.com/rtqichen/beta-tcvae
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/baseline_models/beta-tcvae/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR  --dataset toy-nn/action_sparsity_non_trivial --tcvae --beta 1 --gt_z_dim 10 --gt_x_dim 20 --time_limit 3
```

##### iVAE
Code adapted from: https://github.com/ilkhem/icebeem
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/baseline_models/icebeem/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR  --dataset toy-nn/action_sparsity_non_trivial --method ivae --gt_z_dim 10 --gt_x_dim 20 --time_limit 3
```

##### SlowVAE
Code adapted from: https://github.com/bethgelab/slow_disentanglement
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/baseline_models/slowvae_pcl/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR --dataset toy-nn/temporal_sparsity_non_trivial --gt_z_dim 10 --gt_x_dim 20 --time_limit 3
```

##### PCL
Code adapted from: https://github.com/bethgelab/slow_disentanglement/tree/baselines
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/baseline_models/slowvae_pcl/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR --dataset toy-nn/temporal_sparsity_non_trivial --pcl --r_func mlp --gt_z_dim 10 --gt_x_dim 20 --time_limit 3
```


