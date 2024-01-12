# Disentanglement via Mechanism Sparsity

This repository contains the code used to run the experiments in the papers:<br>
[1] [Disentanglement via Mechanism Sparsity Regularization: A New Principle for Nonlinear ICA](https://arxiv.org/abs/2107.10098) (CLeaR2022)<br>
[2] [Nonparametric Partial Disentanglement via Mechanism Sparsity: Sparse Actions, Interventions and Sparse Temporal Dependencies](https://arxiv.org/abs/2401.04890) (Preprint)<br> 
By Sébastien Lachapelle, Pau Rodríguez López, Yash Sharma, Katie Everett, Rémi Le Priol, Alexandre Lacoste, Simon Lacoste-Julien

### Environment:

Tested on python 3.7.

See `requirements.txt`.

### Action-sparsity experiment
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
DATASET=<name of dataset>
python disentanglement_via_mechanism_sparsity/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR --mode vae --dataset toy-nn/action_sparsity_non_trivial --freeze_g --freeze_gc --z_dim 10 --gt_z_dim 10 --gt_x_dim 20 --n_lag 0 --time_limit 3
```

### Time-sparsity experiment
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR --mode vae --dataset toy-nn/temporal_sparsity_non_trivial --freeze_g --freeze_gc --z_dim 10 --gt_z_dim 10 --gt_x_dim 20 --n_lag 1 --full_seq --time_limit 3
```

### Adding penalty regularization
In the minimal commands provided above, all regularizations are deactivated (via the `--freeze_g` and `--freeze_gc` flags). 
To activate the penalty regularization for say the temporal mask G^z, replace `--freeze_g` by `--g_reg_coeff COEFF_VALUE`. 
Same syntax works also for the action mask G^a (named `gc` in the code). Here's the correspondence between the mask names in the code (left) and in the paper (right):

`g` = Mask G^z (Time sparsity)

`gc` = Mask G^a (Action sparsity)

### Adding constraint regularization
To activate the constraint regularization for say the temporal mask G^z, replace `--freeze_g` by `--g_constraint UPPER_BOUND`. 
Same syntax works also for the action mask G^a (named `gc` in the code). 
The experiments were all performed with `--constraint_scedule 150000` and `--dual_restarts`. 
The option `--set_constraint_to_gt` will automatically set the upper bound of the constraint to the optimal value for the ground-truth graph.

### Synthetic datasets (referencing to sections of [2])
Here's a list of the synthetic datasets used. The data is generated before training, so no need to download anything.

#### Section 8.1 (Used in both [1] and [2])

- `--dataset toy-nn/action_sparsity_trivial` 
- `--dataset toy-nn/action_sparsity_non_trivial`
- `--dataset toy-nn/action_sparsity_non_trivial_no_suff_var`
- `--dataset toy-nn/action_sparsity_non_trivial_k=2`
- `--dataset toy-nn/temporal_sparsity_trivial`
- `--dataset toy-nn/temporal_sparsity_non_trivial`
- `--dataset toy-nn/temporal_sparsity_non_trivial_no_suff_var`
- `--dataset toy-nn/temporal_sparsity_non_trivial_k=2`

#### Section 8.2 (Used only in [2])
- `--dataset toy-nn/action_sparsity_non_trivial --graph_name graph_action_3_easy`
- `--dataset toy-nn/action_sparsity_non_trivial --graph_name graph_action_3_hard`
- `--dataset toy-nn/temporal_sparsity_non_trivial --graph_name graph_temporal_3_easy`
- `--dataset toy-nn/temporal_sparsity_non_trivial --graph_name graph_temporal_3_hard`
- `--dataset toy-nn/action_sparsity_non_trivial --rand_g_density PROBA_OF_EDGE`
- `--dataset toy-nn/temporal_sparsity_non_trivial --rand_g_density PROBA_OF_EDGE`


#### Datasets Used only in [1]
- `--dataset toy-nn/action_sparsity_non_trivial_no_graph_crit`
- `--dataset toy-nn/temporal_sparsity_non_trivial_no_graph_crit` 

### Baselines
#### TCVAE
Code adapted from: https://github.com/rtqichen/beta-tcvae
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/baseline_models/beta-tcvae/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR  --dataset toy-nn/action_sparsity_non_trivial --tcvae --beta 1 --gt_z_dim 10 --gt_x_dim 20 --time_limit 3
```

#### iVAE
Code adapted from: https://github.com/ilkhem/icebeem
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/baseline_models/icebeem/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR  --dataset toy-nn/action_sparsity_non_trivial --method ivae --gt_z_dim 10 --gt_x_dim 20 --time_limit 3
```

#### SlowVAE
Code adapted from: https://github.com/bethgelab/slow_disentanglement
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/baseline_models/slowvae_pcl/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR --dataset toy-nn/temporal_sparsity_non_trivial --gt_z_dim 10 --gt_x_dim 20 --time_limit 3
```

#### PCL
Code adapted from: https://github.com/bethgelab/slow_disentanglement/tree/baselines
```
OUTPUT_DIR=<where to save experiment>
DATAROOT=<where data is located>
python disentanglement_via_mechanism_sparsity/baseline_models/slowvae_pcl/train.py --dataroot $DATAROOT --output_dir $OUTPUT_DIR --dataset toy-nn/temporal_sparsity_non_trivial --pcl --r_func mlp --gt_z_dim 10 --gt_x_dim 20 --time_limit 3
```


