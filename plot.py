import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_matrix(matrix, title="", row_label="", col_label="", col_to_mark=[], row_to_mark=[], kl_values=None, row_names=[], col_names=[], vmin=0., vmax=1.):
    fig = plt.figure()
    if kl_values is not None:
        ax = fig.add_subplot(2, 1, 1)
        ax_kl = fig.add_subplot(2, 1, 2)
    else:
        ax = fig.add_subplot(1, 1, 1)
    ax.matshow(matrix, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_xticklabels([''] + col_names)
    ax.set_yticklabels([''] + row_names)

    # Loop over data dimensions and create text annotations.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = "{:.2f}".format(matrix[i, j])
            if i in row_to_mark:
                text += "*"
            if j in col_to_mark:
                text += "*"
            ax.text(j, i, text, ha="center", va="center", color="w")

    if kl_values is not None:
        ax_kl.matshow(kl_values.reshape(1, -1), vmin=0)
        ax_kl.set_xlabel("KL values")

    return fig

def plot_weighted_adjacency_vs_steps(weighted_adjacency, gt_adjacency, iterations=None):
    num_row = weighted_adjacency.shape[1]
    num_col = weighted_adjacency.shape[2]
    if iterations is None:
        iterations = range(weighted_adjacency.shape[0])
    assert weighted_adjacency.shape[0] == len(iterations)

    fig, ax1 = plt.subplots()

    # Plot weight of incorrect edges
    for i in range(num_row):
        for j in range(num_col):
            if gt_adjacency[i, j] == 1:
                color = 'g'  # correct edge
            else:
                color = 'r'  # incorrect edge
            y = weighted_adjacency[:, i, j]
            ax1.plot(iterations, y, color, linewidth=1)

    fig.tight_layout()
    return fig

