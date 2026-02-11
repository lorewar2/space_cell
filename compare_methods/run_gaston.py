import sys
import os
from collections import defaultdict
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import torch

import sys
from importlib import reload

import gaston
from gaston import neural_net,cluster_plotting, dp_related, segmented_fit, model_selection
from gaston import binning_and_plotting, isodepth_scaling, run_slurm_scripts, parse_adata
from gaston import spatial_gene_classification, plot_cell_types, filter_genes, process_NN_output

import seaborn as sns
import math

COUNT_FILE = "./data/cerebellum_count.csv"
COOR_FILE = "./data/cerebellum_coor.csv"
GLM_FILE = "./data/cerebellum_glm.csv"

def main():
    #train_gaston()
    plot_iso_depth()
    return

def train_gaston():
    # script made for slide_seq cerebellum
    # loading data
    # glm
    glm_df = pd.read_csv(GLM_FILE)
    glm_df.set_index("cell_id", inplace=True)
    # coordinates
    spatial_df = pd.read_csv(COOR_FILE, usecols=[0, 1, 2])
    spatial_df = spatial_df.rename(columns={spatial_df.columns[0]: "cell_id"})
    spatial_df.set_index("cell_id", inplace=True)
    # only common cells
    common_cells = spatial_df.index.intersection(glm_df.index)
    spatial_df = spatial_df.loc[common_cells]
    glm_df = glm_df.loc[common_cells]

    S = spatial_df.to_numpy()
    A = glm_df.to_numpy()
    # training
    S_torch, A_torch = neural_net.load_rescale_input_data(S, A)
    ######################################
    # NEURAL NET PARAMETERS (USER CAN CHANGE)
    # architectures are encoded as list, eg [20,20] means two hidden layers of size 20 hidden neurons
    isodepth_arch = [20, 20] # architecture for isodepth neural network d(x,y) : R^2 -> R 
    expression_fn_arch = [20, 20] # architecture for 1-D expression function h(w) : R -> R^G

    num_epochs = 10000 # number of epochs to train NN (NOTE: it is sometimes beneficial to train longer)
    checkpoint = 500 # save model after number of epochs = multiple of checkpoint
    out_dir = './data' # folder to save model runs
    optimizer = "adam"
    num_restarts = 30
    device = 'cpu' # change to 'cpu' if you don't have a GPU

    ######################################

    seed_list=range(num_restarts)
    for seed in seed_list:
        print(f'training neural network for seed {seed}')
        out_dir_seed=f"{out_dir}/rep{seed}"
        os.makedirs(out_dir_seed, exist_ok=True)
        mod, loss_list = neural_net.train(S_torch, A_torch,
                            S_hidden_list = isodepth_arch, A_hidden_list = expression_fn_arch, 
                            epochs = num_epochs, checkpoint = checkpoint, device = device,
                            save_dir = out_dir_seed, optim = optimizer, seed = seed, save_final = True)
    return

def plot_iso_depth():
    # # load count data
    counts_df = pd.read_csv(COUNT_FILE, index_col=0)  # genes as rows, cell_ids as columns
    counts_df = counts_df.T  # now rows are cells, columns are genes
    # load coor data
    coor_df = pd.read_csv(COOR_FILE, usecols=[0, 1, 2])
    coor_df = coor_df.rename(columns={coor_df.columns[0]: "cell_id"})
    coor_df.set_index("cell_id", inplace=True)
    # only common cells
    common_cells = coor_df.index.intersection(counts_df.index)
    coor_df = coor_df.loc[common_cells]
    counts_df = counts_df.loc[common_cells]
    # to numpy
    counts_mat = counts_df.to_numpy()
    coords_mat = coor_df.to_numpy()
    # MODEL TRAINED ABOVE
    gaston_model, A, S = process_NN_output.process_files('./data/gaston_cerebellum_trained')
    # domain num selection by kneed
    # model_selection.plot_ll_curve(gaston_model, A, S, max_domain_num=8, start_from=2, num_buckets=100)
    # CHANGE FOR YOUR APPLICATION: use number of domains from above!
    num_layers = 4 

    # identify labels
    gaston_isodepth, gaston_labels = dp_related.get_isodepth_labels(gaston_model, A, S, num_layers)

    # DATASET-SPECIFIC: so domains are ordered oligodendrocyte to molecular, with increasing isodepth
    gaston_isodepth = np.max(gaston_isodepth) - gaston_isodepth
    gaston_labels = (num_layers - 1) - gaston_labels

    # scaling
    scale_factor=64/100 # since 64 pixels = 100 microns in slide-seq image

    # WITH VISUALIZATION
    # gaston_isodepth=isodepth_scaling.adjust_isodepth(gaston_isodepth, gaston_labels, coords_mat, 
    #                                  q_vals=[0.2, 0.05, 0.15, 0.3], visualize=True, figsize=(12,12),num_rows=2)

    # WITHOUT
    gaston_isodepth=isodepth_scaling.adjust_isodepth(gaston_isodepth, gaston_labels, coords_mat, 
                                    q_vals=[0.2, 0.05, 0.15, 0.3], scale_factor=scale_factor)

    # plotting
    show_streamlines = True
    cluster_plotting.plot_isodepth(gaston_isodepth, S, gaston_model, figsize=(7, 6), streamlines = show_streamlines, cmap = 'Reds',
                              neg_gradient = True) # since we did isodepth -> -1*isodepth above, we also need to do gradient -> -1*gradient
    
    labels=['Oligodendrocyte layer', 'Granule layer', 'Purkinje-Bergmann layer', 'Molecular layer']

    # WITHOUT CUSTOM COLORS
    # cluster_plotting.plot_clusters(gaston_labels, S, figsize=(6,6), colors=None, 
    #                                color_palette=plt.cm.Dark2, s=10,labels=labels,lgd=True)
    plt.savefig("clusters1.png", dpi=300, bbox_inches="tight")
    plt.close()
    # TO PLOT WITH CUSTOM COLORS:
    domain_colors=['C6', 'mediumseagreen', 'darkmagenta', 'C8']
    cluster_plotting.plot_clusters(gaston_labels, S, figsize=(6,6), 
                                colors=domain_colors, s=10,labels=labels,lgd=True)
    plt.savefig("clusters2.png", dpi=300, bbox_inches="tight")
    plt.close()
    return

if __name__ == '__main__':
    main()