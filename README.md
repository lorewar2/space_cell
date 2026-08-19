# SpaceCell

Cell-type and spatial-domain clustering for spatial transcriptomics data, combining histology-aware expression imputation, GLM-PCA + Gaussian mixture clustering, and negative-binomial refinement with Leiden community detection.

# Overview

SpaceCell assigns cells (or spots) in spatial transcriptomics data to distinct spatial domains through a three-stage pipeline:

Gene expression imputation (optional). An H&E histology image and the raw spatial expression are fused by a cross-attention module — UNI morphological features and GAT neighbourhood-expression features combined via bidirectional cross-attention, to produce a denser, higher-fidelity expression matrix. When no histology image is available, the pipeline starts from the raw expression at stage 2.
Initial clustering. Expression is reduced with GLM-PCA and clustered with a Gaussian mixture model. Each cluster is spatially sub-clustered to find its contiguous domains, whose centres become reference points for the next stage.
Refinement. A seed cell per cluster is chosen to minimise a combined negative-binomial × spatial loss to the domain centres. Cells are added iteratively to maximise cluster separability, leaving ambiguous cells unassigned. A weighted graph, edge weights inverse to the negative-binomial × spatial distance, with assigned cells anchored — is partitioned by Leiden to produce the final clustering.
Installation

# Usage

from space_cell import load_data, cluster_with_gmm, do_spatial_clustering_on_top_of_gmm, run_leiden

data = load_data()                                    # load counts, coords, GLM-PCA, labels
gmm_clusters = cluster_with_gmm(data)                 # initial GMM clustering
result = do_spatial_clustering_on_top_of_gmm(         # spatial sub-clustering + domain centres
    data, gmm_clusters, expected_n_clusters=7)
weights_for_leiden(data)                              # build + save the weighted graph
pred, ari, f1 = run_leiden(data)                      # final Leiden clustering + evaluation