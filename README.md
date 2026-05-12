# space_cell
Spatial Transcriptomics Border Cell Analysis
Overview
This project analyzes spatial transcriptomics data (Visium) to investigate the transcriptomic identity of cells at the boundaries between cortical layers. The central question is: do border cells transcriptomically belong to the layer they are spatially assigned to, or do they show affinity toward the neighboring layer?
Data
The pipeline works with four inputs:

Gene expression counts — raw UMI count matrix (cells × genes)
Spatial coordinates — x/y positions of each cell on the tissue slide
GLM-PCA embeddings — dimensionality-reduced representation of gene expression
Ground truth annotations — manually assigned cortical layer labels per cell

Pipeline
1. Data Loading
All four data sources are aligned by cell barcode into a single data structure, ensuring consistent indexing across counts, coordinates, embeddings, and labels.
2. GLM-PCA Distribution per Cluster
For each cortical layer and each principal component, the distribution of GLM-PCA values is visualized as boxplots, with outliers shown explicitly. This gives an overview of how transcriptomically distinct each layer is in reduced space.
3. Border Cell Detection
Cells are classified as border cells if any of their spatial neighbours (within a fixed radius) belong to a different ground-truth cluster. This purely spatial definition identifies cells sitting at the physical boundary between two layers.
4. Interior Clump Detection
DBSCAN clustering is applied within each layer to identify small, tight spatial clumps of cells (configurable size range) that sit away from the cluster boundary. These clumps serve as a transcriptomically pure reference for each layer — cells that are unambiguously interior to their assigned layer.
5. Border Cell Assignment via Graph + Leiden
For a pair of adjacent layers, the pipeline:

Selects border cells from both layers that are within a user-defined spatial distance threshold of the opposing layer's border
Computes a transcriptomic distance from each border cell to the interior clump average of each layer, using one of two methods:

GLM-PCA — Euclidean distance in PCA space
Poisson NLL — negative Poisson log-likelihood of the cell's raw counts under the clump's mean expression, a statistically principled measure of transcriptomic similarity


Builds a weighted graph (igraph) where border cells are nodes, and edge weights reflect similarity to each layer's clump average (higher weight = more similar)
Runs Leiden community detection on the graph to assign each border cell to a transcriptomic community

6. Visualization
Results are displayed as:

Spatial scatter plot — all cells in the background (ground truth colours), border cells highlighted by either ground truth label or Leiden community assignment
Distance scatter plot — each border cell plotted by its distance to Layer A vs Layer B clump average, revealing which layer it transcriptomically resembles more

Key Parameters
ParameterDescriptionradiusSpatial neighbourhood radius for border cell detectiondistance_thresholdMaximum spatial distance to opposing layer for border cell selectionmethod"glm_pca" or "poisson" — transcriptomic distance metriccolor_by"ground_truth" or "community" — plot colouring modedbscan_epsDBSCAN neighbourhood radius for interior clump detectionmin_clump / max_clumpSize bounds for interior clumps
Motivation
Cortical layer annotations in spatial transcriptomics are often assigned based on spatial position alone. However, gene expression does not always respect these boundaries cleanly. This pipeline quantifies the degree to which border cells are transcriptomically ambiguous — belonging spatially to one layer but expressionally closer to another — which has implications for understanding layer identity, gradients, and annotation accuracy.
