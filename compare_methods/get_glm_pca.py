import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glmpca import glmpca
import sys
from sklearn.cluster import KMeans

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path> <penalty>")
        sys.exit(1)
    output_file_path = sys.argv[1]  # Get file path from command-line argument
    penalty = int(sys.argv[2].strip())
    glm_pca(output_file_path, penalty)

def glm_pca(output_file_path, penalty):
    # load the data
    counts_df = pd.read_csv("./data/spatial_data.csv", index_col=0)  # genes as rows, cell_ids as columns
    counts_df = counts_df.T  # now rows are cells, columns are genes
    counts_mat = counts_df.to_numpy()   # shape: (N_cells, N_genes)

    # OPTIONAL: filter out cells with UMI below spot_umi_threshold
    spot_umi_threshold = 1_000

    spots_to_keep=np.sum(counts_mat, axis=1)>=spot_umi_threshold
    print(f'number of removed spots: {counts_mat.shape[0]-spots_to_keep.sum()}')
    counts_mat=counts_mat[spots_to_keep,:]
    cell_ids = counts_df.index.to_numpy()  # keep cell IDs

    # GLM-PCA parameters
    num_dims = 8 # 2 * number of clusters
    penalty = penalty # may need to increase if this is too small

    # CHANGE THESE PARAMETERS TO REDUCE RUNTIME
    num_iters = 30
    eps = 1e-8
    num_genes = 18_000

    counts_mat_glmpca = counts_mat[:,np.argsort(np.sum(counts_mat, axis=0))[-num_genes:]]
    print(counts_mat_glmpca.shape)
    glmpca_res = glmpca.glmpca(counts_mat_glmpca.T, num_dims, fam="poi", penalty=penalty, verbose=True,
                            ctl = {"maxIter":num_iters, "eps":eps, "optimizeTheta":True})
    A = glmpca_res['factors'] # should be of size N x num_dims, where each column is a PC

    # save to csv the results
    glm_pca_df = pd.DataFrame(
        A,
        index=cell_ids,
        columns=[f'PC{i+1}' for i in range(A.shape[1])]
    )
    glm_pca_df.insert(0, 'cell_id', glm_pca_df.index)
    glm_pca_df.to_csv("./data/" + output_file_path, index=False)
    return

def show_plot():
    # load data
    glm_df = pd.read_csv("./data/cerebellum_glm.csv")
    glm_df.set_index("cell_id", inplace=True)

    A = glm_df.to_numpy()  # (N_cells, num_PCs)

    spatial_df = pd.read_csv("./data/cerebellum_coor.csv", usecols=[0, 1, 2]) # 0, 35, 36
    print(spatial_df)
    spatial_df = spatial_df.rename(columns={spatial_df.columns[0]: "cell_id"})
    spatial_df.set_index("cell_id", inplace=True)

    common_cells = spatial_df.index.intersection(glm_df.index)

    spatial_df = spatial_df.loc[common_cells]
    glm_df = glm_df.loc[common_cells]

    coords_mat = spatial_df.to_numpy()
    A = glm_df.to_numpy()
    # cluster using kmeans to test

    n_clusters=4 # CHANGE to number of experts
    kmeans=KMeans(n_clusters=4)
    kmeans.fit(A)
    kmeans_labels=kmeans.labels_
    fig,ax=plt.subplots(figsize=(5,5))
    for t in np.unique(kmeans_labels):
        plt.scatter(coords_mat[kmeans_labels==t,0],coords_mat[kmeans_labels==t,1],s=2)
    plt.axis('off')
    plt.show()

    # plot
    R = 2
    C = 4
    fig, axs = plt.subplots(R, C, figsize=(20, 10))

    for r in range(R):
        for c in range(C):
            i = r * C + c
            axs[r, c].scatter(
                coords_mat[:, 0],
                coords_mat[:, 1],
                c=A[:, i],
                cmap="Reds",
                s=3
            )
            axs[r, c].set_title(f"GLM-PC{i}")
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])

    plt.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    show_plot()
    #main()