import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glmpca import glmpca

def main():
    # Load and match the cell_ids 
    spatial_df = pd.read_csv("./data/spatial_metadata.csv")
    spatial_df = spatial_df.rename(columns={spatial_df.columns[0]: 'cell_id'})
    spatial_df.set_index('cell_id', inplace=True)

    counts_df = pd.read_csv("./data/spatial_data.csv", index_col=0)  # genes as rows, cell_ids as columns
    counts_df = counts_df.T  # now rows are cells, columns are genes

    common_cells = spatial_df.index.intersection(counts_df.index)
    print(f"Number of common cells: {len(common_cells)}")

    spatial_df = spatial_df.loc[common_cells]
    counts_df = counts_df.loc[common_cells]

    coords_mat = spatial_df.to_numpy()  # shape: (N_cells, 2)
    counts_mat = counts_df.to_numpy()   # shape: (N_cells, N_genes)

    # OPTIONAL: filter out cells with UMI below spot_umi_threshold
    spot_umi_threshold=1000

    spots_to_keep=np.sum(counts_mat, axis=1)>=spot_umi_threshold
    print(f'number of removed spots: {counts_mat.shape[0]-spots_to_keep.sum()}')
    counts_mat=counts_mat[spots_to_keep,:]
    coords_mat=coords_mat[spots_to_keep,:]

    coords_mat = spatial_df.to_numpy()      # (N_cells, 2)
    counts_mat = counts_df.to_numpy()       # (N_cells, N_genes)
    cell_ids = spatial_df.index.to_numpy()  # keep cell IDs

    # GLM-PCA parameters
    num_dims=8 # 2 * number of clusters
    penalty=10_000 # may need to increase if this is too small

    # CHANGE THESE PARAMETERS TO REDUCE RUNTIME
    num_iters=30
    eps=1e-4
    num_genes=18_000

    counts_mat_glmpca=counts_mat[:,np.argsort(np.sum(counts_mat, axis=0))[-num_genes:]]
    glmpca_res=glmpca.glmpca(counts_mat_glmpca.T, num_dims, fam="poi", penalty=penalty, verbose=True,
                            ctl = {"maxIter":num_iters, "eps":eps, "optimizeTheta":True})
    A = glmpca_res['factors'] # should be of size N x num_dims, where each column is a PC

    # save to csv the results
    glm_pca_df = pd.DataFrame(
        A,
        index=cell_ids,
        columns=[f'PC{i+1}' for i in range(A.shape[1])]
    )
    glm_pca_df.insert(0, 'cell_id', glm_pca_df.index)
    glm_pca_df.to_csv("./data/glm_pca_data.csv", index=False)
    return

def show_plot():
    #A = np.loadtxt("./data/spatial_metadata.csv", delimiter=',', skiprows=1)  # skip header
    coords_mat = np.genfromtxt("./data/spatial_metadata.csv", delimiter=',', skip_header=1, usecols=(34, 35))
    A = np.loadtxt("./data/glm_pca_data.csv", delimiter=',', skiprows=1)
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
                cmap='Reds',
                s=3
            )
            axs[r, c].set_title(f'GLM-PC{i}')
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
    plt.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    #show_plot()
    main()