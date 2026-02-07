import numpy as np
import matplotlib.pyplot as plt
from glmpca import glmpca

def main():
    # load the data
    coords_mat = np.genfromtxt("./data/spatial_metadata.csv", delimiter=',', skip_header=1, usecols=(34, 35))
    counts_mat = np.genfromtxt("./data/spatial_data.csv", delimiter=',', skip_header=1)
    # transpose
    counts_mat = counts_mat.T

    # OPTIONAL: filter out cells with UMI below spot_umi_threshold
    spot_umi_threshold=0

    spots_to_keep=np.sum(counts_mat, axis=1)>=spot_umi_threshold
    print(f'number of removed spots: {counts_mat.shape[0]-spots_to_keep.sum()}')
    counts_mat=counts_mat[spots_to_keep,:]
    coords_mat=coords_mat[spots_to_keep,:]
    # GLM-PCA parameters
    num_dims=8 # 2 * number of clusters
    penalty=10 # may need to increase if this is too small

    # CHANGE THESE PARAMETERS TO REDUCE RUNTIME
    num_iters=30
    eps=1e-4
    num_genes=30000

    counts_mat_glmpca=counts_mat[:,np.argsort(np.sum(counts_mat, axis=0))[-num_genes:]]
    glmpca_res=glmpca.glmpca(counts_mat_glmpca.T, num_dims, fam="poi", penalty=penalty, verbose=True,
                            ctl = {"maxIter":num_iters, "eps":eps, "optimizeTheta":True})
    A = glmpca_res['factors'] # should be of size N x num_dims, where each column is a PC

    # save to csv the results
    header = ','.join([f'PC{i+1}' for i in range(A.shape[1])])
    np.savetxt("./data/glm_pca_data.csv", A, delimiter=',', header=header, comments='')
    # visualize top GLM-PCs
    R=2
    C=4
    fig,axs=plt.subplots(R,C,figsize=(20,10))
    for r in range(R):
        for c in range(C):
            i=r*C+c
            axs[r,c].scatter(coords_mat[:,0], coords_mat[:,1], c=A[:,i],cmap='Reds',s=3)
            axs[r,c].set_title(f'GLM-PC{i}')
    return

if __name__ == '__main__':
    main()