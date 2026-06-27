import anndata as ad
import squidpy as sq
import cellcharter as cc
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything

seed_everything(12345)
scvi.settings.seed = 12345
COUNT_FILE = "./data/visium_count.csv"
COOR_FILE = "./data/visium_coor.csv"

def main():
    # read counts
    adata_spatial = sc.read_csv(COUNT_FILE)
    adata_spatial = adata_spatial.T
    # read coordinates and load them to adata
    fov_df = pd.read_csv(COOR_FILE, usecols=[0, 4, 5])
    fov_df.columns = ["cell_id", "x", "y"]

    fov_df = fov_df.set_index("cell_id")
    fov_df = fov_df.loc[adata_spatial.obs_names]
    adata_spatial.obsm["spatial_fov"] = fov_df[["x", "y"]].to_numpy()
    adata_spatial.obs["sample"] = 'FOV1'
    adata_spatial.uns['spatial_fov'] = {s: {} for s in adata_spatial.obs['sample'].unique()}
    adata_spatial.obs['sample'] = pd.Categorical(adata_spatial.obs['sample'])
    #print(adata_spatial.obsm["spatial_fov"])
    # filter genes + cells
    sc.pp.filter_genes(adata_spatial, min_counts=3)
    sc.pp.filter_cells(adata_spatial, min_counts=3)

    adata_spatial.layers["counts"] = adata_spatial.X.copy()
    # normalize + log
    sc.pp.normalize_total(adata_spatial, target_sum=1e6)
    sc.pp.log1p(adata_spatial)

    # pca
    scvi.model.SCVI.setup_anndata(
        adata_spatial,
        layer="counts",
    )

    model = scvi.model.SCVI(adata_spatial)
    model.train(early_stopping=True, enable_progress_bar=True)
    adata_spatial.obsm['X_scVI'] = model.get_latent_representation(adata_spatial).astype(np.float32)
    # cell charter clustering ++
    sq.gr.spatial_neighbors(adata_spatial, library_key='sample', coord_type='generic', delaunay=True, spatial_key='spatial_fov', percentile=99)
    cc.gr.aggregate_neighbors(adata_spatial, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter', sample_key='sample')

    autok = cc.tl.ClusterAutoK(
        n_clusters=(2,10),
        max_runs=10,
        convergence_tol=0.001
    )
    autok.fit(adata_spatial, use_rep='X_cellcharter')
    cc.pl.autok_stability(autok)
    # save stability plot here
    plt.savefig("cellcharter_stab_visi.png", dpi=300, bbox_inches="tight")
    plt.close()
    adata_spatial.obs['cluster_cellcharter'] = autok.predict(adata_spatial, use_rep='X_cellcharter')
    sq.pl.spatial_scatter(
        adata_spatial,
        color=['cluster_cellcharter'],
        library_key='sample',
        size=60,
        img=None,
        spatial_key='spatial_fov',
        palette='Set2',
        figsize=(5,5),
        ncols=1,
        library_id=['FOV1']
    )
    # save clustered
    plt.savefig("cellcharter_clus_visi.png", dpi=300, bbox_inches="tight")
    plt.close()
    # save to csv
    df = pd.DataFrame({
        "cell_id": adata_spatial.obs_names,
        "x": adata_spatial.obsm["spatial_fov"][:, 0],
        "y": adata_spatial.obsm["spatial_fov"][:, 1],
        "cluster": adata_spatial.obs["cluster_cellcharter"].values
    })

    df.to_csv("cell_clusters_visi.csv", index=False)
    return

if __name__ == '__main__':
    main()