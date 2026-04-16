import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def cluster_and_plot(
    glm_file,
    spatial_file,
    output_plot="kmeans_spatial.png",
    output_csv="kmeans_clusters.csv",
    n_clusters=7,
    random_state=0
):
    glm_df = pd.read_csv(glm_file)
    glm_df.set_index("cell_barcode", inplace=True)

    glm_df = glm_df.select_dtypes(include=[np.number])

    spatial_df = pd.read_csv(spatial_file, usecols=[0, 4, 5])
    spatial_df = spatial_df.rename(columns={
        spatial_df.columns[0]: "cell_id",
        spatial_df.columns[1]: "x",
        spatial_df.columns[2]: "y"
    })
    spatial_df.set_index("cell_id", inplace=True)

    common_cells = spatial_df.index.intersection(glm_df.index)

    if len(common_cells) == 0:
        raise ValueError("No overlapping barcodes found.")

    spatial_df = spatial_df.loc[common_cells]
    glm_df = glm_df.loc[common_cells]

    coords_mat = spatial_df.to_numpy()
    A = glm_df.to_numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(A)

    cluster_df = pd.DataFrame({
        "cell_barcode": common_cells,
        "cluster": kmeans_labels,
        "x": coords_mat[:, 0],
        "y": coords_mat[:, 1]
    })
    cluster_df.to_csv(output_csv, index=False)

    plt.figure(figsize=(6, 6))

    for t in np.unique(kmeans_labels):
        plt.scatter(
            coords_mat[kmeans_labels == t, 0],
            coords_mat[kmeans_labels == t, 1],
            s=3,
            label=f"C{t}"
        )

    plt.axis('off')
    plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()

    return cluster_df

if __name__ == "__main__":
    cluster_and_plot("glm_scrna_p10_final.csv", "spatial_coor.txt")