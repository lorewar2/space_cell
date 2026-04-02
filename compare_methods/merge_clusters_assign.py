import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def main():
    merge_clusters_glm_pca()
    #merge_clusters_raw_counts()
    return

def merge_clusters_raw_counts():
    k = 5
    # data load
    counts_df = pd.read_csv("./data/cerebellum_count.csv", index_col=0)  
    counts_df = counts_df.T
    spatial_df = pd.read_csv("./data/cerebellum_coor.csv", usecols=[0, 1, 2])
    spatial_df = spatial_df.rename(columns={spatial_df.columns[0]: "cell_id"}).set_index("cell_id")
    leiden_df = pd.read_csv("./leiden_cere.csv", usecols=[0, 1])
    leiden_df.columns = ["cell_id", "community"]
    leiden_df = leiden_df.set_index("cell_id")

    # common cells only
    common_cells = counts_df.index.intersection(spatial_df.index)
    counts_df = counts_df.loc[common_cells]
    spatial_df = spatial_df.loc[common_cells]

    merged_df = counts_df.join(leiden_df.reindex(counts_df.index), how="left")
    merged_df["community"] = merged_df["community"].fillna(-1)

    # separate valid and missing
    valid_df = merged_df[merged_df["community"] != -1].copy()
    missing_df = merged_df[merged_df["community"] == -1].copy()

    community_means = valid_df.groupby("community").mean()
    cluster_map = {c: c for c in community_means.index}

    while len(community_means) > k:
        comm_list = community_means.index.tolist()
        min_dist = np.inf
        best_pair = None

        # compute L1 distance only pair by pair
        for i, comm_i in enumerate(comm_list):
            mean_i = community_means.loc[comm_i].values
            for comm_j in comm_list[i+1:]:
                mean_j = community_means.loc[comm_j].values
                dist = np.sum(np.abs(mean_i - mean_j))  # L1 distance
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (comm_i, comm_j)

        # merge j into i
        comm_i, comm_j = best_pair
        valid_df.loc[valid_df["community"] == comm_j, "community"] = comm_i

        # update cluster map
        for key in cluster_map:
            if cluster_map[key] == comm_j:
                cluster_map[key] = comm_i

        # recompute community means
        community_means = valid_df.groupby("community").mean()

    final_df = pd.concat([valid_df, missing_df])
    final_output = final_df[["community"]].join(spatial_df)
    final_output = final_output.reset_index()
    final_output.columns = ["cell_id", "community_id", "x", "y"]
    final_output.to_csv("./data/merged_leiden.csv", index=False)
    return final_output

def merge_clusters_glm_pca():
    k = 5
    glm_df = pd.read_csv("./data/cerebellum_glm.csv").set_index("cell_id")

    spatial_df = pd.read_csv("./data/cerebellum_coor.csv", usecols=[0, 1, 2])
    spatial_df = spatial_df.rename(columns={spatial_df.columns[0]: "cell_id"})
    spatial_df = spatial_df.set_index("cell_id")

    leiden_df = pd.read_csv("./leiden_cere.csv", usecols=[0, 1])
    leiden_df.columns = ["cell_id", "community"]
    leiden_df = leiden_df.set_index("cell_id")

    common_cells = glm_df.index.intersection(spatial_df.index)
    glm_df = glm_df.loc[common_cells]
    spatial_df = spatial_df.loc[common_cells]

    merged_df = glm_df.join(leiden_df.reindex(glm_df.index), how="left")

    merged_df["community"] = merged_df["community"].fillna(-1)

    valid_df = merged_df[merged_df["community"] != -1].copy()
    missing_df = merged_df[merged_df["community"] == -1].copy()

    community_means = valid_df.groupby("community").mean()

    cluster_map = {c: c for c in community_means.index}

    while len(community_means) > k:
        dist_matrix = cdist(community_means.values, community_means.values)
        np.fill_diagonal(dist_matrix, np.inf)

        # find closest pair
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        comm_i = community_means.index[i]
        comm_j = community_means.index[j]

        # merge j into i
        valid_df.loc[valid_df["community"] == comm_j, "community"] = comm_i

        # update cluster map
        for key in cluster_map:
            if cluster_map[key] == comm_j:
                cluster_map[key] = comm_i

        # recompute means
        community_means = valid_df.groupby("community").mean()

    final_df = pd.concat([valid_df, missing_df])

    final_output = final_df[["community"]].join(spatial_df)
    final_output = final_output.reset_index()
    final_output.columns = ["cell_id", "community_id", "x", "y"]

    final_output.to_csv("./data/merged_leiden.csv", index=False)

    return final_output

if __name__ == "__main__":
    main()