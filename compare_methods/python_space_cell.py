import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

def main():
    data = load_data()
    #fig, axes = glm_pca_map(data)
    # find cells in clusters 
    all_cell_per_cluster = find_cells_in_cluster(data)

    # initialize the cluster centers based on ground truth
    #avg_per_cluster_counts, avg_per_cluster_glm_pca, avg_per_cluster_pca = initialize_cluster_centers(data, all_cell_per_cluster)

    # for two close by cells plot the loss for each cell from each cluster center
    #plot_using_different_loss_functions(data, all_cell_per_cluster, avg_per_cluster_counts, avg_per_cluster_glm_pca, avg_per_cluster_pca)
    # do kmeans++ and gaussian mixture model on glm pca data and plot the clustering
    #kmeans_plot(data, all_cell_per_cluster)

    cluster_with_gmm_refine_negbi(data, all_cell_per_cluster)
    return

def cluster_with_gmm_refine_negbi(data, all_cell_per_cluster):
    n_clusters = 10
    random_state = 10

    keys     = list(all_cell_per_cluster.keys())
    spatial  = data["spatial"]
    labels   = data["labels"]
    counts   = data["counts"]
    glm_pca  = data["glm_pca"]
    pca      = data["pca"]
    
    return

def kmeans_plot(data, all_cell_per_cluster):
    n_clusters = 7
    random_state = 10

    keys     = list(all_cell_per_cluster.keys())
    spatial  = data["spatial"]
    labels   = data["labels"]
    counts   = data["counts"]
    glm_pca  = data["glm_pca"]
    pca      = data["pca"]

    shared_bcs = glm_pca.index.intersection(labels.index)
    X          = glm_pca.loc[shared_bcs].values.astype(float)
    y_true     = labels.loc[shared_bcs].values

    # kmeans ++
    # km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    # y_pred = km.fit_predict(X)
    
    # gmm
    gmm = GaussianMixture(
        n_components    = n_clusters,
        covariance_type = "full",
        init_params     = "k-means++",
        n_init          = 5,
        random_state    = random_state,
        max_iter        = 300,
    )
    gmm.fit(X)
    y_pred = gmm.predict(X)
    

    # rand index
    ari = adjusted_rand_score(y_true, y_pred)
    f1  = aligned_f1(y_true, y_pred)
    print(f"kmeans on GLM-PCA  |  k={n_clusters}  |  ARI = {ari:.4f}  |  F1 = {f1:.4f}")
    pred_series = pd.Series(y_pred, index=shared_bcs, name="kmeans_cluster")

    # plot
    gt_unique  = sorted(labels.unique())
    gt_cmap    = plt.cm.get_cmap("tab10", len(gt_unique))
    gt_colors  = {v: gt_cmap(i) for i, v in enumerate(gt_unique)}
    gt_handles = [mpatches.Patch(color=gt_colors[v], label=v) for v in gt_unique]

    pred_unique = sorted(np.unique(y_pred))
    km_cmap     = plt.cm.get_cmap("tab10", len(pred_unique))
    km_colors   = {v: km_cmap(i) for i, v in enumerate(pred_unique)}
    km_handles  = [mpatches.Patch(color=km_colors[v], label=f"Cluster {v}")
                   for v in pred_unique]
    coords = spatial.loc[shared_bcs, ["x", "y"]].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # left: ground truth
    axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=[gt_colors[v] for v in y_true],
        s=20, alpha=0.95, linewidths=0.3, edgecolors="white"
    )
    axes[0].legend(handles=gt_handles, fontsize=7, loc="upper right")
    axes[0].set_title("Ground truth\n(spatial view)", fontsize=11)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # right: k-means++
    axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=[km_colors[v] for v in y_pred],
        s=20, alpha=0.95, linewidths=0.3, edgecolors="white"
    )
    axes[1].legend(handles=km_handles, fontsize=7, loc="upper right")
    axes[1].set_title(f"gaussian on GLM-PCA  (k={n_clusters})\nARI = {ari:.3f} F1= {f1:.3f}",
                      fontsize=11)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    plt.suptitle("GLM-PCA gaussian vs ground truth", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"gaussian_glmpca_k{n_clusters}.png", dpi=150, bbox_inches="tight")
    plt.show()

    return

def aligned_f1(y_true, y_pred):
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    # build cost matrix: -overlap so Hungarian minimises = max overlap
    cost = np.zeros((len(pred_labels), len(true_labels)))
    for i, p in enumerate(pred_labels):
        for j, t in enumerate(true_labels):
            cost[i, j] = -np.sum((y_pred == p) & (y_true == t))

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred_labels[r]: true_labels[c] for r, c in zip(row_ind, col_ind)}

    y_pred_mapped = np.array([mapping.get(p, -1) for p in y_pred])
    return f1_score(y_true, y_pred_mapped, average="macro")

def plot_using_different_loss_functions(data, all_cell_per_cluster, avg_per_cluster_counts, avg_per_cluster_glm_pca, avg_per_cluster_pca):
    keys = list(all_cell_per_cluster.keys())
    spatial  = data["spatial"]
    labels   = data["labels"]
    counts = data["counts"]
    glm_pca = data["glm_pca"]
    pca = data["pca"]

    # calculate and plot for visium
    for i in range(0, len(keys) - 1):
        first_cluster = keys[i]
        first_cluster_avg_glm = avg_per_cluster_glm_pca[i]
        first_cluster_avg_counts = avg_per_cluster_counts[i]
        first_cluster_avg_pca = avg_per_cluster_pca[i]

        second_cluster = keys[i + 1]
        second_cluster_avg_glm = avg_per_cluster_glm_pca[i + 1]
        second_cluster_avg_counts = avg_per_cluster_counts[i + 1]
        second_cluster_avg_pca = avg_per_cluster_pca[i + 1]

        mixed_barcodes = all_cell_per_cluster[first_cluster] + all_cell_per_cluster[second_cluster]

        # calcuale glm pca loss 
        records = []
        for bc in mixed_barcodes:
            dist_l1, dist_l2 = glm_pca_distance(bc, first_cluster_avg_glm, second_cluster_avg_glm, glm_pca)
            records.append({"barcode": bc, "dist_layer1": dist_l1, "dist_layer2": dist_l2})
        dist_df_glm = pd.DataFrame(records).set_index("barcode")
        # calculate pca loss
        records = []
        for bc in mixed_barcodes:
            dist_l1, dist_l2 = glm_pca_distance(bc, first_cluster_avg_pca, second_cluster_avg_pca, pca)
            records.append({"barcode": bc, "dist_layer1": dist_l1, "dist_layer2": dist_l2})
        dist_df_pca = pd.DataFrame(records).set_index("barcode")
        # calculate poisson loss
        records = []
        for bc in mixed_barcodes:
            dist_l1, dist_l2 = poisson_distance(bc, first_cluster_avg_counts, second_cluster_avg_counts, counts)
            records.append({"barcode": bc, "dist_layer1": dist_l1, "dist_layer2": dist_l2})
        dist_df_poisson = pd.DataFrame(records).set_index("barcode")
        # calculcate negative binomial loss
        records = []
        for bc in mixed_barcodes:
            dist_l1, dist_l2 = negative_binomial_distance(bc, first_cluster_avg_counts, second_cluster_avg_counts, counts)
            records.append({"barcode": bc, "dist_layer1": dist_l1, "dist_layer2": dist_l2})
        dist_df_negbi = pd.DataFrame(records).set_index("barcode")

        # Plot
        unique_vals    = sorted(labels.unique())
        cmap           = plt.cm.get_cmap("tab10", len(unique_vals))
        val_to_color   = {v: cmap(i) for i, v in enumerate(unique_vals)}
        cell_color_fn  = lambda bc: val_to_color[labels[bc]]
        legend_handles = [mpatches.Patch(color=val_to_color[v], label=v)
                          for v in unique_vals]
        color_title    = "Ground Truth"
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        all_coords = spatial[["x", "y"]].values
        all_bcs    = spatial.index.values
        gt_cmap    = plt.cm.get_cmap("Set2", len(labels.unique()))
        gt_colors  = {cl: gt_cmap(i) for i, cl in enumerate(sorted(labels.unique()))}
        axes[0].scatter(all_coords[:, 0], all_coords[:, 1],
                        c=[gt_colors[labels[bc]] for bc in all_bcs],
                        s=8, alpha=0.12, linewidths=0)

        bc_colors = [cell_color_fn(bc) for bc in mixed_barcodes]
        bc_coords = spatial.loc[mixed_barcodes, ["x", "y"]].values
        axes[0].scatter(bc_coords[:, 0], bc_coords[:, 1],
                        c=bc_colors, s=30, alpha=0.9,
                        linewidths=0.4, edgecolors="white")

        axes[0].legend(handles=legend_handles, fontsize=7, loc="upper right")

        axes[0].set_title(f"Border cells — {color_title}\n(spatial view)", fontsize=11)
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
        axes[0].set_aspect("equal")

        d1 = dist_df_poisson.loc[mixed_barcodes, "dist_layer1"].values
        d2 = dist_df_poisson.loc[mixed_barcodes, "dist_layer2"].values
        axes[1].scatter(d1, d2, c=bc_colors, s=15, alpha=0.55,
                        linewidths=0.3, edgecolors="white")
        axes[1].axline((0, 0), slope=1, color="grey", linewidth=0.8,
                    linestyle="--", label="equal distance")
        axes[1].set_xlabel("loss to cc1 avg", fontsize=9)
        axes[1].set_ylabel("loss to cc2 avg", fontsize=9)
        axes[1].set_title(f"Loss to cluster center avg — {color_title}\n(method)", fontsize=11)
        #axes[1].set_xlim(0, 100)
        #axes[1].set_ylim(0, 100)

        axes[1].legend(handles=legend_handles + [
            mpatches.Patch(color="grey", label="Equal distance")], fontsize=7)
        plt.tight_layout()
        plt.show()
        plt.savefig("poisson_cere" + str(i))

    return


def glm_pca_distance(bc, avg1, avg2, glm_pca):
    vec     = glm_pca.loc[bc].values.astype(float)
    dist_l1 = float(np.linalg.norm(vec - avg1))
    dist_l2 = float(np.linalg.norm(vec - avg2))
    return dist_l1, dist_l2

def poisson_distance(bc, avg1, avg2, counts):
    """
    Negative Poisson log-likelihood:
    NLL = -sum( y * log(mu) - mu - log(y!) )
    """
    eps = 1e-8
    y   = counts.loc[bc].values.astype(float)

    def nll(mu, y):
        mu = np.maximum(mu, eps)
        return float(np.sum(mu - y * np.log(mu) + gammaln(y + 1)))

    return nll(avg1, y), nll(avg2, y)

def negative_binomial_distance(bc, avg1, avg2, counts, theta=10.0):
    """
    Negative Binomial negative log-likelihood (NB2 parameterization)

    Var(Y) = mu + mu^2 / theta

    Parameters
    ----------
    bc : barcode/index
    avg1, avg2 : mean expression vectors
    counts : dataframe of counts
    theta : dispersion parameter
        Larger theta -> approaches Poisson

    Returns
    -------
    nll(avg1), nll(avg2)
    """
    eps = 1e-8
    y   = counts.loc[bc].values.astype(float)

    def nll(mu, y, theta):
        mu = np.maximum(mu, eps)

        # NB log-likelihood
        ll = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (theta + mu))
            + y * np.log(mu / (theta + mu))
        )

        return float(-np.sum(ll))

    return nll(avg1, y, theta), nll(avg2, y, theta)

def initialize_cluster_centers(data, all_cell_per_cluster):
    keys = all_cell_per_cluster.keys()
    spatial  = data["spatial"]
    labels   = data["labels"]
    counts = data["counts"]
    glm_pca = data["glm_pca"]
    pca = data["pca"]
    clusters = sorted(labels.unique())

    coords   = spatial[["x", "y"]].values
    barcodes = spatial.index.values

    # average calculation counts for poisson and negative bi
    avg_per_cluster_counts = list()
    for key in keys:
        cells_with_key = all_cell_per_cluster[key]
        avg_per_cluster_counts.append(counts.loc[cells_with_key].mean(axis=0).values.astype(float))
    print(avg_per_cluster_counts)
    # average calculation using glm pca
    avg_per_cluster_glm_pca = list()
    for key in keys:
        cells_with_key = all_cell_per_cluster[key]
        avg_per_cluster_glm_pca.append(glm_pca.loc[cells_with_key].mean(axis=0).values.astype(float))
    print(avg_per_cluster_glm_pca)
    # average calculation using pca
    avg_per_cluster_pca = list()
    for key in keys:
        cells_with_key = all_cell_per_cluster[key]
        avg_per_cluster_pca.append(pca.loc[cells_with_key].mean(axis=0).values.astype(float))
    print(avg_per_cluster_pca)
    return avg_per_cluster_counts, avg_per_cluster_glm_pca, avg_per_cluster_pca

def find_cells_in_cluster(data):
    # load data
    spatial  = data["spatial"]
    labels   = data["labels"]
    clusters = sorted(labels.unique())

    coords   = spatial[["x", "y"]].values
    barcodes = spatial.index.values

    # separate the cells to clusters
    cluster_barcodes = {cl: [] for cl in clusters}
    for i, bc in enumerate(barcodes):
        own_cluster        = labels[bc]
        cluster_barcodes[own_cluster].append(bc)
    return cluster_barcodes

def border_finder(data, radius=150):
    """
    Border cells: cells that have at least one neighbour within `radius`
    belonging to a different ground-truth cluster.

    Parameters
    ----------
    data   : dict   output of load_data()
    radius : float  spatial distance threshold

    Returns
    -------
    border_barcodes : dict  {cluster_name: [barcode, ...]}
    """
    spatial  = data["spatial"]
    labels   = data["labels"]
    clusters = sorted(labels.unique())

    coords   = spatial[["x", "y"]].values
    barcodes = spatial.index.values
    tree     = cKDTree(coords)

    cmap      = plt.cm.get_cmap("tab10", len(clusters))
    color_map = {cl: cmap(i) for i, cl in enumerate(clusters)}

    border_barcodes = {cl: [] for cl in clusters}

    for i, bc in enumerate(barcodes):
        neighbour_idxs = tree.query_ball_point(coords[i], r=radius)
        neighbour_idxs = [j for j in neighbour_idxs if j != i]

        own_cluster        = labels[bc]
        neighbour_clusters = {labels[barcodes[j]] for j in neighbour_idxs}

        # border = at least one neighbour is from a different cluster
        if neighbour_clusters - {own_cluster}:
            border_barcodes[own_cluster].append(bc)

    # ── plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    # all cells faint in background
    ax.scatter(coords[:, 0], coords[:, 1],
               c=[color_map[labels[bc]] for bc in barcodes],
               s=8, alpha=0.12, linewidths=0)

    # border cells solid
    for cl, bcs in border_barcodes.items():
        if not bcs:
            continue
        bc_coords = spatial.loc[bcs, ["x", "y"]].values
        ax.scatter(bc_coords[:, 0], bc_coords[:, 1],
                   color=color_map[cl], s=20, alpha=0.95,
                   linewidths=0, label=f"{cl}  (n={len(bcs)})")

    ax.legend(fontsize=7, markerscale=1.5, loc="upper right")
    ax.set_title(f"Border cells per cluster  (radius={radius})", fontsize=12)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("border_cells.png", dpi=150, bbox_inches="tight")
    #plt.show()

    return border_barcodes

def poisson_loss(x, lam, eps=1e-8):
    lam = np.clip(lam, eps, None)
    return np.sum(lam - x * np.log(lam))

def border_cell_assignment(
    data,
    selected_cells,
    border_barcodes,
    beta=1.0,
    n_neighbors=6,
    n_iters=5,
    plot=True
):
    spatial  = data["spatial"]
    counts   = data["counts"]

    # --- flatten border_barcodes ---
    if isinstance(border_barcodes, dict):
        border_barcodes = [bc for bcs in border_barcodes.values() for bc in bcs]

    clusters = list(selected_cells.keys())

    # --- 1. normalize counts (log1p CPM style) ---
    libsize = counts.sum(axis=1).values.reshape(-1, 1)
    norm_counts = counts.values / (libsize + 1e-8) * 1e4
    log_counts = np.log1p(norm_counts)

    log_counts = np.array(log_counts)
    barcodes_all = counts.index.values
    bc_to_idx = {bc: i for i, bc in enumerate(barcodes_all)}

    # --- 2. fit Gaussian per cluster ---
    cluster_means = {}
    cluster_covs = {}

    for clust in clusters:
        idx = [bc_to_idx[bc] for bc in selected_cells[clust]]
        X = log_counts[idx]

        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 0.01

        cluster_means[clust] = mu
        cluster_covs[clust] = cov

    inv_covs = {cl: np.linalg.inv(cluster_covs[cl]) for cl in clusters}

    # --- 3. build spatial neighbor graph ---
    coords = spatial[["x", "y"]].values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # --- 4. initialize labels (GMM only) ---
    labels = {}

    for bc in border_barcodes:
        i = bc_to_idx[bc]
        x = log_counts[i]

        losses = {}
        for clust in clusters:
            mu = cluster_means[clust]
            inv_cov = inv_covs[clust]

            diff = x - mu
            mahal = diff.T @ inv_cov @ diff
            losses[clust] = mahal

        labels[bc] = min(losses, key=losses.get)

    # --- 5. HMRF iterations (ICM) ---
    for _ in range(n_iters):
        for bc in border_barcodes:
            i = bc_to_idx[bc]
            x = log_counts[i]

            neighbor_idxs = indices[i]
            neighbor_bcs = [barcodes_all[j] for j in neighbor_idxs if barcodes_all[j] in labels]

            losses = {}

            for clust in clusters:
                # --- expression term ---
                mu = cluster_means[clust]
                inv_cov = inv_covs[clust]

                diff = x - mu
                expr_loss = diff.T @ inv_cov @ diff

                # --- spatial term ---
                disagree = sum(labels.get(nb) != clust for nb in neighbor_bcs)
                spatial_loss = beta * disagree

                losses[clust] = expr_loss + spatial_loss

            labels[bc] = min(losses, key=losses.get)

    # --- 6. plot ---
    if plot:
        plt.figure(figsize=(6, 6))

        # background
        all_coords = spatial[["x", "y"]].values
        plt.scatter(all_coords[:, 0], all_coords[:, 1], s=5, alpha=0.1)

        # core cells
        for clust in clusters:
            sel_coords = spatial.loc[selected_cells[clust]][["x", "y"]].values
            plt.scatter(sel_coords[:, 0], sel_coords[:, 1], s=40, label=f"Core {clust}")

        # border assignments
        for clust in clusters:
            assigned = [bc for bc, c in labels.items() if c == clust]
            if not assigned:
                continue

            bc_coords = spatial.loc[assigned][["x", "y"]].values
            plt.scatter(
                bc_coords[:, 0],
                bc_coords[:, 1],
                marker="x",
                s=60,
                label=f"Border→{clust}"
            )

        plt.legend()
        plt.title("Border Assignment (HMRF)")
        plt.tight_layout()
        plt.show()

    return labels

def load_data():
    data_type = "slideseq"
    # visium paths 
    visium_count = "./data/visium_count.csv"
    visium_coor = "./data/visium_coor.csv"
    visium_glm = "./data/visium_glm.csv"
    visium_manual = "./data/visium_manual.csv"
    # slideseq paths
    slideseq_count = "./data/cerebellum_2_count.csv"
    slideseq_coor = "./data/cerebellum_2_coor.csv"
    slideseq_glm = "./data/cerebellum_2_glm.csv"
    slideseq_manual = "./data/cerebellum_2_manual.csv"
    # selected paths
    if data_type == "slideseq":
        count_path = slideseq_count
        coor_path = slideseq_coor
        glm_path = slideseq_glm
        manual_path = slideseq_manual
    else:
        count_path = visium_count
        coor_path = visium_coor
        glm_path = visium_glm
        manual_path = visium_manual

    # Load counts matrix (genes x cells)
    counts_df = pd.read_csv(count_path, index_col=0)
    print(counts_df)
    print("Count loading done")
    # counts_df.index = gene names, counts_df.columns = cell barcodes

    # Load spatial coordinates (barcodes, ..., x, y)
    coor_df = pd.read_csv(coor_path, index_col=0)
    spatial = coor_df.iloc[:, [-2, -1]]          # last two columns = x, y
    spatial.columns = ["x", "y"]
    spatial.index.name = "barcode"
    print(spatial)
    print("Coordinate loading done")

    # Load GLM-PCA embeddings (first two columns are redundant barcodes)
    glm_df = pd.read_csv(glm_path, index_col=0)
    if data_type == "visium":
        glm_df = glm_df.iloc[:, 1:] # drop the redundant barcode column for visium
    glm_df.index.name = "barcode"
    print(glm_df)
    print("GLM PCA loading done")

    # Load ground-truth annotations (col 1 = index, col 1 = barcode, col 2 = label)
    if data_type == "visium":
        gt_df = pd.read_csv(manual_path, index_col=0)  # index_col=0 for visium
        ground_truth = gt_df.iloc[:, [0, 1]]
    else:
        gt_df = pd.read_csv(manual_path)  # index_col=0 for visium
        ground_truth = gt_df.iloc[:, [0, 2]]
    ground_truth.columns = ["barcode", "cell_type"]
    ground_truth = ground_truth.set_index("barcode")["cell_type"]

    # Keep only selected cell types
    if data_type == "slideseq":
        keep_types = [
            "Astrocytes",
            "Granule",
            "MLI2",
            "MLI1",
            "Bergmann",
            "Purkinje",
            "Oligo",
        ]
        ground_truth = ground_truth[ground_truth.isin(keep_types)]
    print(ground_truth)
    print("Manual Annotation loading done")


    # Align everything to the shared set of barcodes
    barcodes = counts_df.columns.intersection(spatial.index) \
                                .intersection(glm_df.index) \
                                .intersection(ground_truth.index)
    print(barcodes)
    spatial     = spatial.loc[barcodes]                 # cells  × 2
    glm_pca     = glm_df.loc[barcodes]                  # cells  × n_components
    labels      = ground_truth.loc[barcodes]            # cells  (Series)
    counts = counts_df[barcodes] # cells  × genes
    counts = counts.T 

    # Get PCA embeddings
    n_pca_components = 50
    pca_input = counts.values
    print(pca_input)
    # Scale 
    pca_input = StandardScaler().fit_transform(pca_input)

    print(f"Computing PCA with {n_pca_components} components")
    pca_model = PCA(n_components=n_pca_components)

    pca_embeddings = pca_model.fit_transform(pca_input)

    pca = pd.DataFrame(
        pca_embeddings,
        index=counts.index,
        columns=[f"PC{i+1}" for i in range(n_pca_components)]
    )

    print(pca)
    print("PCA done")
    
    return {
        "barcodes": barcodes,
        "counts": counts,           # DataFrame: cells × genes
        "spatial": spatial,         # DataFrame: cells × {x, y}
        "glm_pca": glm_pca,         # DataFrame: cells × GLM-PCA components
        "pca": pca,              # DataFrame: cells × PCA components
        "labels": labels,           # Series: cells → cell_type string
    }

if __name__ == "__main__":
    main()