import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm
import igraph as ig
import leidenalg
from matplotlib.colors import Normalize
from scipy.special import gammaln 

def main():
    data = load_data()
    #fig, axes = glm_pca_map(data)
    all_cell_per_cluster = find_cells_in_cluster(data)
    print(all_cell_per_cluster.keys())
    print(len(all_cell_per_cluster["Layer 1"]))
    print(len(all_cell_per_cluster["Layer 2"]))
    print(len(all_cell_per_cluster["Layer 3"]))
    print(len(all_cell_per_cluster["Layer 4"]))
    print(len(all_cell_per_cluster["Layer 5"]))
    print(len(all_cell_per_cluster["Layer 6"]))
    print(len(all_cell_per_cluster["WM"]))
    #border_barcodes = border_finder(data)
    #selected_cells = middle_cells_in_cluster(data)
    #border_cell_assignment(data, selected_cells, border_barcodes, plot=True)
    #minimal_test(border_barcodes, selected_cells, data)
    return

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

def minimal_test(border_barcodes, cell_clumps, data, color_by="ground_truth",
                 distance_threshold=500.0, method="poisson"):
    """
    Parameters
    ----------
    color_by            : str   "ground_truth" or "community"
    distance_threshold  : float spatial distance threshold for border cell selection
    method              : str   "glm_pca" or "poisson"
                                glm_pca  — Euclidean distance in GLM-PCA space
                                poisson  — Poisson deviance from clump mean counts
    """
    glm_pca = data["glm_pca"]
    spatial = data["spatial"]
    labels  = data["labels"]
    counts  = data["counts"]        # genes × cells DataFrame
    print(glm_pca)

    print(counts)
    # ── 1. collect border cells within spatial threshold ─────────────────
    bc_to_coord = {bc: coord for bc, coord in zip(spatial.index.values,
                                                   spatial[["x", "y"]].values)}

    layer2_bcs = [bc for bc in border_barcodes["Layer 3"] if bc in bc_to_coord]
    layer3_bcs = [bc for bc in border_barcodes["Layer 4"] if bc in bc_to_coord]

    layer2_coords = np.array([bc_to_coord[bc] for bc in layer2_bcs])
    layer3_coords = np.array([bc_to_coord[bc] for bc in layer3_bcs])

    dist_layer2 = [np.min(np.linalg.norm(layer3_coords - coord, axis=1)) for coord in layer2_coords]
    dist_layer3 = [np.min(np.linalg.norm(layer2_coords - coord, axis=1)) for coord in layer3_coords]

    closer_layer2 = [bc for bc, d in zip(layer2_bcs, dist_layer2) if d <= distance_threshold]
    closer_layer3 = [bc for bc, d in zip(layer3_bcs, dist_layer3) if d <= distance_threshold]

    print(f"Layer 1 border cells within threshold: {len(closer_layer2)}")
    print(f"Layer 2 border cells within threshold: {len(closer_layer3)}")

    mixed_barcodes = closer_layer2 + closer_layer3

    layer1_clump = list(cell_clumps["Layer 3"])
    layer2_clump = list(cell_clumps["Layer 4"])

    # ── 2. distance function — swap method here ──────────────────────────
    def glm_pca_distance(bc, avg1, avg2):
        vec     = glm_pca.loc[bc].values.astype(float)
        dist_l1 = float(np.linalg.norm(vec - avg1))
        dist_l2 = float(np.linalg.norm(vec - avg2))
        return dist_l1, dist_l2

    def poisson_distance(bc, avg1, avg2):
        """
        Negative Poisson log-likelihood:
        NLL = -sum( y * log(mu) - mu - log(y!) )
        Lower NLL = cell is more likely to come from that clump's distribution.
        """
        eps = 1e-8
        y   = counts.loc[bc].values.astype(float)

        def nll(mu, y):
            mu = np.maximum(mu, eps)
            return float(np.sum(mu - y * np.log(mu) + gammaln(y + 1)))

        return nll(avg1, y), nll(avg2, y)

    # ── 3. compute clump averages depending on method ────────────────────
    if method == "glm_pca":
        avg1       = glm_pca.loc[layer1_clump].mean(axis=0).values.astype(float)
        avg2       = glm_pca.loc[layer2_clump].mean(axis=0).values.astype(float)
        dist_fn    = glm_pca_distance
        dist_label = ("GLM-PCA distance to Layer1 avg", "GLM-PCA distance to Layer2 avg")

    elif method == "poisson":
        # mean count vector across clump cells (genes,)
        avg1       = counts.loc[layer1_clump].mean(axis=0).values.astype(float)
        avg2       = counts.loc[layer2_clump].mean(axis=0).values.astype(float)
        dist_fn    = poisson_distance
        dist_label = ("Poisson deviance to Layer1 avg", "Poisson deviance to Layer2 avg")

    else:
        raise ValueError("method must be 'glm_pca' or 'poisson'")

    # ── 4. compute losses ────────────────────────────────────────────────
    records = []
    for bc in mixed_barcodes:
        dist_l1, dist_l2 = dist_fn(bc, avg1, avg2)
        records.append({"barcode": bc, "dist_layer1": dist_l1, "dist_layer2": dist_l2})
    dist_df = pd.DataFrame(records).set_index("barcode")

    # ── 5. igraph + Leiden ───────────────────────────────────────────────
    node_ids   = ["Layer1_avg", "Layer2_avg"] + mixed_barcodes
    node_index = {name: i for i, name in enumerate(node_ids)}

    edges, weights = [], []
    for bc in mixed_barcodes:
        edges.append((node_index["Layer1_avg"], node_index[bc]))
        weights.append(float(1 / (dist_df.loc[bc, "dist_layer1"] + 1e-8)))
        edges.append((node_index["Layer2_avg"], node_index[bc]))
        weights.append(float(1 / (dist_df.loc[bc, "dist_layer2"] + 1e-8)))

    g = ig.Graph(n=len(node_ids), edges=edges, directed=False)
    g.vs["name"]   = node_ids
    g.vs["is_avg"] = [True, True] + [False] * len(mixed_barcodes)
    g.es["weight"] = weights

    part              = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights=weights, resolution_parameter=1.0
    )
    membership        = part.membership
    g.vs["community"] = membership

    community_series = pd.Series(
        {g.vs[node_index[bc]]["name"]: g.vs[node_index[bc]]["community"]
         for bc in mixed_barcodes},
        name="community"
    )

    # ── 6. color setup ───────────────────────────────────────────────────
    if color_by == "ground_truth":
        unique_vals    = sorted(labels.unique())
        cmap           = plt.cm.get_cmap("tab10", len(unique_vals))
        val_to_color   = {v: cmap(i) for i, v in enumerate(unique_vals)}
        cell_color_fn  = lambda bc: val_to_color[labels[bc]]
        legend_handles = [mpatches.Patch(color=val_to_color[v], label=v)
                          for v in unique_vals]
        color_title    = "Ground Truth"

    elif color_by == "community":
        unique_vals    = sorted(set(community_series.values))
        n              = len(unique_vals)
        norm           = Normalize(vmin=0, vmax=n - 1)
        cmap           = cm.get_cmap("gist_rainbow", n)
        val_to_color   = {v: cmap(norm(i)) for i, v in enumerate(unique_vals)}
        cell_color_fn  = lambda bc: val_to_color[community_series[bc]]
        legend_handles = []
        color_title    = "Leiden Community"

    else:
        raise ValueError("color_by must be 'ground_truth' or 'community'")

    # ── 7. plots ─────────────────────────────────────────────────────────
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

    if legend_handles:
        axes[0].legend(handles=legend_handles, fontsize=7, loc="upper right")
    else:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=axes[0], shrink=0.6).set_label("Community index", fontsize=8)

    axes[0].set_title(f"Border cells — {color_title}\n(spatial view)", fontsize=11)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    d1 = dist_df.loc[mixed_barcodes, "dist_layer1"].values
    d2 = dist_df.loc[mixed_barcodes, "dist_layer2"].values
    axes[1].scatter(d1, d2, c=bc_colors, s=25, alpha=0.85,
                    linewidths=0.3, edgecolors="white")
    axes[1].axline((0, 0), slope=1, color="grey", linewidth=0.8,
                   linestyle="--", label="equal distance")
    axes[1].set_xlabel(dist_label[0], fontsize=9)
    axes[1].set_ylabel(dist_label[1], fontsize=9)
    axes[1].set_title(f"Loss to clump avg — {color_title}\n(method: {method})", fontsize=11)

    if legend_handles:
        axes[1].legend(handles=legend_handles + [
            mpatches.Patch(color="grey", label="Equal distance")], fontsize=7)
    else:
        sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm2.set_array([])
        plt.colorbar(sm2, ax=axes[1], shrink=0.6).set_label("Community index", fontsize=8)

    plt.suptitle(f"Layer1 & Layer2 border cells — {color_title} | method: {method}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"minimal_test_{color_by}_{method}.png", dpi=150, bbox_inches="tight")
    plt.show()

    return g, part, dist_df, community_series

def glm_pca_map(data):
    """
    For each PC, plot one boxplot per cluster side by side.

    Parameters
    ----------
    data : dict
        Output of load_data(), containing 'glm_pca' and 'labels'.
    """
    glm_pca  = data["glm_pca"]       # DataFrame: cells × PCs
    labels   = data["labels"]        # Series:    cells → cell_type

    clusters = sorted(labels.unique())   # 7 layers
    pcs      = glm_pca.columns          # all PC columns
    n_pcs    = len(pcs)

    fig, axes = plt.subplots(
        n_pcs, 1,
        figsize=(10, 4 * n_pcs),
        constrained_layout=True
    )

    # Make axes always iterable
    if n_pcs == 1:
        axes = [axes]

    for ax, pc in zip(axes, pcs):

        # Collect values per cluster in sorted order
        grouped = [glm_pca.loc[labels == cl, pc].values for cl in clusters]

        bp = ax.boxplot(
            grouped,
            labels=clusters,
            patch_artist=True,      # filled boxes
            showfliers=True,        # show outliers as points
            flierprops=dict(
                marker="o",
                markersize=3,
                linestyle="none",
                markerfacecolor="#e74c3c",
                markeredgecolor="#e74c3c",
                alpha=0.6,
            ),
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor="#3498db", alpha=0.7),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(pc, fontsize=10, fontweight="bold")
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("GLM-PCA value", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("GLM-PCA distribution per cluster", fontsize=13, fontweight="bold")
    plt.savefig("glm_pca_map.png", dpi=150, bbox_inches="tight")
    #plt.show()

    return fig, axes

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

def middle_cells_in_cluster(data, n_cells=20, plot=True):
    spatial  = data["spatial"]
    labels   = data["labels"]
    clusters = sorted(labels.unique())
    
    coords   = spatial[["x", "y"]].values
    barcodes = spatial.index.values

    selected_cells = {}

    if plot:
        plt.figure(figsize=(6, 6))

    for clust in clusters:
        # mask for cluster
        mask = labels == clust
        
        cluster_coords = coords[mask]
        cluster_barcodes = barcodes[mask]

        # --- 1. find cluster center ---
        center = np.median(cluster_coords, axis=0)

        # --- 2. compute distances ---
        dists = np.linalg.norm(cluster_coords - center, axis=1)

        # --- 3. select n closest ---
        idx = np.argsort(dists)[:n_cells]

        selected_cells[clust] = cluster_barcodes[idx]

        if plot:
            # plot all cells in cluster (light)
            plt.scatter(
                cluster_coords[:, 0],
                cluster_coords[:, 1],
                s=10,
                alpha=0.2
            )

            # plot selected "middle clump" (highlight)
            plt.scatter(
                cluster_coords[idx, 0],
                cluster_coords[idx, 1],
                s=40,
                label=f"Cluster {clust}"
            )

            # plot center
            plt.scatter(
                center[0],
                center[1],
                marker="x",
                s=80
            )

    if plot:
        plt.title("Middle 10-cell clumps per cluster")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return selected_cells

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
    # Load counts matrix (genes x cells)
    counts_df = pd.read_csv("./data/visium_count.csv", index_col=0)
    print("Count loading done")
    # counts_df.index = gene names, counts_df.columns = cell barcodes

    # Load spatial coordinates (barcodes, ..., x, y)
    coor_df = pd.read_csv("./data/visium_coor.csv", index_col=0)
    spatial = coor_df.iloc[:, [-2, -1]]          # last two columns = x, y
    spatial.columns = ["x", "y"]
    spatial.index.name = "barcode"
    print("Coordinate loading done")

    # Load GLM-PCA embeddings (first two columns are redundant barcodes)
    glm_df = pd.read_csv("./data/visium_glm.csv", index_col=0)
    glm_df = glm_df.iloc[:, 1:]                  # drop the redundant barcode column
    glm_df.index.name = "barcode"
    print("GLM PCA loading done")

    # Load ground-truth annotations (col 1 = index, col 1 = barcode, col 2 = label)
    gt_df = pd.read_csv("./data/visium_manual.csv", index_col=0)
    ground_truth = gt_df.iloc[:, [0, 1]]
    ground_truth.columns = ["barcode", "cell_type"]
    ground_truth = ground_truth.set_index("barcode")["cell_type"]
    print("Manual Annotation loading done")

    # Align everything to the shared set of barcodes
    barcodes = counts_df.columns.intersection(spatial.index) \
                                .intersection(glm_df.index) \
                                .intersection(ground_truth.index)

    counts      = counts_df[barcodes]                   # cells  × genes
    spatial     = spatial.loc[barcodes]                 # cells  × 2
    glm_pca     = glm_df.loc[barcodes]                  # cells  × n_components
    labels      = ground_truth.loc[barcodes]            # cells  (Series)
    print("Intersecting done")
    counts = counts.T
    print("Flipping counts done")
    print(counts)
    return {
        "barcodes":   barcodes,
        "counts":     counts,       # DataFrame: cells × genes
        "spatial":    spatial,      # DataFrame: cells × {x, y}
        "glm_pca":    glm_pca,      # DataFrame: cells × PCA components
        "labels":     labels,       # Series:    cells → cell_type string
    }

if __name__ == "__main__":
    main()