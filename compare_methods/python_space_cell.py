import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
import random
import copy
from collections import Counter
import time

def main():
    # load data
    data = load_data()
    #fig, axes = glm_pca_map(data)
    # find cells in clusters 
    #all_cell_per_cluster = find_cells_in_cluster(data)

    # initialize the cluster centers based on ground truth
    #avg_per_cluster_counts, avg_per_cluster_glm_pca, avg_per_cluster_pca = initialize_cluster_centers(data, all_cell_per_cluster)

    # for two close by cells plot the loss for each cell from each cluster center
    #plot_using_different_loss_functions(data, all_cell_per_cluster, avg_per_cluster_counts, avg_per_cluster_glm_pca, avg_per_cluster_pca)
    # do kmeans++ and gaussian mixture model on glm pca data and plot the clustering
    #kmeans_plot(data, all_cell_per_cluster)
    # cluster + refine
    cluster_with_gmm_refine_negbi(data)
    return

def cluster_with_gmm_refine_negbi(data):
    n_clusters = 6
    random_state = 10
    show_gaussian_cluster_result = True
    spatial  = data["spatial"]
    labels   = data["labels"]
    counts   = data["counts"]
    glm_pca  = data["glm_pca"]
    pca      = data["pca"]
    umi      = data["umi"]

    # this doesnt need to be done, only loading data with groundtruth
    shared_bcs = glm_pca.index.intersection(labels.index)
    X          = glm_pca.loc[shared_bcs].values.astype(float)
    y_true     = labels.loc[shared_bcs].values

    # first cluster with gmm using glm pca data
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
    # Look into how gmm clustered
    if show_gaussian_cluster_result == True:
        gt_counts = Counter(zip(y_pred, y_true))
        for cluster_id in range(n_clusters):
            breakdown = {
                gt: cnt
                for (cl, gt), cnt in gt_counts.items()
                if cl == cluster_id
            }
            total = sum(breakdown.values())
            print(f"Cluster {cluster_id} (n={total})")
            for gt, cnt in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
                print(f"    {gt}: {cnt}")
    # get the list of cells in each cluster
    shared_bcs_arr = np.array(shared_bcs)
    gmm_clusters = [
        shared_bcs_arr[y_pred == c].tolist()
        for c in range(n_clusters)
    ]
    for c, bcs in enumerate(gmm_clusters):
        print(f"GMM cluster {c}: {len(bcs)} cells")

    # for each cluster make a stack/list whatever
    cluster_stacks = [[] for _ in range(n_clusters)] 
    # initialize the cluster stacks
    cluster_stacks = find_best_cells_to_add_to_each_stack_init(cluster_stacks, gmm_clusters, counts, umi, 0)
    # find the initial cells to add to each stack
    for run in range(0, 1_000):
        cluster_stacks = find_best_cells_to_add_to_each_stack_iter(cluster_stacks, gmm_clusters, counts, umi, run)
        # check if the cluster stacks corrospond to one ground truth or multiple
        flat = [(bc, i) for i, stack in enumerate(cluster_stacks) for bc in stack]
        gt_counts = Counter((my_cluster, labels[bc]) for bc, my_cluster in flat)
        for i in range(len(cluster_stacks)):
            breakdown = {gt: cnt for (cl, gt), cnt in gt_counts.items() if cl == i}
            total     = sum(breakdown.values())
            print(f"Cluster {i}  (n={total}): {breakdown}")
    # if not separable by -ve binomial after a number of retries make a note of the unsepearable clusters (stacks) and disable one stack randomly and continue the process

    # when no more cells can be added, end the process, assign the disabled stack to the closest stack, assign unassigned cells to the closest and stack

    # check the ari and f1 score of the clustering

    return

def find_best_cells_to_add_to_each_stack_iter(original_cluster_stacks, gmm_clusters, counts, umi, run):
    # based on the run increase theta
    theta = 10 / max(run / 10, 1)
    cluster_stacks = copy.deepcopy(original_cluster_stacks)
    # assign random cell from each cluster to the stack, and get the lowest -bi loss cell and max loss with other cells
    best_score_loss = float('-inf')
    for cluster_stack_index in range(0, len(cluster_stacks)):
        start_time = time.perf_counter()
        cc_for_stacks = initialize_cluster_centers_for_stacks(cluster_stacks, counts)
        # for each stack select the next cell with best score (lowest loss with stack and highest loss with other stacks)
        available = [bc for bc in gmm_clusters[cluster_stack_index] if bc not in cluster_stacks[cluster_stack_index]]
        if not available:
            print(cluster_stack_index, "full!")
            continue
        # only select few of the available (half or 500 cells)
        num_cell_to_process = int(min(max(len(available) / 2, 1), 50))
        available = random.sample(available, k = num_cell_to_process)
        print("Number of cells to process in stack", cluster_stack_index, "cells", len(available), "theta", theta)
        # go through all the available cells and choose
        not_updated = True
        best_cell = None
        best_score_loss = float("-inf")
        inseparable_count = 0
        for cell_index, cell in enumerate(available):
            #cluster_stacks[cluster_stack_index].append(cell)
            #old_cc = cc_for_stacks[cluster_stack_index]
            # update only the required using the cell barcode
            #new_cc = initialize_cluster_centers_for_stacks(None, counts, update_single=True, new_barcode=cell, current_mean=old_cc, current_count=len(cluster_stacks[cluster_stack_index]) - 1)
            #new_cc = initialize_cluster_centers_for_stacks(cluster_stacks, counts, True, cluster_stack_index)
            # check if cc's are the same
            #print("equal?", np.allclose(new_cc_1, new_cc))
            #cc_for_stacks[cluster_stack_index] = new_cc
            # loss calculation need to update this function
            own_cc_loss_total, other_cc_loss_total, separable = loss_cells_per_cc_2(cell, cluster_stack_index, cc_for_stacks, counts, umi, theta)
            score_loss = other_cc_loss_total / (own_cc_loss_total + 1e-8)
            if cell_index % int(len(available) / 5) == 0:
                print("Current stack {} cell {} current score {} best score {}".format(cluster_stack_index, cell_index, score_loss, best_score_loss))
            if score_loss > best_score_loss and separable == True:
                best_score_loss = score_loss
                best_cell = cell
                not_updated = False
            if separable == False:
                inseparable_count += 1
            #cluster_stacks[cluster_stack_index].pop()
            #cc_for_stacks[cluster_stack_index] = old_cc
        if best_cell != None or not_updated != True:
            cluster_stacks[cluster_stack_index].append(best_cell)
        end_time = time.perf_counter()
        print(f"Elapsed time for stack: {(end_time - start_time):.6f} seconds, inseparable cells {inseparable_count}")
    return cluster_stacks

def loss_cells_per_cc_2 (cell, cluster_stack_index, cc_for_stacks, counts, umi , theta):
    own_cc_loss = 0.0
    other_cc_loss_total = 0.0
    separable = True
    other_cc_loss_array = []
    # caluculate the loss from each cc to cell
    for cc_index, cc_for_stack in enumerate(cc_for_stacks):
        # cells own stack
        if cluster_stack_index == cc_index:
            # calculate own loss here
            own_cc_loss = negative_binomial_distance2(cell, cc_for_stack, counts, theta) / umi.loc[cell]
            #print("equal?", np.allclose(counts.loc[cell].values.astype(float), cc_for_stack), "own_cc_loss", own_cc_loss)
        # other stacks
        else:
            # calculate other loss here
            other_cc_loss = negative_binomial_distance2(cell, cc_for_stack, counts, theta) / umi.loc[cell]
            other_cc_loss_array.append(other_cc_loss)
            other_cc_loss_total += other_cc_loss
    for other_cc_loss in other_cc_loss_array:
        if own_cc_loss > other_cc_loss:
            separable = False
    return (own_cc_loss, other_cc_loss_total, separable)

def find_best_cells_to_add_to_each_stack_init(original_cluster_stacks, gmm_clusters, counts, umi, run):
    cluster_stacks = copy.deepcopy(original_cluster_stacks)
    # assign random cell from each cluster to the stack, and get the lowest -bi loss cell and max loss with other cells
    best_score_loss = float('-inf')
    best_score_seed = 0
    best_cluster_stack = []
    not_updated = True
    # initial stack finding, run as much as possible and get the lowest loss cells
    for seed in range(11_700, 11_800):
        # set random seed
        random.seed(seed)
        for cluster_stack_index in range(0, len(cluster_stacks)):
            # find available cells (barcodes)
            random_cell_for_cluster = random.choice(gmm_clusters[cluster_stack_index])
            # insert the random cell
            cluster_stacks[cluster_stack_index].append(random_cell_for_cluster)
        # initialize cluster centers for each stack
        cc_for_stacks = initialize_cluster_centers_for_stacks(cluster_stacks, counts)
        # check the -vebinomial loss of cluster center vs cells (should be min for current stack and max for other stacks)
        own_cc_loss_total, other_cc_loss_total, separable = loss_cells_per_cc(cc_for_stacks, cluster_stacks, counts, umi)
        # print whether separable, orgin cluster loss and other cluster loss  
        score_loss = other_cc_loss_total / (own_cc_loss_total + 1e-8)
        if score_loss > best_score_loss and separable == True:
            best_score_loss = score_loss
            best_score_seed = seed
            best_cluster_stack = copy.deepcopy(cluster_stacks)
            not_updated = False
        if seed % 100 == 0:
            print("own loss", own_cc_loss_total, "other loss", other_cc_loss_total, "separable", separable)
            print("score", best_score_loss, "best_seed", best_score_seed, "current_seed", seed, "cells_per_stack", run + 1)
        # go back to the original
        cluster_stacks = copy.deepcopy(original_cluster_stacks)
    if not_updated:
        print("ERROR NOT UPDATED THE STACK!!!")
    return best_cluster_stack

def loss_cells_per_cc (cc_for_stacks, cluster_stacks, counts, umi):
    own_cc_loss_total = 0.0
    other_cc_loss_total = 0.0
    separable = True
    # for each cell in cluster stacks, calculate the -ve binomial loss with each cc,
    for cluster_index, cell_list in enumerate(cluster_stacks):
        for cell in cell_list:
            own_cc_loss = 0.0
            other_cc_loss_array = []
            for cc_index, cc_for_stack in enumerate(cc_for_stacks):
                # calculate the own cc loss 
                if cc_index == cluster_index:
                    own_cc_loss = negative_binomial_distance2(cell, cc_for_stack, counts, theta=1) / umi.loc[cell]
                    #print("equal?", np.allclose(counts.loc[cell].values.astype(float), cc_for_stack), "own_cc_loss", own_cc_loss)
                    own_cc_loss_total += own_cc_loss
                # calculate the other cc loss
                else:
                    other_cc_loss = negative_binomial_distance2(cell, cc_for_stack, counts, theta=1) / umi.loc[cell]
                    other_cc_loss_array.append(other_cc_loss)
                    other_cc_loss_total += other_cc_loss
            for other_cc_loss in other_cc_loss_array:
                if own_cc_loss > other_cc_loss:
                    separable = False
    #print(loss_array)
    #print(own_cc_loss_total, other_cc_loss_total, separable)
    return (own_cc_loss_total, other_cc_loss_total, separable)

def loss_cells_per_cc_with_prior_knowledge():
    # old cc own loss and other loss is known so using additive rule only calculate diff between old vs new and new cell loss

    return

def L(obs, ref, theta=0.01, eps=1e-8):
    return nll(ref, obs, theta, eps) # nll(mu=ref, y=obs)

def own_after_add(own_old, n_old, mu_old, mu_new, y_c, theta=0.01):
    shift = n_old * (L(mu_old, mu_new, theta) - L(mu_old, mu_old, theta))
    return own_old + shift + L(y_c, mu_new, theta)

def other_after_add(other_old, n_other, mu_bar_other, mu_old, mu_new, theta=0.01):
    return other_old + n_other * (L(mu_bar_other, mu_new, theta)
                                  - L(mu_bar_other, mu_old, theta))

def negative_binomial_distance2(bc, avg, counts, theta=5):
    eps = 1e-8
    y   = counts.loc[bc].values.astype(float)
    return nll(avg, y, theta, eps)

def nll(mu, y, theta, eps = 1e-8):
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

def initialize_cluster_centers_for_stacks(cluster_stacks, counts, only_init_one=False, init_one=0, update_single=False, new_barcode=None, current_mean=None, current_count=0):
    avg_per_cluster_counts = list()
    # only update one cell, to do test this part, run with update sinle and stack and check
    if update_single:
        new_value = counts.loc[new_barcode].values.astype(float)
        if current_mean is None or current_count == 0:
            return new_value
        current_mean = np.asarray(current_mean, dtype=float)
        return current_mean + (new_value - current_mean) / (current_count + 1)
    # only one stack
    if only_init_one == True:
        return counts.loc[cluster_stacks[init_one]].mean(axis=0).values.astype(float)
    # calculate all cluster stack stuff
    for cluster_stack in cluster_stacks:
        avg_per_cluster_counts.append(
            counts.loc[cluster_stack].mean(axis=0).values.astype(float)
        )
    return avg_per_cluster_counts

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
    umi   = data["umi"]
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
            dist_l1 = dist_l1 / umi.loc[bc]
            dist_l2 = dist_l2 / umi.loc[bc]
            records.append({"barcode": bc, "dist_layer1": dist_l1, "dist_layer2": dist_l2})
        dist_df_poisson = pd.DataFrame(records).set_index("barcode")
        # calculcate negative binomial loss
        records = []
        for bc in mixed_barcodes:
            dist_l1, dist_l2 = negative_binomial_distance(bc, first_cluster_avg_counts, second_cluster_avg_counts, counts)
            dist_l1 = dist_l1 / umi.loc[bc]
            dist_l2 = dist_l2 / umi.loc[bc]
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

        d1 = dist_df_negbi.loc[mixed_barcodes, "dist_layer1"].values
        d2 = dist_df_negbi.loc[mixed_barcodes, "dist_layer2"].values
        axes[1].scatter(d1, d2, c=bc_colors, s=15, alpha=0.55,
                        linewidths=0.3, edgecolors="white")
        axes[1].axline((0, 0), slope=1, color="grey", linewidth=0.8,
                    linestyle="--", label="equal distance")
        axes[1].set_xlabel("log(loss to cc1 avg) / umi", fontsize=9)
        axes[1].set_ylabel("log(loss to cc2 avg) / umi", fontsize=9)
        axes[1].set_title(f"Loss to cluster center avg — {color_title}\n(method)", fontsize=11)
        #axes[1].set_xlim(0, 7.5)
        #axes[1].set_ylim(0, 7.5)

        axes[1].legend(handles=legend_handles + [
            mpatches.Patch(color="grey", label="Equal distance")], fontsize=7)
        plt.tight_layout()
        plt.savefig("negbi_slide" + str(i))
        plt.show()
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

def negative_binomial_distance(bc, avg1, avg2, counts, theta=0.01):
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
    return nll(avg1, y, theta, eps), nll(avg2, y, theta, eps)

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

def poisson_loss(x, lam, eps=1e-8):
    lam = np.clip(lam, eps, None)
    return np.sum(lam - x * np.log(lam))

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
    # calculate the umi per cell (total counts)
    umi_per_cell = counts.sum(axis=1).astype(int)
    umi_per_cell.name = "umi"
    print(f"UMI per cell — min: {umi_per_cell.min()}, "
          f"median: {umi_per_cell.median()}, max: {umi_per_cell.max()}")
    
    # get PCA embeddings
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
        "pca": pca,                 # DataFrame: cells × PCA components
        "labels": labels,           # Series: cells → cell_type string
        "umi": umi_per_cell         # Series:    cells → total UMI count
    }

if __name__ == "__main__":
    main()