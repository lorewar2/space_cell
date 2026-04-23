import igraph as ig
import leidenalg
import csv
from collections import Counter
from sklearn.metrics import adjusted_rand_score


def main():
    # load graph
    G = ig.Graph.Read_GraphML("cere.graphml")
    print(G.summary())

    # node labels
    if "label" in G.vs.attributes():
        node_labels = G.vs["label"]
    else:
        node_labels = [str(v.index) for v in G.vs]

    # weights
    weights = G.es["weight"] if "weight" in G.es.attributes() else None

    # Leiden partition
    part = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=0.0000001
    )

    # ---- Load CellCharter clusters ----
    other_map = {}
    coords_map = {}

    with open("./data/visium_cellcharter.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            cell_id = row[0]
            x = row[1]
            y = row[2]
            community = row[3]

            other_map[cell_id] = community
            coords_map[cell_id] = (x, y)

    # ---- Load Manual clusters ----
    manual_map = {}

    with open("./data/visium_manual.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            cell_id = row[1]
            manual_cluster = row[2]
            manual_map[cell_id] = manual_cluster

    # ---- Build Leiden map ----
    leiden_map = {}
    community_sizes = {}

    for comm_id, community in enumerate(part):
        size = len(community)
        community_sizes[comm_id] = size
        print(f"Leiden Community {comm_id}: {size} nodes")
        for node_index in community:
            leiden_map[str(node_labels[node_index])] = comm_id

    # ---- Top communities ----
    top10 = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:20]
    top10_ids = {cid for cid, _ in top10}

    print("\nTop 20 largest communities:")
    for cid, size in top10:
        print(f"Community {cid}: {size} nodes")

    # ---- Write CSV ----
    with open("leiden_cere.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_name", "community_id", "x", "y"])

        for cell, comm in leiden_map.items():
            if comm not in top10_ids:
                continue

            if cell in coords_map:
                x, y = coords_map[cell]
            else:
                x, y = "", ""

            writer.writerow([cell, comm, x, y])

    print("\nCSV written with only the 20 largest Leiden communities.")

    # ---- CellCharter stats ----
    print("\nCellCharter:")
    counts = Counter(other_map.values())
    for comm_id, size in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"Cluster {comm_id}: {size} cells")

    # ---- ARI: CellCharter vs Leiden ----
    common_cells = set(leiden_map.keys()) & set(other_map.keys())
    print(f"\nIntersecting cells (CellCharter): {len(common_cells)}")

    if common_cells:
        leiden_labels = [leiden_map[c] for c in common_cells]
        other_labels = [other_map[c] for c in common_cells]

        ari = adjusted_rand_score(other_labels, leiden_labels)
        print(f"Adjusted Rand Index (CellCharter vs Leiden): {ari:.6f}")
    else:
        print("No overlapping cells found (CellCharter).")

    # ---- ARI: Manual vs Leiden ----
    common_manual_cells = set(leiden_map.keys()) & set(manual_map.keys())
    print(f"\nIntersecting cells (Manual): {len(common_manual_cells)}")

    if common_manual_cells:
        leiden_labels_manual = [leiden_map[c] for c in common_manual_cells]
        manual_labels = [manual_map[c] for c in common_manual_cells]

        ari_manual = adjusted_rand_score(manual_labels, leiden_labels_manual)
        print(f"Adjusted Rand Index (Manual vs Leiden): {ari_manual:.6f}")
    else:
        print("No overlapping cells found (Manual).")

    # ---- Breakdown: Leiden vs Manual ----
    print("\nLeiden community composition (Manual clusters):")

    community_to_manual = {}

    for cell, comm in leiden_map.items():
        if cell not in manual_map:
            continue

        manual_cluster = manual_map[cell]

        if comm not in community_to_manual:
            community_to_manual[comm] = Counter()

        community_to_manual[comm][manual_cluster] += 1

    for comm_id, counter in sorted(community_to_manual.items()):
        total_cells = sum(counter.values())
        if total_cells < 2:
            continue
        print(f"\nLeiden Community {comm_id}:")

        
        print(f"  Total cells with manual labels: {total_cells}")

        for cluster_id, count in counter.most_common():
            print(f"    Manual Cluster {cluster_id}: {count} cells")

        # Flag mixed communities
        if len(counter) > 1:
            print("Mixed community (maps to multiple manual clusters)")

        # Optional purity metric
        dominant = max(counter.values())
        purity = dominant / total_cells
        print(f"  Purity: {purity:.3f}")


if __name__ == "__main__":
    main()