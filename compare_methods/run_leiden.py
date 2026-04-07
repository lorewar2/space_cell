import igraph as ig
import leidenalg
import csv
from sklearn.metrics import adjusted_rand_score
from collections import Counter

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
        resolution_parameter=0.0000000001
    )

    other_map = {}
    coords_map = {}

    with open("cellcharter_cere.csv", newline="") as f:
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
        # Build Leiden map and compute sizes
    leiden_map = {}
    community_sizes = {}

    for comm_id, community in enumerate(part):
        size = len(community)
        community_sizes[comm_id] = size
        print(f"Leiden Community {comm_id}: {size} nodes")
        for node_index in community:
            leiden_map[str(node_labels[node_index])] = comm_id

    top10 = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:20]
    top10_ids = {cid for cid, _ in top10}

    print("\nTop 10 largest communities:")
    for cid, size in top10:
        print(f"Community {cid}: {size} nodes")

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

    print("\nCSV written with only the 10 largest Leiden communities.")
    # print other cluster sizes
    print("\nCellcharter:")
    counts = Counter(other_map.values())
    for comm_id, size in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"Cluster {comm_id}: {size} cells")

    common_cells = set(leiden_map.keys()) & set(other_map.keys())
    print(f"\nIntersecting cells: {len(common_cells)}")

    leiden_labels = [leiden_map[c] for c in common_cells]
    other_labels = [other_map[c] for c in common_cells]

    if common_cells:
        ari = adjusted_rand_score(other_labels, leiden_labels)
        print(f"Adjusted Rand Index: {ari:.6f}")
    else:
        print("No overlapping cells found.")


if __name__ == "__main__":
    main()