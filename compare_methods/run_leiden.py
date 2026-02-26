import igraph as ig
import leidenalg
import csv
from sklearn.metrics import adjusted_rand_score
from collections import Counter

def main():
    # load the graph from GraphML
    G = ig.Graph.Read_GraphML("cosmx.graphml")
    print(G.summary())

    # node labels
    if "label" in G.vs.attributes():
        node_labels = G.vs["label"]
    else:
        node_labels = [str(v.index) for v in G.vs]

    # Leiden clustering using weights if present
    weights = G.es["weight"] if "weight" in G.es.attributes() else None

    part = leidenalg.find_partition(
        G,
        leidenalg.RBERVertexPartition,
        weights=weights,
        resolution_parameter=0.01
    )

    # community sizes
    for i, community in enumerate(part):
        print(f"Community {i}: {len(community)} nodes")

    # Build dict: cell_id -> leiden community
    leiden_map = {}
    for comm_id, community in enumerate(part):
        for node_index in community:
            leiden_map[str(node_labels[node_index])] = comm_id

    # write Leiden CSV
    # with open("test.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["cell_name", "community_id"])
    #     for cell, comm in leiden_map.items():
    #         writer.writerow([cell, comm])

    print("CSV file 'community_assignments_cosmx.csv' written successfully.")

    other_map = {}
    with open("cellcharter_cosmx.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 4:
                continue
            cell_id = row[0]
            community = row[3]
            other_map[cell_id] = community

    print("\nOther clustering communities:")
    counts = Counter(other_map.values())
    for comm_id, size in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"Community {comm_id}: {size} cells")
    # common
    common_cells = set(leiden_map.keys()) & set(other_map.keys())
    print(f"Intersecting cells: {len(common_cells)}")

    leiden_labels = []
    other_labels = []

    for cell in common_cells:
        leiden_labels.append(leiden_map[cell])
        other_labels.append(other_map[cell])

    # ari
    if len(common_cells) > 0:
        ari = adjusted_rand_score(other_labels, leiden_labels)
        print(f"Adjusted Rand Index: {ari:.6f}")
    else:
        print("No overlapping cells found. ARI not computed.")


if __name__ == "__main__":
    main()