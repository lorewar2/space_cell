import igraph as ig
import leidenalg
import csv

def main():
    # load the graph from GraphML
    G = ig.Graph.Read_GraphML("languages.graphml")
    print(G.summary())

    if "label" in G.vs.attributes():
        node_labels = G.vs["label"]
    else:
        node_labels = [str(v.index) for v in G.vs]

    # Leiden clustering using weights if present
    if "weight" in G.es.attributes():
        weights = G.es["weight"]
    else:
        weights = None

    part = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=0.01
    )

    # community 
    community_sizes = [len(c) for c in part]
    for i, size in enumerate(community_sizes):
        print(f"Community {i}: {size} nodes")

    # csv
    with open("community_assignments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_name", "community_id"])
        for comm_id, community in enumerate(part):
            for node_index in community:
                writer.writerow([node_labels[node_index], comm_id])

    print("CSV file 'community_assignments.csv' written successfully.")

if __name__ == "__main__":
    main()