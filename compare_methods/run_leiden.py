import leidenalg
import igraph as ig

def main():
    G = ig.Graph.Read_DOT("graph.dot")
    print(G.summary())
    # Edge list from Rust code
    edges = [
        ("Fortran", "C", 0.5),
        ("Fortran", "LISP", 0.3),
        ("Fortran", "MATLAB", 0.6),
        ("C", "C++", 0.9),
        ("C", "Go", 0.6),
        ("LISP", "ML", 0.5),
        ("LISP", "OCaml", 0.2),
        ("LISP", "Haskell", 0.2),
        ("LISP", "Ruby", 0.5),
        ("LISP", "Julia", 0.6),
        ("ML", "OCaml", 0.8),
        ("ML", "Haskell", 0.5),
        ("OCaml", "Haskell", 0.3),
        ("OCaml", "F#", 0.6),
        ("Haskell", "Julia", 0.2),
        ("C++", "Python", 0.32),
        ("C++", "Ruby", 0.2),
        ("C++", "C#", 0.5),
        ("Python", "F#", 0.2),
        ("Python", "Julia", 0.4),
        ("C#", "F#", 0.3),
    ]

    # Collect unique node names
    nodes = sorted({u for u, v, _ in edges} | {v for u, v, _ in edges})

    # Create graph with named vertices
    G = ig.Graph()
    G.add_vertices(nodes)

    # Add edges by index
    name_to_index = {name: i for i, name in enumerate(nodes)}
    edge_indices = [(name_to_index[u], name_to_index[v]) for u, v, _ in edges]
    weights = [w for _, _, w in edges]

    G.add_edges(edge_indices)
    G.es["weight"] = weights

    # Leiden clustering using weights
    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition, weights=G.es["weight"])
    print(part)
    return


if __name__ == '__main__':
    main()