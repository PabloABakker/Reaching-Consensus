"""
Helper functions for the Distributed Signal Processing project.
Made by Pablo Bakker and Ines Marques.

"""

# Imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



# Graph constructors
def random_geometric_network(n, area_length, comm_radius, seed=None):
    """
    Place n sensors uniformly in [0, area_length]^2 and connect any pair
    within Euclidean distance comm_radius.

    Returns
    -------
    G : networkx.Graph
    positions : (n, 2) ndarray of node coordinates
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, area_length, size=(n, 2))

    G = nx.Graph()
    for i, p in enumerate(positions):
        G.add_node(i, pos=tuple(p))

    # Pairwise distances.
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    iu, ju = np.triu_indices(n, k=1)
    mask = dist[iu, ju] <= comm_radius
    for i, j, d in zip(iu[mask], ju[mask], dist[iu, ju][mask]):
        G.add_edge(int(i), int(j), weight=float(d))

    return G, positions

def grid_network(grid_size, area_length, neighbours="4"):
    """
    Place grid_size x grid_size sensors on a regular lattice spanning
    [0, area_length]^2.

    neighbours = "4"  -> rook adjacency (up/down/left/right)
    neighbours = "8"  -> king adjacency (also diagonals)
    """
    n = grid_size * grid_size
    if grid_size > 1:
        spacing = area_length / (grid_size - 1)
    else:
        spacing = 0.0

    positions = np.array([[i * spacing, j * spacing] for i in range(grid_size) for j in range(grid_size)])

    if neighbours == "4":
        radius = spacing * 1.01
    elif neighbours == "8":
        radius = spacing * np.sqrt(2) * 1.01
    else:
        raise ValueError("neighbours must be '4' or '8'")

    G = nx.Graph()
    for i, p in enumerate(positions):
        G.add_node(i, pos=tuple(p))

    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu, ju = np.triu_indices(n, k=1)
    mask = dist[iu, ju] <= radius
    for i, j, d in zip(iu[mask], ju[mask], dist[iu, ju][mask]):
        G.add_edge(int(i), int(j), weight=float(d))

    return G, positions



# Theory & helpers
def critical_radius_rgg(n, area_size):
    """
    Critical connectivity radius for an Random Geometric Graph:
    """
    return area_size * np.sqrt(2*np.log(n) / n)



# Information
def network_summary(G, name=""):
    """Print useful structural properties for the consensus analysis later."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    connected = nx.is_connected(G)
    avg_deg = 2 * m / n
    print(f"--- {name} ---")
    print(f"  nodes              : {n}")
    print(f"  edges              : {m}")
    print(f"  connected          : {connected}")
    print(f"  average degree     : {avg_deg:.2f}")
    if connected:
        # Algebraic connectivity = 2nd smallest Laplacian eigenvalue.
        # Larger lambda_2 -> faster consensus convergence.
        L = nx.laplacian_matrix(G).toarray().astype(float)
        eigvals = np.sort(np.linalg.eigvalsh(L))
        print(f"  lambda_2 : {eigvals[1]:.4f}")
        print(f"  lambda_max : {eigvals[-1]:.4f}")
    print()



# Visualization
def plot_network(G, positions, title, ax):
    pos_dict = {i: positions[i] for i in range(len(positions))}
    nx.draw_networkx_edges(G, pos=pos_dict, edge_color="lightgray", ax=ax)
    nx.draw_networkx_nodes(G, pos=pos_dict, node_size=40, node_color="steelblue", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)