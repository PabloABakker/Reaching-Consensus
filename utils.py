"""
Helper functions for the Distributed Signal Processing project.
Made by Pablo Bakker and Ines Marques.

"""

# Imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



# Connectivity experiment for choosing n
def estimate_connectivity_probability(n_values, comm_radius, area_length=100, n_trials=500, base_seed=0):
    results = []

    for n in n_values:
        connected_count = 0
        avg_degrees = []

        for trial in range(n_trials):
            seed = base_seed + 10_000 * n + trial
            G, _ = random_geometric_network(
                n=n,
                area_length=area_length,
                comm_radius=comm_radius,
                seed=seed
            )

            if nx.is_connected(G):
                connected_count += 1

            avg_degrees.append(2 * G.number_of_edges() / n)

        results.append({
            "n": n,
            "connectivity_probability": connected_count / n_trials,
            "mean_average_degree": np.mean(avg_degrees)
        })

    return results



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


# Function to get the second largest eigenvalue from the averaging matrix
def lambda2_W(G):
    """ E[W] = I - L/(2|E|)"""
    m = G.number_of_edges()
    L = nx.laplacian_matrix(G).toarray().astype(float)
    lam2_L = np.sort(np.linalg.eigvalsh(L))[1]
    return 1.0 - lam2_L / (2 * m)



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

    # Check lambda_2 which is related to the consensus convergence.
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigvals_L = np.sort(np.linalg.eigvalsh(L))
    lambda2 = lambda2_W(G)
    print(f"  lambda_2 : {lambda2:.4f}")
    print(f"  lambda_max : {eigvals_L[0]:.4f}\n")



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



# Random Gossip
def randomized_gossip_average(G, x0, n_iterations=10_000, packet_loss_prob=0.0, seed=None, track_every=10):
    """
    Randomized pairwise gossip for average consensus

    At each iteration:
      1. Pick a random edge (i, j) uniformly
      2. With probability 1 - packet_loss_prob, nodes i and j exchange values
      3. Both update to their pairwise average

    Parameters
    ----------
    G : Connected communication graph
    x0 : Initial sensor measurements
    n_iterations : Number of gossip iterations
    packet_loss_prob : Bernoulli packet failure probability
    seed : Random seed
    track_every : Store error every `track_every` iterations

    Returns
    -------
    x : Final node values
    errors : Consensus error history
    transmissions : Transmission counts corresponding to errors
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()

    # Compute average for reference
    true_average = np.mean(x0)

    # Get the edges of the graph
    edges = list(G.edges())
    if len(edges) == 0:
        raise ValueError("Graph has no edges.")

    # Keep track of history
    errors = []
    transmissions = []

    denom = max(np.linalg.norm(x0 - true_average)**2, 1.0)

    # Gossip
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            err = np.linalg.norm(x - true_average)**2 
            errors.append(err/denom)
            transmissions.append(k*2)

        if k == n_iterations:
            break

        i, j = edges[rng.integers(len(edges))]

        # Bernoulli packet loss
        if rng.random() < packet_loss_prob:
            continue

        avg_ij = 0.5 * (x[i] + x[j])
        x[i] = avg_ij
        x[j] = avg_ij

    return x, np.array(errors), np.array(transmissions)


# Broadcast one way gossip, same inputs and outputs as regular gossip
def randomized_broadcast_gossip(G, x0, n_iterations, alpha=0.5, packet_loss_prob=0.0, seed=None, track_every=10):
    """
    Broadcast gossip

    At each iteration:
      1. pick a random node i
      2. i broadcasts x_i to all neighbours
      3. each neighbour j updates  x_j <- (1 - alpha) * x_j + alpha * x_i
      4. i itself does not update

    Cost per broadcast: d_i transmissions same accounting as broadcast PDMM, so the two are directly comparable.

    Does not preserve the network mean:
    The consensus value reached is a random variable that depends on the broadcast sequence
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)

    # Calculate true average for reference
    true_average = float(np.mean(x0))

    # Get neighbours
    neighbours = [list(G.neighbors(i)) for i in range(n)]

    # Keep track of history
    errors, transmissions = [], []
    n_tx = 0

    denom = max(np.linalg.norm(x0 - true_average)**2, 1.0)


    # Run boradcast one-way gossip
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            errors.append(np.linalg.norm(x - true_average) ** 2 / denom)
            transmissions.append(n_tx)
        if k == n_iterations:
            break

        i = int(rng.integers(n))
        if not neighbours[i]:
            continue
        x_i = x[i]
        for j in neighbours[i]:
            if rng.random() >= packet_loss_prob:
                x[j] = (1 - alpha) * x[j] + alpha * x_i
            n_tx += 1

    return x, np.array(errors), np.array(transmissions)


# Sum-Weight broadcast gossip, same inputs and outputs as regular gossip
def sum_weight_broadcast(G, x0, n_iterations, packet_loss_prob=0.0, seed=None, track_every=10):
    """
    Push-sum average consensus (Kempe-Dobra-Gehrke 2003), broadcast variant.

    Each node i holds (s_i, w_i) = (a_i, 1) initially; its estimate is s_i/w_i.
    When node i activates it splits its mass into d_i + 1 equal shares,
    keeps one, and sends one to each neighbour (who add it to their own).

    Conserves sum(s) = sum(a) and sum(w) = n exactly (no packet loss),
    so s_i/w_i -> mean(a) at every node with NO degree bias.

    Cost per activation: d_i transmissions.

    WARNING: naive packet loss breaks conservation (mass leaks). For the
    packet-loss experiment, either expect divergence or use a robust variant.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(x0, dtype=float).copy()
    n = len(a)
   
    # Get true average for reference
    true_average = float(np.mean(x0))

    # Get neighbours
    neighbours = [list(G.neighbors(i)) for i in range(n)]
    s = a.copy()
    w = np.ones(n)

    # Keep track of history
    errors, transmissions = [], []
    n_tx = 0

    denom = max(np.linalg.norm(a - true_average)**2, 1.0)


    # Run sum-weight gossip
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            est = s / w
            errors.append(np.linalg.norm(est - true_average) ** 2 / denom)
            transmissions.append(n_tx)
        if k == n_iterations:
            break

        i = int(rng.integers(n))
        di = len(neighbours[i])
        if di == 0:
            continue

        beta = 1.0 / (di + 1)
        s_share, w_share = beta * s[i], beta * w[i]
        s[i], w[i] = s_share, w_share # keep one share
        for j in neighbours[i]:
            if rng.random() >= packet_loss_prob:
                s[j] += s_share
                w[j] += w_share
            n_tx += 1

    return s / w, np.array(errors), np.array(transmissions)


# Robust sum-Weight broadcast gossip, same inputs and outputs as regular gossip
def robust_sum_weight_broadcast(G, x0, n_iterations, packet_loss_prob=0.0, seed=None, track_every=10):
    """
    Robust (loss-tolerant) sum-weight, broadcast variant

    Each directed edge i->j carries the cumulative mass i has ever sent toward
    j. The receiver remembers the cumulative mass it has absorbed and applies
    only the difference, so a lost packet is recovered by the next successful
    one. Conserves total mass (held + in-flight) exactly under any packet loss.

    Cost per activation: d_i transmissions.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(x0, dtype=float).copy()
    n = len(a)
    
    # Get true average for reference
    true_average = float(np.mean(x0))

    # Get neighbours
    neighbours = [list(G.neighbors(i)) for i in range(n)]
    s = a.copy()
    w = np.ones(n)

    # Initialize cumulative sums and weights
    sent_s = {(i, j): 0.0 for i in range(n) for j in neighbours[i]}
    sent_w = {(i, j): 0.0 for i in range(n) for j in neighbours[i]}
    recv_s = {(j, i): 0.0 for i in range(n) for j in neighbours[i]}
    recv_w = {(j, i): 0.0 for i in range(n) for j in neighbours[i]}

    # Keep track of history
    errors, transmissions = [], []
    n_tx = 0

    denom = max(np.linalg.norm(a - true_average)**2, 1.0)


    # Run robust sum-weight
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            est = s / w
            errors.append(np.linalg.norm(est - true_average) ** 2 / denom)
            transmissions.append(n_tx)
        if k == n_iterations:
            break

        i = int(rng.integers(n))
        di = len(neighbours[i])
        if di == 0:
            continue

        beta = 1.0 / (di + 1)
        s_out, w_out = beta * s[i], beta * w[i]
        s[i], w[i] = s_out, w_out # keep one share

        for j in neighbours[i]:
            sent_s[(i, j)] += s_out # record in flight (survives loss)
            sent_w[(i, j)] += w_out
            if rng.random() >= packet_loss_prob:
                s[j] += sent_s[(i, j)] - recv_s[(j, i)] # absorb the catch-up
                w[j] += sent_w[(i, j)] - recv_w[(j, i)]
                recv_s[(j, i)] = sent_s[(i, j)]
                recv_w[(j, i)] = sent_w[(i, j)]
            n_tx += 1

    return s / w, np.array(errors), np.array(transmissions)




# Average PDMM
def pdmm_average_broadcast(G, x0, n_iterations=20_000, c=0.4, packet_loss_prob=0.0, alpha=1, seed=None, track_every=10):
    """
    Asynchronous (broadcast) PDMM for average consensus.

    At each iteration:
      1. pick a random node i (uniform)
      2. node i computes its x_i from current incoming z's
      3. node i broadcasts y[i|j] to every neighbour j
      4. each neighbour j updates z[j|i] = -y[i|j]

    Cost per iteration: d_i transmissions (one broadcast to d_i neighbours).
    More transmission-efficient than sync PDMM on dense graphs.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(x0, dtype=float).copy()
    n = len(a)

    # Compute true average for reference
    true_average = float(np.mean(x0))

    # Get neighbours and cardinality
    neighbours = [list(G.neighbors(i)) for i in range(n)]
    degrees = np.array([len(neighbours[i]) for i in range(n)], dtype=float)

    # Initialize z
    z = {(i, j): 0.0 for i in range(n) for j in neighbours[i]}

    # Keep track of history
    errors, transmissions = [], []
    n_tx = 0

    denom = max(np.linalg.norm(a - true_average)**2, 1.0)

    # x update
    def current_x():
        return np.array([(a[i] - sum(z[(i, j)] for j in neighbours[i])) / (1.0 + c * degrees[i]) for i in range(n)])

    # PDMM loop
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            x = current_x()
            errors.append(np.linalg.norm(x - true_average) ** 2 / denom)
            transmissions.append(n_tx)
        if k == n_iterations:
            break

        i = int(rng.integers(n))
        if degrees[i] == 0:
            continue

        x_i = (a[i] - sum(z[(i, j)] for j in neighbours[i])) / (1.0 + c * degrees[i])
        for j in neighbours[i]:
            y_ij = z[(i, j)] + 2.0 * c * x_i
            if rng.random() >= packet_loss_prob:
                z[(j, i)] = (1 - alpha) * z[(j, i)] + alpha * (-y_ij)
            n_tx += 1

    return current_x(), np.array(errors), np.array(transmissions)



# Class to implement Differential quantizer with error feedback (sigma-delta)
class sigma_delta_quantizer():
    def __init__(self, delta, n_bits=None):
        self.delta = delta
        self.q_prev = 0.0      # last reconstructed value (shared by both ends)
        self.e = 0.0           # accumulated rounding error
        self.max_level = (2**(n_bits - 1) - 1) if n_bits else None

    def step(self, v):
        target = v + self.e  # add back previous error
        idx = np.round((target - self.q_prev) / self.delta)
        if self.max_level is not None:
            idx = np.clip(idx, -self.max_level, self.max_level)
        r_q = self.delta * idx
        self.q_prev += r_q
        self.e = target - self.q_prev # leftover -> next step
        return self.q_prev
    


# Quantized randomized gossip
def randomized_gossip_quantized(G, x0, n_iterations, n_bits=8, r_max=10.0, seed=None, track_every=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float).copy(); n = len(x)

    # Get true average for reference
    true_average = float(np.mean(x0))

    # Get edges
    edges = list(G.edges())

    # One quantizer per directed link
    delta = 2 * r_max / (2**n_bits - 1)
    Q = {(i, j): sigma_delta_quantizer(delta, n_bits) for i in range(n) for j in G.neighbors(i)}

    # Keep track of history
    errors, txs, n_tx = [], [], 0

    denom = max(np.linalg.norm(x0 - true_average)**2, 1.0)

    # Run quantized random gosip
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            errors.append(np.linalg.norm(x - true_average)**2 / denom)
            txs.append(n_tx)
        if k == n_iterations: break

        i, j = edges[rng.integers(len(edges))]
        # each node transmits a quantized version of its value
        xi_recv = Q[(i, j)].step(x[i])  
        xj_recv = Q[(j, i)].step(x[j]) 
        x[i] = 0.5*(x[i] + xj_recv)
        x[j] = 0.5*(x[j] + xi_recv)
        n_tx += 2

    return x, np.array(errors), np.array(txs)




# Quantized PDMM
def pdmm_average_broadcast_quantized(G, x0, n_iterations, c=0.4, n_bits=8, r_max=10.0, alpha=1, seed=None, track_every=10):
    rng = np.random.default_rng(seed)
    a = np.asarray(x0, float).copy(); n = len(a)
    
    # Get true average for reference
    true_average = float(np.mean(a))

    # Get neighbours
    nbr = [list(G.neighbors(i)) for i in range(n)]
    deg = np.array([len(nbr[i]) for i in range(n)], float)

    # Initialize z
    z = {(i, j): 0.0 for i in range(n) for j in nbr[i]}

    # Make quantized links between each node
    delta = 2 * r_max / (2**n_bits - 1)
    Q = {(i, j): sigma_delta_quantizer(delta, n_bits) for i in range(n) for j in nbr[i]}

    # Keep track of history
    errors, txs, n_tx = [], [], 0

    denom = max(np.linalg.norm(a - true_average)**2, 1.0)

    def current_x():
        return np.array([(a[i] - sum(z[(i, j)] for j in nbr[i])) / (1 + c*deg[i])
                         for i in range(n)])

    # Run quantized PDMM
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            errors.append(np.linalg.norm(current_x() - true_average)**2 / denom)
            txs.append(n_tx)
        if k == n_iterations: break

        i = int(rng.integers(n))
        if deg[i] == 0: continue
        x_i = (a[i] - sum(z[(i, j)] for j in nbr[i])) / (1 + c*deg[i])
        for j in nbr[i]:
            y_ij = z[(i, j)] + 2*c*x_i
            y_recv = Q[(i, j)].step(y_ij)
            z[(j, i)] = (1 - alpha) * z[(j, i)] + alpha * (-y_recv)
            n_tx += 1

    return current_x(), np.array(errors), np.array(txs)



# Add long range links to the graphs
def add_long_range_links(G, positions, n_links, min_dist=0.0, seed=None):
    """Add n_links random shortcut edges between non-adjacent node pairs."""
    rng = np.random.default_rng(seed)
    G = G.copy()
    n = G.number_of_nodes()
    added = 0
    attempts = 0
    while added < n_links and attempts < 100 * n_links:
        i, j = rng.integers(n), rng.integers(n)
        attempts += 1
        if i == j or G.has_edge(i, j):
            continue
        d = np.linalg.norm(positions[i] - positions[j])
        if d < min_dist: # optionally force them to be "long"
            continue
        G.add_edge(int(i), int(j), weight=float(d))
        added += 1
    return G


# Check when a certain tolerance has been hit
def tx_to_tol(err, tx, tol):
    m = err < tol
    return int(tx[np.argmax(m)]) if m.any() else np.nan



# Helper for median PDMM
def prox_abs_median(a_i, b, q):
    """
    argmin_x  |x - a_i| + b*x + 0.5*q*x^2,   q = c*d_i, b = sum_j z_{i|j}.
    Closed-form solution (soft-threshold at the kink x = a_i).
    """
    x_hi = (-b - 1.0) / q # active branch: derivative +1
    if x_hi > a_i:
        return x_hi
    x_lo = (-b + 1.0) / q # active branch: derivative -1
    if x_lo < a_i:
        return x_lo
    return a_i  # kink: subgradient absorbs the rest



# Median PDMM
def pdmm_median_broadcast(G, x0, n_iterations, c=0.4, packet_loss_prob=0.0, seed=None, track_every=10, alpha=1.0):
    rng = np.random.default_rng(seed)
    a = np.asarray(x0, float).copy(); n = len(a)

    # Get true median for reference
    true_median = float(np.median(a))

    # Get neighbours
    nbr = [list(G.neighbors(i)) for i in range(n)]
    deg = np.array([len(nbr[i]) for i in range(n)], float)
    z = {(i, j): 0.0 for i in range(n) for j in nbr[i]}

    # Keep track of history
    errors, txs, n_tx = [], [], 0

    denom = max(np.linalg.norm(a - true_median)**2, 1.0)

    def current_x():
        return np.array([prox_abs_median(a[i], sum(z[(i, j)] for j in nbr[i]), c*deg[i]) if deg[i] > 0 else a[i] for i in range(n)])

    # Run median PDMM
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            errors.append(np.linalg.norm(current_x() - true_median)**2 / denom)
            txs.append(n_tx)
        if k == n_iterations: break

        i = int(rng.integers(n))
        if deg[i] == 0: continue
        b = sum(z[(i, j)] for j in nbr[i])
        x_i = prox_abs_median(a[i], b, c*deg[i])
        for j in nbr[i]:
            y_ij = z[(i, j)] + 2*c*x_i
            if rng.random() >= packet_loss_prob:
                z[(j, i)] = (1-alpha)*z[(j, i)] + alpha*(-y_ij)
            n_tx += 1

    return current_x(), np.array(errors), np.array(txs)



# Quantized median PDMM
def pdmm_median_quantized(G, x0, n_iterations, c=0.4, n_bits=8, r_max=10.0, seed=None, track_every=10, alpha=1):
    """Broadcast median PDMM with sigma-delta quantization."""
    rng = np.random.default_rng(seed)
    a = np.asarray(x0, float).copy(); n = len(a)
    
    # Calculate true median for reference
    true_median = float(np.median(a))

    # Get neighbours
    nbr = [list(G.neighbors(i)) for i in range(n)]
    deg = np.array([len(nbr[i]) for i in range(n)], float)

    # Initialize z
    z = {(i, j): 0.0 for i in range(n) for j in nbr[i]}

    # Quantize
    delta = 2 * r_max / (2**n_bits - 1)
    Q = {(i, j): sigma_delta_quantizer(delta, n_bits) for i in range(n) for j in nbr[i]}

    # Keep track of history
    errors, txs, n_tx = [], [], 0

    denom = max(np.linalg.norm(a - true_median)**2, 1.0)

    def current_x():
        return np.array([prox_abs_median(a[i], sum(z[(i, j)] for j in nbr[i]), c*deg[i]) if deg[i] > 0 else a[i] for i in range(n)])

    # Run quantized median PDMM
    for k in range(n_iterations + 1):
        if k % track_every == 0:
            errors.append(np.linalg.norm(current_x() - true_median)**2 / denom)
            txs.append(n_tx)
        if k == n_iterations: break

        i = int(rng.integers(n))
        if deg[i] == 0: continue
        b = sum(z[(i, j)] for j in nbr[i])
        x_i = prox_abs_median(a[i], b, c*deg[i])
        for j in nbr[i]:
            y_ij = z[(i, j)] + 2*c*x_i
            y_recv = Q[(i, j)].step(y_ij)
            z[(j, i)] = (1 - alpha) * z[(j, i)] + alpha * (-y_recv)
            n_tx += 1

    return current_x(), np.array(errors), np.array(txs)