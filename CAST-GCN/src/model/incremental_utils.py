"""
Incremental GNN Utilities — Ported from DynamicGNN-main notebook.

Provides:
- Adjacency structure building with self-loops
- Full and incremental AX (normalized adjacency × features) computation
- k-hop subgraph extraction
- SVD-based row importance and gradient masking
"""

import torch
import numpy as np
import networkx as nx
from collections import defaultdict


# =========================================================================
# Adjacency & AX Cache
# =========================================================================

def build_adj_structures(edge_index, num_nodes):
    """
    Build adjacency lists (including self-loop) and degrees from PyG edge_index.

    Returns:
        adjacency: dict node -> set(neighbors incl self)
        deg: dict node -> degree (including self-loop)
        nodes_sorted: list of node indices
        node_to_idx: identity mapping (since nodes are 0-indexed)
    """
    nodes_sorted = list(range(num_nodes))
    node_to_idx = {n: n for n in nodes_sorted}

    adjacency = defaultdict(set)
    # Add self-loops
    for n in nodes_sorted:
        adjacency[n].add(n)

    # Add edges
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adjacency[s].add(d)
        adjacency[d].add(s)

    deg = {n: len(adjacency[n]) for n in nodes_sorted}

    return adjacency, deg, nodes_sorted, node_to_idx


def compute_AX_full(adjacency, deg, nodes_sorted, node_to_idx, X):
    """
    Compute A_hat @ X where A_hat = D^{-1/2} A D^{-1/2} (A includes self-loops).
    X: tensor of shape (N, F). Returns tensor (N, F).

    This is a row-wise computation for correctness. For large graphs,
    this can be slow but is exact.
    """
    if isinstance(X, np.ndarray):
        X_t = torch.tensor(X, dtype=torch.float)
    else:
        X_t = X.clone().detach().cpu().float()

    N, F = X_t.shape
    AX = torch.zeros(N, F)

    # Precompute inverse sqrt degrees
    inv_sqrt_deg = {}
    for n in nodes_sorted:
        d = deg.get(n, 1)
        inv_sqrt_deg[n] = 1.0 / np.sqrt(max(d, 1e-12))

    for i, n in enumerate(nodes_sorted):
        inv_i = inv_sqrt_deg[n]
        acc = torch.zeros(F)
        for m in adjacency.get(n, {n}):
            j = node_to_idx[m]
            acc += inv_i * inv_sqrt_deg.get(m, 1.0) * X_t[j]
        AX[i] = acc

    return AX


def compute_AX_sparse(edge_index, num_nodes, X, device=None):
    """
    Compute A_hat @ X using sparse matrix multiplication.
    Fully vectorized and runs on target device.
    """
    if device is None:
        device = X.device if torch.is_tensor(X) else torch.device('cpu')
        
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float, device=device)
    else:
        X = X.to(device).float()

    edge_index = edge_index.to(device)

    # Add self-loops
    self_loops = torch.arange(num_nodes, dtype=torch.long, device=device)
    self_loop_idx = torch.stack([self_loops, self_loops], dim=0)
    edge_index_sl = torch.cat([edge_index, self_loop_idx], dim=1)

    # Make undirected (if not already)
    edge_index_sl = torch.cat([edge_index_sl, edge_index_sl.flip(0)], dim=1)
    edge_index_sl = torch.unique(edge_index_sl, dim=1)

    # Compute degree
    row = edge_index_sl[0]
    deg = torch.zeros(num_nodes, device=device)
    ones = torch.ones(edge_index_sl.shape[1], device=device)
    deg.scatter_add_(0, row, ones)

    # D^{-1/2}
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # Normalize: edge_weight = D^{-1/2}[src] * D^{-1/2}[dst]
    edge_weight = deg_inv_sqrt[edge_index_sl[0]] * deg_inv_sqrt[edge_index_sl[1]]

    # Sparse matmul
    adj = torch.sparse_coo_tensor(edge_index_sl, edge_weight, (num_nodes, num_nodes), device=device)
    AX = torch.sparse.mm(adj, X)

    return AX


def update_AX_rows(adjacency, deg, nodes_sorted, node_to_idx, X, AX, rows_to_update):
    """
    Incrementally recompute AX rows for affected nodes and their neighbors.
    """
    if isinstance(X, np.ndarray):
        X_t = torch.tensor(X, dtype=torch.float)
    else:
        X_t = X.clone().cpu().float()

    AX_new = AX.clone()

    # Recompute degrees
    for n in adjacency:
        deg[n] = len(adjacency[n])

    inv_sqrt_deg = {n: 1.0 / np.sqrt(max(deg[n], 1e-12)) for n in adjacency}

    # Expand to include normalization dependents
    expanded = set(rows_to_update)
    for n in rows_to_update:
        for nbr in adjacency.get(n, set()):
            expanded.add(nbr)

    # Rebuild AX rows
    for n in expanded:
        i = node_to_idx[n]
        if i >= AX_new.shape[0]:
            continue
        acc = torch.zeros_like(AX_new[i])
        inv_i = inv_sqrt_deg.get(n, 1.0)
        for m in adjacency.get(n, {n}):
            j = node_to_idx.get(m, None)
            if j is not None and j < X_t.shape[0]:
                acc += inv_i * inv_sqrt_deg.get(m, 1.0) * X_t[j]
        AX_new[i] = acc

    return AX_new


def nodes_for_AX_update(adjacency, changed_nodes):
    """Return set of nodes whose AX rows must be recomputed."""
    rows = set()
    for n in changed_nodes:
        rows.add(n)
        rows.update(adjacency.get(n, []))
    return rows


# =========================================================================
# Subgraph Extraction
# =========================================================================

def build_k_hop_subgraph_from_edge_index(edge_index, affected_nodes, num_nodes, k=2):
    """
    Build k-hop subgraph around affected nodes using edge_index.

    Returns:
        sub_nodes: sorted list of node indices in the subgraph
        sub_edge_index: edge_index for the subgraph (remapped to local indices)
        local_to_global: mapping from local to global node indices
        global_to_local: mapping from global to local node indices
    """
    # Build adjacency list
    adj = defaultdict(set)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].add(d)
        adj[d].add(s)

    frontier = set(affected_nodes)
    visited = set(frontier)

    for _ in range(k):
        new_nodes = set()
        for u in frontier:
            new_nodes.update(adj.get(u, set()))
        new_nodes -= visited
        visited |= new_nodes
        frontier = new_nodes

    sub_nodes = sorted(visited)
    global_to_local = {g: l for l, g in enumerate(sub_nodes)}

    # Build sub edge_index
    sub_src = []
    sub_dst = []
    sub_set = set(sub_nodes)
    for s, d in zip(src, dst):
        if s in sub_set and d in sub_set:
            sub_src.append(global_to_local[s])
            sub_dst.append(global_to_local[d])

    if sub_src:
        sub_edge_index = torch.tensor([sub_src, sub_dst], dtype=torch.long)
    else:
        sub_edge_index = torch.zeros(2, 0, dtype=torch.long)

    return sub_nodes, sub_edge_index, sub_nodes, global_to_local


# =========================================================================
# SVD-Based Row Importance
# =========================================================================

def compute_important_rows(W, k=5, top_k=5):
    """
    Compute importance scores for weight matrix rows using SVD leverage scores.

    Args:
        W: weight tensor (m x n)
        k: number of top singular vectors to use
        top_k: number of important rows to return

    Returns:
        indices of top-k important rows
    """
    U, S, Vh = torch.linalg.svd(W.detach(), full_matrices=False)
    k = min(k, U.shape[1])
    U_k = U[:, :k]

    # Leverage scores
    scores = torch.sum(U_k ** 2, dim=1)
    top_k = min(top_k, scores.shape[0])

    important_rows = torch.topk(scores, top_k).indices
    return important_rows


def mask_gradients(W, important_rows):
    """Zero out gradients for all rows except important ones."""
    if W.grad is None:
        return
    mask = torch.zeros_like(W.grad)
    mask[important_rows, :] = 1.0
    W.grad *= mask


def compute_weight_svd(model, tag="model"):
    """Compute SVD of all Linear layers inside the model."""
    svd_dict = {}
    for name, param in model.named_parameters():
        if "weight" in name and len(param.shape) == 2:
            W = param.detach().cpu()
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                svd_dict[name] = S.numpy()
            except Exception as e:
                print(f"[{tag}] SVD failed for {name}: {e}")
    return svd_dict


def compare_svd(svd_old, svd_new, tag="comparison"):
    """Compare SVD singular values between two model states."""
    drifts = {}
    for key in svd_old:
        if key in svd_new:
            min_len = min(len(svd_old[key]), len(svd_new[key]))
            diff = np.linalg.norm(svd_old[key][:min_len] - svd_new[key][:min_len])
            drifts[key] = diff
    return drifts


# =========================================================================
# Edge Diff Computation
# =========================================================================

def compute_edge_diff(old_edge_index, new_edge_index, num_nodes=None):
    """
    Compute added and removed edges between two snapshots.
    Vectorized using 1D integer hashing for extreme speed.
    """
    if num_nodes is None:
        num_nodes = max(
            old_edge_index.max().item() if old_edge_index.numel() > 0 else 0,
            new_edge_index.max().item() if new_edge_index.numel() > 0 else 0
        ) + 1

    device = old_edge_index.device
    new_edge_index = new_edge_index.to(device)

    old_hash = old_edge_index[0] * num_nodes + old_edge_index[1]
    new_hash = new_edge_index[0] * num_nodes + new_edge_index[1]

    # Added edges
    added_mask = ~torch.isin(new_hash, old_hash)
    added_edges_tensor = new_edge_index[:, added_mask]
    added_edges = added_edges_tensor.t().tolist()

    # Removed edges
    removed_mask = ~torch.isin(old_hash, new_hash)
    removed_edges_tensor = old_edge_index[:, removed_mask]
    removed_edges = removed_edges_tensor.t().tolist()

    # Affected nodes
    affected_nodes = set()
    if added_edges_tensor.numel() > 0:
        affected_nodes.update(added_edges_tensor.flatten().tolist())
    if removed_edges_tensor.numel() > 0:
        affected_nodes.update(removed_edges_tensor.flatten().tolist())

    return added_edges, removed_edges, affected_nodes
