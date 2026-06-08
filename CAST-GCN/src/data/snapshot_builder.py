"""
Snapshot Builder: Convert continuous temporal edges into discrete graph snapshots.

Supports two modes:
  1. Cumulative: Given temporal edges (u, v, t), divides the time range into N
     equal-duration windows and builds cumulative PyG Data objects.
  2. Prebuilt: Loads pre-computed snapshots (from preprocess_snapshot_datasets.py)
     that already contain edge_index, new_edges, and removed_edges per snapshot.
     Used for datasets with edge deletions (AS-733, AS-CAIDA, SocioPatterns).
"""

import torch
import numpy as np
from torch_geometric.data import Data


def build_snapshots(edge_file, num_snapshots=50, feature_dim=32):
    """
    Build graph snapshots from a temporal edge file.
    Auto-detects whether the file contains prebuilt snapshots (with deletions)
    or a raw edge stream (cumulative only).

    Args:
        edge_file: Path to .pt file
        num_snapshots: Number of time windows to create (ignored for prebuilt)
        feature_dim: Dimension of node feature embeddings

    Returns:
        list of PyG Data objects (one per snapshot), metadata dict
    """
    data = torch.load(edge_file, weights_only=False)

    # Dispatch to prebuilt loader if applicable
    if data.get('snapshot_mode') == 'prebuilt':
        return build_snapshots_prebuilt(data, feature_dim)

    u_all = data['u'].long()
    v_all = data['v'].long()
    t_all = data['t'].float()
    num_nodes = int(data.get('num_nodes', max(u_all.max(), v_all.max()) + 1))

    # Sort by time
    sort_idx = torch.argsort(t_all)
    u_all = u_all[sort_idx]
    v_all = v_all[sort_idx]
    t_all = t_all[sort_idx]

    t_min = t_all[0].item()
    t_max = t_all[-1].item()
    time_range = t_max - t_min

    # Divide into equal-duration windows
    boundaries = [t_min + (i + 1) * time_range / num_snapshots for i in range(num_snapshots)]

    snapshots = []
    metadata = {
        'num_nodes': num_nodes,
        'num_snapshots': num_snapshots,
        'feature_dim': feature_dim,
        't_min': t_min,
        't_max': t_max,
        'edges_per_snapshot': [],
        'cumulative_edges': [],
    }

    # Assign each edge to a time window
    edge_window = torch.zeros(len(t_all), dtype=torch.long)
    for i, boundary in enumerate(boundaries):
        if i == 0:
            mask = t_all <= boundary
        else:
            mask = (t_all > boundaries[i - 1]) & (t_all <= boundary)
        edge_window[mask] = i

    # Any edge after last boundary goes to last window
    edge_window[t_all > boundaries[-1]] = num_snapshots - 1

    for snap_idx in range(num_snapshots):
        # Cumulative: all edges up to and including this snapshot
        cum_mask = edge_window <= snap_idx
        u_snap = u_all[cum_mask]
        v_snap = v_all[cum_mask]

        # New edges in this snapshot only
        new_mask = edge_window == snap_idx
        u_new = u_all[new_mask]
        v_new = v_all[new_mask]

        # Build undirected edge_index (add both directions)
        if len(u_snap) > 0:
            edge_index = torch.stack([
                torch.cat([u_snap, v_snap]),
                torch.cat([v_snap, u_snap])
            ], dim=0)
            # Remove self-loops
            non_self = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, non_self]
            # Remove duplicates
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        # New edges (for evaluation — these are the ones we want to predict)
        if len(u_new) > 0:
            new_edge_index = torch.stack([
                torch.cat([u_new, v_new]),
                torch.cat([v_new, u_new])
            ], dim=0)
            non_self = new_edge_index[0] != new_edge_index[1]
            new_edge_index = new_edge_index[:, non_self]
            new_edge_index = torch.unique(new_edge_index, dim=1)
        else:
            new_edge_index = torch.zeros(2, 0, dtype=torch.long)

        snap_data = Data(
            edge_index=edge_index,
            num_nodes=num_nodes,
            new_edges=new_edge_index,
            snapshot_idx=snap_idx,
        )

        metadata['edges_per_snapshot'].append(int(new_edge_index.shape[1] // 2))
        metadata['cumulative_edges'].append(int(edge_index.shape[1] // 2))

        snapshots.append(snap_data)

    print(f"Built {num_snapshots} snapshots from {len(u_all)} temporal edges")
    print(f"  Nodes: {num_nodes}")
    print(f"  Time range: {t_min:.0f} — {t_max:.0f}")
    print(f"  Edges per snapshot (new): min={min(metadata['edges_per_snapshot'])}, "
          f"max={max(metadata['edges_per_snapshot'])}, "
          f"avg={np.mean(metadata['edges_per_snapshot']):.0f}")
    print(f"  Cumulative edges: {metadata['cumulative_edges'][0]} → {metadata['cumulative_edges'][-1]}")

    return snapshots, metadata


def build_snapshots_prebuilt(data, feature_dim=32):
    """
    Load prebuilt snapshots from a .pt file produced by
    preprocess_snapshot_datasets.py.

    Each snapshot already contains edge_index, new_edges, and removed_edges.
    This is used for datasets with edge deletions (AS-733, AS-CAIDA, etc.).

    Args:
        data: dict loaded from .pt file with keys:
              'snapshots', 'num_nodes', 'num_snapshots', 'has_deletions'
        feature_dim: Dimension of node features

    Returns:
        list of PyG Data objects, metadata dict
    """
    raw_snaps = data['snapshots']
    num_nodes = data['num_nodes']
    num_snapshots = data['num_snapshots']
    has_deletions = data.get('has_deletions', False)

    snapshots = []
    metadata = {
        'num_nodes': num_nodes,
        'num_snapshots': num_snapshots,
        'feature_dim': feature_dim,
        'has_deletions': has_deletions,
        'snapshot_mode': 'prebuilt',
        'edges_per_snapshot': [],
        'added_per_snapshot': [],
        'removed_per_snapshot': [],
    }

    for s in raw_snaps:
        edge_index = s['edge_index']
        new_edges = s['new_edges']
        removed_edges = s.get('removed_edges', torch.zeros(2, 0, dtype=torch.long))

        snap_data = Data(
            edge_index=edge_index,
            num_nodes=num_nodes,
            new_edges=new_edges,
            removed_edges=removed_edges,
            snapshot_idx=s['snapshot_idx'],
        )
        snapshots.append(snap_data)

        metadata['edges_per_snapshot'].append(int(edge_index.shape[1] // 2))
        metadata['added_per_snapshot'].append(int(new_edges.shape[1] // 2))
        metadata['removed_per_snapshot'].append(int(removed_edges.shape[1] // 2))

    total_added = sum(metadata['added_per_snapshot'])
    total_removed = sum(metadata['removed_per_snapshot'])

    print(f"Loaded {num_snapshots} prebuilt snapshots ({num_nodes:,} nodes)")
    print(f"  Edges per snapshot: min={min(metadata['edges_per_snapshot']):,}, "
          f"max={max(metadata['edges_per_snapshot']):,}, "
          f"avg={np.mean(metadata['edges_per_snapshot']):.0f}")
    print(f"  Total additions: {total_added:,}, deletions: {total_removed:,}")
    if has_deletions:
        print(f"  ⚡ Dataset has edge DELETIONS (non-cumulative)")

    return snapshots, metadata


def compute_node_features(snapshot, num_nodes, feature_dim=32):
    """
    Compute structural node features from the graph snapshot.
    Uses degree + random positional encoding.

    Args:
        snapshot: PyG Data object with edge_index
        num_nodes: Total number of nodes
        feature_dim: Feature dimension

    Returns:
        Tensor of shape (num_nodes, feature_dim)
    """
    edge_index = snapshot.edge_index

    # Compute degree
    degree = torch.zeros(num_nodes)
    if edge_index.shape[1] > 0:
        unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
        degree[unique_nodes] = counts.float()

    # Normalize degree
    max_deg = degree.max().clamp(min=1.0)
    norm_degree = degree / max_deg

    # Log degree
    log_degree = torch.log1p(degree)
    log_degree = log_degree / log_degree.max().clamp(min=1.0)

    # Build feature matrix: [normalized_degree, log_degree, ...]
    # Pad with learnable positional encoding dimensions
    features = torch.zeros(num_nodes, feature_dim)
    features[:, 0] = norm_degree
    features[:, 1] = log_degree

    # Add some structural noise for remaining dimensions
    # (these will be refined during training via backprop)
    if feature_dim > 2:
        torch.manual_seed(42)
        features[:, 2:] = torch.randn(num_nodes, feature_dim - 2) * 0.01

    return features


if __name__ == '__main__':
    import os
    import sys

    edge_file = 'processed/superuser_compact.pt'
    if not os.path.exists(edge_file):
        print(f"Error: {edge_file} not found. Run preprocess.py first.")
        sys.exit(1)

    snapshots, metadata = build_snapshots(edge_file, num_snapshots=50, feature_dim=32)

    # Add node features to each snapshot
    print("\nComputing node features...")
    for snap in snapshots:
        snap.x = compute_node_features(snap, metadata['num_nodes'], feature_dim=32)

    # Save
    save_path = 'processed/askubuntu_snapshots.pt'
    torch.save({
        'snapshots': snapshots,
        'metadata': metadata,
    }, save_path)
    print(f"\nSaved {len(snapshots)} snapshots to {save_path}")

    # Print summary
    print(f"\nTrain snapshots (0-39): cumulative edges range "
          f"{metadata['cumulative_edges'][0]} — {metadata['cumulative_edges'][39]}")
    print(f"Test snapshots (40-49): cumulative edges range "
          f"{metadata['cumulative_edges'][40]} — {metadata['cumulative_edges'][49]}")
