"""
Preprocessor for snapshot-format datasets (AS-733, AS-CAIDA) and
contact-stream datasets (SocioPatterns Primary School).

These datasets naturally exhibit edge DELETIONS between snapshots,
unlike the cumulative StackExchange/WikiTalk datasets.

Usage:
    python preprocess_snapshot_datasets.py --name as733
    python preprocess_snapshot_datasets.py --name ascaida
    python preprocess_snapshot_datasets.py --name primaryschool
    python preprocess_snapshot_datasets.py --all
"""

import argparse
import os
import glob
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


# ── Dataset definitions ──────────────────────────────────────────
RAW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'scripts', 'data', 'raw')


def parse_as733(raw_dir, num_snapshots=50):
    """
    Parse AS-733 dataset: 733 daily edge-list files.
    Each file is a complete snapshot of the AS graph for that day.
    Subsample to `num_snapshots` equally-spaced snapshots.

    Returns list of (edge_set, filename) tuples.
    """
    pattern = os.path.join(raw_dir, 'as-733', 'as*.txt')
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"No AS-733 files found in {raw_dir}/as-733/"
    print(f"  Found {len(files)} daily snapshots")

    # Subsample to num_snapshots equally-spaced
    indices = np.linspace(0, len(files) - 1, num_snapshots, dtype=int)
    files = [files[i] for i in indices]
    print(f"  Subsampled to {len(files)} snapshots")

    snapshots = []
    all_nodes = set()

    for f in tqdm(files, desc="Parsing AS-733"):
        edges = set()
        with open(f, 'r') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    if u != v:  # skip self-loops
                        edges.add((min(u, v), max(u, v)))  # undirected
                        all_nodes.add(u)
                        all_nodes.add(v)
        snapshots.append((edges, os.path.basename(f)))

    return snapshots, all_nodes


def parse_ascaida(raw_dir, num_snapshots=50):
    """
    Parse AS-CAIDA dataset: 122 periodic edge-list files.
    Each file has columns: FromNodeId  ToNodeId  Relationship
    We treat all relationships as undirected edges.
    Subsample to min(num_snapshots, 122) equally-spaced snapshots.
    """
    pattern = os.path.join(raw_dir, 'as-caida', 'as-caida*.txt')
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"No AS-CAIDA files found in {raw_dir}/as-caida/"
    print(f"  Found {len(files)} periodic snapshots")

    n = min(num_snapshots, len(files))
    indices = np.linspace(0, len(files) - 1, n, dtype=int)
    files = [files[i] for i in indices]
    print(f"  Subsampled to {len(files)} snapshots")

    snapshots = []
    all_nodes = set()

    for f in tqdm(files, desc="Parsing AS-CAIDA"):
        edges = set()
        with open(f, 'r') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    if u != v:
                        edges.add((min(u, v), max(u, v)))
                        all_nodes.add(u)
                        all_nodes.add(v)
        snapshots.append((edges, os.path.basename(f)))

    return snapshots, all_nodes


def parse_primaryschool(raw_dir, num_snapshots=50):
    """
    Parse SocioPatterns Primary School contact network.
    Format: tab-separated "t i j Ci Cj"
    Contacts are 20-second intervals. We slice into num_snapshots windows.
    Each window contains ONLY the edges active in that window (non-cumulative).
    """
    csv_path = os.path.join(raw_dir, 'primaryschool.csv')
    gz_path = csv_path + '.gz'

    if os.path.exists(csv_path):
        path = csv_path
    elif os.path.exists(gz_path):
        path = gz_path
    else:
        raise FileNotFoundError(
            f"Primary school data not found. Expected {csv_path} or {gz_path}\n"
            f"Download from: http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/"
        )

    print(f"  Reading {path}")
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['t', 'i', 'j', 'Ci', 'Cj'],
                     compression='gzip' if path.endswith('.gz') else None)

    print(f"  Loaded {len(df):,} contact events")

    t_min, t_max = df['t'].min(), df['t'].max()
    time_range = t_max - t_min

    # Divide into windows
    boundaries = [t_min + (k + 1) * time_range / num_snapshots
                  for k in range(num_snapshots)]

    snapshots = []
    all_nodes = set()

    for k in tqdm(range(num_snapshots), desc="Building windows"):
        lo = t_min if k == 0 else boundaries[k - 1]
        hi = boundaries[k]
        mask = (df['t'] > lo) & (df['t'] <= hi) if k > 0 else (df['t'] <= hi)
        sub = df[mask]

        edges = set()
        for _, row in sub.iterrows():
            u, v = int(row['i']), int(row['j'])
            if u != v:
                edges.add((min(u, v), max(u, v)))
                all_nodes.add(u)
                all_nodes.add(v)

        snapshots.append((edges, f"window_{k}"))

    return snapshots, all_nodes


def build_and_save(name, snapshots_raw, all_nodes):
    """
    Convert raw snapshot data to .pt format with edge diffs.

    For each snapshot, compute:
      - edge_index: all edges in this snapshot (undirected)
      - new_edges: edges ADDED since previous snapshot
      - removed_edges: edges REMOVED since previous snapshot
    """
    # Remap nodes to contiguous 0-indexed
    node_list = sorted(all_nodes)
    node_map = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    print(f"  Total unique nodes: {num_nodes:,}")

    snapshot_data = []
    prev_edges = set()

    for snap_idx, (edges, label) in enumerate(snapshots_raw):
        # Remap edges
        remapped = set()
        for u, v in edges:
            if u in node_map and v in node_map:
                ru, rv = node_map[u], node_map[v]
                remapped.add((min(ru, rv), max(ru, rv)))

        # Compute diffs
        added = remapped - prev_edges
        removed = prev_edges - remapped

        # Build undirected edge_index
        if remapped:
            src = [u for u, v in remapped] + [v for u, v in remapped]
            dst = [v for u, v in remapped] + [u for u, v in remapped]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Build new_edges (added)
        if added:
            a_src = [u for u, v in added] + [v for u, v in added]
            a_dst = [v for u, v in added] + [u for u, v in added]
            new_edge_index = torch.tensor([a_src, a_dst], dtype=torch.long)
            new_edge_index = torch.unique(new_edge_index, dim=1)
        else:
            new_edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Build removed_edges
        if removed:
            r_src = [u for u, v in removed] + [v for u, v in removed]
            r_dst = [v for u, v in removed] + [u for u, v in removed]
            removed_edge_index = torch.tensor([r_src, r_dst], dtype=torch.long)
            removed_edge_index = torch.unique(removed_edge_index, dim=1)
        else:
            removed_edge_index = torch.zeros(2, 0, dtype=torch.long)

        snapshot_data.append({
            'edge_index': edge_index,
            'new_edges': new_edge_index,
            'removed_edges': removed_edge_index,
            'snapshot_idx': snap_idx,
            'label': label,
        })

        prev_edges = remapped

    # Summary stats
    total_added = sum(s['new_edges'].shape[1] // 2 for s in snapshot_data)
    total_removed = sum(s['removed_edges'].shape[1] // 2 for s in snapshot_data)
    edge_counts = [s['edge_index'].shape[1] // 2 for s in snapshot_data]

    print(f"  Snapshots: {len(snapshot_data)}")
    print(f"  Edges per snapshot: min={min(edge_counts):,}, max={max(edge_counts):,}, "
          f"avg={np.mean(edge_counts):.0f}")
    print(f"  Total additions: {total_added:,}")
    print(f"  Total deletions: {total_removed:,}")
    print(f"  Deletion ratio: {total_removed / max(total_added, 1):.2%}")

    # Save
    os.makedirs("processed", exist_ok=True)
    out_path = os.path.join("processed", f"{name}_snapshots.pt")
    torch.save({
        'snapshots': snapshot_data,
        'num_nodes': num_nodes,
        'num_snapshots': len(snapshot_data),
        'snapshot_mode': 'prebuilt',  # signals to snapshot_builder
        'has_deletions': True,
    }, out_path)
    print(f"  ✅ Saved → {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────

DATASET_PARSERS = {
    'as733': ('AS-733 (Autonomous Systems, 733 daily)', parse_as733),
    'ascaida': ('AS-CAIDA (Autonomous Systems, 122 periodic)', parse_ascaida),
    'primaryschool': ('SocioPatterns Primary School', parse_primaryschool),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess snapshot/contact datasets")
    parser.add_argument("--name", type=str, default=None,
                        choices=list(DATASET_PARSERS.keys()),
                        help="Dataset name")
    parser.add_argument("--num_snapshots", type=int, default=50,
                        help="Number of snapshots to produce (default: 50)")
    parser.add_argument("--all", action="store_true",
                        help="Process all datasets")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.all:
        names = list(DATASET_PARSERS.keys())
    elif args.name:
        names = [args.name]
    else:
        parser.error("Provide --name or --all")

    for name in names:
        desc, parse_fn = DATASET_PARSERS[name]
        print(f"\n{'=' * 60}")
        print(f"Processing: {desc}")
        print(f"{'=' * 60}")
        try:
            snapshots_raw, all_nodes = parse_fn(RAW_ROOT, args.num_snapshots)
            build_and_save(name, snapshots_raw, all_nodes)
        except (FileNotFoundError, AssertionError) as e:
            print(f"  ⚠️  Skipping {name}: {e}")
