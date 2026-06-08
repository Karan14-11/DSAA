"""
Preprocess 5 diverse datasets into temporal graph snapshots for DyCond.

Datasets:
  1. CollegeMsg      — social messaging (temporal)
  2. Bitcoin-Alpha   — trust/financial (temporal)
  3. Euroroad        — road network (static → simulated temporal)
  4. PP-Pathways     — protein interaction (static → simulated temporal)
  5. Cit-HepPh       — physics citation (temporal via dates)

Each produces 50 snapshots saved as:
    data/processed/{name}/snap_{i}.pt

Usage:
    python preprocess_all_datasets.py
    python preprocess_all_datasets.py --dataset collegemsg
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# Raw data directory (relative to this script)
# ═══════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, 'data', 'raw')
OUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')


# ═══════════════════════════════════════════════════════════════
# 1. CollegeMsg — space-separated: SRC DST UNIXTS
# ═══════════════════════════════════════════════════════════════
def load_collegemsg():
    path = os.path.join(RAW_DIR, 'CollegeMsg.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"CollegeMsg.txt not found at {path}")
    df = pd.read_csv(path, sep=' ', header=None, names=['u', 'v', 't'], comment='#')
    print(f"  CollegeMsg: {len(df)} edges loaded")
    return df, 'window'  # social = window mode


# ═══════════════════════════════════════════════════════════════
# 2. Bitcoin-Alpha — comma-separated: SOURCE,TARGET,RATING,TIME
# ═══════════════════════════════════════════════════════════════
def load_bitcoin_alpha():
    path = os.path.join(RAW_DIR, 'soc-sign-bitcoinalpha.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"soc-sign-bitcoinalpha.csv not found at {path}")
    df = pd.read_csv(path, header=None, names=['u', 'v', 'rating', 't'], comment='#')
    df = df[['u', 'v', 't']]  # drop rating
    print(f"  Bitcoin-Alpha: {len(df)} edges loaded")
    return df, 'cumulative'  # trust persists


# ═══════════════════════════════════════════════════════════════
# 3. Euroroad — MTX-like format (static graph)
# ═══════════════════════════════════════════════════════════════
def load_euroroad():
    path = os.path.join(RAW_DIR, 'road-euroroad.edges')
    if not os.path.exists(path):
        # Try .mtx format
        path = os.path.join(RAW_DIR, 'road-euroroad.mtx')
        if not os.path.exists(path):
            raise FileNotFoundError(f"road-euroroad.edges/.mtx not found in {RAW_DIR}")

    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    edges.append((u, v))
                except ValueError:
                    continue

    df = pd.DataFrame(edges, columns=['u', 'v'])
    # Simulate temporal ordering: assign pseudo-timestamps by edge index
    # (simulates network discovery/construction)
    df['t'] = np.arange(len(df), dtype=float)
    print(f"  Euroroad: {len(df)} edges loaded (static → simulated temporal)")
    return df, 'cumulative'  # road edges persist once built


# ═══════════════════════════════════════════════════════════════
# 4. PP-Pathways — CSV: Gene1,Gene2 (static, CR line endings)
# ═══════════════════════════════════════════════════════════════
def load_pp_pathways():
    path = os.path.join(RAW_DIR, 'PP-Pathways_ppi.csv')
    if not os.path.exists(path):
        # Try Decagon variant
        path = os.path.join(RAW_DIR, 'PP-Decagon_ppi.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"PP-Pathways_ppi.csv not found in {RAW_DIR}")

    # Handle carriage-return line endings
    with open(path, 'rb') as f:
        content = f.read().decode('utf-8')

    lines = content.replace('\r\n', '\n').replace('\r', '\n').strip().split('\n')

    edges = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('Gene'):
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
            except ValueError:
                continue

    df = pd.DataFrame(edges, columns=['u', 'v'])

    # Subsample to top N nodes by degree for tractability
    MAX_NODES = 5000
    all_nodes = pd.concat([df['u'], df['v']])
    top_nodes = set(all_nodes.value_counts().head(MAX_NODES).index)
    df = df[df['u'].isin(top_nodes) & df['v'].isin(top_nodes)].copy()

    # Simulate temporal ordering: shuffle edges then assign timestamps
    # (simulates protein interaction discovery over time)
    np.random.seed(42)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['t'] = np.arange(len(df), dtype=float)

    print(f"  PP-Pathways: {len(df)} edges loaded (subsampled to {MAX_NODES} nodes, static → simulated temporal)")
    return df, 'cumulative'  # interactions persist once discovered


# ═══════════════════════════════════════════════════════════════
# 5. Cit-HepPh — citation with dates (temporal)
# ═══════════════════════════════════════════════════════════════
def load_cit_hepph():
    edge_path = os.path.join(RAW_DIR, 'Cit-HepPh.txt')
    date_path = os.path.join(RAW_DIR, 'cit-HepPh-dates.txt')
    if not os.path.exists(edge_path) or not os.path.exists(date_path):
        raise FileNotFoundError(f"Cit-HepPh files not found in {RAW_DIR}")

    # Load dates
    dates = pd.read_csv(date_path, sep='\t', header=None, comment='#',
                        names=['node_id', 'date'])
    dates['date'] = pd.to_datetime(dates['date'], errors='coerce')
    dates = dates.dropna()
    node_to_time = pd.Series(dates['date'].values, index=dates['node_id']).to_dict()

    # Load edges
    edges = pd.read_csv(edge_path, sep='\t', header=None, comment='#',
                        names=['u', 'v'])

    # Map edge time = publication date of source paper
    edges['time'] = edges['u'].map(node_to_time)
    edges = edges.dropna(subset=['time'])

    # Convert to unix timestamp
    edges['t'] = edges['time'].astype(np.int64) // 10**9
    df = edges[['u', 'v', 't']].copy()

    # Subsample to top N nodes
    MAX_NODES = 20000
    all_nodes = pd.concat([df['u'], df['v']])
    top_nodes = set(all_nodes.value_counts().head(MAX_NODES).index)
    df = df[df['u'].isin(top_nodes) & df['v'].isin(top_nodes)].copy()

    print(f"  Cit-HepPh: {len(df)} edges loaded (subsampled to {MAX_NODES} nodes)")
    return df, 'cumulative'  # citations persist


# ═══════════════════════════════════════════════════════════════
# Snapshot Generator
# ═══════════════════════════════════════════════════════════════
def generate_snapshots(name, df, output_dir, num_snapshots=50, max_nodes=None, mode='window'):
    """
    Convert edge DataFrame to PyG snapshot files.

    Args:
        name: dataset name (used for output directory)
        df: DataFrame with columns [u, v, t]
        output_dir: base output directory
        num_snapshots: number of time windows
        max_nodes: cap on number of nodes (None = use all)
        mode: 'window' (edges exist only in their bucket) or
              'cumulative' (edges persist into future buckets)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {name} | Mode: {mode} | Snapshots: {num_snapshots}")
    print(f"{'='*60}")

    # Sort by time
    df = df.sort_values('t').reset_index(drop=True)

    # Remap node IDs to compact 0-indexed
    all_nodes = pd.concat([df['u'], df['v']])
    if max_nodes is not None:
        top_nodes = all_nodes.value_counts().head(max_nodes).index
        df = df[df['u'].isin(top_nodes) & df['v'].isin(top_nodes)].copy()
        all_nodes = pd.concat([df['u'], df['v']])

    unique_nodes = sorted(all_nodes.unique())
    node_map = {n: i for i, n in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)

    df['u'] = df['u'].map(node_map)
    df['v'] = df['v'].map(node_map)

    print(f"  Nodes: {num_nodes} | Edges: {len(df)}")

    # Time bins
    t_start = df['t'].min()
    t_end = df['t'].max()
    t_range = t_end - t_start
    if t_range == 0:
        t_range = 1.0

    time_step = t_range / num_snapshots

    # Assign edges to buckets
    snapshot_edges = [[] for _ in range(num_snapshots)]

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Binning {name}", leave=False):
        u, v, t = int(row['u']), int(row['v']), float(row['t'])
        idx = int((t - t_start) / time_step)
        idx = min(idx, num_snapshots - 1)

        if mode == 'window':
            snapshot_edges[idx].append([u, v])
        elif mode == 'cumulative':
            for future_idx in range(idx, num_snapshots):
                snapshot_edges[future_idx].append([u, v])

    # Save snapshots
    save_path = os.path.join(output_dir, name)
    os.makedirs(save_path, exist_ok=True)

    edge_counts = []
    for i in range(num_snapshots):
        edges = snapshot_edges[i]
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            # Remove self-loops
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            # Remove duplicates
            edge_index = torch.unique(edge_index, dim=1)

        # Class label: 0=citation-like, 1=social-like
        y_label = torch.tensor([0], dtype=torch.long)
        x_dummy = torch.ones((num_nodes, 1))

        data = Data(x=x_dummy, edge_index=edge_index, y=y_label, num_nodes=num_nodes)
        torch.save(data, os.path.join(save_path, f'snap_{i}.pt'))
        edge_counts.append(edge_index.shape[1] // 2)

    print(f"  Saved {num_snapshots} snapshots to {save_path}")
    print(f"  Edge counts: {edge_counts[0]} → {edge_counts[-1]} "
          f"(min={min(edge_counts)}, max={max(edge_counts)})")

    return num_nodes


# ═══════════════════════════════════════════════════════════════
# Dataset Registry
# ═══════════════════════════════════════════════════════════════
DATASETS = {
    'collegemsg': {
        'loader': load_collegemsg,
        'max_nodes': None,  # small enough
        'num_snapshots': 50,
    },
    'bitcoin_alpha': {
        'loader': load_bitcoin_alpha,
        'max_nodes': None,
        'num_snapshots': 50,
    },
    'euroroad': {
        'loader': load_euroroad,
        'max_nodes': None,
        'num_snapshots': 50,
    },
    'pp_pathways': {
        'loader': load_pp_pathways,
        'max_nodes': None,  # already subsampled in loader
        'num_snapshots': 50,
    },
    'cit_hepph': {
        'loader': load_cit_hepph,
        'max_nodes': None,  # already subsampled in loader
        'num_snapshots': 50,
    },
}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess datasets for DyCond")
    parser.add_argument('--dataset', type=str, default=None,
                        help='Process single dataset (e.g., collegemsg). Default: all')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.dataset:
        datasets_to_process = {args.dataset: DATASETS[args.dataset]}
    else:
        datasets_to_process = DATASETS

    for name, cfg in datasets_to_process.items():
        try:
            df, mode = cfg['loader']()
            generate_snapshots(
                name, df, OUT_DIR,
                num_snapshots=cfg['num_snapshots'],
                max_nodes=cfg['max_nodes'],
                mode=mode
            )
        except FileNotFoundError as e:
            print(f"  ⚠️ Skipping {name}: {e}")
        except Exception as e:
            print(f"  ❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ All done! Snapshots saved to {OUT_DIR}")
