"""
Snapshot Statistics Monitor — prints per-snapshot edge counts for all datasets.

Usage:
    python3 snapshot_stats.py                    # All datasets
    python3 snapshot_stats.py --dataset askubuntu # Single dataset
"""
import torch
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data.snapshot_builder import build_snapshots

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'mathoverflow':  'processed/mathoverflow_compact.pt',
    'askubuntu':     'processed/askubuntu_compact.pt',
    'superuser':     'processed/superuser_compact.pt',
    'wikitalk':      'processed/wikitalk_compact.pt',
    'as733':         'processed/as733_snapshots.pt',
    'ascaida':       'processed/ascaida_snapshots.pt',
}


def print_snapshot_stats(name, data_file, num_snapshots=50):
    print(f"\n{'='*80}")
    print(f"  Dataset: {name}")
    print(f"{'='*80}")

    path = os.path.join(PROJECT_ROOT, data_file)
    if not os.path.exists(path):
        print(f"  ⚠️  File not found: {path}")
        return

    snapshots, meta = build_snapshots(path, num_snapshots=num_snapshots, feature_dim=32)
    num_nodes = meta['num_nodes']

    # Header
    print(f"\n  {'Snap':>4s}  {'Total Edges':>12s}  {'New Edges':>10s}  {'Removed':>10s}  "
          f"{'Δ Edges':>8s}  {'Avg Deg':>8s}  {'Active Nodes':>13s}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*13}")

    prev_total = 0
    total_new = 0
    total_removed = 0

    for i, snap in enumerate(snapshots):
        total_edges = snap.edge_index.shape[1] // 2  # undirected
        new_edges = snap.new_edges.shape[1] // 2 if snap.new_edges is not None else 0
        removed = snap.removed_edges.shape[1] // 2 if hasattr(snap, 'removed_edges') and snap.removed_edges is not None and snap.removed_edges.numel() > 0 else 0
        delta = total_edges - prev_total
        active_nodes = snap.edge_index.unique().numel() if snap.edge_index.numel() > 0 else 0
        avg_deg = (2 * total_edges) / max(active_nodes, 1)

        marker = ""
        if i == 39:
            marker = " ◀ train/test split"

        print(f"  {i:4d}  {total_edges:12,d}  {new_edges:10,d}  {removed:10,d}  "
              f"{delta:+8,d}  {avg_deg:8.2f}  {active_nodes:13,d}{marker}")

        prev_total = total_edges
        total_new += new_edges
        total_removed += removed

    # Summary
    print(f"\n  Summary:")
    print(f"    Nodes:               {num_nodes:,d}")
    print(f"    Final edges:         {prev_total:,d}")
    print(f"    Total new edges:     {total_new:,d}")
    print(f"    Total removed edges: {total_removed:,d}")
    if total_removed > 0:
        print(f"    Deletion ratio:      {total_removed / max(total_new, 1) * 100:.1f}%")
    print(f"    Avg new/snapshot:    {total_new / num_snapshots:.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Snapshot Statistics Monitor")
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name, or omit for all')
    args = parser.parse_args()

    if args.dataset:
        if args.dataset in DATASETS:
            print_snapshot_stats(args.dataset, DATASETS[args.dataset])
        else:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(DATASETS.keys())}")
    else:
        for name, path in DATASETS.items():
            print_snapshot_stats(name, path)
