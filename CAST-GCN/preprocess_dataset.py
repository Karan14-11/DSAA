"""
General-purpose preprocessor for temporal edge-list datasets.

Usage:
    python preprocess_dataset.py --name superuser --raw scripts/data/raw/sx-superuser-a2q.txt
    python preprocess_dataset.py --all           # process all known datasets

Input: space-separated file with columns  u  v  t  (no header)
Output: processed/<name>_compact.pt  with keys {u, v, t, num_nodes}
"""

import argparse
import os
import torch
import pandas as pd

# ── Known datasets ────────────────────────────────────────────────
# Maps dataset name → (raw file path relative to project root, has_header)
RAW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'scripts', 'data', 'raw')

DATASETS = {
    'askubuntu':    ('sx-askubuntu-a2q.txt',    False),
    'superuser':    ('sx-superuser-a2q.txt',    False),
    'mathoverflow': ('sx-mathoverflow-a2q.txt',  False),
    'stackoverflow':('sx-stackoverflow-a2q.txt', False),
    'wikitalk':     ('wiki-talk-temporal.txt',   False),
}


def preprocess(name, raw_path):
    assert os.path.exists(raw_path), f"Raw file not found: {raw_path}"

    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"  Source: {raw_path}")
    print(f"{'='*60}")

    df = pd.read_csv(raw_path, header=None, sep=r'\s+',
                     comment='#',           # skip comment lines
                     on_bad_lines='skip')   # skip malformed rows
    # Keep only first 3 columns (u, v, t)
    df = df.iloc[:, :3]
    df.columns = ['u', 'v', 't']
    df = df.dropna()

    print(f"  Loaded {len(df):,} edges")

    u = torch.tensor(df['u'].values, dtype=torch.long)
    v = torch.tensor(df['v'].values, dtype=torch.long)
    t = torch.tensor(df['t'].values, dtype=torch.float)

    # Remove self-loops
    mask = u != v
    u, v, t = u[mask], v[mask], t[mask]
    print(f"  After removing self-loops: {len(u):,} edges")

    # Remap node IDs to 0-indexed contiguous range
    nodes = torch.unique(torch.cat([u, v]))
    node_map = {int(n): i for i, n in enumerate(nodes.tolist())}
    u_new = torch.tensor([node_map[int(x)] for x in u.tolist()], dtype=torch.long)
    v_new = torch.tensor([node_map[int(x)] for x in v.tolist()], dtype=torch.long)
    num_nodes = len(nodes)

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Time range: {t.min().item():.0f} — {t.max().item():.0f}")

    os.makedirs("processed", exist_ok=True)
    out_path = os.path.join("processed", f"{name}_compact.pt")
    torch.save({
        'u': u_new,
        'v': v_new,
        't': t,
        'num_nodes': num_nodes,
    }, out_path)
    print(f"  ✅ Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess temporal edge-list dataset")
    parser.add_argument("--name", type=str, default=None,
                        help="Dataset name (e.g. superuser)")
    parser.add_argument("--raw", type=str, default=None,
                        help="Path to raw edge-list file (u v t)")
    parser.add_argument("--all", action="store_true",
                        help="Process all known datasets")
    args = parser.parse_args()

    if args.all:
        for name, (filename, _) in DATASETS.items():
            raw_path = os.path.join(RAW_ROOT, filename)
            if os.path.exists(raw_path):
                preprocess(name, raw_path)
            else:
                print(f"\n⚠️  Skipping {name}: {raw_path} not found")
    elif args.name and args.raw:
        preprocess(args.name, args.raw)
    else:
        parser.error("Provide --name and --raw, or use --all")
