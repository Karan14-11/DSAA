"""
Unified DyCond Inference — generate graph timelines from trained checkpoints.

For each checkpoint, loads the model, takes snap_0 as seed, and generates
a temporal rollout of graph snapshots.

Usage:
    python infer_multi_dataset.py --dataset collegemsg --loss_type deg_slope --checkpoints 500 1000 1500 2000
"""

import torch
from tqdm import tqdm
import os
import sys
import re
import argparse
from torch_geometric.data import Data

# Add Edge Repo to path
EDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Edge Repo', 'EDGE')
sys.path.insert(0, EDGE_DIR)

from model import get_model, add_model_args


def force_safety(graph, capacity, device):
    if not hasattr(graph, "batch_nodes") or graph.batch_nodes is None:
        graph.batch_nodes = torch.arange(graph.num_nodes, device=device)
    graph.batch_nodes = graph.batch_nodes % capacity
    return graph


def get_real_max_nodes(dataset_name, data_root):
    path = os.path.join(data_root, 'processed', dataset_name)
    files = [f for f in os.listdir(path) if f.startswith("snap_")]
    max_id = 0
    for f in files:
        try:
            data = torch.load(os.path.join(path, f), weights_only=False)
            if data.edge_index.numel() > 0:
                max_id = max(max_id, data.edge_index.max().item())
            max_id = max(max_id, data.num_nodes)
        except:
            pass
    return max_id + 1000


def generate_for_checkpoint(args, checkpoint_path, data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 🚀 Generating with {checkpoint_path} on {device} ---")

    model = get_model(args, initial_graph_sampler=None)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    capacity = model.node_emb.weight.size(0)

    # Load seed graph
    seed_path = os.path.join(data_root, 'processed', args.dataset, 'snap_0.pt')
    prev_graph = torch.load(seed_path, weights_only=False).to(device)

    # Disable label conditioning (CRITICAL)
    prev_graph.y = torch.zeros(prev_graph.num_nodes, dtype=torch.long, device=device)
    prev_graph.batch = torch.zeros(prev_graph.num_nodes, dtype=torch.long, device=device)
    prev_graph = force_safety(prev_graph, capacity, device)

    generated = [prev_graph.clone().cpu()]
    sampling_steps = 50

    for step in tqdm(range(1, args.steps + 1), desc="Temporal rollout"):
        num_nodes = prev_graph.num_nodes
        num_edges = prev_graph.edge_index.size(1)

        # Initialize noisy canvas
        rand_src = torch.randint(0, num_nodes, (num_edges,), device=device)
        rand_dst = torch.randint(0, num_nodes, (num_edges,), device=device)

        curr_graph = Data(
            x=torch.ones(num_nodes, 1, device=device),
            edge_index=torch.stack([rand_src, rand_dst])
        )
        curr_graph.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        curr_graph.batch_nodes = torch.arange(num_nodes, device=device)
        curr_graph.y = torch.zeros(num_nodes, dtype=torch.long, device=device)
        curr_graph = force_safety(curr_graph, capacity, device)

        # Reverse diffusion
        time_pairs = list(zip(
            reversed(range(sampling_steps)),
            reversed(range(1, sampling_steps + 1))
        ))

        with torch.no_grad():
            for t_curr, t_prev in time_pairs:
                model_t = torch.tensor(
                    [t_prev * (args.diffusion_steps // sampling_steps)],
                    device=device
                )

                node_emb, edge_head = model(
                    curr_graph, prev_graph, model_t, curr_graph.y
                )

                # Sparse candidate set
                k = min(curr_graph.edge_index.size(1), num_nodes * 2)

                cand_src = torch.cat([
                    curr_graph.edge_index[0][:k],
                    torch.randint(0, num_nodes, (k,), device=device)
                ])
                cand_dst = torch.cat([
                    curr_graph.edge_index[1][:k],
                    torch.randint(0, num_nodes, (k,), device=device)
                ])

                scores = edge_head(
                    torch.cat([node_emb[cand_src], node_emb[cand_dst]], dim=1)
                ).squeeze()

                probs = torch.sigmoid(scores)
                mask = torch.bernoulli(probs).bool()

                if mask.sum() == 0:
                    continue

                curr_graph.edge_index = torch.stack(
                    [cand_src[mask], cand_dst[mask]]
                )

        generated.append(curr_graph.clone().cpu())
        prev_graph = curr_graph

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyCond Multi-Dataset Inference")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dycond_loss_type", type=str, required=True,
                        choices=["edge_only", "deg_hist", "deg_slope", "deg_combined", "pr_hist", "pr_slope", "pr_combined", "combined"])
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Checkpoint epochs or 'best' to evaluate")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of generation steps")
    parser.add_argument("--candidates", type=int, default=20000,
                        help="Number of candidate edges per step")
    parser.add_argument("--edge_threshold", type=float, default=0.5)


    # Model args
    add_model_args(parser)

    parser.set_defaults(
        arch="SparseDyCond",
        max_nodes=50000,
        diffusion_steps=1000,
        diffusion_dim=256,
        num_heads=[8, 8, 8, 8, 8, 8],
        emb_dim=256,
    )

    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # Set correct max_nodes
    real_max = get_real_max_nodes(args.dataset, data_root)
    args.max_nodes = real_max
    args.num_nodes = real_max

    ckpt_dir = os.path.join('checkpoints', f'{args.dataset}_{args.dycond_loss_type}')
    results_dir = os.path.join('results', args.dataset, args.dycond_loss_type)
    os.makedirs(results_dir, exist_ok=True)

    for epoch in tqdm(args.checkpoints, desc="Generating checkpoints"):
        # Try multiple checkpoint naming conventions
        if epoch == "best":
            candidates = [os.path.join(ckpt_dir, 'best.pth')]
        else:
            candidates = [
                os.path.join(ckpt_dir, f'epoch_{epoch}.pth'),
                os.path.join(ckpt_dir, 'best.pth'),
            ]

        ckpt_path = None
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

        if ckpt_path is None:
            print(f"⚠️ Skipping epoch {epoch}: no checkpoint found in {ckpt_dir}")
            continue

        generated = generate_for_checkpoint(args, ckpt_path, data_root)

        save_path = os.path.join(results_dir, f'generated_timeline_epoch_{epoch}.pt')
        torch.save(generated, save_path)
        print(f"✅ Saved: {save_path}")
