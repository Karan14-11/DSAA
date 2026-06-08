"""
Unified DyCond Training Script — supports 3 loss types on any dataset.

Loss types:
  1. edge_only  — baseline edge scoring (BCE), no structural guidance
  2. deg_hist   — edge loss + degree histogram matching (MSE after warmup)
  3. deg_slope  — edge loss + degree log-log slope matching

Usage:
    python train_multi_dataset.py --dataset collegemsg --loss_type edge_only --epochs 2000
    python train_multi_dataset.py --dataset pp_pathways --loss_type deg_slope --epochs 2000

Must be run from within Edge Repo/EDGE/ directory (or have its modules on path).
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import sys
import re
import json

# Add Edge Repo to path
EDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Edge Repo', 'EDGE')
sys.path.insert(0, EDGE_DIR)

from datasets.paired_loader import get_paired_loader
from model import get_model, add_model_args


# ═══════════════════════════════════════════════════════════════
# Helpers (from existing DyCond training scripts)
# ═══════════════════════════════════════════════════════════════

def get_real_max_nodes(dataset_name, data_root):
    """Scan dataset for max node ID."""
    print(f"   -> Scanning {dataset_name} for max node ID...")
    path = os.path.join(data_root, 'processed', dataset_name)

    files = [f for f in os.listdir(path) if f.startswith("snap_")]
    files.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    max_id = 0
    for f in tqdm(files, desc="Scanning", leave=False):
        try:
            data = torch.load(os.path.join(path, f), weights_only=False)
            if data.edge_index.numel() > 0:
                max_id = max(max_id, data.edge_index.max().item())
            max_id = max(max_id, data.num_nodes)
        except:
            pass

    safe_max = max_id + 1000
    print(f"   -> Max ID: {max_id}. Model capacity: {safe_max}")
    return safe_max


def force_safety(graph, capacity, device):
    if not hasattr(graph, "batch_nodes") or graph.batch_nodes is None:
        graph.batch_nodes = torch.arange(graph.num_nodes, device=device)
    graph.batch_nodes = graph.batch_nodes % capacity
    return graph


def get_beta_schedule(steps, type="cosine"):
    if type == "cosine":
        steps += 1
        s = 0.008
        x = torch.linspace(0, steps, steps)
        alphas = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        return torch.clamp(betas, 0, 0.999)
    else:
        return torch.linspace(1e-4, 0.02, steps)


def apply_noise(curr_graph, t, betas, device):
    alpha_bars = torch.cumprod(1 - betas, dim=0).to(device)
    probs_keep = alpha_bars[t]
    if probs_keep.dim() > 0:
        edge_probs = probs_keep[curr_graph.batch[curr_graph.edge_index[0]]]
    else:
        edge_probs = probs_keep
    mask = torch.rand(curr_graph.edge_index.size(1), device=device) < edge_probs
    return curr_graph.edge_index[:, mask]


def score_edges(edge_index, batch_nodes, node_emb, edge_head):
    src, dst = edge_index
    gsrc = batch_nodes[src]
    gdst = batch_nodes[dst]
    return edge_head(torch.cat([node_emb[gsrc], node_emb[gdst]], dim=1))


# ═══════════════════════════════════════════════════════════════
# Structural Loss Functions
# ═══════════════════════════════════════════════════════════════

def degree_histogram(edge_index, num_nodes, bins=20):
    """Degree histogram for Loss 2 (deg-hist matching)."""
    deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    deg = torch.log1p(deg)
    hist = torch.histc(deg, bins=bins, min=0,
                       max=deg.max() if deg.max() > 0 else 1.0)
    return hist / (hist.sum() + 1e-8)


def degree_loglog_slope(deg, eps=1e-8):
    """Compute power-law slope for Loss 3 (deg-slope matching)."""
    deg = deg[deg >= 1]
    if deg.numel() < 10:
        return None
    deg_sorted, _ = torch.sort(deg, descending=True)
    ranks = torch.arange(1, deg_sorted.numel() + 1, device=deg.device)
    log_deg = torch.log(deg_sorted + eps)
    log_rank = torch.log(ranks.float() + eps)
    x = log_rank - log_rank.mean()
    y = log_deg - log_deg.mean()
    slope = (x * y).sum() / (x.pow(2).sum() + eps)
    return slope


def degree_slope_loss(real_edge_index, gen_edge_index, num_nodes, device):
    """MSE between real and generated degree slopes."""
    real_deg = torch.zeros(num_nodes, device=device)
    gen_deg = torch.zeros(num_nodes, device=device)
    real_deg.scatter_add_(0, real_edge_index[0],
                          torch.ones(real_edge_index.size(1), device=device))
    gen_deg.scatter_add_(0, gen_edge_index[0],
                         torch.ones(gen_edge_index.size(1), device=device))
    real_slope = degree_loglog_slope(real_deg)
    gen_slope = degree_loglog_slope(gen_deg)
    if real_slope is None or gen_slope is None:
        return torch.tensor(0.0, device=device)
    return F.mse_loss(gen_slope, real_slope)


def dynamic_pagerank(edge_index, num_nodes, alpha=0.85, iters=3, eps=1e-8):
    """Compute PageRank of a graph in a vectorized/differentiable manner."""
    device = edge_index.device
    if edge_index.numel() == 0:
        return torch.ones(num_nodes, device=device) / num_nodes
    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=device)
    deg.index_add_(0, src, torch.ones(edge_index.size(1), device=device))
    deg_inv = 1.0 / (deg + eps)
    pr = torch.ones(num_nodes, device=device) / num_nodes
    for _ in range(iters):
        msg = pr[src] * deg_inv[src]
        agg = torch.zeros_like(pr)
        agg.index_add_(0, dst, msg)
        pr = alpha * agg + (1 - alpha) / num_nodes
    pr = pr / (pr.sum() + eps)
    return pr


def pagerank_histogram(edge_index, num_nodes, bins=20, alpha=0.85, iters=3):
    """PageRank histogram for PageRank histogram matching."""
    pr = dynamic_pagerank(edge_index, num_nodes, alpha=alpha, iters=iters)
    pr = torch.log1p(pr * num_nodes)
    hist = torch.histc(pr, bins=bins, min=0,
                       max=pr.max() if pr.max() > 0 else 1.0)
    return hist / (hist.sum() + 1e-8)


def pagerank_loglog_slope(pr, eps=1e-8):
    """Compute log-log slope for PageRank rank distribution."""
    pr = pr[pr > 0]
    if pr.numel() < 10:
        return None
    pr_sorted, _ = torch.sort(pr, descending=True)
    ranks = torch.arange(1, pr_sorted.numel() + 1, device=pr.device)
    log_pr = torch.log(pr_sorted + eps)
    log_rank = torch.log(ranks.float() + eps)
    x = log_rank - log_rank.mean()
    y = log_pr - log_pr.mean()
    slope = (x * y).sum() / (x.pow(2).sum() + eps)
    return slope


def pagerank_slope_loss(real_edge_index, gen_edge_index, num_nodes, device, alpha=0.85, iters=3):
    """MSE between real and generated PageRank slopes."""
    real_pr = dynamic_pagerank(real_edge_index, num_nodes, alpha=alpha, iters=iters)
    gen_pr = dynamic_pagerank(gen_edge_index, num_nodes, alpha=alpha, iters=iters)
    real_slope = pagerank_loglog_slope(real_pr)
    gen_slope = pagerank_loglog_slope(gen_pr)
    if real_slope is None or gen_slope is None:
        return torch.tensor(0.0, device=device)
    return F.mse_loss(gen_slope, real_slope)


# ═══════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"DyCond Training: {args.dataset} | Loss: {args.dycond_loss_type}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Data root for paired loader
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # Scan for max nodes
    real_max = get_real_max_nodes(args.dataset, data_root)
    args.max_nodes = real_max
    args.num_nodes = real_max

    # Use Edge Repo's data root for paired loader
    train_loader = get_paired_loader(data_root, args.dataset, args.batch_size, split="train")
    val_loader = get_paired_loader(data_root, args.dataset, args.batch_size, split="val")

    # Build model
    model = get_model(args, initial_graph_sampler=None).to(device)
    capacity = model.node_emb.weight.size(0)
    print(f"   -> Model Table Size: {capacity}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    betas = get_beta_schedule(args.diffusion_steps).to(device)

    # Checkpoint directory
    ckpt_dir = os.path.join('checkpoints', f'{args.dataset}_{args.dycond_loss_type}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training log
    log = {
        'edge_loss': [],
        'deg_hist_loss': [],
        'deg_slope_loss': [],
        'pr_hist_loss': [],
        'pr_slope_loss': [],
        'total_loss': [],
        'val_loss': []
    }
    best_val = float("inf")

    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        model.train()
        total_edge_loss = 0.0
        total_deg_hist_loss = 0.0
        total_deg_slope_loss = 0.0
        total_pr_hist_loss = 0.0
        total_pr_slope_loss = 0.0
        total_combined_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            prev_graph, curr_graph = (
                (batch[0], batch[1]) if isinstance(batch, list)
                else (batch.prev_graph, batch.curr_graph)
            )

            prev_graph = force_safety(prev_graph.to(device), capacity, device)
            curr_graph = force_safety(curr_graph.to(device), capacity, device)

            t = torch.randint(0, args.diffusion_steps, (curr_graph.num_graphs,), device=device)
            noisy_edges = apply_noise(curr_graph, t, betas, device)
            noisy_graph = curr_graph.clone()
            noisy_graph.edge_index = noisy_edges
            noisy_graph.batch_nodes = curr_graph.batch_nodes

            optimizer.zero_grad()
            node_emb, edge_head = model(noisy_graph, prev_graph, t, curr_graph.y)

            # ── Positive edges ──
            pos_scores = score_edges(
                curr_graph.edge_index, curr_graph.batch_nodes, node_emb, edge_head
            )

            # ── Negative sampling ──
            num_nodes = curr_graph.batch_nodes.size(0)
            neg_src = torch.randint(0, num_nodes, (pos_scores.size(0),), device=device)
            neg_dst = torch.randint(0, num_nodes, (pos_scores.size(0),), device=device)
            neg_scores = edge_head(torch.cat([
                node_emb[curr_graph.batch_nodes[neg_src]],
                node_emb[curr_graph.batch_nodes[neg_dst]],
            ], dim=1))

            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()
            edge_loss = pos_loss + neg_loss

            # ── Structural components (Always computed for tracking/logging) ──
            struct_loss_deg = torch.tensor(0.0, device=device)
            if epoch >= args.degree_warmup:
                real_hist = degree_histogram(
                    curr_graph.edge_index, num_nodes, args.degree_bins
                )
                gen_hist = degree_histogram(
                    noisy_graph.edge_index, num_nodes, args.degree_bins
                )
                struct_loss_deg = F.mse_loss(gen_hist, real_hist)

            with torch.no_grad():
                gen_probs = torch.sigmoid(pos_scores).view(-1)
                gen_mask = gen_probs > 0.5
                gen_edges = curr_graph.edge_index[:, gen_mask]

            struct_loss_slope = degree_slope_loss(
                curr_graph.edge_index, gen_edges, num_nodes, device
            )

            # ── PageRank structural components (Always computed for tracking/logging) ──
            struct_loss_pr = torch.tensor(0.0, device=device)
            if epoch >= args.pr_warmup:
                real_pr_hist = pagerank_histogram(
                    curr_graph.edge_index, num_nodes, args.pr_bins, alpha=args.pr_alpha, iters=args.pr_iters
                )
                gen_pr_hist = pagerank_histogram(
                    noisy_graph.edge_index, num_nodes, args.pr_bins, alpha=args.pr_alpha, iters=args.pr_iters
                )
                struct_loss_pr = F.mse_loss(gen_pr_hist, real_pr_hist)

            struct_loss_pr_slope = pagerank_slope_loss(
                curr_graph.edge_index, gen_edges, num_nodes, device, alpha=args.pr_alpha, iters=args.pr_iters
            )

            # ── Select loss based on configuration ──
            if args.dycond_loss_type == 'edge_only':
                loss = edge_loss
            elif args.dycond_loss_type == 'deg_hist':
                loss = edge_loss + args.lambda_deg * struct_loss_deg
            elif args.dycond_loss_type == 'deg_slope':
                loss = edge_loss + args.lambda_slope * struct_loss_slope
            elif args.dycond_loss_type == 'pr_hist':
                loss = edge_loss + args.lambda_pr * struct_loss_pr
            elif args.dycond_loss_type == 'pr_slope':
                loss = edge_loss + args.lambda_pr_slope * struct_loss_pr_slope
            elif args.dycond_loss_type == 'pr_combined':
                loss = edge_loss + args.lambda_pr * struct_loss_pr + args.lambda_pr_slope * struct_loss_pr_slope
            else:  # combined (default: joint degree slope + PageRank slope)
                loss = edge_loss + args.lambda_slope * struct_loss_slope + args.lambda_pr_slope * struct_loss_pr_slope

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_edge_loss += edge_loss.item()
            total_deg_hist_loss += struct_loss_deg.item()
            total_deg_slope_loss += struct_loss_slope.item()
            total_pr_hist_loss += struct_loss_pr.item()
            total_pr_slope_loss += struct_loss_pr_slope.item()
            total_combined_loss += loss.item()

        avg_edge = total_edge_loss / max(len(train_loader), 1)
        avg_deg_hist = total_deg_hist_loss / max(len(train_loader), 1)
        avg_deg_slope = total_deg_slope_loss / max(len(train_loader), 1)
        avg_pr_hist = total_pr_hist_loss / max(len(train_loader), 1)
        avg_pr_slope = total_pr_slope_loss / max(len(train_loader), 1)
        avg_total = total_combined_loss / max(len(train_loader), 1)

        log['edge_loss'].append(avg_edge)
        log['deg_hist_loss'].append(avg_deg_hist)
        log['deg_slope_loss'].append(avg_deg_slope)
        log['pr_hist_loss'].append(avg_pr_hist)
        log['pr_slope_loss'].append(avg_pr_slope)
        log['total_loss'].append(avg_total)

        # ── Validation ──
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            count = 0

            with torch.no_grad():
                for batch in val_loader:
                    prev_graph, curr_graph = (
                        (batch[0], batch[1]) if isinstance(batch, list)
                        else (batch.prev_graph, batch.curr_graph)
                    )
                    prev_graph = force_safety(prev_graph.to(device), capacity, device)
                    curr_graph = force_safety(curr_graph.to(device), capacity, device)

                    t = torch.randint(0, args.diffusion_steps, (curr_graph.num_graphs,), device=device)
                    noisy_edges = apply_noise(curr_graph, t, betas, device)
                    noisy_graph = curr_graph.clone()
                    noisy_graph.edge_index = noisy_edges
                    noisy_graph.batch_nodes = curr_graph.batch_nodes

                    node_emb, edge_head = model(noisy_graph, prev_graph, t, curr_graph.y)
                    pos_scores = score_edges(
                        curr_graph.edge_index, curr_graph.batch_nodes, node_emb, edge_head
                    )
                    num_nodes = curr_graph.batch_nodes.size(0)
                    neg_src = torch.randint(0, num_nodes, (pos_scores.size(0),), device=device)
                    neg_dst = torch.randint(0, num_nodes, (pos_scores.size(0),), device=device)
                    neg_scores = edge_head(torch.cat([
                        node_emb[curr_graph.batch_nodes[neg_src]],
                        node_emb[curr_graph.batch_nodes[neg_dst]],
                    ], dim=1))

                    loss = (
                        -torch.log(torch.sigmoid(pos_scores) + 1e-10).mean()
                        - torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()
                    )
                    val_loss += loss.item()
                    count += 1

            avg_val = val_loss / max(count, 1)
            log['val_loss'].append(avg_val)

            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Edge {avg_edge:.4f} | "
                f"DegHist {avg_deg_hist:.4f} | "
                f"Slope {avg_deg_slope:.4f} | "
                f"PRHist {avg_pr_hist:.4f} | "
                f"PRSlope {avg_pr_slope:.4f} | "
                f"Total {avg_total:.4f} | "
                f"Val {avg_val:.4f}"
            )

            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'best.pth'))
                print("   >>> New Best Model Saved")

        # ── Periodic checkpoints ──
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth'))

    # Save training log
    with open(os.path.join(ckpt_dir, 'training_log.json'), 'w') as f:
        json.dump(log, f)

    print(f"\n✅ Training complete. Checkpoints in: {ckpt_dir}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyCond Multi-Dataset Training")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (collegemsg, bitcoin_alpha, euroroad, pp_pathways, cit_hepph)")
    parser.add_argument("--dycond_loss_type", type=str, default="combined",
                        choices=["edge_only", "deg_hist", "deg_slope", "deg_combined", "pr_hist", "pr_slope", "pr_combined", "combined"],
                        help="Loss function type")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N epochs")

    # Degree histogram loss params
    parser.add_argument("--lambda_deg", type=float, default=0.05)
    parser.add_argument("--degree_bins", type=int, default=20)
    parser.add_argument("--degree_warmup", type=int, default=200)

    # Degree slope loss params
    parser.add_argument("--lambda_slope", type=float, default=0.1)

    # PageRank loss params
    parser.add_argument("--lambda_pr", type=float, default=0.05,
                        help="Weight for PageRank histogram loss")
    parser.add_argument("--lambda_pr_slope", type=float, default=0.5,
                        help="Weight for PageRank slope loss")
    parser.add_argument("--pr_bins", type=int, default=20,
                        help="Number of bins for PageRank histogram")
    parser.add_argument("--pr_warmup", type=int, default=200,
                        help="Epochs of warmup before PageRank histogram loss is active")
    parser.add_argument("--pr_alpha", type=float, default=0.85,
                        help="Damping factor for PageRank")
    parser.add_argument("--pr_iters", type=int, default=3,
                        help="Power iteration steps for PageRank")

    # Model args (from EDGE)
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

    # Run from the dycond directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train(args)
