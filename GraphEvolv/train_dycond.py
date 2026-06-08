import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import re

# --- Imports ---
from datasets.paired_loader import get_paired_loader
from model import get_model, add_model_args


# ============================================================
# 1. Robust Max Node Scanner
# ============================================================
def get_real_max_nodes(dataset_name):
    print(f"   -> Scanning {dataset_name} for max node ID (Full Scan)...")
    path = f"data/processed/{dataset_name}"

    files = [f for f in os.listdir(path) if f.startswith("snap_")]
    files.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    max_id = 0
    for f in tqdm(files, desc="Scanning Files", leave=False):
        try:
            data = torch.load(os.path.join(path, f))
            if data.edge_index.numel() > 0:
                max_id = max(max_id, data.edge_index.max().item())
            max_id = max(max_id, data.num_nodes)
        except:
            pass

    safe_max = max_id + 1000
    print(f"   -> Found Max ID: {max_id}. Setting model capacity to: {safe_max}")
    return safe_max


# ============================================================
# 2. Force Safety (NO edge_index modulo)
# ============================================================
def force_safety(graph, capacity, device):
    if not hasattr(graph, "batch_nodes") or graph.batch_nodes is None:
        graph.batch_nodes = torch.arange(graph.num_nodes, device=device)

    graph.batch_nodes = graph.batch_nodes % capacity
    return graph


# ============================================================
# 3. Noise Schedule
# ============================================================
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


# ============================================================
# 4. Noise Injector
# ============================================================
def apply_noise(curr_graph, t, betas, device):
    alpha_bars = torch.cumprod(1 - betas, dim=0).to(device)
    probs_keep = alpha_bars[t]

    if probs_keep.dim() > 0:
        edge_probs = probs_keep[curr_graph.batch[curr_graph.edge_index[0]]]
    else:
        edge_probs = probs_keep

    mask = torch.rand(curr_graph.edge_index.size(1), device=device) < edge_probs
    return curr_graph.edge_index[:, mask]


# ============================================================
# 5. Edge Scoring Helper (SAFE)
# ============================================================
def score_edges(edge_index, batch_nodes, node_emb, edge_head):
    src, dst = edge_index
    gsrc = batch_nodes[src]
    gdst = batch_nodes[dst]
    return edge_head(torch.cat([node_emb[gsrc], node_emb[gdst]], dim=1))


# ============================================================
# 6. DEGREE DISTRIBUTION LOSS (NEW)
# ============================================================
def degree_histogram(edge_index, num_nodes, bins=20):
    deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    deg = torch.log1p(deg)  # stabilizes power-law

    hist = torch.histc(
        deg,
        bins=bins,
        min=0,
        max=deg.max() if deg.max() > 0 else 1.0
    )
    return hist / (hist.sum() + 1e-8)


# ============================================================
# 7. Training Loop
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 Starting Training on {device} ---")

    real_max = get_real_max_nodes(args.dataset)
    args.max_nodes = real_max
    args.num_nodes = real_max

    train_loader = get_paired_loader("data", args.dataset, args.batch_size, split="train")
    val_loader = get_paired_loader("data", args.dataset, args.batch_size, split="val")

    model = get_model(args, initial_graph_sampler=None).to(device)
    capacity = model.node_emb.weight.size(0)
    print(f"   -> Model Table Size: {capacity}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    betas = get_beta_schedule(args.diffusion_steps).to(device)

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_deg_loss = 0

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

            # --- Positive edges ---
            pos_scores = score_edges(
                curr_graph.edge_index,
                curr_graph.batch_nodes,
                node_emb,
                edge_head,
            )

            # --- Negative sampling ---
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

            # --- Degree loss (NEW) ---
            if epoch >= args.degree_warmup:
                real_hist = degree_histogram(
                    curr_graph.edge_index,
                    curr_graph.batch_nodes.size(0),
                    args.degree_bins,
                )

                gen_hist = degree_histogram(
                    noisy_graph.edge_index,
                    noisy_graph.batch_nodes.size(0),
                    args.degree_bins,
                )

                deg_loss = F.mse_loss(gen_hist, real_hist)
            else:
                deg_loss = torch.tensor(0.0, device=device)

            loss = edge_loss + args.lambda_deg * deg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_deg_loss += deg_loss.item()

        # --- Validation ---
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
                        curr_graph.edge_index,
                        curr_graph.batch_nodes,
                        node_emb,
                        edge_head,
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
                        -torch.log(1 - torch.sigmoid(neg_scores) + 1e-10).mean()
                    )

                    val_loss += loss.item()
                    count += 1

            avg_val = val_loss / count
            print(
                f"Epoch {epoch+1} | "
                f"Train {total_loss/len(train_loader):.4f} | "
                f"Deg {total_deg_loss/len(train_loader):.4f} | "
                f"Val {avg_val:.4f}"
            )

            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), "dycond_best.pth")
                print("   >>> New Best Model Saved")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"dycond_checkpoint_{epoch}.pth")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="reddit")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Degree loss args
    parser.add_argument("--lambda_deg", type=float, default=0.05)
    parser.add_argument("--degree_bins", type=int, default=20)
    parser.add_argument("--degree_warmup", type=int, default=200)

    add_model_args(parser)

    parser.set_defaults(
        arch="SparseDyCond",
        max_nodes=50000,
        diffusion_steps=1000,
        diffusion_dim=256,
        num_heads=[8, 8, 8, 8, 8, 8],
    )

    args = parser.parse_args()
    train(args)
