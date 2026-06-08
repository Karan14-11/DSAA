import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from datasets.paired_loader import get_paired_loader
from model import get_model, add_model_args


# -------------------------
# SAFETY
# -------------------------
def force_safety(graph, capacity, device):
    if not hasattr(graph, "batch_nodes") or graph.batch_nodes is None:
        graph.batch_nodes = torch.arange(graph.num_nodes, device=device)
    graph.batch_nodes = graph.batch_nodes % capacity
    return graph

# -------------------------
# SUBGRAPH SAMPLING (fully vectorized)
# -------------------------
def sample_subgraph(edge_index, max_nodes=2000, device='cuda'):
    nodes = torch.unique(edge_index.view(-1))
    if nodes.numel() > max_nodes:
        nodes = nodes[torch.randperm(nodes.numel(), device=device)[:max_nodes]]
    # Map old node indices to 0..num_sub_nodes-1
    node_map = torch.full((edge_index.max() + 1,), -1, device=device, dtype=torch.long)
    node_map[nodes] = torch.arange(len(nodes), device=device)
    mask = torch.isin(edge_index[0], nodes) & torch.isin(edge_index[1], nodes)
    src_sub = node_map[edge_index[0, mask]]
    dst_sub = node_map[edge_index[1, mask]]
    return src_sub, dst_sub, nodes


# -------------------------
# DEGREE REGULARIZER (vectorized)
# -------------------------
def degree_regularizer_sampled(edge_scores, edge_index, max_nodes=2000):
    """
    Vectorized degree regularizer using a sampled subgraph.
    """
    device = edge_scores.device
    probs = torch.sigmoid(edge_scores)

    src_sub, dst_sub, nodes = sample_subgraph(edge_index, max_nodes, device)
    if src_sub.numel() == 0:
        return torch.tensor(0., device=device)

    # Compute predicted degree for sampled nodes
    pred_deg = torch.zeros(len(nodes), device=device)
    pred_deg.index_add_(0, src_sub, probs[:src_sub.size(0)])  # take same number of probs as sampled edges

    # Normalize and compute deviation
    pred_deg = pred_deg / (pred_deg.sum() + 1e-8)
    return torch.mean(torch.abs(pred_deg - pred_deg.mean()))


# -------------------------
# DYNAMIC PAGERANK (vectorized)
# -------------------------
def dynamic_pagerank(edge_index, edge_weights, num_nodes, alpha=0.85, iters=3, eps=1e-8):
    device = edge_index.device
    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=device)
    deg.index_add_(0, src, edge_weights)
    deg_inv = 1.0 / (deg + eps)
    pr = torch.ones(num_nodes, device=device) / num_nodes
    for _ in range(iters):
        msg = pr[src] * edge_weights * deg_inv[src]
        agg = torch.zeros_like(pr)
        agg.index_add_(0, dst, msg)
        pr = alpha * agg + (1 - alpha) / num_nodes
    pr = pr / (pr.sum() + eps)
    return pr

# -------------------------
# RECIPROCITY REGULARIZER (vectorized)
# -------------------------
def reciprocity_regularizer_sampled(edge_scores, edge_index, samples=5000):
    device = edge_scores.device
    probs = torch.sigmoid(edge_scores)
    E = edge_index.size(1)
    if E == 0:
        return torch.tensor(0., device=device)
    idx = torch.randint(0, E, (min(samples, E),), device=device)
    return torch.mean(torch.abs(probs[idx] - probs[idx].mean()))

# -------------------------
# TRAINING
# -------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training SparseDyCond (Degree-Controlled) on {device}")

    loader = get_paired_loader("data", args.dataset, args.batch_size, split="train",num_workers=8)
    model = get_model(args, initial_graph_sampler=None).to(device)
    capacity = model.node_emb.weight.size(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for prev_graph, curr_graph in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            prev_graph = force_safety(prev_graph.to(device), capacity, device)
            curr_graph = force_safety(curr_graph.to(device), capacity, device)
            optimizer.zero_grad()

            t = torch.zeros(prev_graph.num_graphs, device=device, dtype=torch.long)

            # Forward pass
            node_emb, scores = model(prev_graph, prev_graph, t, curr_graph.y)
            # node_h = node_emb[curr_graph.batch_nodes]
            # src, dst = curr_graph.edge_index
            # edge_h = torch.cat([node_h[src], node_h[dst]], dim=1)
            # scores = edge_head(edge_h).view(-1)
            # probs = torch.sigmoid(scores)

            # -----------------------------
            # Losses
            # -----------------------------
            # BCE
            bce = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))

            # Degree
            # deg_loss = degree_regularizer_sampled(scores, curr_graph.edge_index, max_nodes=2000)

            # Dynamic PageRank
            # pr_real = dynamic_pagerank(curr_graph.edge_index, torch.ones_like(scores), curr_graph.num_nodes)
            # pr_gen  = dynamic_pagerank(curr_graph.edge_index, probs, curr_graph.num_nodes)
            # pr_loss = torch.mean(torch.abs(pr_real - pr_gen))

            # Reciprocity
            # reciprocity_loss = reciprocity_regularizer_sampled(scores, curr_graph.edge_index)

            # Total
            loss = bce 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(
            # f"Epoch {epoch} | BCE {bce.item():.3f} | Deg {deg_loss.item():.3f} | "
            # f"PR {pr_loss.item():.3f} | #edges {curr_graph.edge_index.size(1)} | "
            # f"#pos_edges {(probs>0.5).sum().item()}"
        )

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"dycond_citation_{epoch}.pth")




# # -------------------------
# # MAIN
# # -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="stackoverflow")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_deg", type=float, default=1.0)
    parser.add_argument("--lambda_pr", type=float, default=0.05)

    add_model_args(parser)
    parser.set_defaults(
        arch="SparseDyCond",
        diffusion_dim=256,
        max_nodes=21000,
    )

    args = parser.parse_args()
    train(args)
