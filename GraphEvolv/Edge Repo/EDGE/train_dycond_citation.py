import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os

from datasets.paired_loader import get_paired_loader
from model import get_model, add_model_args


# ---------------------------------------------------
# SAFETY
# ---------------------------------------------------
def force_safety(graph, capacity, device):
    if not hasattr(graph, "batch_nodes") or graph.batch_nodes is None:
        graph.batch_nodes = torch.arange(graph.num_nodes, device=device)
    graph.batch_nodes = graph.batch_nodes % capacity
    return graph


# ---------------------------------------------------
# DELTA EDGES (monotonic growth)
# ---------------------------------------------------
def delta_edges(prev_edge_index, curr_edge_index):
    if prev_edge_index.numel() == 0:
        return curr_edge_index

    # Encode edges as unique integers
    max_node = max(
        prev_edge_index.max().item(),
        curr_edge_index.max().item()
    ) + 1

    prev_hash = prev_edge_index[0] * max_node + prev_edge_index[1]
    curr_hash = curr_edge_index[0] * max_node + curr_edge_index[1]

    mask = ~torch.isin(curr_hash, prev_hash)
    return curr_edge_index[:, mask]


# ---------------------------------------------------
# EDGE SCORING
# ---------------------------------------------------
def score_edges(edge_index, batch_nodes, node_emb, edge_head):
    src, dst = edge_index
    gsrc = batch_nodes[src]
    gdst = batch_nodes[dst]
    return edge_head(
        torch.cat([node_emb[gsrc], node_emb[gdst]], dim=1)
    ).view(-1)


# ---------------------------------------------------
# DEGREE MASS REGULARIZER (GLOBAL, LIGHTWEIGHT)
# ---------------------------------------------------
def degree_mass_loss(real_edge_index, gen_edge_index, num_nodes, k=100):
    if gen_edge_index.numel() == 0:
        return torch.tensor(0.0, device=real_edge_index.device)

    real_deg = torch.bincount(
        real_edge_index.view(-1),
        minlength=num_nodes
    ).float()

    gen_deg = torch.bincount(
        gen_edge_index.view(-1),
        minlength=num_nodes
    ).float()

    k = min(k, num_nodes)

    real_mass = torch.topk(real_deg, k).values.sum() / (real_deg.sum() + 1e-8)
    gen_mass  = torch.topk(gen_deg,  k).values.sum() / (gen_deg.sum()  + 1e-8)

    return (real_mass - gen_mass).abs()


# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 Training Citation SparseDyCond on {device} ---")

    loader = get_paired_loader(
        "data",
        args.dataset,
        args.batch_size,
        split="train"
    )

    model = get_model(args, initial_graph_sampler=None).to(device)
    capacity = model.node_emb.weight.size(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for prev_graph, curr_graph in tqdm(
            loader, desc=f"Epoch {epoch}", leave=False
        ):
            prev_graph = force_safety(prev_graph.to(device), capacity, device)
            curr_graph = force_safety(curr_graph.to(device), capacity, device)

            delta = delta_edges(
                prev_graph.edge_index,
                curr_graph.edge_index
            ).to(device)

            if delta.numel() == 0:
                continue

            optimizer.zero_grad()

            # ---------------------------------------------------
            # Zero-noise (citation graphs are monotonic)
            # ---------------------------------------------------
            t = torch.zeros(
                prev_graph.num_graphs,
                device=device,
                dtype=torch.long
            )

            node_emb, edge_head = model(
                prev_graph, prev_graph, t, curr_graph.y
            )

            # Positive edges = new citations
            pos_scores = score_edges(
                delta,
                curr_graph.batch_nodes,
                node_emb,
                edge_head
            )

            # Negative sampling (same sources, random destinations)
            # num_neg = min(pos_scores.size(0), 20000)
            num_neg = pos_scores.size(0)


            src = delta[0]
            neg_dst = torch.randint(
                0, curr_graph.num_nodes,
                (num_neg,),
                device=device
            )

            neg_scores = edge_head(
                torch.cat([
                    node_emb[curr_graph.batch_nodes[src]],
                    node_emb[curr_graph.batch_nodes[neg_dst]],
                ], dim=1)
            ).view(-1)

            # ---------------------------------------------------
            # BCE loss
            # ---------------------------------------------------
            bce_loss = (
                F.binary_cross_entropy_with_logits(
                    pos_scores, torch.ones_like(pos_scores)
                ) +
                F.binary_cross_entropy_with_logits(
                    neg_scores, torch.zeros_like(neg_scores)
                )
            )

            # ---------------------------------------------------
            # Degree mass regularization
            # ---------------------------------------------------
            gen_edges = torch.cat(
                [prev_graph.edge_index, delta],
                dim=1
            )

            deg_loss = degree_mass_loss(
                curr_graph.edge_index,
                gen_edges,
                curr_graph.num_nodes,
                k=100
            )

            loss = bce_loss + 0.05 * deg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(
            f"Epoch {epoch} | "
            f"Loss {total_loss / max(steps,1):.4f}"
        )

        if epoch % 50 == 0 or epoch == args.epochs-1:
            torch.save(
                model.state_dict(),
                f"dycond_citation_{epoch}.pth"
            )


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="citation")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    add_model_args(parser)

    parser.set_defaults(
        arch="SparseDyCond",
        diffusion_dim=256,
        max_nodes=21000,   # MUST match preprocessing
    )

    args = parser.parse_args()
    train(args)
