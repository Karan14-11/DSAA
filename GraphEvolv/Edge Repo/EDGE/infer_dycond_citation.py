import torch
from tqdm import tqdm
from model import get_model
import argparse
import os
from torch_geometric.data import Data


# -------------------------
# SAFETY
# -------------------------
def force_safety(graph, capacity, device):
    graph = graph.to(device)
    if not hasattr(graph, "batch_nodes") or graph.batch_nodes is None:
        graph.batch_nodes = torch.arange(
            graph.num_nodes, device=device
        )
    else:
        graph.batch_nodes = graph.batch_nodes.to(device)

    graph.batch_nodes = graph.batch_nodes % capacity
    return graph


# -------------------------
# PER-NODE DEGREE BUDGET
# -------------------------
def degree_budget_decode(src, dst, probs, deg_budget):
    keep = torch.zeros_like(probs, dtype=torch.bool)
    for u in torch.unique(src):
        idx = (src == u)
        k = deg_budget[u].item()
        if k <= 0:
            continue
        topk = torch.topk(probs[idx], min(k, idx.sum())).indices
        keep[idx.nonzero()[topk]] = True
    return keep


# -------------------------
# GENERATION
# -------------------------
def generate_for_checkpoint(args, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Generating with {ckpt_path}")

    model = get_model(args, initial_graph_sampler=None)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()

    capacity = model.node_emb.weight.size(0)

    seed = torch.load(f"data/processed/{args.dataset}/snap_0.pt").to(device)
    seed.batch = torch.zeros(seed.num_nodes, device=device)
    seed.y = torch.zeros(seed.num_nodes, device=device)
    seed = force_safety(seed, capacity, device)

    graphs = [seed.cpu()]
    prev = seed

    for _ in range(args.steps):
        num_nodes = prev.num_nodes

        src = torch.arange(num_nodes, device=device)
        dst = torch.randint(0, num_nodes, (num_nodes,), device=device)

        with torch.no_grad():
            t = torch.zeros(prev.num_graphs, device=device, dtype=torch.long)

            node_emb, edge_head = model(prev, prev, t, prev.y)

            scores = edge_head(
                torch.cat([node_emb[src], node_emb[dst]], dim=1)
            ).view(-1)

            probs = torch.sigmoid(scores)

        true_deg = torch.bincount(
            prev.edge_index[0], minlength=num_nodes
        )

        mask = degree_budget_decode(src, dst, probs, true_deg)

        new_edges = torch.stack([src[mask], dst[mask]])

        curr = Data(
            x=prev.x,
            edge_index=torch.cat([prev.edge_index, new_edges], dim=1)
        )

        curr = force_safety(curr, capacity, device)

        graphs.append(curr.cpu())
        prev = curr

    return graphs


# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="citation")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)

    parser.add_argument("--arch", type=str, default="SparseDyCond")
    parser.add_argument("--max_nodes", type=int, default=21000)
    parser.add_argument("--diffusion_dim", type=int, default=256)

    parser.add_argument("--num_heads", type=int, nargs="*", default=[8, 8, 8, 8, 8])
    parser.add_argument("--diffusion_steps", type=int, default=1000)

    parser.add_argument("--loss_type", type=str, default="vb_kl")
    parser.add_argument("--dp_rate", type=float, default=0.)
    parser.add_argument("--final_prob_node", type=float, nargs="*", default=None)
    parser.add_argument("--final_prob_edge", type=float, nargs="*", default=[0.9, 0.1])
    parser.add_argument("--parametrization", type=str, default="x0")
    parser.add_argument("--sample_time_method", type=str, default="importance")
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--norm", type=str, default="None")

    parser.add_argument("--candidates", type=int, default=20000,
                        help="Candidate citation edges per step")

    args = parser.parse_args()

    for ck in args.checkpoints:
        path = f"dycond_citation_{ck}.pth"
        if not os.path.exists(path):
            continue

        graphs = generate_for_checkpoint(args, path)
        os.makedirs("results/citation", exist_ok=True)
        torch.save(graphs, f"results/citation/gen_{ck}.pt")
