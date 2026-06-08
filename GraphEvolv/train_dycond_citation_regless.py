import torch
from tqdm import tqdm
from model import get_model
import os
import argparse
from torch_geometric.data import Data


# -----------------------------
# SAFETY
# -----------------------------
def force_safety(graph, capacity, device):
    if not hasattr(graph, "batch_nodes") or graph.batch_nodes is None:
        graph.batch_nodes = torch.arange(graph.num_nodes, device=device)
    graph.batch_nodes = graph.batch_nodes % capacity
    return graph


# -----------------------------
# DELTA EDGE GENERATION
# -----------------------------
def generate_for_checkpoint(args, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 🚀 Citation Generation with {checkpoint_path} on {device} ---")

    # Load model
    model = get_model(args, initial_graph_sampler=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    capacity = model.node_emb.weight.size(0)

    # Load seed graph (t = 0)
    seed_path = f"data/processed/{args.dataset}/snap_0.pt"
    prev_graph = torch.load(seed_path).to(device)

    prev_graph.y = torch.zeros(prev_graph.num_nodes, dtype=torch.long, device=device)
    prev_graph.batch = torch.zeros(prev_graph.num_nodes, dtype=torch.long, device=device)
    prev_graph = force_safety(prev_graph, capacity, device)

    generated = [prev_graph.clone().cpu()]

    # --------------------------------------
    # Temporal rollout
    # --------------------------------------
    for step in tqdm(range(1, args.steps + 1), desc="Temporal rollout"):
        num_nodes = prev_graph.num_nodes

        with torch.no_grad():
            # Dummy time (citation model is time-agnostic)
            t = torch.zeros(1, device=device, dtype=torch.long)

            node_emb, edge_head = model(
                prev_graph,
                prev_graph,
                t=t,
                y=prev_graph.y
            )

            # ----------------------------------
            # Candidate edges: (u -> v), u < v
            # ----------------------------------
            num_candidates = args.candidates
            src = torch.randint(0, num_nodes, (num_candidates,), device=device)
            dst = torch.randint(0, num_nodes, (num_candidates,), device=device)

            scores = edge_head(
                torch.cat([node_emb[src], node_emb[dst]], dim=1)
            ).squeeze()

            probs = torch.sigmoid(scores)

            # Threshold-based edge addition
            mask = probs > args.edge_threshold

            new_edges = torch.stack([src[mask], dst[mask]])

            # ----------------------------------
            # Merge edges (monotonic growth)
            # ----------------------------------
            if new_edges.numel() > 0:
                combined_edges = torch.cat(
                    [prev_graph.edge_index, new_edges], dim=1
                )
            else:
                combined_edges = prev_graph.edge_index

            curr_graph = Data(
                x=prev_graph.x,
                edge_index=combined_edges,
                num_nodes=num_nodes
            )

            curr_graph.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
            curr_graph.batch_nodes = torch.arange(num_nodes, device=device)
            curr_graph.y = torch.zeros(num_nodes, dtype=torch.long, device=device)

            curr_graph = force_safety(curr_graph, capacity, device)

        generated.append(curr_graph.clone().cpu())
        prev_graph = curr_graph

    return generated


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Same interface as Reddit
    parser.add_argument("--dataset", type=str, default="citation")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)

    # Model args
    parser.add_argument("--arch", type=str, default="SparseDyCond")
    parser.add_argument("--max_nodes", type=int, default=50000)
    parser.add_argument("--diffusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, nargs="*", default=[8, 8, 8, 8, 8])
    parser.add_argument("--diffusion_steps", type=int, default=1000)

    # Citation-specific inference knobs
    parser.add_argument("--candidates", type=int, default=20000,
                        help="Number of candidate citation edges per step")
    parser.add_argument("--edge_threshold", type=float, default=0.5,
                        help="Probability threshold for adding edges")

    # Dummy args (required by model)
    parser.add_argument("--loss_type", type=str, default="vb_kl")
    parser.add_argument("--dp_rate", type=float, default=0.)
    parser.add_argument("--final_prob_node", type=float, nargs="*", default=None)
    parser.add_argument("--final_prob_edge", type=float, nargs="*", default=[0.9, 0.1])
    parser.add_argument("--parametrization", type=str, default="x0")
    parser.add_argument("--sample_time_method", type=str, default="importance")
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--norm", type=str, default="None")

    args = parser.parse_args()

    for ckpt_epoch in args.checkpoints:
        ckpt_path = f"dycond_citation_{ckpt_epoch}.pth"
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Skipping missing checkpoint: {ckpt_path}")
            continue

        generated = generate_for_checkpoint(args, ckpt_path)

        save_dir = f"results/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)

        save_path = f"{save_dir}/generated_timeline_epoch_{ckpt_epoch}.pt"
        torch.save(generated, save_path)
        print(f"✅ Saved: {save_path}")

