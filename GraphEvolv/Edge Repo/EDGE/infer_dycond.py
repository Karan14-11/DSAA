import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import get_model
import os
import argparse
from torch_geometric.data import Data

# -----------------------------
# SAFETY: batch_nodes modulo
# -----------------------------
def force_safety(graph, capacity, device):
    if not hasattr(graph, 'batch_nodes') or graph.batch_nodes is None:
        graph.batch_nodes = torch.arange(graph.num_nodes, device=device)
    graph.batch_nodes = graph.batch_nodes % capacity
    return graph


# -----------------------------
# MAIN GENERATION FUNCTION
# -----------------------------
def generate_for_checkpoint(args, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- 🚀 Generating with {checkpoint_path} on {device} ---")

    # 1. Load model
    model = get_model(args, initial_graph_sampler=None)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    capacity = model.node_emb.weight.size(0)

    # 2. Load seed graph
    seed_path = f"data/processed/{args.dataset}/snap_0.pt"
    prev_graph = torch.load(seed_path).to(device)

    # Disable label conditioning (CRITICAL)
    prev_graph.y = torch.zeros(prev_graph.num_nodes, dtype=torch.long, device=device)

    prev_graph.batch = torch.zeros(prev_graph.num_nodes, dtype=torch.long, device=device)
    prev_graph = force_safety(prev_graph, capacity, device)

    generated = [prev_graph.clone().cpu()]

    sampling_steps = 50

    # 3. Temporal rollout
    for step in range(args.steps):
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


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='reddit')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--checkpoints', type=int, nargs="+", required=True)

    # MODEL ARGS (must match training)
    parser.add_argument('--arch', type=str, default='SparseDyCond')
    parser.add_argument('--max_nodes', type=int, default=25000)
    parser.add_argument('--diffusion_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, nargs="*", default=[8, 8, 8, 8, 8, 8])
    parser.add_argument('--diffusion_steps', type=int, default=1000)

    # Dummy args
    parser.add_argument('--loss_type', type=str, default='vb_kl')
    parser.add_argument('--dp_rate', type=float, default=0.)
    parser.add_argument('--final_prob_node', type=float, nargs="*", default=None)
    parser.add_argument('--final_prob_edge', type=float, nargs="*", default=[0.9, 0.1])
    parser.add_argument('--parametrization', type=str, default='x0')
    parser.add_argument('--sample_time_method', type=str, default='importance')
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--norm', type=str, default='None')

    args = parser.parse_args()

    for ckpt_epoch in args.checkpoints:
        ckpt_path = f"dycond_checkpoint_{ckpt_epoch}.pth"
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Skipping missing checkpoint: {ckpt_path}")
            continue

        generated = generate_for_checkpoint(args, ckpt_path)

        save_dir = f"results/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)

        save_path = f"{save_dir}/generated_timeline_epoch_{ckpt_epoch}.pt"
        torch.save(generated, save_path)
        print(f"✅ Saved: {save_path}")

    
    ckpt_path = f"dycond_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Skipping missing checkpoint: {ckpt_path}")
        

    generated = generate_for_checkpoint(args, ckpt_path)

    save_dir = f"results/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/generated_timeline_epoch_{ckpt_epoch}.pt"
    torch.save(generated, save_path)
    print(f"✅ Saved: {save_path}")