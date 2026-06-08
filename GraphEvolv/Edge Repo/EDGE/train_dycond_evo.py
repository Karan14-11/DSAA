import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from datasets.paired_loader import get_paired_loader
from model import get_model, add_model_args


# -------------------------------------------------
# REGULARIZERS (CORE RESEARCH IDEA)
# -------------------------------------------------


# -------------------------------------------------
# STRUCTURAL REGULARIZERS
# -------------------------------------------------

def degree_regularizer(edge_index, edge_scores, num_nodes):
    """
    Encourages degree distribution stability
    """
    probs = torch.sigmoid(edge_scores)
    src = edge_index[0]

    deg = torch.zeros(num_nodes, device=probs.device)
    deg.index_add_(0, src, probs)

    # Penalize degree variance explosion
    return torch.var(deg)


def pagerank_regularizer(edge_index, edge_scores, num_nodes, alpha=0.85, iters=3):
    """
    Encourages global importance consistency
    """
    device = edge_scores.device
    probs = torch.sigmoid(edge_scores)

    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=device)
    deg.index_add_(0, src, probs)
    deg_inv = 1.0 / (deg + 1e-8)

    pr = torch.ones(num_nodes, device=device) / num_nodes

    for _ in range(iters):
        msg = pr[src] * probs * deg_inv[src]
        agg = torch.zeros_like(pr)
        agg.index_add_(0, dst, msg)
        pr = alpha * agg + (1 - alpha) / num_nodes

    return torch.var(pr)


def reciprocity_regularizer(edge_index, edge_scores, samples=5000):
    """
    Encourages bidirectional consistency
    (cheap approximation)
    """
    device = edge_scores.device
    probs = torch.sigmoid(edge_scores)
    E = edge_index.size(1)

    if E == 0:
        return torch.tensor(0., device=device)

    idx = torch.randint(0, E, (min(samples, E),), device=device)
    return torch.mean(torch.abs(probs[idx] - probs[idx].mean()))



def sparsity_regularizer(edge_scores, target_density):
    """
    Control graph density
    """
    probs = torch.sigmoid(edge_scores)
    return torch.abs(probs.mean() - target_density)


def memory_energy_regularizer(node_memory):
    """
    Prevent node states from exploding
    """
    norms = torch.norm(node_memory, dim=1)
    return torch.var(norms)


# -------------------------------------------------
# TRAIN ONE EPOCH
# -------------------------------------------------

def train_epoch(model, optimizer, graph, t, y, args):
    model.train()
    optimizer.zero_grad()

    # ---- Forward ----
    edge_scores, node_memory = model(
        graph,
        t,
        y
    )

    # ---- Base likelihood (implicit edge existence) ----
    base_loss = F.binary_cross_entropy_with_logits(
        edge_scores,
        torch.ones_like(edge_scores)
    )

    # ---- Regularizers ----
    deg_loss = degree_regularizer(
        graph.edge_index,
        edge_scores,
        model.num_nodes
    )

    sparse_loss = sparsity_regularizer(
        edge_scores,
        args.target_density
    )

    mem_loss = memory_energy_regularizer(node_memory)

    # ---- Total loss ----
    loss = (
        base_loss
        + args.lambda_deg * deg_loss
        + args.lambda_sparse * sparse_loss
        + args.lambda_mem * mem_loss
    )

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "base": base_loss.item(),
        "deg": deg_loss.item(),
        "sparse": sparse_loss.item(),
        "mem": mem_loss.item()
    }


# -------------------------------------------------
# MAIN TRAIN LOOP
# -------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    loader = get_paired_loader(

        root_dir="data",
        name= args.dataset,
        batch_size=args.batch_size,
    )

    # ---- Model ----
    model = get_model(args,initial_graph_sampler=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training on {args.dataset} with {sum(p.numel() for p in model.parameters())} params")

    # ---- Epoch loop ----
    for epoch in range(1, args.epochs + 1):
        logs = []

        # for prev_graph, curr_graph in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        #     prev_graph = prev_graph.to(device)
        #     curr_graph = curr_graph.to(device)


        #     optimizer.zero_grad()

        #     # Time (implicit)
        #     t = torch.zeros(curr_graph.num_graphs, device=device)

        #     # Class conditioning
        #     y = curr_graph.y

        #     # Forward: predict edges of curr_graph conditioned on prev_graph
        #     node_emb, edge_scores = model(
        #         curr_graph=curr_graph,
        #         prev_graph=prev_graph,
        #         t=t,
        #         y=y
        #     )

        #     # Base loss (edge existence)
        #     bce = F.binary_cross_entropy_with_logits(
        #         edge_scores,
        #         torch.ones_like(edge_scores)
        #     )

        #     # (Optional) property regularizers go here


        #     deg_loss = degree_regularizer(
        #         curr_graph.edge_index,
        #         edge_scores,
        #         curr_graph.num_nodes
        #     )

        #     pr_loss = pagerank_regularizer(
        #         curr_graph.edge_index,
        #         edge_scores,
        #         curr_graph.num_nodes
        #     )

        #     rec_loss = reciprocity_regularizer(
        #         curr_graph.edge_index,
        #         edge_scores
        #     )

        #     # -----------------------------
        #     # Final loss
        #     # -----------------------------
        #     loss = (
        #         bce
        #         + args.lambda_deg * deg_loss
        #         + args.lambda_pr * pr_loss
        #         + args.lambda_rec * rec_loss
        #     )

        #     # loss = (
        #     #     bce
        #     #     + args.lambda_deg * deg_loss
        #     #     # + args.lambda_pr * pr_loss
        #     #     # + args.lambda_rec * rec_loss
        #     # )
        #     # loss = bce
        #     loss.backward()
        #     optimizer.step()




        for prev_graph, curr_graph in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            prev_graph = prev_graph.to(device)
            curr_graph = curr_graph.to(device)

            optimizer.zero_grad()

            # 1. Update memory using past
            model.update_memory(prev_graph.edge_index)

            # 2. Predict current edges
            logits = model(curr_graph.edge_index)

            bce = F.binary_cross_entropy_with_logits(
                logits, curr_graph.edge_label.float()
            )

            # 3. Structural regularizers (OUTSIDE model)
            deg_loss = degree_regularizer(logits, curr_graph)
            pr_loss  = pagerank_regularizer(logits, curr_graph)
            rec_loss = reciprocity_regularizer(logits, curr_graph)

            loss = (
                bce
                + args.lambda_deg * deg_loss
                + args.lambda_pr  * pr_loss
                + args.lambda_rec * rec_loss
            )

            loss.backward()
            optimizer.step()


        

       


        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")


# -------------------------------------------------
# ENTRY
# -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---- Training args ----
    parser.add_argument("--dataset", type=str, default="stackoverflow")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)

    # ---- Regularizer weights ----
    parser.add_argument("--lambda_deg", type=float, default=1.0)
    parser.add_argument("--lambda_pr", type=float, default=0.05)
    parser.add_argument("--lambda_rec", type=float, default=0.01)
    parser.add_argument("--target_density", type=float, default=0.01)

    # ---- Model args ----
    add_model_args(parser)
    parser.set_defaults(
        arch="SparseDyCond",
        emb_dim=256,
        num_nodes=21000,
    )

    args = parser.parse_args()
    train(args)
