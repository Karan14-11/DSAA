import torch
import os
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.sparse.linalg import eigsh
import scipy.linalg
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def safe_to_nx(data):
    try:
        G = to_networkx(data, to_undirected=True)
        if G.number_of_nodes() == 0:
            return None
        return G
    except Exception:
        return None


def degree_and_clustering(G, max_nodes=5000):
    deg = np.array([d for _, d in G.degree()])

    if G.number_of_nodes() > max_nodes:
        nodes = np.random.choice(list(G.nodes()), max_nodes, replace=False)
        clus = list(nx.clustering(G, nodes).values())
    else:
        clus = list(nx.clustering(G).values())

    return np.array(deg), np.array(clus)


def spectral_density(G, k=50, max_nodes=3000):
    """
    Fast + safe spectral proxy
    """
    n = G.number_of_nodes()
    if n < 5:
        return np.zeros(k)

    # Subsample nodes if graph is large
    if n > max_nodes:
        nodes = np.random.choice(list(G.nodes()), max_nodes, replace=False)
        G = G.subgraph(nodes)

    try:
        L = nx.normalized_laplacian_matrix(G)
        kk = min(k, L.shape[0] - 2)

        if kk <= 0:
            return np.zeros(k)

        # Use LARGEST magnitude (much more stable)
        evals = eigsh(
            L,
            k=kk,
            which="LM",
            return_eigenvectors=False,
            tol=1e-2,
            maxiter=500
        )

    except Exception:
        return np.zeros(k)

    if len(evals) < k:
        evals = np.pad(evals, (0, k - len(evals)))

    return np.real(evals)


def mmd(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return wasserstein_distance(a, b)


def largest_cc_size(G):
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


# -------------------------------------------------
# Plotting
# -------------------------------------------------

def plot_distribution(real, gen, title, xlabel, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(real, bins=50, density=True, alpha=0.6, label="Real")
    plt.hist(gen, bins=50, density=True, alpha=0.6, label="Generated")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

def evaluate(dataset, checkpoints):
    real_path = f"data/processed/{dataset}"
    gen_dir = f"results/{dataset}"
    plot_dir = f"{gen_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Load real snapshots (last 10)
    print("Computing real graph statistics...")
    real_stats = {"deg": [], "clus": [], "eig": []}

    real_files = sorted(
        [f for f in os.listdir(real_path)],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )[-10:]

    for f in tqdm(real_files):
        G = safe_to_nx(torch.load(os.path.join(real_path, f)))
        if G is None:
            continue
        d, c = degree_and_clustering(G)
        e = spectral_density(G)
        real_stats["deg"].extend(d)
        real_stats["clus"].extend(c)
        real_stats["eig"].extend(e)

    print("\n--- RESULTS ---")
    print(f"{'Epoch':>6} | {'Deg MMD':>9} | {'Clus MMD':>9} | {'Spec MMD':>9} | {'Edges':>9} | {'LCC':>9}")
    print("-" * 70)

    for tag in checkpoints:
        gen_path = f"{gen_dir}/generated_timeline_epoch_{tag}.pt"
        if not os.path.exists(gen_path):
            continue

        gen_stats = {"deg": [], "clus": [], "eig": []}
        edges = 0
        lcc = 0

        gen_snaps = torch.load(gen_path)[-10:]

        for data in gen_snaps:
            G = safe_to_nx(data)
            if G is None:
                continue
            edges += G.number_of_edges()
            lcc += largest_cc_size(G)

            d, c = degree_and_clustering(G)
            e = spectral_density(G)

            gen_stats["deg"].extend(d)
            gen_stats["clus"].extend(c)
            gen_stats["eig"].extend(e)

        edges //= max(len(gen_snaps), 1)
        lcc //= max(len(gen_snaps), 1)

        # ---- Print table row
        print(
            f"{tag:>6} | "
            f"{mmd(real_stats['deg'], gen_stats['deg']):9.4f} | "
            f"{mmd(real_stats['clus'], gen_stats['clus']):9.4f} | "
            f"{mmd(real_stats['eig'], gen_stats['eig']):9.4f} | "
            f"{edges:9d} | "
            f"{lcc:9d}"
        )

        # ---- Plots
        plot_distribution(
            real_stats["deg"], gen_stats["deg"],
            f"Degree Distribution (Epoch {tag})",
            "Degree",
            f"{plot_dir}/degree_epoch_{tag}.png"
        )

        plot_distribution(
            real_stats["clus"], gen_stats["clus"],
            f"Clustering Coefficient (Epoch {tag})",
            "Clustering Coefficient",
            f"{plot_dir}/clustering_epoch_{tag}.png"
        )

        plot_distribution(
            real_stats["eig"], gen_stats["eig"],
            f"Spectral Density (Epoch {tag})",
            "Eigenvalue",
            f"{plot_dir}/spectral_epoch_{tag}.png"
        )


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="reddit")
    parser.add_argument(
        "--epochs",
        type=str,
        default="best,1300,1500,1900",
        help="Comma-separated checkpoints"
    )
    args = parser.parse_args()

    checkpoints = [e.strip() for e in args.epochs.split(",")]
    evaluate(args.dataset, checkpoints)
