import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.stats import wasserstein_distance
from torch_geometric.utils import to_networkx
import os
import argparse
from tqdm import tqdm

# -----------------------
# METRIC HELPERS
# -----------------------

def get_degree_clustering(G):
    degrees = [d for _, d in G.degree()]

    if G.number_of_nodes() > 5000:
        nodes = np.random.choice(list(G.nodes()), 5000, replace=False)
        clus_dict = nx.clustering(G, nodes=nodes)
    else:
        clus_dict = nx.clustering(G)

    return np.array(degrees), np.array(list(clus_dict.values()))


def get_spectral_density(G, k=100):
    if G.number_of_nodes() < 5 or G.number_of_edges() == 0:
        return np.zeros(k)

    L = nx.normalized_laplacian_matrix(G)

    try:
        k_eff = min(k, G.number_of_nodes() - 2)
        evals = eigsh(L, k=k_eff, which='SM', return_eigenvectors=False)
    except Exception:
        try:
            evals = scipy.linalg.eigvalsh(L.todense())[:k]
        except Exception:
            evals = np.zeros(k)

    if len(evals) < k:
        evals = np.pad(evals, (0, k - len(evals)))

    return evals


def compute_mmd(a, b):
    if len(a) == 0 or len(b) == 0:
        return 1.0
    return wasserstein_distance(a, b)


def graph_sanity(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    lcc = max((len(c) for c in nx.connected_components(G)), default=0)
    return m, lcc


# -----------------------
# PLOTTING
# -----------------------

def plot_hist(real, gen, title, xlabel, save_path, log=False):
    plt.figure(figsize=(6, 5))

    if log:
        bins = np.logspace(np.log10(max(1, min(real))), np.log10(max(real)), 50)
        plt.xscale('log')
        plt.yscale('log')
    else:
        bins = 50

    plt.hist(real, bins=bins, density=True, alpha=0.6, label='Real')
    plt.hist(gen, bins=bins, density=True, alpha=0.6, label='Generated')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()


# -----------------------
# MAIN EVAL
# -----------------------

def evaluate(dataset, epochs):
    print(f"\n--- 🔍 ICML Multi-Checkpoint Evaluation ({dataset}) ---")

    # Load real graphs
    real_snaps = []
    for i in range(40, 50):
        try:
            real_snaps.append(torch.load(f"data/processed/{dataset}/snap_{i}.pt"))
        except Exception:
            break

    real_stats = {
        'deg': [], 'clus': [], 'eig': [],
        'edges': [], 'lcc': []
    }

    print("Computing real graph statistics...")
    for d in tqdm(real_snaps):
        G = to_networkx(d, to_undirected=True)
        deg, clus = get_degree_clustering(G)
        eig = get_spectral_density(G)
        m, lcc = graph_sanity(G)

        real_stats['deg'].extend(deg)
        real_stats['clus'].extend(clus)
        real_stats['eig'].extend(eig)
        real_stats['edges'].append(m)
        real_stats['lcc'].append(lcc)

    print("\n--- RESULTS ---")
    print(f"{'Epoch':>6} | {'Deg MMD':>8} | {'Clus MMD':>9} | {'Spec MMD':>9} | {'Edges':>8} | {'LCC':>8}")
    print("-" * 70)

    for epoch in epochs:
        gen_path = f"results/{dataset}/generated_timeline_epoch_{epoch}.pt"
        if not os.path.exists(gen_path):
            print(f"{epoch:>6} | ❌ missing")
            continue

        gen_snaps = torch.load(gen_path)

        gen_stats = {
            'deg': [], 'clus': [], 'eig': [],
            'edges': [], 'lcc': []
        }

        for d in tqdm(gen_snaps, desc=f"Epoch {epoch}", leave=False):
            G = to_networkx(d, to_undirected=True)
            deg, clus = get_degree_clustering(G)
            eig = get_spectral_density(G)
            m, lcc = graph_sanity(G)

            gen_stats['deg'].extend(deg)
            gen_stats['clus'].extend(clus)
            gen_stats['eig'].extend(eig)
            gen_stats['edges'].append(m)
            gen_stats['lcc'].append(lcc)

        mmd_deg = compute_mmd(real_stats['deg'], gen_stats['deg'])
        mmd_clus = compute_mmd(real_stats['clus'], gen_stats['clus'])
        mmd_eig = compute_mmd(real_stats['eig'], gen_stats['eig'])

        print(f"{epoch:>6} | {mmd_deg:8.4f} | {mmd_clus:9.4f} | {mmd_eig:9.4f} | "
              f"{np.mean(gen_stats['edges']):8.0f} | {np.mean(gen_stats['lcc']):8.0f}")

        # Save plots
        plot_dir = f"results/{dataset}/plots_epoch_{epoch}"
        os.makedirs(plot_dir, exist_ok=True)

        plot_hist(real_stats['deg'], gen_stats['deg'],
                  "Degree Distribution (Log-Log)",
                  "Degree", f"{plot_dir}/degree.png", log=True)

        plot_hist(real_stats['clus'], gen_stats['clus'],
                  "Clustering Coefficient",
                  "Clustering", f"{plot_dir}/clustering.png")

        plot_hist(real_stats['eig'], gen_stats['eig'],
                  "Spectral Density",
                  "Eigenvalue", f"{plot_dir}/spectral.png")


# -----------------------
# ENTRY
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='reddit')
    parser.add_argument('--epochs', type=int, nargs="+", required=True)
    args = parser.parse_args()

    evaluate(args.dataset, args.epochs)
