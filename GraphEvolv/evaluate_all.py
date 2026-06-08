"""
Comprehensive evaluation of generated graphs across all datasets and losses.

Computes:
  1. MMD metrics (Degree, Clustering, Spectral)
  2. Graph-level statistics (edges, LCC, avg/max degree, triangles, density)
  3. Power-law exponent fitting
  4. Distribution plots
  5. Cross-dataset summary table (LaTeX-formatted)

Usage:
    python evaluate_all.py
    python evaluate_all.py --dataset collegemsg --loss_type deg_slope --epochs 500,1000
"""

import torch
import os
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import json
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# Graph Statistics Helpers
# ═══════════════════════════════════════════════════════════════

def safe_to_nx(data):
    try:
        G = to_networkx(data, to_undirected=True)
        if G.number_of_nodes() == 0:
            return None
        return G
    except Exception:
        return None


def degree_and_clustering(G, max_nodes=500):
    deg = np.array([d for _, d in G.degree()])
    if G.number_of_nodes() > max_nodes:
        nodes = np.random.choice(list(G.nodes()), max_nodes, replace=False)
        clus = list(nx.clustering(G, nodes).values())
    else:
        clus = list(nx.clustering(G).values())
    return deg, np.array(clus)


def spectral_density(G, k=50, max_nodes=1000):
    n = G.number_of_nodes()
    if n < 5:
        return np.zeros(k)
    if n > max_nodes:
        nodes = np.random.choice(list(G.nodes()), max_nodes, replace=False)
        G = G.subgraph(nodes)
    try:
        L = nx.normalized_laplacian_matrix(G)
        # Use direct dense solver which is extremely fast and guaranteed to converge
        L_dense = L.toarray()
        evals = np.linalg.eigvalsh(L_dense)
        # Sort by absolute value (magnitude) to match "LM" (Largest Magnitude) behavior of eigsh
        evals = evals[np.argsort(np.abs(evals))]
        kk = min(k, L.shape[0] - 2)
        if kk <= 0:
            return np.zeros(k)
        evals = evals[-kk:]
    except Exception:
        return np.zeros(k)
    if len(evals) < k:
        evals = np.pad(evals, (0, k - len(evals)))
    return np.real(evals)


def mmd(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return wasserstein_distance(a, b)


def compute_pagerank(G, alpha=0.85):
    try:
        return list(nx.pagerank(G, alpha=alpha).values())
    except Exception:
        return [1.0 / max(G.number_of_nodes(), 1)] * G.number_of_nodes()


def largest_cc_size(G):
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


def fit_power_law_exponent(deg):
    """Fit power-law exponent using MLE (Clauset et al.)."""
    deg = deg[deg >= 1]
    if len(deg) < 10:
        return None, None
    # Simple MLE for power-law: alpha = 1 + n / sum(ln(x/xmin))
    x_min = max(deg.min(), 1)
    n = len(deg)
    alpha = 1 + n / np.sum(np.log(deg / x_min))
    # KS statistic (goodness of fit)
    sorted_deg = np.sort(deg)
    cdf_empirical = np.arange(1, len(sorted_deg) + 1) / len(sorted_deg)
    cdf_theoretical = 1 - (sorted_deg / x_min) ** (-(alpha - 1))
    ks_stat = np.max(np.abs(cdf_empirical - cdf_theoretical))
    return alpha, ks_stat


def graph_statistics(G):
    """Compute comprehensive graph statistics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {}

    degs = [d for _, d in G.degree()]
    avg_deg = np.mean(degs) if degs else 0
    max_deg = max(degs) if degs else 0
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    lcc = largest_cc_size(G)
    triangles = sum(nx.triangles(G).values()) // 3 if n < 1000 and m < 5000 else -1

    return {
        'nodes': n,
        'edges': m,
        'avg_degree': avg_deg,
        'max_degree': max_deg,
        'density': density,
        'lcc': lcc,
        'triangles': triangles,
    }


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_distribution(real, gen, title, xlabel, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(real, bins=50, density=True, alpha=0.6, label="Real", color='#2196F3')
    plt.hist(gen, bins=50, density=True, alpha=0.6, label="Generated", color='#FF5722')
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_dataset(dataset, loss_type, checkpoints, data_root, results_root):
    """Evaluate generated graphs vs real graphs for a single dataset+loss."""
    real_path = os.path.join(data_root, 'processed', dataset)
    gen_dir = os.path.join(results_root, dataset, loss_type)

    if not os.path.exists(real_path):
        print(f"  ⚠️ Real data not found: {real_path}")
        return None

    # Load real snapshots (last 10)
    print(f"\n  Computing real graph statistics for {dataset}...")
    real_stats = {"deg": [], "clus": [], "eig": [], "pr": []}
    real_graph_stats = []

    real_files = sorted(
        [f for f in os.listdir(real_path) if f.startswith('snap_')],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )[-10:]

    for f in tqdm(real_files, desc="Real graphs", leave=False):
        G = safe_to_nx(torch.load(os.path.join(real_path, f), weights_only=False))
        if G is None:
            continue
        d, c = degree_and_clustering(G)
        e = spectral_density(G)
        pr = compute_pagerank(G)
        real_stats["deg"].extend(d.tolist())
        real_stats["clus"].extend(c.tolist())
        real_stats["eig"].extend(e.tolist())
        real_stats["pr"].extend(pr)
        real_graph_stats.append(graph_statistics(G))

    # Power-law of real graph
    real_deg_arr = np.array(real_stats["deg"])
    real_alpha, real_ks = fit_power_law_exponent(real_deg_arr)

    # Evaluate each checkpoint
    results = []

    print(f"\n  {'Epoch':>6} | {'Deg MMD':>9} | {'Clus MMD':>9} | {'Spec MMD':>9} | {'PR MMD':>9} | "
          f"{'Edges':>9} | {'LCC':>9} | {'AvgDeg':>8} | {'MaxDeg':>8} | "
          f"{'α_gen':>8} | {'KS':>8}")
    print("-" * 122)

    for tag in tqdm(checkpoints, desc="Evaluating checkpoints"):
        gen_path = os.path.join(gen_dir, f'generated_timeline_epoch_{tag}.pt')
        if not os.path.exists(gen_path):
            print(f"  ⚠️ Skipping epoch {tag}: {gen_path} not found")
            continue

        gen_stats = {"deg": [], "clus": [], "eig": [], "pr": []}
        gen_graph_stats = []

        gen_snaps = torch.load(gen_path, weights_only=False)[-10:]

        for idx, data in enumerate(gen_snaps):
            n_nodes = data.num_nodes if hasattr(data, 'num_nodes') else '?'
            n_edges = data.edge_index.size(1) if hasattr(data, 'edge_index') else '?'
            print(f"    -> [Snap {idx+1}/10] Nodes: {n_nodes}, Edges: {n_edges}")
            print(f"    -> [Snap {idx+1}/10] converting to NetworkX...")
            G = safe_to_nx(data)
            if G is None:
                continue
            print(f"    -> [Snap {idx+1}/10] degree & clustering...")
            d, c = degree_and_clustering(G)
            print(f"    -> [Snap {idx+1}/10] spectral density...")
            e = spectral_density(G)
            pr = compute_pagerank(G)
            gen_stats["deg"].extend(d.tolist())
            gen_stats["clus"].extend(c.tolist())
            gen_stats["eig"].extend(e.tolist())
            gen_stats["pr"].extend(pr)
            print(f"    -> [Snap {idx+1}/10] graph stats...")
            gen_graph_stats.append(graph_statistics(G))

        # MMDs
        deg_mmd = mmd(real_stats["deg"], gen_stats["deg"])
        clus_mmd = mmd(real_stats["clus"], gen_stats["clus"])
        spec_mmd = mmd(real_stats["eig"], gen_stats["eig"])
        pr_mmd = mmd(real_stats["pr"], gen_stats["pr"])

        # Graph stats (averages)
        avg_edges = int(np.mean([s['edges'] for s in gen_graph_stats])) if gen_graph_stats else 0
        avg_lcc = int(np.mean([s['lcc'] for s in gen_graph_stats])) if gen_graph_stats else 0
        avg_deg = np.mean([s['avg_degree'] for s in gen_graph_stats]) if gen_graph_stats else 0
        max_deg = int(np.mean([s['max_degree'] for s in gen_graph_stats])) if gen_graph_stats else 0

        # Power-law fit
        gen_deg_arr = np.array(gen_stats["deg"])
        gen_alpha, gen_ks = fit_power_law_exponent(gen_deg_arr)

        # Print
        print(
            f"  {tag:>6} | {deg_mmd:9.4f} | {clus_mmd:9.4f} | {spec_mmd:9.4f} | {pr_mmd:9.4f} | "
            f"{avg_edges:9d} | {avg_lcc:9d} | {avg_deg:8.2f} | {max_deg:8d} | "
            f"{gen_alpha if gen_alpha else 0:8.3f} | {gen_ks if gen_ks else 0:8.4f}"
        )

        result = {
            'epoch': tag,
            'deg_mmd': deg_mmd,
            'clus_mmd': clus_mmd,
            'spec_mmd': spec_mmd,
            'pr_mmd': pr_mmd,
            'edges': avg_edges,
            'lcc': avg_lcc,
            'avg_degree': avg_deg,
            'max_degree': max_deg,
            'power_law_alpha': gen_alpha,
            'power_law_ks': gen_ks,
        }
        results.append(result)

        # Plots
        plot_dir = os.path.join(gen_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        plot_distribution(
            real_stats["deg"], gen_stats["deg"],
            f"{dataset} Degree Dist ({loss_type}, Epoch {tag})",
            "Degree", os.path.join(plot_dir, f'degree_epoch_{tag}.png')
        )
        plot_distribution(
            real_stats["clus"], gen_stats["clus"],
            f"{dataset} Clustering ({loss_type}, Epoch {tag})",
            "Clustering Coefficient",
            os.path.join(plot_dir, f'clustering_epoch_{tag}.png')
        )
        plot_distribution(
            real_stats["eig"], gen_stats["eig"],
            f"{dataset} Spectral ({loss_type}, Epoch {tag})",
            "Eigenvalue",
            os.path.join(plot_dir, f'spectral_epoch_{tag}.png')
        )
        plot_distribution(
            real_stats["pr"], gen_stats["pr"],
            f"{dataset} PageRank ({loss_type}, Epoch {tag})",
            "PageRank",
            os.path.join(plot_dir, f'pagerank_epoch_{tag}.png')
        )

    # Save results
    eval_output = {
        'dataset': dataset,
        'loss_type': loss_type,
        'real_power_law_alpha': real_alpha,
        'real_power_law_ks': real_ks,
        'real_graph_stats': real_graph_stats,
        'results': results,
    }

    os.makedirs(gen_dir, exist_ok=True)
    with open(os.path.join(gen_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_output, f, indent=2, default=str)

    return eval_output


def generate_cross_dataset_table(results_root, datasets, loss_types):
    """Generate cross-dataset summary table (LaTeX)."""
    print("\n" + "=" * 100)
    print("CROSS-DATASET SUMMARY TABLE")
    print("=" * 100)

    loss_headers = [f"Loss ({lt})" for lt in loss_types]
    header = f"{'Dataset':<18} | {'Domain':<12} | " + " | ".join([f"{lh:>14}" for lh in loss_headers]) + " | Best"
    print(header)
    print("-" * (36 + 17 * len(loss_types)))

    domain_map = {
        'collegemsg': 'Social',
        'bitcoin_alpha': 'Financial',
        'euroroad': 'Road',
        'pp_pathways': 'Protein',
        'cit_hepph': 'Citation',
    }

    latex_rows = []

    for ds in datasets:
        row = {'dataset': ds, 'domain': domain_map.get(ds, '?')}
        best_val = float('inf')
        best_loss = '?'

        for lt in loss_types:
            eval_path = os.path.join(results_root, ds, lt, 'eval_results.json')
            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    data = json.load(f)
                if data.get('results'):
                    # Best degree MMD across checkpoints
                    best_deg = min(r['deg_mmd'] for r in data['results'])
                    row[lt] = best_deg
                    if best_deg < best_val:
                        best_val = best_deg
                        best_loss = lt
                else:
                    row[lt] = None
            else:
                row[lt] = None

        row['best'] = best_loss

        # Print
        vals = []
        for lt in loss_types:
            v = row.get(lt)
            if v is not None:
                vals.append(f"{v:14.4f}")
            else:
                vals.append(f"{'N/A':>14}")

        print(f"  {ds:<16} | {row['domain']:<12} | {' | '.join(vals)} | {best_loss:>6}")

        # LaTeX
        latex_vals = []
        for lt in loss_types:
            v = row.get(lt)
            if v is not None:
                if lt == best_loss:
                    latex_vals.append(f"\\textbf{{{v:.4f}}}")
                else:
                    latex_vals.append(f"{v:.4f}")
            else:
                latex_vals.append("---")

        latex_rows.append(
            f"  {ds.replace('_', '-')} & {row['domain']} & "
            f"{' & '.join(latex_vals)} \\\\"
        )

    # Print LaTeX table
    col_spec = "ll" + "c" * len(loss_types)
    headers_latex = " & ".join([f"\\textbf{{{lt.replace('_', ' ').title()}}}" for lt in loss_types])

    print("\n\n% === LaTeX Table ===")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Cross-dataset comparison of best Degree MMD (lower is better).}")
    print("\\label{tab:cross_dataset}")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    print(f"\\textbf{{Dataset}} & \\textbf{{Domain}} & {headers_latex} \\\\")
    print("\\midrule")
    for row in latex_rows:
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def print_formatted_summary(results_root, dataset, loss_types, checkpoints):
    loss_mapping = {
        'edge_only': 'Ledge',
        'deg_hist': 'Ldeg',
        'deg_slope': 'Lslope',
        'deg_combined': 'Ldeg_combined',
        'pr_hist': 'Lpr_hist',
        'pr_slope': 'Lpr_slope',
        'pr_combined': 'Lpr_combined',
        'combined': 'Lcombined'
    }

    print("\n" + "=" * 50)
    print(f"FORMATTED SUMMARY FOR DATASET: {dataset.upper()}")
    print("=" * 50)

    for lt in loss_types:
        header_name = loss_mapping.get(lt, lt)
        print(header_name)
        print("Epoch Deg_MMD Clus_MMD Spec_MMD PR_MMD Edges LCC")
        eval_path = os.path.join(results_root, dataset, lt, 'eval_results.json')
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                data = json.load(f)

            results_by_epoch = {str(r['epoch']): r for r in data.get('results', [])}
            for tag in checkpoints:
                r = results_by_epoch.get(str(tag))
                if r:
                    deg = r['deg_mmd']
                    clus = r['clus_mmd']
                    spec = r['spec_mmd']
                    pr = r.get('pr_mmd', 0.0)
                    edges = r['edges']
                    lcc = r['lcc']
                    print(f"{tag} {deg:.4f} {clus:.4f} {spec:.4f} {pr:.4f} {edges:,} {lcc:,}")
                else:
                    print(f"{tag} N/A N/A N/A N/A N/A N/A")
        else:
            for tag in checkpoints:
                print(f"{tag} N/A N/A N/A N/A N/A N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyCond Multi-Dataset Evaluation")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Evaluate single dataset (default: all)")
    parser.add_argument("--dycond_loss_type", "--loss_type", dest="dycond_loss_type", type=str, default=None,
                        choices=["edge_only", "deg_hist", "deg_slope", "deg_combined", "pr_hist", "pr_slope", "pr_combined", "combined"],
                        help="Evaluate single loss type (default: all)")
    parser.add_argument("--epochs", type=str, default="250,500,750,1000,best",
                        help="Comma-separated checkpoint epochs to evaluate")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    data_root = 'data'
    results_root = 'results'

    all_datasets = ['collegemsg', 'bitcoin_alpha', 'euroroad', 'pp_pathways', 'cit_hepph']
    # all_loss_types = ['edge_only', 'deg_hist', 'deg_slope', 'deg_combined', 'pr_hist', 'pr_slope', 'pr_combined', 'combined']
    all_loss_types = ['edge_only','deg_slope','combined']


    datasets = [args.dataset] if args.dataset else all_datasets
    loss_types = [args.dycond_loss_type] if args.dycond_loss_type else all_loss_types
    checkpoints = [e.strip() for e in args.epochs.split(",")]

    for ds in tqdm(datasets, desc="Evaluation Datasets"):
        for lt in loss_types:
            print(f"\n{'='*60}")
            print(f"Evaluating: {ds} | Loss: {lt}")
            print(f"{'='*60}")
            evaluate_dataset(ds, lt, checkpoints, data_root, results_root)

    # Print formatted tables for all evaluated datasets
    for ds in datasets:
        print_formatted_summary(results_root, ds, loss_types, checkpoints)

    # Cross-dataset table
    if len(datasets) > 1 and len(loss_types) > 1:
        generate_cross_dataset_table(results_root, datasets, all_loss_types)
