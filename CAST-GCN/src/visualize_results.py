"""
Visualize experiment results: comparison plots for all strategies.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(path='results/experiment_results.json'):
    with open(path) as f:
        return json.load(f)


STRATEGY_LABELS = {
    'gcn_static': 'GCN (Static)',
    'gcn_retrain': 'GCN (Retrain)',
    'inc_no_update': 'Inc-GCN (No Update)',
    'inc_cached_ax': 'Inc-GCN (Cached AX)',
    'inc_subgraph': 'Inc-GCN (Subgraph)',
    'inc_svd': 'Inc-GCN (SVD Selective)',
}

COLORS = {
    'gcn_static': '#e74c3c',
    'gcn_retrain': '#3498db',
    'inc_no_update': '#95a5a6',
    'inc_cached_ax': '#2ecc71',
    'inc_subgraph': '#f39c12',
    'inc_svd': '#9b59b6',
}

MARKERS = {
    'gcn_static': 'x',
    'gcn_retrain': 's',
    'inc_no_update': 'v',
    'inc_cached_ax': 'o',
    'inc_subgraph': 'D',
    'inc_svd': '^',
}


def plot_auc_over_snapshots(data, save_path='results/auc_comparison.png'):
    """Plot AUC-ROC over test snapshots."""
    fig, ax = plt.subplots(figsize=(12, 6))

    num_train = data['metadata']['num_train']
    results = data['results']

    for strategy in results:
        aucs = [r['auc'] for r in results[strategy]]
        x = list(range(num_train, num_train + len(aucs)))
        ax.plot(x, aucs,
                label=STRATEGY_LABELS.get(strategy, strategy),
                color=COLORS.get(strategy, 'gray'),
                marker=MARKERS.get(strategy, 'o'),
                markersize=6, linewidth=2, alpha=0.85)

    ax.set_xlabel('Snapshot Index', fontsize=13)
    ax.set_ylabel('AUC-ROC', fontsize=13)
    ax.set_title('Link Prediction AUC-ROC Across Test Snapshots', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_ap_over_snapshots(data, save_path='results/ap_comparison.png'):
    """Plot Average Precision over test snapshots."""
    fig, ax = plt.subplots(figsize=(12, 6))

    num_train = data['metadata']['num_train']
    results = data['results']

    for strategy in results:
        aps = [r['ap'] for r in results[strategy]]
        x = list(range(num_train, num_train + len(aps)))
        ax.plot(x, aps,
                label=STRATEGY_LABELS.get(strategy, strategy),
                color=COLORS.get(strategy, 'gray'),
                marker=MARKERS.get(strategy, 'o'),
                markersize=6, linewidth=2, alpha=0.85)

    ax.set_xlabel('Snapshot Index', fontsize=13)
    ax.set_ylabel('Average Precision', fontsize=13)
    ax.set_title('Link Prediction Average Precision Across Test Snapshots', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_time_comparison(data, save_path='results/time_comparison.png'):
    """Bar chart comparing total time for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    summary = data['summary']
    strategies = list(summary.keys())
    total_times = [summary[s]['total_time'] for s in strategies]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    colors = [COLORS.get(s, 'gray') for s in strategies]

    bars = ax.barh(labels, total_times, color=colors, edgecolor='white', linewidth=1.5)

    for bar, t in zip(bars, total_times):
        ax.text(bar.get_width() + max(total_times) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{t:.1f}s', va='center', fontsize=10)

    ax.set_xlabel('Total Time (seconds)', fontsize=13)
    ax.set_title('Computation Time Comparison', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_summary_table(data, save_path='results/summary_table.png'):
    """Render summary metrics as a table image."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    summary = data['summary']
    strategies = list(summary.keys())

    headers = ['Strategy', 'Mean AUC', 'Std AUC', 'Mean AP', 'Std AP', 'Total Time (s)']
    rows = []
    for s in strategies:
        d = summary[s]
        rows.append([
            STRATEGY_LABELS.get(s, s),
            f"{d['mean_auc']:.4f}",
            f"{d['std_auc']:.4f}",
            f"{d['mean_ap']:.4f}",
            f"{d['std_ap']:.4f}",
            f"{d['total_time']:.2f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#3498db'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_text_props(color='white', fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_svd_drift(data, save_path='results/svd_drift.png'):
    """Plot SVD spectral drift over test snapshots."""
    svd_drifts = data.get('svd_drifts', {})
    if not svd_drifts:
        print("No SVD drift data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    num_train = data['metadata']['num_train']

    for strategy in svd_drifts:
        drifts = svd_drifts[strategy]
        if not drifts:
            continue
        # Sum drifts across all layers
        total_drifts = []
        for d in drifts:
            total_drifts.append(sum(d.values()) if d else 0.0)

        x = list(range(num_train, num_train + len(total_drifts)))
        ax.plot(x, total_drifts,
                label=STRATEGY_LABELS.get(strategy, strategy),
                color=COLORS.get(strategy, 'gray'),
                marker=MARKERS.get(strategy, 'o'),
                markersize=6, linewidth=2, alpha=0.85)

    ax.set_xlabel('Snapshot Index', fontsize=13)
    ax.set_ylabel('Spectral L2 Drift (sum over layers)', fontsize=13)
    ax.set_title('SVD Spectral Drift vs Initial Model', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def generate_all_plots(results_path='results/experiment_results.json', output_dir=None):
    """Generate all comparison plots."""
    data = load_results(results_path)
    if output_dir is None:
        output_dir = os.path.dirname(results_path) or 'results'
    os.makedirs(output_dir, exist_ok=True)

    plot_auc_over_snapshots(data, save_path=os.path.join(output_dir, 'auc_comparison.png'))
    plot_ap_over_snapshots(data, save_path=os.path.join(output_dir, 'ap_comparison.png'))
    plot_time_comparison(data, save_path=os.path.join(output_dir, 'time_comparison.png'))
    plot_summary_table(data, save_path=os.path.join(output_dir, 'summary_table.png'))
    plot_svd_drift(data, save_path=os.path.join(output_dir, 'svd_drift.png'))

    print(f"\nAll plots generated in {output_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results/experiment_results.json')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    generate_all_plots(args.results, args.output_dir)
