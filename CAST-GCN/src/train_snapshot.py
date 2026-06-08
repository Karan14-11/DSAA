"""
Main Experiment: Snapshot-based Dynamic Link Prediction (GPU-ready)

Compares 6 strategies:
  1. GCN Static       — baseline GCN trained once, no updates
  2. GCN Retrain      — baseline GCN retrained from scratch each snapshot
  3. Inc-GCN NoUpdate — IncrementalGCN, no update (stale model)
  4. Inc-GCN CachedAX — IncrementalGCN with incremental AX cache updates + fine-tune
  5. Inc-GCN Subgraph — IncrementalGCN with subgraph-local fine-tuning
  6. Inc-GCN SVD      — IncrementalGCN with SVD row-selective gradient masking

Usage:
    python train_snapshot.py                    # defaults (CPU)
    python train_snapshot.py --device cuda      # GPU
    python train_snapshot.py --epochs 200       # more epochs
"""
import os, sys, copy, time, json, argparse
import torch
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.snapshot_builder import build_snapshots, compute_node_features
from model.gcn_link_predictor import GCNLinkPredictor, train_gcn_link, eval_gcn_link
from model.incremental_gcn_link import (
    IncrementalGCNLink, train_incremental_link, eval_incremental_link,
    fine_tune_incremental, fine_tune_svd_selective,
)
from model.incremental_utils import (
    compute_AX_sparse, build_adj_structures, update_AX_rows,
    nodes_for_AX_update, compute_edge_diff, compute_weight_svd, compare_svd,
    build_k_hop_subgraph_from_edge_index,
)

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def negative_sample(num_nodes, num_neg):
    src = torch.randint(0, num_nodes, (num_neg,))
    dst = torch.randint(0, num_nodes, (num_neg,))
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


@torch.no_grad()
def eval_deletion_prediction(model, snap, prev_edge_index, model_type, prev_ax=None, device='cpu', prev_x=None):
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.eval()

    num_nodes = snap.num_nodes
    pos_del = getattr(snap, 'removed_edges', None)
    if pos_del is None or pos_del.shape[1] == 0:
        return {'auc_del': float('nan'), 'ap_del': float('nan')}

    if prev_edge_index.numel() == 0 or snap.edge_index.numel() == 0:
        return {'auc_del': float('nan'), 'ap_del': float('nan')}

    # Hashing nodes to identify surviving edges
    old_hash = prev_edge_index[0].cpu() * num_nodes + prev_edge_index[1].cpu()
    new_hash = snap.edge_index[0].cpu() * num_nodes + snap.edge_index[1].cpu()
    
    surviving_mask = torch.isin(old_hash, new_hash)
    surviving_edges = prev_edge_index[:, surviving_mask]

    if surviving_edges.shape[1] == 0:
        return {'auc_del': float('nan'), 'ap_del': float('nan')}

    num_pos = pos_del.shape[1]
    num_neg = min(surviving_edges.shape[1], num_pos)
    
    perm = torch.randperm(surviving_edges.shape[1])[:num_neg]
    neg_del = surviving_edges[:, perm]

    if model_type == 'gcn':
        # Use previous features and previous edge_index to prevent leakage
        x = prev_x.to(device) if prev_x is not None else snap.x.to(device)
        edge_index = prev_edge_index.to(device)
        z = model.encode(x, edge_index)
        pos_score = model.decode(z, pos_del.to(device)).cpu()
        neg_score = model.decode(z, neg_del.to(device)).cpu()
    else:
        # Use previous AX and previous edge_index to prevent leakage
        ax = prev_ax.to(device) if prev_ax is not None else (snap_ax.to(device) if 'snap_ax' in locals() else None)
        edge_index = prev_edge_index.to(device)
        z = model.encode(ax, edge_index)
        pos_score = model.decode(z, pos_del.to(device)).cpu()
        neg_score = model.decode(z, neg_del.to(device)).cpu()

    # Smaller score = higher chance of deletion
    scores = torch.cat([-pos_score, -neg_score]).sigmoid().detach().cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_score.shape[0]),
        torch.zeros(neg_score.shape[0])
    ]).numpy()

    if len(set(labels)) > 1:
        auc_del = roc_auc_score(labels, scores)
        ap_del = average_precision_score(labels, scores)
    else:
        auc_del = 0.5
        ap_del = 0.5

    return {'auc_del': auc_del, 'ap_del': ap_del}


def train_model_full(model, data, num_nodes, epochs, lr, device, model_type, desc="Training"):
    """Train a model from scratch with tqdm progress bar."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Sample training edges (cap at 50k for speed)
    max_train = min(data.edge_index.shape[1], 50000)
    perm = torch.randperm(data.edge_index.shape[1])[:max_train]
    pos_sub = data.edge_index[:, perm]
    neg_sub = negative_sample(num_nodes, max_train)

    pbar = tqdm(range(epochs), desc=desc, leave=False)
    for epoch in pbar:
        if model_type == 'gcn':
            loss = train_gcn_link(model, data, pos_sub, neg_sub, optimizer, device)
        else:
            loss = train_incremental_link(
                model, data.ax, data.edge_index, pos_sub, neg_sub, optimizer, device
            )
        pbar.set_postfix(loss=f"{loss:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════

def run_experiment(config, args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    num_total  = config['snapshot']['num_total']
    num_train  = config['snapshot']['num_train']
    feature_dim = config['model_gcn']['feature_dim']
    gcn_cfg    = config['model_gcn']
    inc_cfg    = config['incremental']
    reg_cfg    = config.get('regularization', {})
    epochs     = args.epochs or config['training']['epochs']
    lr         = config['training']['lr']

    # Regularization weights
    lambda_deg = reg_cfg.get('lambda_degree_dist', 0.05)
    lambda_nbr = reg_cfg.get('lambda_neighborhood', 0.05)
    lambda_div = reg_cfg.get('lambda_diversity', 0.02)

    # ── 1. Build snapshots ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Building snapshots")
    print("=" * 60)

    snapshots, meta = build_snapshots(
        config['data_file'], num_snapshots=num_total, feature_dim=feature_dim
    )
    num_nodes = meta['num_nodes']

    if config.get('reverse_mode', False):
        print("⚡ Running in REVERSE (backward) mode for deletion analysis!")
        # 1. Store the forward new_edges before reversing
        forward_new_edges = [snap.new_edges.clone() if snap.new_edges is not None else torch.zeros(2, 0, dtype=torch.long) for snap in snapshots]
        
        # 2. Reverse the snapshots list
        snapshots = snapshots[::-1]
        
        # 3. Recompute new_edges and removed_edges for the reversed sequence:
        for i in range(len(snapshots)):
            snap = snapshots[i]
            snap.snapshot_idx = i  # Update index in reversed sequence
            
            if i == 0:
                snap.new_edges = torch.zeros(2, 0, dtype=torch.long)
                snap.removed_edges = torch.zeros(2, 0, dtype=torch.long)
            else:
                snap.new_edges = torch.zeros(2, 0, dtype=torch.long)
                forward_idx = len(snapshots) - i
                snap.removed_edges = forward_new_edges[forward_idx]

    print("Computing node features...")
    for s in tqdm(snapshots, desc="Features", leave=False):
        s.x = compute_node_features(s, num_nodes, feature_dim)

    # ── 2. Initial training on snapshot[num_train-1] ───────────────
    train_snap = snapshots[num_train - 1]
    train_snap.x = compute_node_features(train_snap, num_nodes, feature_dim)
    train_snap.ax = compute_AX_sparse(train_snap.edge_index, num_nodes, train_snap.x, device=device)

    print("\n" + "=" * 60)
    print(f"STEP 2: Initial training (snapshot {num_train-1})")
    print(f"  {num_nodes:,} nodes | {train_snap.edge_index.shape[1]//2:,} edges")
    print("=" * 60)

    # Baseline GCN
    t0 = time.time()
    gcn_model = GCNLinkPredictor(feature_dim, gcn_cfg['hidden_dim'],
                                  gcn_cfg['embed_dim'], gcn_cfg['dropout'])
    gcn_model = train_model_full(gcn_model, train_snap, num_nodes, epochs, lr,
                                  device, 'gcn', desc="GCN initial train")
    t_gcn_init = time.time() - t0
    print(f"  GCN trained in {t_gcn_init:.2f}s")

    # IncrementalGCN
    t0 = time.time()
    inc_model = IncrementalGCNLink(feature_dim, gcn_cfg['hidden_dim'],
                                    gcn_cfg['embed_dim'], gcn_cfg['dropout'])
    inc_model = train_model_full(inc_model, train_snap, num_nodes, epochs, lr,
                                  device, 'incremental', desc="Inc-GCN initial train")
    t_inc_init = time.time() - t0
    print(f"  Inc-GCN trained in {t_inc_init:.2f}s")

    svd_init = compute_weight_svd(inc_model, "init")

    # ── 3. Evaluate on test snapshots ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 3: Evaluating snapshots {num_train}–{num_total-1}")
    print("=" * 60)

    STRATEGIES = ['gcn_static', 'gcn_retrain',
                  'inc_no_update', 'inc_cached_ax', 'inc_subgraph', 'inc_svd']
    results = {s: [] for s in STRATEGIES}
    times   = {s: [] for s in STRATEGIES}
    svd_drifts = {s: [] for s in STRATEGIES}

    # Running copies for incremental strategies
    inc_cached   = copy.deepcopy(inc_model)
    inc_subgraph = copy.deepcopy(inc_model)
    inc_svd_m    = copy.deepcopy(inc_model)

    prev_edge_index = train_snap.edge_index.clone()
    prev_ax         = train_snap.ax.clone()

    ft_epochs = inc_cfg['finetune_epochs']
    ft_lr     = inc_cfg['finetune_lr']

    for snap_idx in tqdm(range(num_train, num_total), desc="Test snapshots"):
        snap = snapshots[snap_idx]
        snap.x = compute_node_features(snap, num_nodes, feature_dim)

        # Test edges = new edges in this snapshot (or removed edges if reverse_mode)
        test_pos = snap.removed_edges if config.get('reverse_mode', False) else snap.new_edges
        if test_pos.shape[1] == 0:
            for s in STRATEGIES:
                results[s].append({'auc': float('nan'), 'ap': float('nan')})
                times[s].append(0.0)
            continue

        test_neg = negative_sample(num_nodes, test_pos.shape[1])

        # Fine-tuning edges (subsample for speed)
        max_ft = min(snap.edge_index.shape[1], 20000)
        perm_ft = torch.randperm(snap.edge_index.shape[1])[:max_ft]
        ft_pos = snap.edge_index[:, perm_ft]
        ft_neg = negative_sample(num_nodes, max_ft)

        # Edge diff for incremental strategies
        added, removed, affected = compute_edge_diff(prev_edge_index, snap.edge_index)

        # Fetch previous features for GCN evaluation
        prev_x = snapshots[snap_idx - 1].x

        # ── Strategy 1: GCN Static ──
        t0 = time.time()
        if config.get('reverse_mode', False):
            r = eval_deletion_prediction(gcn_model, snap, prev_edge_index, 'gcn', device=device, prev_x=prev_x)
            r = {
                'auc': r['auc_del'], 'ap': r['ap_del'],
                'auc_del': r['auc_del'], 'ap_del': r['ap_del'],
                'f1': 0.0, 'hit_rate': 0.0, 'pos_score': 0.0
            }
        else:
            r = eval_gcn_link(gcn_model, snap, test_pos, test_neg, device)
            r_del = eval_deletion_prediction(gcn_model, snap, prev_edge_index, 'gcn', device=device, prev_x=prev_x)
            r.update(r_del)
        times['gcn_static'].append(time.time() - t0)
        results['gcn_static'].append(r)

        # ── Strategy 2: GCN Retrain ──
        t0 = time.time()
        gcn_re = GCNLinkPredictor(feature_dim, gcn_cfg['hidden_dim'],
                                   gcn_cfg['embed_dim'], gcn_cfg['dropout'])
        gcn_re = train_model_full(gcn_re, snap, num_nodes,
                                   max(epochs // 2, 30), lr, device, 'gcn',
                                   desc=f"GCN retrain s{snap_idx}")
        if config.get('reverse_mode', False):
            r = eval_deletion_prediction(gcn_re, snap, prev_edge_index, 'gcn', device=device, prev_x=prev_x)
            r = {
                'auc': r['auc_del'], 'ap': r['ap_del'],
                'auc_del': r['auc_del'], 'ap_del': r['ap_del'],
                'f1': 0.0, 'hit_rate': 0.0, 'pos_score': 0.0
            }
        else:
            r = eval_gcn_link(gcn_re, snap, test_pos, test_neg, device)
            r_del = eval_deletion_prediction(gcn_re, snap, prev_edge_index, 'gcn', device=device, prev_x=prev_x)
            r.update(r_del)
        times['gcn_retrain'].append(time.time() - t0)
        results['gcn_retrain'].append(r)

        # ── Strategy 3: Inc-GCN No Update ──
        t0 = time.time()
        snap_ax = compute_AX_sparse(snap.edge_index, num_nodes, snap.x, device=device)
        if config.get('reverse_mode', False):
            r = eval_deletion_prediction(inc_model, snap, prev_edge_index, 'incremental', prev_ax, device)
            r = {
                'auc': r['auc_del'], 'ap': r['ap_del'],
                'auc_del': r['auc_del'], 'ap_del': r['ap_del'],
                'f1': 0.0, 'hit_rate': 0.0, 'pos_score': 0.0
            }
        else:
            r = eval_incremental_link(inc_model, snap_ax, snap.edge_index,
                                       test_pos, test_neg, device)
            r_del = eval_deletion_prediction(inc_model, snap, prev_edge_index, 'incremental', prev_ax, device)
            r.update(r_del)
        times['inc_no_update'].append(time.time() - t0)
        results['inc_no_update'].append(r)

        # ── Strategy 4: Cached AX ──
        t0 = time.time()
        cached_ax = compute_AX_sparse(snap.edge_index, num_nodes, snap.x, device=device)
        m_c = copy.deepcopy(inc_cached)
        m_c = fine_tune_incremental(m_c, cached_ax, snap.edge_index,
                                     ft_pos, ft_neg, ft_epochs, ft_lr, device,
                                     num_nodes, lambda_deg, lambda_nbr, lambda_div)
        if config.get('reverse_mode', False):
            r = eval_deletion_prediction(m_c, snap, prev_edge_index, 'incremental', prev_ax, device)
            r = {
                'auc': r['auc_del'], 'ap': r['ap_del'],
                'auc_del': r['auc_del'], 'ap_del': r['ap_del'],
                'f1': 0.0, 'hit_rate': 0.0, 'pos_score': 0.0
            }
        else:
            r = eval_incremental_link(m_c, cached_ax, snap.edge_index,
                                       test_pos, test_neg, device)
            r_del = eval_deletion_prediction(m_c, snap, prev_edge_index, 'incremental', prev_ax, device)
            r.update(r_del)
        times['inc_cached_ax'].append(time.time() - t0)
        results['inc_cached_ax'].append(r)
        svd_c = compute_weight_svd(m_c)
        svd_drifts['inc_cached_ax'].append(compare_svd(svd_init, svd_c))
        inc_cached = m_c

        # ── Strategy 5: Subgraph ──
        t0 = time.time()
        sub_ax = compute_AX_sparse(snap.edge_index, num_nodes, snap.x, device=device)
        if len(affected) > 0:
            sub_nodes, sub_ei, _, g2l = build_k_hop_subgraph_from_edge_index(
                snap.edge_index, affected, num_nodes, k=inc_cfg['k_hop']
            )
            sub_set = set(sub_nodes)
            ss, sd = [], []
            for i in range(ft_pos.shape[1]):
                s, d = ft_pos[0, i].item(), ft_pos[1, i].item()
                if s in sub_set and d in sub_set:
                    ss.append(s); sd.append(d)
            if ss:
                sub_ft_pos = torch.tensor([ss, sd], dtype=torch.long)
                sub_ft_neg = negative_sample(num_nodes, len(ss))
            else:
                sub_ft_pos, sub_ft_neg = ft_pos, ft_neg
        else:
            sub_ft_pos, sub_ft_neg = ft_pos, ft_neg

        m_s = copy.deepcopy(inc_subgraph)
        m_s = fine_tune_incremental(m_s, sub_ax, snap.edge_index,
                                     sub_ft_pos, sub_ft_neg, ft_epochs, ft_lr, device,
                                     num_nodes, lambda_deg, lambda_nbr, lambda_div)
        if config.get('reverse_mode', False):
            r = eval_deletion_prediction(m_s, snap, prev_edge_index, 'incremental', prev_ax, device)
            r = {
                'auc': r['auc_del'], 'ap': r['ap_del'],
                'auc_del': r['auc_del'], 'ap_del': r['ap_del'],
                'f1': 0.0, 'hit_rate': 0.0, 'pos_score': 0.0
            }
        else:
            r = eval_incremental_link(m_s, sub_ax, snap.edge_index,
                                       test_pos, test_neg, device)
            r_del = eval_deletion_prediction(m_s, snap, prev_edge_index, 'incremental', prev_ax, device)
            r.update(r_del)
        times['inc_subgraph'].append(time.time() - t0)
        results['inc_subgraph'].append(r)
        inc_subgraph = m_s

        # ── Strategy 6: SVD Selective ──
        t0 = time.time()
        svd_ax = compute_AX_sparse(snap.edge_index, num_nodes, snap.x, device=device)
        m_v = copy.deepcopy(inc_svd_m)
        m_v = fine_tune_svd_selective(
            m_v, svd_ax, snap.edge_index, ft_pos, ft_neg,
            ft_epochs, ft_lr, inc_cfg['svd_k'], inc_cfg['svd_top_k'], device,
            num_nodes, lambda_deg, lambda_nbr, lambda_div
        )
        if config.get('reverse_mode', False):
            r = eval_deletion_prediction(m_v, snap, prev_edge_index, 'incremental', prev_ax, device)
            r = {
                'auc': r['auc_del'], 'ap': r['ap_del'],
                'auc_del': r['auc_del'], 'ap_del': r['ap_del'],
                'f1': 0.0, 'hit_rate': 0.0, 'pos_score': 0.0
            }
        else:
            r = eval_incremental_link(m_v, svd_ax, snap.edge_index,
                                       test_pos, test_neg, device)
            r_del = eval_deletion_prediction(m_v, snap, prev_edge_index, 'incremental', prev_ax, device)
            r.update(r_del)
        times['inc_svd'].append(time.time() - t0)
        results['inc_svd'].append(r)
        svd_v = compute_weight_svd(m_v)
        svd_drifts['inc_svd'].append(compare_svd(svd_init, svd_v))
        inc_svd_m = m_v

        # Update state
        prev_edge_index = snap.edge_index.clone()
        prev_ax = snap_ax.clone()

        # Print snapshot summary
        tqdm.write(
            f"  Snap {snap_idx} | "
            f"GCN-R: {results['gcn_retrain'][-1]['f1']:.2f}f {results['gcn_retrain'][-1]['auc']:.2f}a {results['gcn_retrain'][-1].get('auc_del', 0.0):.2f}d | "
            f"Inc-Sub: {results['inc_subgraph'][-1]['f1']:.2f}f {results['inc_subgraph'][-1]['auc']:.2f}a {results['inc_subgraph'][-1].get('auc_del', 0.0):.2f}d | "
            f"Inc-SVD: {results['inc_svd'][-1]['f1']:.2f}f {results['inc_svd'][-1]['auc']:.2f}a {results['inc_svd'][-1].get('auc_del', 0.0):.2f}d"
        )

    # ── 4. Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    LABELS = {
        'gcn_static':    'GCN (Static)',
        'gcn_retrain':   'GCN (Retrain)',
        'inc_no_update': 'Inc-GCN (NoUpdate)',
        'inc_cached_ax': 'Inc-GCN (CachedAX)',
        'inc_subgraph':  'Inc-GCN (Subgraph)',
        'inc_svd':       'Inc-GCN (SVD)',
    }

    summary = {}
    print(f"\n{'Strategy':<25s} {'Mean F1':>10s} {'Mean HitRate':>15s} {'Mean PosScore':>15s} {'Mean AUC':>10s} "
          f"{'Mean AP':>10s} {'Mean AUC-Del':>12s} {'Mean AP-Del':>12s} {'Total Time':>12s}")
    print("-" * 140)

    for s in STRATEGIES:
        aucs = [r['auc'] for r in results[s] if not np.isnan(r.get('auc', np.nan))]
        aps  = [r['ap']  for r in results[s] if not np.isnan(r.get('ap', np.nan))]
        pos  = [r['pos_score'] for r in results[s] if not np.isnan(r.get('pos_score', np.nan))]
        hits = [r['hit_rate'] for r in results[s] if not np.isnan(r.get('hit_rate', np.nan))]
        f1s  = [r.get('f1', 0) for r in results[s] if not np.isnan(r.get('f1', np.nan))]
        aucs_del = [r['auc_del'] for r in results[s] if not np.isnan(r.get('auc_del', np.nan))]
        aps_del  = [r['ap_del']  for r in results[s] if not np.isnan(r.get('ap_del', np.nan))]
        tt = sum(times[s])
        at = np.mean(times[s]) if times[s] else 0
        summary[s] = {
            'mean_f1':  float(np.mean(f1s))  if f1s else 0,
            'mean_hit': float(np.mean(hits)) if hits else 0,
            'mean_pos': float(np.mean(pos))  if pos else 0,
            'mean_auc': float(np.mean(aucs)) if aucs else 0,
            'std_auc':  float(np.std(aucs))  if aucs else 0,
            'mean_ap':  float(np.mean(aps))  if aps else 0,
            'std_ap':   float(np.std(aps))   if aps else 0,
            'mean_auc_del': float(np.mean(aucs_del)) if aucs_del else 0.0,
            'std_auc_del':  float(np.std(aucs_del))  if aucs_del else 0.0,
            'mean_ap_del':  float(np.mean(aps_del))  if aps_del else 0.0,
            'std_ap_del':   float(np.std(aps_del))   if aps_del else 0.0,
            'total_time': tt,
            'avg_time': at,
        }
        d = summary[s]
        print(f"  {LABELS[s]:<23s} {d['mean_f1']:10.4f} {d['mean_hit']:15.4f} {d['mean_pos']:15.4f} {d['mean_auc']:10.4f} "
              f"{d['mean_ap']:10.4f} {d['mean_auc_del']:12.4f} {d['mean_ap_del']:12.4f} {tt:10.2f}s")

    # ── 5. Save ────────────────────────────────────────────────────
    output = {
        'results':  {k: [dict(r) for r in v] for k, v in results.items()},
        'times':    times,
        'summary':  summary,
        'svd_drifts': {k: [{kk: float(vv) for kk, vv in d.items()} if d else {}
                        for d in v] for k, v in svd_drifts.items()},
        'metadata': {
            'dataset': getattr(args, 'dataset', 'unknown'),
            'num_nodes': num_nodes,
            'num_train': num_train,
            'num_test':  num_total - num_train,
            'init_time_gcn': t_gcn_init,
            'init_time_inc': t_inc_init,
            'device': str(device),
            'epochs': epochs,
        }
    }

    dataset_name = getattr(args, 'dataset', 'default')
    results_dir = os.path.join('results', dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    from visualize_results import generate_all_plots
    generate_all_plots(results_path, output_dir=results_dir)

    return output


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamic GNN Link Prediction Experiment")
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (auto-resolves config from configs/<dataset>.yaml)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: from config)')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path (overrides --dataset)')
    parser.add_argument('--reverse', action='store_true',
                        help='Run the snapshot sequence in reverse (backward) to evaluate deletion prediction.')
    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Auto-resolve config from dataset name
    if args.config:
        config_path = args.config
    elif args.dataset:
        config_path = os.path.join(project_root, 'configs', f'{args.dataset}.yaml')
    else:
        config_path = os.path.join(project_root, 'configs', 'default.yaml')
        args.dataset = 'default'

    if not os.path.exists(config_path):
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set reverse mode
    if args.reverse:
        config['reverse_mode'] = True

    # Resolve data_file
    data_file = config.get('data_file', 'processed/askubuntu_compact.pt')
    if not os.path.isabs(data_file):
        config['data_file'] = os.path.join(project_root, data_file.lstrip('../'))

    # Override device from CLI
    if args.device:
        config['training']['device'] = args.device
    if not args.device:
        args.device = config['training'].get('device', 'cpu')

    print(f"\n{'#'*60}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Config:  {config_path}")
    print(f"  Data:    {config['data_file']}")
    if config.get('reverse_mode', False):
        print("  Mode:    REVERSE (BACKWARD) DELETION ANALYSIS")
    else:
        print("  Mode:    FORWARD ADDITION ANALYSIS")
    print(f"{'#'*60}")

    os.chdir(project_root)
    run_experiment(config, args)
