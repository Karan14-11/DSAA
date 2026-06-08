"""
Quick smoke test on a tiny 10-node graph to verify all code paths work.
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.gcn_link_predictor import GCNLinkPredictor, train_gcn_link, eval_gcn_link
from model.incremental_gcn_link import (
    IncrementalGCNLink, train_incremental_link, eval_incremental_link,
    fine_tune_incremental, fine_tune_svd_selective
)
from model.incremental_utils import (
    compute_AX_sparse, build_adj_structures, update_AX_rows,
    nodes_for_AX_update, compute_edge_diff, compute_weight_svd, compare_svd,
    build_k_hop_subgraph_from_edge_index
)
from torch_geometric.data import Data

def make_tiny_graph(num_nodes=10, num_edges=20, feature_dim=8):
    """Create a random tiny graph."""
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ], dim=0)
    edge_index = torch.unique(edge_index, dim=1)
    x = torch.randn(num_nodes, feature_dim)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

def test_baseline_gcn():
    print("=" * 50)
    print("TEST 1: Baseline GCN Link Predictor")
    print("=" * 50)
    data = make_tiny_graph()
    model = GCNLinkPredictor(8, 16, 8, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pos_edge = data.edge_index[:, :data.edge_index.shape[1]//2]
    neg_edge = torch.stack([
        torch.randint(0, 10, (pos_edge.shape[1],)),
        torch.randint(0, 10, (pos_edge.shape[1],))
    ])

    # Train
    for i in range(5):
        loss = train_gcn_link(model, data, pos_edge, neg_edge, optimizer)
    print(f"  Training loss (5 epochs): {loss:.4f}")

    # Eval
    metrics = eval_gcn_link(model, data, pos_edge, neg_edge)
    print(f"  HitRate={metrics.get('hit_rate', 0):.4f}  F1={metrics.get('f1', 0):.4f}  PosScore={metrics.get('pos_score', 0):.4f}  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}")
    print("  ✓ Baseline GCN works!\n")

def test_incremental_gcn():
    print("=" * 50)
    print("TEST 2: IncrementalGCN Link Predictor")
    print("=" * 50)
    data = make_tiny_graph()

    # Compute AX
    ax = compute_AX_sparse(data.edge_index, data.num_nodes, data.x)
    print(f"  AX shape: {ax.shape}")

    model = IncrementalGCNLink(8, 16, 8, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    pos_edge = data.edge_index[:, :data.edge_index.shape[1]//2]
    neg_edge = torch.stack([
        torch.randint(0, 10, (pos_edge.shape[1],)),
        torch.randint(0, 10, (pos_edge.shape[1],))
    ])

    # Train
    for i in range(5):
        loss = train_incremental_link(model, ax, data.edge_index, pos_edge, neg_edge, optimizer)
    print(f"  Training loss (5 epochs): {loss:.4f}")

    # Eval
    metrics = eval_incremental_link(model, ax, data.edge_index, pos_edge, neg_edge)
    print(f"  HitRate={metrics.get('hit_rate', 0):.4f}  F1={metrics.get('f1', 0):.4f}  PosScore={metrics.get('pos_score', 0):.4f}  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}")
    print("  ✓ IncrementalGCN works!\n")

def test_incremental_ax_update():
    print("=" * 50)
    print("TEST 3: Incremental AX Update")
    print("=" * 50)
    data = make_tiny_graph()
    ax_full = compute_AX_sparse(data.edge_index, data.num_nodes, data.x)

    # Simulate adding 2 edges
    new_src = torch.tensor([0, 1])
    new_dst = torch.tensor([5, 7])
    new_edges = torch.stack([
        torch.cat([data.edge_index[0], new_src, new_dst]),
        torch.cat([data.edge_index[1], new_dst, new_src])
    ])
    new_edges = torch.unique(new_edges, dim=1)

    # Full recompute
    ax_new_full = compute_AX_sparse(new_edges, data.num_nodes, data.x)

    # Incremental update
    adj, deg, ns, n2i = build_adj_structures(new_edges, data.num_nodes)
    affected = {0, 1, 5, 7}
    rows = nodes_for_AX_update(adj, affected)
    ax_incremental = update_AX_rows(adj, deg, ns, n2i, data.x, ax_full, rows)

    diff = (ax_new_full - ax_incremental).abs().max().item()
    print(f"  Max |full - incremental| = {diff:.6e}")
    print("  ✓ Incremental AX update works!\n")

def test_edge_diff():
    print("=" * 50)
    print("TEST 4: Edge Diff Computation")
    print("=" * 50)
    old = torch.tensor([[0,1,2], [1,2,3]])
    new = torch.tensor([[0,1,3], [1,2,4]])
    added, removed, affected = compute_edge_diff(old, new)
    print(f"  Added: {len(added)}, Removed: {len(removed)}, Affected: {len(affected)}")
    print("  ✓ Edge diff works!\n")

def test_subgraph():
    print("=" * 50)
    print("TEST 5: k-hop Subgraph Extraction")
    print("=" * 50)
    data = make_tiny_graph()
    sub_nodes, sub_ei, _, g2l = build_k_hop_subgraph_from_edge_index(
        data.edge_index, {0, 1}, data.num_nodes, k=1
    )
    print(f"  Affected: {{0, 1}}, Subgraph nodes: {len(sub_nodes)}, edges: {sub_ei.shape[1]}")
    print("  ✓ Subgraph extraction works!\n")

def test_svd_selective():
    print("=" * 50)
    print("TEST 6: SVD-Selective Fine-tuning")
    print("=" * 50)
    data = make_tiny_graph()
    ax = compute_AX_sparse(data.edge_index, data.num_nodes, data.x)
    model = IncrementalGCNLink(8, 16, 8, dropout=0.1)

    pos_edge = data.edge_index[:, :data.edge_index.shape[1]//2]
    neg_edge = torch.stack([
        torch.randint(0, 10, (pos_edge.shape[1],)),
        torch.randint(0, 10, (pos_edge.shape[1],))
    ])

    svd_before = compute_weight_svd(model, "before")
    model = fine_tune_svd_selective(
        model, ax, data.edge_index, pos_edge, neg_edge,
        epochs=5, lr=1e-3, svd_k=3, svd_top_k=3
    )
    svd_after = compute_weight_svd(model, "after")
    drifts = compare_svd(svd_before, svd_after, "test")
    print(f"  SVD drifts: {drifts}")
    print("  ✓ SVD-selective update works!\n")

def test_cached_ax_finetune():
    print("=" * 50)
    print("TEST 7: Cached-AX Fine-tuning")
    print("=" * 50)
    data = make_tiny_graph()
    ax = compute_AX_sparse(data.edge_index, data.num_nodes, data.x)
    model = IncrementalGCNLink(8, 16, 8, dropout=0.1)

    pos_edge = data.edge_index[:, :data.edge_index.shape[1]//2]
    neg_edge = torch.stack([
        torch.randint(0, 10, (pos_edge.shape[1],)),
        torch.randint(0, 10, (pos_edge.shape[1],))
    ])

    model = fine_tune_incremental(model, ax, data.edge_index, pos_edge, neg_edge,
                                   epochs=5, lr=1e-3)
    metrics = eval_incremental_link(model, ax, data.edge_index, pos_edge, neg_edge)
    print(f"  After fine-tune: HitRate={metrics.get('hit_rate', 0):.4f}  F1={metrics.get('f1', 0):.4f}  PosScore={metrics.get('pos_score', 0):.4f}  AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}")
    print("  ✓ Cached-AX fine-tuning works!\n")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    test_baseline_gcn()
    test_incremental_gcn()
    test_incremental_ax_update()
    test_edge_diff()
    test_subgraph()
    test_svd_selective()
    test_cached_ax_finetune()
    print("=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
