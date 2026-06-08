"""
Incremental GCN for Link Prediction.
Adapted from DynamicGNN-main: first layer uses cached AX, second uses GCNConv.
Now with structural regularization losses for better graph structure preservation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from loss.regulizer_loss import (
    degree_distribution_loss, neighborhood_overlap_loss,
    embedding_diversity_loss, degree_regularizer
)


class IncrementalGCNLink(nn.Module):
    """
    First hop: Linear(AX) where AX = precomputed A_hat @ X
    Second hop: GCNConv on hidden features
    Decoder: dot product for link prediction
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, ax, edge_index):
        """Encode using cached AX for first layer."""
        x = self.lin1(ax)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=1)

    def forward(self, ax, edge_index, edge_label_index):
        z = self.encode(ax, edge_index)
        return self.decode(z, edge_label_index)


def train_incremental_link(model, ax, edge_index, train_pos_edge, train_neg_edge,
                           optimizer, device='cpu', num_nodes=None,
                           lambda_deg=0.05, lambda_nbr=0.05, lambda_div=0.02):
    """One training step with structural regularization."""
    model.train()
    optimizer.zero_grad()
    ax = ax.to(device)
    edge_index = edge_index.to(device)
    z = model.encode(ax, edge_index)
    pos_score = model.decode(z, train_pos_edge.to(device))
    neg_score = model.decode(z, train_neg_edge.to(device))
    pos_label = torch.ones(pos_score.shape[0], device=device)
    neg_label = torch.zeros(neg_score.shape[0], device=device)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([pos_label, neg_label])
    link_loss = F.binary_cross_entropy_with_logits(scores, labels)

    # Structural regularization
    n = num_nodes or z.shape[0]
    reg_loss = torch.tensor(0.0, device=device)
    if lambda_deg > 0:
        reg_loss = reg_loss + lambda_deg * degree_distribution_loss(z, edge_index, n)
    if lambda_nbr > 0:
        reg_loss = reg_loss + lambda_nbr * neighborhood_overlap_loss(z, edge_index)
    if lambda_div > 0:
        reg_loss = reg_loss + lambda_div * embedding_diversity_loss(z, edge_index, n)

    loss = link_loss + reg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_incremental_link(model, ax, edge_index, pos_edge, neg_edge, device='cpu'):
    """Evaluate IncrementalGCNLink."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    model.eval()
    ax = ax.to(device)
    edge_index = edge_index.to(device)
    z = model.encode(ax, edge_index)
    pos_score = model.decode(z, pos_edge.to(device)).cpu()
    neg_score = model.decode(z, neg_edge.to(device)).cpu()
    scores = torch.cat([pos_score, neg_score]).sigmoid().numpy()
    labels = torch.cat([
        torch.ones(pos_score.shape[0]),
        torch.zeros(neg_score.shape[0])
    ]).numpy()
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    ap = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0.5
    
    pred_labels = (scores > 0.5).astype(int)
    f1 = f1_score(labels, pred_labels) if len(set(labels)) > 1 else 0.0
    
    # Calculate how many of the new added edges were predicted correctly (> 0.5)
    pos_probs = pos_score.sigmoid()
    pos_correct = (pos_probs > 0.5).sum().item()
    pos_total = pos_score.shape[0]
    hit_rate = pos_correct / max(pos_total, 1)

    return {
        'auc': auc, 
        'ap': ap, 
        'f1': f1,
        'pos_score': pos_probs.mean().item(),
        'hit_rate': hit_rate
    }


def fine_tune_incremental(model, ax, edge_index, train_pos_edge, train_neg_edge,
                          epochs=50, lr=5e-4, device='cpu', num_nodes=None,
                          lambda_deg=0.05, lambda_nbr=0.05, lambda_div=0.02):
    """Fine-tune model after incremental AX update, with structural losses."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        train_incremental_link(model, ax, edge_index, train_pos_edge, train_neg_edge,
                               optimizer, device, num_nodes,
                               lambda_deg, lambda_nbr, lambda_div)
    return model


def fine_tune_svd_selective(model, ax, edge_index, train_pos_edge, train_neg_edge,
                            epochs=50, lr=5e-4, svd_k=5, svd_top_k=5, device='cpu',
                            num_nodes=None,
                            lambda_deg=0.05, lambda_nbr=0.05, lambda_div=0.02):
    """Fine-tune with SVD row-selective masking + structural regularization."""
    from .incremental_utils import compute_important_rows, mask_gradients

    # Compute important rows before training
    imp_W1 = compute_important_rows(model.lin1.weight, k=svd_k, top_k=svd_top_k)
    imp_W2 = compute_important_rows(model.conv2.lin.weight, k=svd_k, top_k=svd_top_k)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    n = num_nodes or 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        ax_d = ax.to(device)
        ei_d = edge_index.to(device)
        z = model.encode(ax_d, ei_d)
        pos_score = model.decode(z, train_pos_edge.to(device))
        neg_score = model.decode(z, train_neg_edge.to(device))
        pos_label = torch.ones(pos_score.shape[0], device=device)
        neg_label = torch.zeros(neg_score.shape[0], device=device)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_label, neg_label])
        link_loss = F.binary_cross_entropy_with_logits(scores, labels)

        # Structural regularization
        reg_loss = torch.tensor(0.0, device=device)
        if lambda_deg > 0:
            reg_loss = reg_loss + lambda_deg * degree_distribution_loss(z, ei_d, n or z.shape[0])
        if lambda_nbr > 0:
            reg_loss = reg_loss + lambda_nbr * neighborhood_overlap_loss(z, ei_d)
        if lambda_div > 0:
            reg_loss = reg_loss + lambda_div * embedding_diversity_loss(z, ei_d, n or z.shape[0])

        loss = link_loss + reg_loss
        loss.backward()
        # Mask gradients — only update important rows
        mask_gradients(model.lin1.weight, imp_W1)
        mask_gradients(model.conv2.lin.weight, imp_W2)
        optimizer.step()

    return model
