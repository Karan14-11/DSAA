"""
Baseline GCN Link Predictor.
A standard 2-layer GCN for link prediction with dot-product decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def train_gcn_link(model, data, train_pos_edge, train_neg_edge, optimizer, device='cpu'):
    model.train()
    optimizer.zero_grad()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    z = model.encode(x, edge_index)
    pos_score = model.decode(z, train_pos_edge.to(device))
    neg_score = model.decode(z, train_neg_edge.to(device))
    pos_label = torch.ones(pos_score.shape[0], device=device)
    neg_label = torch.zeros(neg_score.shape[0], device=device)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([pos_label, neg_label])
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_gcn_link(model, data, pos_edge, neg_edge, device='cpu'):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    z = model.encode(x, edge_index)
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
