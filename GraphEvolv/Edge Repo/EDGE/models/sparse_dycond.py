import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SparseDyCondGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, num_layers=3, num_classes=2, max_nodes=21000, edge_mlp_chunk=50000, emb_dim=None):
        super().__init__()
        if emb_dim is not None:
            hidden_dim = emb_dim
        self.max_nodes = max_nodes
        self.edge_mlp_chunk = edge_mlp_chunk

        self.node_emb = nn.Embedding(num_nodes, hidden_dim)

        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.class_emb = nn.Embedding(num_classes, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.edge_pred_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, curr_graph, prev_graph, t, y):
        device = curr_graph.edge_index.device

        # Node embeddings with precomputed modulo
        h_curr = self.node_emb(curr_graph.batch_nodes % self.node_emb.num_embeddings)
        h_prev = self.node_emb(prev_graph.batch_nodes % self.node_emb.num_embeddings) if prev_graph is not None else torch.zeros_like(h_curr)

        # Time and class embeddings
        t_emb = self.time_emb(t.float().view(-1, 1))
        y_emb = self.class_emb(y)

        # Batch index
        batch_idx = curr_graph.batch if hasattr(curr_graph, 'batch') and curr_graph.batch is not None else torch.zeros(curr_graph.num_nodes, dtype=torch.long, device=device)

        # Combine embeddings + global context
        h = h_curr + h_prev + t_emb[batch_idx] + y_emb[batch_idx]

        # GCN Message Passing
        edge_index = curr_graph.edge_index % self.max_nodes
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.silu(h)

        # Chunked edge MLP for memory efficiency
        src, dst = edge_index
        if src.size(0) > 0:
            edge_logits = []
            for start in range(0, src.size(0), self.edge_mlp_chunk):
                end = start + self.edge_mlp_chunk
                edge_h_chunk = torch.cat([h[src[start:end]], h[dst[start:end]]], dim=1)
                edge_logits.append(self.edge_pred_mlp(edge_h_chunk))
            edge_logits = torch.cat(edge_logits, dim=0).view(-1)

        return h, self.edge_pred_mlp
