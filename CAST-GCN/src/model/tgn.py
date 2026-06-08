]
import torch
import torch.nn as nn
from .utilities import Memory, MeanAggregator, MessageFunction, TemporalEncoder,TimeEncode



class TGN(nn.Module):
    def __init__(self, num_nodes, config):
        super().__init__()
        self.memory = Memory(num_nodes, config['memory_dim'])
        self.message_fn = MessageFunction(config['memory_dim'], config['message_dim'],config['time_dim'])
        self.aggregator = MeanAggregator(config['message_dim'], config['memory_dim'])
        self.encoder = TemporalEncoder(config['memory_dim'], config['embedding_dim'])
        self.time_encoder = TimeEncode(config['memory_dim'])
        self.device = "cuda"

    def forward(self, u, v, t):
        u = u.to(self.device)
        v = v.to(self.device)
        t = t.to(self.device)
        h_u = self.memory.get(u)
        h_v = self.memory.get(v)

        dt_u = (t - self.memory.last_update[u]).float().clamp(min=0)
        dt_v = (t - self.memory.last_update[v]).float().clamp(min=0)

        t_enc_u = self.time_encoder(dt_u)
        t_enc_v = self.time_encoder(dt_v)

        msg_u = self.message_fn(h_u, h_v, t_enc_u)
        msg_v = self.message_fn(h_v, h_u, t_enc_v)

        return h_u, h_v, msg_u, msg_v