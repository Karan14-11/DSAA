import torch
import torch.nn as nn


class Memory(nn.Module):
    def __init__(self, num_nodes, memory_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim), requires_grad=False)
        self.last_update = torch.zeros(num_nodes)


    def get(self, nodes):
        return self.memory[nodes]


    def update(self, nodes, new_memory, t):
        self.memory[nodes] = new_memory.detach()
        self.last_update[nodes] = t.float()


class MessageFunction(nn.Module):
    def __init__(self, memory_dim, message_dim,time_dim):
        super().__init__()
        self.lin = nn.Linear(2 * memory_dim + time_dim, message_dim)


    def forward(self, h_u, h_v, t_enc):
        x = torch.cat([h_u, h_v, t_enc], dim=-1)
        return self.lin(x)


class MeanAggregator(nn.Module):
    def __init__(self, message_dim, memory_dim):
        super().__init__()
        self.lin = nn.Linear(message_dim, memory_dim)


    def forward(self, messages):
        if len(messages) == 0:
            return None
        return self.lin(messages.mean(dim=0))
    
class TemporalEncoder(nn.Module):
    def __init__(self, memory_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(memory_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )


    def forward(self, memory):
        return self.net(memory)
    

class TimeEncode(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(1, dim)

    def forward(self, dt):
        return torch.cos(self.lin(dt.unsqueeze(-1)))