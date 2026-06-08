import torch
import random


class NegativeSampler:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes


    def sample(self, batch_size):
        return torch.randint(0, self.num_nodes, (batch_size,))