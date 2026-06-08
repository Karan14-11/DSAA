import torch


class TemporalEdgeDataset:
    def __init__(self, edge_file):
        data = torch.load(edge_file)
        self.u = data['u']
        self.v = data['v']
        self.t = data['t']
        self.num_edges = len(self.u)


    def __len__(self):
        return self.num_edges


    def get_batch(self, start, batch_size):
        end = min(start + batch_size, self.num_edges)
        return self.u[start:end], self.v[start:end], self.t[start:end]