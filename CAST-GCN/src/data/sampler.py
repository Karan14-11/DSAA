from collections import defaultdict


class TemporalNeighborSampler:
    def __init__(self, num_neighbors):
        self.num_neighbors = num_neighbors
        self.adj = defaultdict(list)


    def update(self, u, v, t):
        self.adj[u].append((v, t))
        self.adj[v].append((u, t))


    def sample(self, node, t):
        neighbors = [x for x in self.adj[node] if x[1] < t]
        neighbors = neighbors[-self.num_neighbors:]
        return neighbors