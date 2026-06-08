"""
Structural Regularization Losses for Dynamic Link Prediction.

These losses encourage the model to produce embeddings that preserve
the structural properties (degree distribution, local connectivity,
temporal smoothness) of the evolving graph.
"""
import torch
import torch.nn.functional as F


def temporal_smoothness(memory_old, memory_new):
    """Penalize large changes in embeddings between snapshots."""
    return ((memory_old - memory_new) ** 2).mean()


def degree_regularizer(embeddings, degrees):
    """
    Encourage embedding norms to correlate with node degree.
    High-degree nodes should have larger embeddings (more information).
    """
    target = torch.log1p(degrees.float())
    return ((embeddings.norm(dim=1) - target) ** 2).mean()


def degree_distribution_loss(z, edge_index, num_nodes):
    """
    Degree Distribution Matching Loss.

    Encourages predicted edge scores to produce a degree distribution
    that matches the actual graph's degree distribution. This ensures
    the model doesn't collapse to predicting edges only for high-degree
    nodes.

    Computes KL divergence between the actual degree distribution and
    the "soft" degree distribution implied by predicted edge scores.
    """
    # Actual degree distribution (normalized)
    actual_deg = torch.zeros(num_nodes, device=z.device)
    src = edge_index[0]
    ones = torch.ones(src.shape[0], device=z.device)
    actual_deg.scatter_add_(0, src, ones)

    # Only consider nodes that have edges
    active_mask = actual_deg > 0
    if active_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    actual_dist = actual_deg[active_mask]
    actual_dist = actual_dist / actual_dist.sum()

    # Predicted "soft" degree: sum of similarity scores with neighbors
    # For each node, compute average embedding norm as proxy for connectivity
    z_norm = z.norm(dim=1)
    pred_dist = z_norm[active_mask]
    pred_dist = F.softmax(pred_dist, dim=0)

    # KL divergence (actual || predicted)
    actual_dist = actual_dist.clamp(min=1e-8)
    pred_dist = pred_dist.clamp(min=1e-8)
    kl = (actual_dist * (actual_dist.log() - pred_dist.log())).sum()

    return kl


def neighborhood_overlap_loss(z, edge_index):
    """
    Neighborhood Overlap Encouragement.

    Nodes that share many neighbors should have similar embeddings.
    This preserves local clustering structure.

    Efficiently computes this by encouraging connected nodes to have
    similar embedding distributions (not just similar embeddings).
    """
    if edge_index.shape[1] == 0:
        return torch.tensor(0.0, device=z.device)

    src, dst = edge_index[0], edge_index[1]

    # Sample a subset for efficiency (max 10k edges)
    if src.shape[0] > 10000:
        perm = torch.randperm(src.shape[0], device=z.device)[:10000]
        src, dst = src[perm], dst[perm]

    z_src = z[src]
    z_dst = z[dst]

    # Cosine similarity between connected node pairs
    cos_sim = F.cosine_similarity(z_src, z_dst, dim=1)

    # We want connected nodes to be similar (cos_sim close to 1)
    # But not identical (that causes collapse)
    # Target: similarity around 0.5–0.8
    target_sim = 0.7
    loss = ((cos_sim - target_sim) ** 2).mean()

    return loss


def embedding_diversity_loss(z, edge_index, num_nodes):
    """
    Prevent embedding collapse.

    Ensures embeddings don't all converge to the same vector,
    which is a common failure mode in contrastive/link prediction.
    Maximizes variance of embeddings across nodes.
    """
    # Only consider active nodes
    active_nodes = torch.unique(edge_index.flatten())
    if len(active_nodes) < 2:
        return torch.tensor(0.0, device=z.device)

    # Sample for efficiency
    if len(active_nodes) > 5000:
        perm = torch.randperm(len(active_nodes))[:5000]
        active_nodes = active_nodes[perm]

    z_active = z[active_nodes]

    # Variance of embeddings (we want this to be HIGH)
    # Negative of variance = loss to minimize
    var = z_active.var(dim=0).mean()

    # We want variance > threshold, so penalize low variance
    target_var = 0.5
    loss = F.relu(target_var - var)

    return loss


def local_structure_loss(z, edge_index, num_nodes):
    """
    Combined local structure preservation loss.

    Wraps degree distribution + neighborhood overlap + diversity
    into a single call for convenience.

    Returns:
        dict with individual loss components and total
    """
    deg_loss = degree_distribution_loss(z, edge_index, num_nodes)
    nbr_loss = neighborhood_overlap_loss(z, edge_index)
    div_loss = embedding_diversity_loss(z, edge_index, num_nodes)

    return {
        'degree_dist': deg_loss,
        'neighborhood': nbr_loss,
        'diversity': div_loss,
        'total': deg_loss + nbr_loss + div_loss,
    }