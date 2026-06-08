import torch
import torch.nn.functional as F


class LinkPredictionLoss:
    def __call__(self, z_u, z_v, z_neg):
        pos_score = (z_u * z_v).sum(dim=1)
        neg_score = (z_u * z_neg).sum(dim=1)
        return F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()
    
    