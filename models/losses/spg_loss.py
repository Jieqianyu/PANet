import torch.nn as nn
import torch.nn.functional as F


def focal_loss(probs, labels, alpha: float = 0.25, gamma: float = 2, reduction="none"):
    ce_loss = F.binary_cross_entropy(probs, labels, reduction="none")
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss
    return loss.sum()


class AffinityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs, labels, reduction="none"):
        loss = focal_loss(probs, labels, reduction=reduction)
        return loss.sum()
