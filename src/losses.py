import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification with severe class imbalance:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground truth class.
    
    Operates on unscaled raw logits for maximum numerical stability.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,) or (B, 1), targets: (B,) or (B, 1)
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable BCE calculation
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)  # p_t is the probability of the true class

        # alpha weighting: alpha for positive (fraud=1), (1 - alpha) for negative (legit=0)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        focal_loss = alpha_t * modulating_factor * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss using pos_weight.
    pos_weight = N_negative / N_positive to counterbalance class imbalance.
    """
    def __init__(self, pos_weight: float = 27.5, reduction: str = 'mean'):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        pos_weight = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction=self.reduction)
