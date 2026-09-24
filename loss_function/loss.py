import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginAwareCELoss(nn.Module):
    """
    Margin-Aware Cross Entropy Loss

    Clean      -> small weight
    Ambiguous  -> large weight
    Noisy      -> ~0 weight
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        probs = F.softmax(logits, dim=1)  # (N, C)

        p_gt = probs.gather(1, targets.unsqueeze(1))  # (N,1)


        mask = torch.ones_like(probs).scatter_(1, targets.unsqueeze(1), 0.0)
        probs_non_gt = probs * mask
        p_nn, _ = probs_non_gt.max(dim=1, keepdim=True)  # (N,1)


        margin = (p_gt - p_nn) / (p_gt + p_nn + self.eps)
        weights = 1.0 - margin.pow(2)


        ce_loss = F.cross_entropy(logits, targets, reduction="none").unsqueeze(1)

        loss = (1.0 + weights) * ce_loss

        loss = loss.squeeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

