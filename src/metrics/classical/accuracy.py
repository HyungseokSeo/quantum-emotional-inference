"""Accuracy metrics."""
import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Simple accuracy."""
    return (predictions == targets).float().mean().item()


def top_k_accuracy(probs: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """Top-k accuracy."""
    top_k = torch.topk(probs, k, dim=-1).indices
    correct = (top_k == targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()
