"""F1 and related metrics."""
import torch
from typing import Optional


def precision(predictions: torch.Tensor, targets: torch.Tensor, class_idx: int) -> float:
    """Precision for specific class."""
    pred_positive = predictions == class_idx
    true_positive = (predictions == class_idx) & (targets == class_idx)
    
    if pred_positive.sum() == 0:
        return 0.0
    return (true_positive.sum() / pred_positive.sum()).item()


def recall(predictions: torch.Tensor, targets: torch.Tensor, class_idx: int) -> float:
    """Recall for specific class."""
    actual_positive = targets == class_idx
    true_positive = (predictions == class_idx) & (targets == class_idx)
    
    if actual_positive.sum() == 0:
        return 0.0
    return (true_positive.sum() / actual_positive.sum()).item()


def f1_score(predictions: torch.Tensor, targets: torch.Tensor, class_idx: int) -> float:
    """F1 score for specific class."""
    p = precision(predictions, targets, class_idx)
    r = recall(predictions, targets, class_idx)
    
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
