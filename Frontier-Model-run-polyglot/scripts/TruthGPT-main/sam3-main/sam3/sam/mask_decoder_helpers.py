"""
Helper functions for mask decoder operations.
============================================

Common utilities for mask decoder:
- Output token construction
- Mask selection logic
- Stability score computation

Single Responsibility: Provide reusable mask decoder utilities.
"""

import torch
from typing import Tuple, Optional, Callable
from torch import Tensor


def build_output_tokens(
    iou_token: torch.nn.Embedding,
    mask_tokens: torch.nn.Embedding,
    obj_score_token: Optional[torch.nn.Embedding] = None,
    batch_size: int = 1
) -> Tuple[Tensor, int]:
    """
    Build output tokens for mask prediction.
    
    Args:
        iou_token: IoU token embedding
        mask_tokens: Mask tokens embedding
        obj_score_token: Optional object score token embedding
        batch_size: Batch size
        
    Returns:
        Tuple of (output_tokens, offset) where offset is the index where mask tokens start
    """
    offset = 0
    if obj_score_token is not None:
        output_tokens = torch.cat(
            [
                obj_score_token.weight,
                iou_token.weight,
                mask_tokens.weight,
            ],
            dim=0,
        )
        offset = 1
    else:
        output_tokens = torch.cat(
            [iou_token.weight, mask_tokens.weight], dim=0
        )
    
    output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)
    return output_tokens, offset


def select_mask_output(
    masks: Tensor,
    iou_pred: Tensor,
    multimask_output: bool,
    dynamic_multimask_via_stability: bool = False,
    stability_scores: Optional[Tensor] = None,
    stability_thresh: float = 0.98
) -> Tuple[Tensor, Tensor]:
    """
    Select appropriate mask output based on configuration.
    
    Args:
        masks: All predicted masks
        iou_pred: IoU predictions for all masks
        multimask_output: Whether to return multiple masks
        dynamic_multimask_via_stability: Whether to use dynamic multimask selection
        stability_scores: Pre-computed stability scores (if using dynamic selection)
        stability_thresh: Stability threshold for dynamic selection
        
    Returns:
        Tuple of (selected_masks, selected_iou_pred)
    """
    if multimask_output:
        return masks[:, 1:, :, :], iou_pred[:, 1:]
    elif dynamic_multimask_via_stability and stability_scores is not None:
        return _dynamic_multimask_via_stability_with_scores(
            masks, iou_pred, stability_scores, stability_thresh
        )
    else:
        return masks[:, 0:1, :, :], iou_pred[:, 0:1]


def compute_stability_scores(
    mask_logits: Tensor,
    stability_delta: float = 0.05
) -> Tensor:
    """
    Compute stability scores of mask logits based on IoU between thresholds.
    
    Args:
        mask_logits: Mask logits tensor
        stability_delta: Delta for stability threshold
        
    Returns:
        Stability scores tensor
    """
    mask_logits = mask_logits.flatten(-2)
    area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
    area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
    stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
    return stability_scores


def _dynamic_multimask_via_stability_with_scores(
    all_mask_logits: Tensor,
    all_iou_scores: Tensor,
    stability_scores: Tensor,
    stability_thresh: float
) -> Tuple[Tensor, Tensor]:
    """
    Dynamically select mask based on pre-computed stability scores.
    
    Args:
        all_mask_logits: All mask logits
        all_iou_scores: All IoU scores
        stability_scores: Pre-computed stability scores
        stability_thresh: Stability threshold
        
    Returns:
        Tuple of (selected_mask_logits, selected_iou_scores)
    """
    # Best mask from multimask output tokens (1~3)
    multimask_logits = all_mask_logits[:, 1:, :, :]
    multimask_iou_scores = all_iou_scores[:, 1:]
    best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
    batch_inds = torch.arange(
        multimask_iou_scores.size(0), device=all_iou_scores.device
    )
    best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
    best_multimask_logits = best_multimask_logits.unsqueeze(1)
    best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
    best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

    # Mask from singlemask output token 0
    singlemask_logits = all_mask_logits[:, 0:1, :, :]
    singlemask_iou_scores = all_iou_scores[:, 0:1]
    is_stable = stability_scores >= stability_thresh

    # Dynamically fall back to best multimask output upon low stability scores
    mask_logits_out = torch.where(
        is_stable[..., None, None].expand_as(singlemask_logits),
        singlemask_logits,
        best_multimask_logits,
    )
    iou_scores_out = torch.where(
        is_stable.expand_as(singlemask_iou_scores),
        singlemask_iou_scores,
        best_multimask_iou_scores,
    )
    return mask_logits_out, iou_scores_out

