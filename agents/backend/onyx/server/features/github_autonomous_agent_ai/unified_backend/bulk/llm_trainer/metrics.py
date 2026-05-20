"""
Evaluation Metrics Module
==========================

Custom evaluation metrics for LLM training.
Provides metrics beyond simple loss calculation.

Author: BUL System
Date: 2024
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def compute_perplexity(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute perplexity from evaluation predictions.
    
    Perplexity = exp(cross_entropy_loss)
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        
    Returns:
        Dictionary with perplexity metric
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Flatten predictions and labels
    if predictions.ndim > 2:
        predictions = predictions.reshape(-1, predictions.shape[-1])
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    
    # Mask out padding tokens (typically -100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Calculate cross entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    logits = torch.from_numpy(predictions)
    labels_tensor = torch.from_numpy(labels)
    
    loss = loss_fct(logits, labels_tensor).item()
    perplexity = np.exp(loss)
    
    return {"perplexity": float(perplexity)}


def compute_accuracy(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute accuracy from evaluation predictions.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        
    Returns:
        Dictionary with accuracy metric
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Get predicted token IDs
    if predictions.ndim > 2:
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions
    
    # Flatten
    if pred_ids.ndim > 1:
        pred_ids = pred_ids.reshape(-1)
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    
    # Mask out padding tokens
    mask = labels != -100
    pred_ids = pred_ids[mask]
    labels = labels[mask]
    
    # Calculate accuracy
    accuracy = (pred_ids == labels).mean()
    
    return {"accuracy": float(accuracy)}


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute multiple metrics from evaluation predictions.
    
    Computes:
    - Perplexity
    - Accuracy
    - Cross-entropy loss
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Compute perplexity
    try:
        perplexity_metrics = compute_perplexity(eval_pred)
        metrics.update(perplexity_metrics)
    except Exception as e:
        logger.warning(f"Could not compute perplexity: {e}")
    
    # Compute accuracy
    try:
        accuracy_metrics = compute_accuracy(eval_pred)
        metrics.update(accuracy_metrics)
    except Exception as e:
        logger.warning(f"Could not compute accuracy: {e}")
    
    return metrics
