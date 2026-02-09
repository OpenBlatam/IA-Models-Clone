"""
Deep Learning Utilities Module
=============================

Módulo de utilidades para deep learning.
"""

from .device_manager import DeviceManager, get_device_manager
from .losses import (
    TrajectoryLoss,
    FocalLoss,
    ContrastiveLoss,
    get_loss_function
)
from .metrics import (
    Metric,
    MSE,
    MAE,
    RMSE,
    R2Score,
    Accuracy,
    MetricCollection,
    create_regression_metrics,
    create_classification_metrics
)

__all__ = [
    'DeviceManager',
    'get_device_manager',
    'TrajectoryLoss',
    'FocalLoss',
    'ContrastiveLoss',
    'get_loss_function',
    'Metric',
    'MSE',
    'MAE',
    'RMSE',
    'R2Score',
    'Accuracy',
    'MetricCollection',
    'create_regression_metrics',
    'create_classification_metrics'
]
