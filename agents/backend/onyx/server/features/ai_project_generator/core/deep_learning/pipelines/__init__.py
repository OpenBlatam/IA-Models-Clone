"""
Pipelines Module - End-to-End Training and Inference Pipelines
================================================================

Provides high-level pipelines for common workflows:
- Training pipelines
- Inference pipelines
- Evaluation pipelines
- Fine-tuning pipelines
"""

from typing import Optional, Dict, Any
import torch

__all__ = [
    "TrainingPipeline",
    "InferencePipeline",
    "EvaluationPipeline",
    "FineTuningPipeline",
]



