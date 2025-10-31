"""
Evaluation metrics for Blaze AI models.

This module provides comprehensive evaluation metrics for various AI tasks:
- Classification metrics
- Text generation metrics  
- Image generation metrics
- SEO optimization metrics
- Brand voice analysis metrics
"""

from .classification import evaluate_classification
from .text_generation import evaluate_text_generation
from .image_generation import evaluate_image_generation
from .seo_optimization import evaluate_seo_optimization
from .brand_voice import evaluate_brand_voice
from .metrics_registry import EvaluationMetricsRegistry

__all__ = [
    "evaluate_classification",
    "evaluate_text_generation", 
    "evaluate_image_generation",
    "evaluate_seo_optimization",
    "evaluate_brand_voice",
    "EvaluationMetricsRegistry"
]


