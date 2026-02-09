"""Gradio applications module."""

from .prediction_demo import create_prediction_demo
from .text_generation_demo import create_text_generation_demo

__all__ = ["create_prediction_demo", "create_text_generation_demo"]




