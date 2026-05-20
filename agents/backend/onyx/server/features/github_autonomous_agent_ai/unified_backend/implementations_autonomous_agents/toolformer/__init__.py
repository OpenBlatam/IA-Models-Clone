"""
Toolformer: Language Models Can Teach Themselves to Use Tools
================================================================

Paper: "Toolformer: Language Models Can Teach Themselves to Use Tools"
arXiv: 2302.04761

Toolformer enables LMs to learn to use external tools via self-supervised learning.
The model decides which APIs to call, when to call them, what arguments to pass,
and how to incorporate results into future token prediction.
"""

from .toolformer import Toolformer, APICall, ToolformerTrainer

__all__ = ["Toolformer", "APICall", "ToolformerTrainer"]



