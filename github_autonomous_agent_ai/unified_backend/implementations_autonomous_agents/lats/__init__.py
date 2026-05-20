"""
Language Agent Tree Search (LATS)
==================================

Paper: "Language Agent Tree Search Unifies Reasoning, Acting, and Planning"
arXiv: 2412.xxxxx

LATS combines tree search with language models to unify:
- Reasoning: LLM-based state evaluation
- Acting: Tool use and action execution
- Planning: Tree search for optimal paths
"""

from .lats import LATSAgent, LATSTree

__all__ = ["LATSAgent", "LATSTree"]



