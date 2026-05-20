"""
ReAct Framework: Synergizing Reasoning and Acting
==================================================

Paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
arXiv: 2210.03629

ReAct combines reasoning (thinking) and acting in an interleaved manner,
allowing agents to reason about what to do, take actions, and observe results.
"""

from .react import ReActAgent

__all__ = ["ReActAgent"]



