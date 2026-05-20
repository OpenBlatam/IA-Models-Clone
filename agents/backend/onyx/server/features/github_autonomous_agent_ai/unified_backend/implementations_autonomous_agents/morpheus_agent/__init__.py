"""
MORPHEUS Agent Framework
========================

Framework for modeling role from personalized dialogue history.
"""

from .morpheus_agent import (
    MorpheusAgent,
    RoleType,
    DialogueContext,
    DialogueTurn,
    RoleProfile
)

__all__ = [
    "MorpheusAgent",
    "RoleType",
    "DialogueContext",
    "DialogueTurn",
    "RoleProfile"
]


