"""
Residual Submodule
Aggregates various residual connection components.
"""

from .standard import ResidualConnection
from .pre_norm import PreNormResidual
from .post_norm import PostNormResidual
from .gated import GatedResidual

__all__ = [
    "ResidualConnection",
    "PreNormResidual",
    "PostNormResidual",
    "GatedResidual",
]



