from __future__ import annotations

from typing import Iterable, Tuple, Type

import torch
from torch import nn


def apply_weight_norm(module: nn.Module, layer_types: Tuple[Type[nn.Module], ...] = (nn.Linear,)) -> nn.Module:
    for m in module.modules():
        if isinstance(m, layer_types):
            try:
                nn.utils.weight_norm(m)  # type: ignore[arg-type]
            except Exception:
                pass
    return module


def remove_weight_norm(module: nn.Module) -> nn.Module:
    for m in module.modules():
        try:
            nn.utils.remove_weight_norm(m)  # type: ignore[arg-type]
        except Exception:
            pass
    return module


def apply_spectral_norm(
    module: nn.Module,
    layer_types: Tuple[Type[nn.Module], ...] = (nn.Linear,),
    n_power_iterations: int = 1,
) -> nn.Module:
    for m in module.modules():
        if isinstance(m, layer_types):
            try:
                nn.utils.spectral_norm(m, n_power_iterations=n_power_iterations)  # type: ignore[arg-type]
            except Exception:
                pass
    return module


def remove_spectral_norm(module: nn.Module) -> nn.Module:
    for m in module.modules():
        try:
            nn.utils.remove_spectral_norm(m)  # type: ignore[arg-type]
        except Exception:
            pass
    return module


