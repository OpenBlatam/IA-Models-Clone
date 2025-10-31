from typing import Iterable
import torch
from factories.registry import Registry

OPTIMIZERS = Registry()


@OPTIMIZERS.register("adamw")
def build_adamw(params: Iterable, lr: float, weight_decay: float = 0.01, fused: bool = True):
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if torch.cuda.is_available():
        try:
            kwargs["fused"] = bool(fused)
        except TypeError:
            pass
    return torch.optim.AdamW(params, **kwargs)


@OPTIMIZERS.register("lion")
def build_lion(params: Iterable, lr: float, weight_decay: float = 0.0):
    # Placeholder: default to AdamW until Lion is added
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


@OPTIMIZERS.register("adafactor")
def build_adafactor(params: Iterable, lr: float):
    # Placeholder: default to AdamW; real Adafactor requires transformers.optimization
    return torch.optim.AdamW(params, lr=lr)





