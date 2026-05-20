"""
Kaiming Initialization Module

Implements Kaiming/He initialization strategies.
"""

import logging
import math
import torch.nn as nn

logger = logging.getLogger(__name__)


def kaiming_uniform(
    module: nn.Module,
    a: float = math.sqrt(5),
    mode: str = 'fan_in',
    nonlinearity: str = 'relu'
):
    """
    Kaiming/He initialization for ReLU activations.
    
    Args:
        module: PyTorch module to initialize.
        a: Negative slope of the rectifier used after this layer.
        mode: 'fan_in' or 'fan_out'.
        nonlinearity: Nonlinearity function name.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
        logger.debug(f"Applied Kaiming uniform initialization to {type(module).__name__}")



