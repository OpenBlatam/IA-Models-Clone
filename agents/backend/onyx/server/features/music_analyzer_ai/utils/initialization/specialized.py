"""
Specialized Initialization Module

Implements specialized initialization for LSTM and Transformer architectures.
"""

import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


def lstm_weights(module: nn.LSTM):
    """
    Proper LSTM weight initialization.
    
    Args:
        module: LSTM module to initialize.
    """
    for name, param in module.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            nn.init.zeros_(param.data)
            # Set forget gate bias to 1
            n = param.size(0)
            start, end = n // 4, n // 2
            param.data[start:end].fill_(1.0)
    logger.debug(f"Applied LSTM initialization to {type(module).__name__}")


def transformer_weights(module: nn.Module):
    """
    Transformer-specific initialization.
    
    Args:
        module: Transformer module to initialize.
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            # Use Xavier for most layers
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.Embedding):
            nn.init.normal_(submodule.weight, mean=0.0, std=0.02)
        elif isinstance(submodule, nn.LayerNorm):
            nn.init.ones_(submodule.weight)
            nn.init.zeros_(submodule.bias)
    logger.debug(f"Applied Transformer initialization to {type(module).__name__}")



