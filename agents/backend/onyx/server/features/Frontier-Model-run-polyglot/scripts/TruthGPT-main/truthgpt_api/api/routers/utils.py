"""
Utilities Router
===============

Utility endpoints for API operations.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from ..schemas import LayerConfig
from ..services import create_layer_from_config
from ..exceptions import InvalidLayerTypeError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/utils", tags=["utils"])


@router.get("/layers")
async def list_available_layers():
    """
    List all available layer types.
    
    Returns information about all supported layer types and their parameters.
    """
    layers_info = {
        "dense": {
            "description": "Fully connected layer",
            "required_params": ["units"],
            "optional_params": ["activation", "use_bias"]
        },
        "conv2d": {
            "description": "2D convolutional layer",
            "required_params": ["filters", "kernel_size"],
            "optional_params": ["strides", "padding", "activation"]
        },
        "lstm": {
            "description": "LSTM layer for sequence processing",
            "required_params": ["units"],
            "optional_params": ["return_sequences", "dropout"]
        },
        "gru": {
            "description": "GRU layer for sequence processing",
            "required_params": ["units"],
            "optional_params": ["return_sequences", "dropout"]
        },
        "dropout": {
            "description": "Dropout layer for regularization",
            "required_params": ["rate"],
            "optional_params": []
        },
        "batchnormalization": {
            "description": "Batch normalization layer",
            "required_params": [],
            "optional_params": ["momentum", "epsilon"]
        },
        "maxpooling2d": {
            "description": "2D max pooling layer",
            "required_params": ["pool_size"],
            "optional_params": ["strides", "padding"]
        },
        "averagepooling2d": {
            "description": "2D average pooling layer",
            "required_params": ["pool_size"],
            "optional_params": ["strides", "padding"]
        },
        "flatten": {
            "description": "Flatten layer",
            "required_params": [],
            "optional_params": []
        },
        "reshape": {
            "description": "Reshape layer",
            "required_params": ["target_shape"],
            "optional_params": []
        },
        "embedding": {
            "description": "Embedding layer",
            "required_params": ["input_dim", "output_dim"],
            "optional_params": ["mask_zero"]
        }
    }
    
    return {
        "layers": layers_info,
        "total": len(layers_info)
    }


@router.get("/optimizers")
async def list_available_optimizers():
    """
    List all available optimizer types.
    
    Returns information about all supported optimizers and their parameters.
    """
    optimizers_info = {
        "adam": {
            "description": "Adam optimizer",
            "optional_params": ["learning_rate", "beta_1", "beta_2", "epsilon"]
        },
        "sgd": {
            "description": "Stochastic gradient descent optimizer",
            "optional_params": ["learning_rate", "momentum", "nesterov"]
        },
        "rmsprop": {
            "description": "RMSprop optimizer",
            "optional_params": ["learning_rate", "rho", "epsilon"]
        },
        "adagrad": {
            "description": "Adagrad optimizer",
            "optional_params": ["learning_rate", "epsilon"]
        },
        "adamw": {
            "description": "AdamW optimizer",
            "optional_params": ["learning_rate", "beta_1", "beta_2", "epsilon", "weight_decay"]
        }
    }
    
    return {
        "optimizers": optimizers_info,
        "total": len(optimizers_info)
    }


@router.get("/losses")
async def list_available_losses():
    """
    List all available loss function types.
    
    Returns information about all supported loss functions and their parameters.
    """
    losses_info = {
        "sparsecategoricalcrossentropy": {
            "description": "Sparse categorical crossentropy loss",
            "optional_params": ["from_logits"]
        },
        "categoricalcrossentropy": {
            "description": "Categorical crossentropy loss",
            "optional_params": ["from_logits"]
        },
        "binarycrossentropy": {
            "description": "Binary crossentropy loss",
            "optional_params": ["from_logits"]
        },
        "meansquarederror": {
            "description": "Mean squared error loss",
            "optional_params": []
        },
        "mse": {
            "description": "Mean squared error loss (alias)",
            "optional_params": []
        },
        "meanabsoluteerror": {
            "description": "Mean absolute error loss",
            "optional_params": []
        },
        "mae": {
            "description": "Mean absolute error loss (alias)",
            "optional_params": []
        }
    }
    
    return {
        "losses": losses_info,
        "total": len(losses_info)
    }


@router.post("/validate/layer")
async def validate_layer_config(config: LayerConfig):
    """
    Validate a layer configuration.
    
    Checks if the layer configuration is valid without creating the layer.
    """
    try:
        layer = create_layer_from_config(config)
        return {
            "valid": True,
            "message": "Layer configuration is valid",
            "layer_type": type(layer).__name__
        }
    except InvalidLayerTypeError as e:
        return {
            "valid": False,
            "message": str(e),
            "error": "InvalidLayerTypeError"
        }
    except Exception as e:
        return {
            "valid": False,
            "message": str(e),
            "error": type(e).__name__
        }

