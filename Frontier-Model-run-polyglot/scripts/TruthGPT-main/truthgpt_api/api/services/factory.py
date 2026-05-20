"""
Factory Functions
================

Factory functions for creating layers, optimizers, and loss functions.
"""

from typing import Dict, Any, Type, TypeVar
import truthgpt as tg
from ..schemas import LayerConfig
from ..exceptions import InvalidLayerTypeError, InvalidOptimizerError, InvalidLossError

T = TypeVar('T')


def _create_from_map(
    item_type: str,
    type_map: Dict[str, Type[T]],
    params: Dict[str, Any],
    error_class: type
) -> T:
    """
    Generic factory function for creating instances from type maps.
    
    Args:
        item_type: Type name (normalized to lowercase)
        type_map: Dictionary mapping type names to classes
        params: Parameters to pass to the class constructor
        error_class: Exception class to raise if type not found
        
    Returns:
        Instance of the requested type
        
    Raises:
        error_class: If type is not found in map
    """
    normalized_type = item_type.lower()
    item_class = type_map.get(normalized_type)
    if item_class is None:
        raise error_class(item_type)
    return item_class(**params)


def create_layer_from_config(config: LayerConfig) -> Any:
    """
    Create a layer from configuration.
    
    Args:
        config: Layer configuration
        
    Returns:
        Layer instance
        
    Raises:
        InvalidLayerTypeError: If layer type is unknown
    """
    layer_map = {
        "dense": tg.layers.Dense,
        "conv2d": tg.layers.Conv2D,
        "lstm": tg.layers.LSTM,
        "gru": tg.layers.GRU,
        "dropout": tg.layers.Dropout,
        "batchnormalization": tg.layers.BatchNormalization,
        "maxpooling2d": tg.layers.MaxPooling2D,
        "averagepooling2d": tg.layers.AveragePooling2D,
        "flatten": tg.layers.Flatten,
        "reshape": tg.layers.Reshape,
        "embedding": tg.layers.Embedding,
    }
    
    return _create_from_map(config.type, layer_map, config.params, InvalidLayerTypeError)


def create_optimizer_from_config(optimizer: str, params: Dict[str, Any]) -> Any:
    """
    Create an optimizer from configuration.
    
    Args:
        optimizer: Optimizer type name
        params: Optimizer parameters
        
    Returns:
        Optimizer instance
        
    Raises:
        InvalidOptimizerError: If optimizer type is unknown
    """
    optimizer_map = {
        "adam": tg.optimizers.Adam,
        "sgd": tg.optimizers.SGD,
        "rmsprop": tg.optimizers.RMSprop,
        "adagrad": tg.optimizers.Adagrad,
        "adamw": tg.optimizers.AdamW,
    }
    
    return _create_from_map(optimizer, optimizer_map, params, InvalidOptimizerError)


def create_loss_from_config(loss: str, params: Dict[str, Any]) -> Any:
    """
    Create a loss function from configuration.
    
    Args:
        loss: Loss function type name
        params: Loss function parameters
        
    Returns:
        Loss function instance
        
    Raises:
        InvalidLossError: If loss type is unknown
    """
    loss_map = {
        "sparsecategoricalcrossentropy": tg.losses.SparseCategoricalCrossentropy,
        "categoricalcrossentropy": tg.losses.CategoricalCrossentropy,
        "binarycrossentropy": tg.losses.BinaryCrossentropy,
        "meansquarederror": tg.losses.MeanSquaredError,
        "mse": tg.losses.MeanSquaredError,
        "meanabsoluteerror": tg.losses.MeanAbsoluteError,
        "mae": tg.losses.MeanAbsoluteError,
    }
    
    return _create_from_map(loss, loss_map, params, InvalidLossError)

