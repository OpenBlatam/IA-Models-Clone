"""
Optimizer Constants
===================

Centralized constants for optimizer default parameters.
Eliminates hardcoded values across multiple files.

Single Responsibility: Define default values for optimizer parameters.
"""

# Learning rate defaults
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SGD_LEARNING_RATE = 0.01

# Adam/AdamW parameter defaults
DEFAULT_BETA_1 = 0.9
DEFAULT_BETA_2 = 0.999
DEFAULT_EPSILON = 1e-7

# RMSprop parameter defaults
DEFAULT_RHO = 0.9
DEFAULT_MOMENTUM = 0.0

# Weight decay defaults
DEFAULT_WEIGHT_DECAY = 0.01

# AMSGrad defaults
DEFAULT_AMSGRAD = False

# Nesterov defaults
DEFAULT_NESTEROV = False

# Centered defaults (for RMSprop)
DEFAULT_CENTERED = False

# Optimizer type constants
OPTIMIZER_TYPES = {
    'ADAM': 'adam',
    'SGD': 'sgd',
    'RMSPROP': 'rmsprop',
    'ADAGRAD': 'adagrad',
    'ADAMW': 'adamw'
}

# Supported optimizer types
SUPPORTED_OPTIMIZER_TYPES = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamw']

# AMSGrad supported types (list)
AMSGRAD_SUPPORTED_TYPES = ['adam', 'adamw']

# AMSGrad supported types (set for fast lookups)
AMSGRAD_SUPPORTED = {'adam', 'adamw'}

# TensorFlow optimizer types (set for fast lookups)
TENSORFLOW_OPTIMIZER_TYPES = {'adam', 'sgd'}

# ════════════════════════════════════════════════════════════════════════════
# Backend name constants
BACKEND_OPTIMIZATION_CORE = 'optimization_core'
BACKEND_PYTORCH = 'pytorch'
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def normalize_optimizer_type(optimizer_type: str) -> str:
    """
    Normalize optimizer type to lowercase.
    
    Args:
        optimizer_type: Optimizer type string
        
    Returns:
        Lowercase optimizer type
    """
    return optimizer_type.lower()


def is_amsgrad_supported(optimizer_type: str) -> bool:
    """
    Check if optimizer type supports AMSGrad.
    
    Args:
        optimizer_type: Optimizer type string
        
    Returns:
        True if AMSGrad is supported
    """
    return normalize_optimizer_type(optimizer_type) in AMSGRAD_SUPPORTED


def is_tensorflow_optimizer_type(optimizer_type: str) -> bool:
    """
    Check if optimizer type is a TensorFlow-inspired optimizer.
    
    Args:
        optimizer_type: Optimizer type string
        
    Returns:
        True if it's a TensorFlow optimizer type
    """
    return normalize_optimizer_type(optimizer_type) in TENSORFLOW_OPTIMIZER_TYPES


def is_supported_optimizer_type(optimizer_type: str) -> bool:
    """
    Check if optimizer type is supported.
    
    Args:
        optimizer_type: Optimizer type string
        
    Returns:
        True if optimizer type is supported
    """
    return normalize_optimizer_type(optimizer_type) in SUPPORTED_OPTIMIZER_TYPES

