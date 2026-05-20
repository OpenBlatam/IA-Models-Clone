"""
TruthGPT API - TensorFlow-like Interface for TruthGPT
====================================================

A TensorFlow-like API for TruthGPT that provides familiar interfaces
for building, training, and optimizing neural networks.

Now integrated with optimization_core for enhanced performance and
access to advanced optimization techniques.

Usage:
    import truthgpt as tg
    
    # Create a model
    model = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(10, activation='softmax')
    ])
    
    # Compile and train (automatically uses optimization_core if available)
    model.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    
    # Check if optimization_core is being used
    optimizer_config = model.optimizer.get_config()
    if optimizer_config.get('using_optimization_core'):
        print("Using optimization_core for enhanced performance!")
"""

# Core API imports
from .models import Sequential, Model, Functional
from .layers import (
    Dense, Conv2D, LSTM, GRU, Dropout, BatchNormalization,
    MaxPooling2D, AveragePooling2D, Flatten, Reshape,
    Embedding, Attention, MultiHeadAttention
)
from .optimizers import Adam, SGD, RMSprop, Adagrad, AdamW
from .losses import (
    SparseCategoricalCrossentropy, CategoricalCrossentropy,
    BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
)
from .metrics import Accuracy, Precision, Recall, F1Score
from .utils import to_categorical, normalize, get_data, save_model, load_model

# Version information
__version__ = "1.0.0"
__author__ = "TruthGPT Team"
__description__ = "TensorFlow-like API for TruthGPT"

# Main API exports
__all__ = [
    # Models
    'Sequential', 'Model', 'Functional',
    
    # Layers
    'Dense', 'Conv2D', 'LSTM', 'GRU', 'Dropout', 'BatchNormalization',
    'MaxPooling2D', 'AveragePooling2D', 'Flatten', 'Reshape',
    'Embedding', 'Attention', 'MultiHeadAttention',
    
    # Optimizers
    'Adam', 'SGD', 'RMSprop', 'Adagrad', 'AdamW',
    
    # Losses
    'SparseCategoricalCrossentropy', 'CategoricalCrossentropy',
    'BinaryCrossentropy', 'MeanSquaredError', 'MeanAbsoluteError',
    
    # Metrics
    'Accuracy', 'Precision', 'Recall', 'F1Score',
    
    # Utils
    'to_categorical', 'normalize', 'get_data', 'save_model', 'load_model'
]









