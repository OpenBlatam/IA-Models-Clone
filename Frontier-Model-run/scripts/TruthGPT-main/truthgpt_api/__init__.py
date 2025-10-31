"""
TruthGPT API - TensorFlow-like Interface for TruthGPT
====================================================

A TensorFlow-like API for TruthGPT that provides familiar interfaces
for building, training, and optimizing neural networks.

Usage:
    import truthgpt as tg
    
    # Create a model
    model = tg.Sequential([
        tg.layers.Dense(128, activation='relu'),
        tg.layers.Dropout(0.2),
        tg.layers.Dense(10, activation='softmax')
    ])
    
    # Compile and train
    model.compile(
        optimizer=tg.optimizers.Adam(learning_rate=0.001),
        loss=tg.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.fit(x_train, y_train, epochs=10, batch_size=32)
"""

# Core API imports
from .models import Sequential, Model, Functional
from .layers import *
from .optimizers import *
from .losses import *
from .metrics import *
from .utils import *

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









