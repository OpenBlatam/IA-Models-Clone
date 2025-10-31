"""
🎯 Modular Machine Learning Project Structure

This package implements the key convention: "Create modular code structures with separate files 
for models, data loading, training, and evaluation."

The structure follows industry best practices for maintainable, scalable ML codebases.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy ML Team"

# Import main modules for easy access
from .models import *
from .data_loading import *
from .training import *
from .evaluation import *
from .utils import *
from .configs import *

__all__ = [
    # Models
    "BaseModel", "ClassificationModel", "RegressionModel", "GenerationModel",
    "TransformerModel", "DiffusionModel", "CustomModel",
    
    # Data Loading
    "BaseDataLoader", "ImageDataLoader", "TextDataLoader", "TabularDataLoader",
    "DataPreprocessor", "DataAugmenter", "DataValidator",
    
    # Training
    "BaseTrainer", "ClassificationTrainer", "RegressionTrainer", "GenerationTrainer",
    "TrainingConfig", "TrainingLoop", "TrainingMonitor",
    
    # Evaluation
    "BaseEvaluator", "ClassificationEvaluator", "RegressionEvaluator", "GenerationEvaluator",
    "MetricsCalculator", "ResultsVisualizer", "ModelComparison",
    
    # Utils
    "Logger", "ConfigManager", "CheckpointManager", "ExperimentTracker",
    
    # Configs
    "ModelConfig", "DataConfig", "TrainingConfig", "EvaluationConfig"
]






