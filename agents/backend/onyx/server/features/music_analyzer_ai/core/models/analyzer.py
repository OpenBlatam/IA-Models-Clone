"""
Deep Music Analyzer Module

Orchestrates multiple deep learning models for music analysis.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from .genre_classifier import DeepGenreClassifier
from .mood_detector import DeepMoodDetector
from .multitask import MultiTaskMusicModel
from .transformer_encoder import TransformerMusicEncoder


class DeepMusicAnalyzer:
    """
    Deep learning analyzer with multiple model architectures.
    
    Args:
        device: Device to run models on.
        compile_models: If True, compile models for faster inference.
    """
    
    def __init__(self, device: str = "cpu", compile_models: bool = True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.device = device
        self.models: Dict[str, nn.Module] = {}
        self.compile_models = compile_models and hasattr(torch, 'compile')
        self._initialize_models()
        logger.debug(f"Initialized DeepMusicAnalyzer with device='{device}', compile_models={compile_models}")
    
    def _initialize_models(self):
        """Initialize all deep models."""
        try:
            # Deep Genre Classifier
            self.models["genre_classifier"] = DeepGenreClassifier(
                input_size=169,
                num_genres=10,
                hidden_layers=[512, 512, 256, 256, 128, 128],
                use_residual=True
            ).to(self.device)
            
            # Deep Mood Detector
            self.models["mood_detector"] = DeepMoodDetector(
                input_channels=13,
                num_moods=6,
                cnn_channels=[32, 64, 128],
                lstm_hidden=256,
                lstm_layers=2
            ).to(self.device)
            
            # Multi-task Model
            self.models["multi_task"] = MultiTaskMusicModel(
                input_size=169,
                num_genres=10,
                num_moods=6,
                num_instruments=15,
                shared_layers=[512, 512, 256]
            ).to(self.device)
            
            # Transformer Encoder
            self.models["transformer_encoder"] = TransformerMusicEncoder(
                input_dim=169,
                embed_dim=256,
                num_heads=8,
                num_layers=4
            ).to(self.device)
            
            # Set to eval mode and compile for speed
            for name, model in self.models.items():
                model.eval()
                # Compile for faster inference
                if self.compile_models:
                    try:
                        self.models[name] = torch.compile(model, mode="reduce-overhead")
                        logger.info(f"Compiled {name} model")
                    except Exception as e:
                        logger.warning(f"Could not compile {name}: {e}")
            
            logger.info(f"Initialized {len(self.models)} models on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing models: {e}", exc_info=True)
            raise
    
    def analyze(self, features: torch.Tensor, model_name: Optional[str] = None) -> Dict:
        """
        Analyze music features using specified model.
        
        Args:
            features: Input features tensor.
            model_name: Name of model to use (None for all models).
        
        Returns:
            Dictionary of model predictions.
        """
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            model = self.models[model_name]
            with torch.no_grad():
                output = model(features.to(self.device))
            return {model_name: output}
        else:
            results = {}
            for name, model in self.models.items():
                with torch.no_grad():
                    results[name] = model(features.to(self.device))
            return results


def get_deep_analyzer(device: str = "cpu", compile_models: bool = True) -> DeepMusicAnalyzer:
    """
    Factory function to get DeepMusicAnalyzer instance.
    
    Args:
        device: Device to run models on.
        compile_models: If True, compile models for faster inference.
    
    Returns:
        DeepMusicAnalyzer instance.
    """
    return DeepMusicAnalyzer(device=device, compile_models=compile_models)



