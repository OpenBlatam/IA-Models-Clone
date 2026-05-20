"""
Hybrid audio separation model.
Combines multiple separation models for improved results.
Refactored to use constants.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np

from .base_separator import BaseSeparatorModel
from .constants import DEFAULT_NUM_SOURCES, DEFAULT_SAMPLE_RATE, DEFAULT_MODEL_TYPE
from .demucs_model import DemucsModel
from .spleeter_model import SpleeterModel


class HybridSeparatorModel(BaseSeparatorModel):
    """
    Hybrid model that combines multiple separation models.
    
    Uses ensemble methods to combine predictions from different models.
    """
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        num_sources: int = DEFAULT_NUM_SOURCES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        **kwargs
    ):
        """
        Initialize hybrid separator model.
        
        Args:
            models: List of model names to use (e.g., ['demucs', 'spleeter'])
            weights: Weights for each model (default: equal weights)
            num_sources: Number of sources
            sample_rate: Audio sample rate
        """
        super().__init__(num_sources=num_sources, sample_rate=sample_rate, **kwargs)
        
        if models is None:
            models = [DEFAULT_MODEL_TYPE, 'spleeter']
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError("weights must have same length as models")
        
        self.models = models
        self.weights = weights
        self.separators = []
        self._load_models()
    
    def _load_models(self):
        """Load all separator models."""
        for model_name in self.models:
            if model_name.lower() == 'demucs':
                separator = DemucsModel(
                    num_sources=self.num_sources,
                    sample_rate=self.sample_rate
                )
            elif model_name.lower() == 'spleeter':
                separator = SpleeterModel(
                    stems=self.num_sources,
                    sample_rate=self.sample_rate
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            self.separators.append(separator)
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Combines predictions from all models using weighted averaging.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Dictionary of separated sources
        """
        all_predictions = []
        
        # Get predictions from all models
        for separator in self.separators:
            prediction = separator.forward(audio)
            all_predictions.append(prediction)
        
        # Combine predictions
        source_names = list(all_predictions[0].keys())
        combined = {}
        
        for source_name in source_names:
            weighted_sum = None
            total_weight = 0.0
            
            for i, prediction in enumerate(all_predictions):
                if source_name in prediction:
                    source_audio = prediction[source_name]
                    weight = self.weights[i]
                    
                    if weighted_sum is None:
                        weighted_sum = source_audio * weight
                    else:
                        # Ensure same shape
                        if weighted_sum.shape != source_audio.shape:
                            # Resample or pad if needed
                            min_len = min(weighted_sum.shape[-1], source_audio.shape[-1])
                            weighted_sum = weighted_sum[..., :min_len]
                            source_audio = source_audio[..., :min_len]
                        weighted_sum = weighted_sum + source_audio * weight
                    
                    total_weight += weight
            
            if weighted_sum is not None and total_weight > 0:
                combined[source_name] = weighted_sum / total_weight
        
        return combined
    
    def separate(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Separate audio using hybrid model.
        
        Args:
            audio_path: Path to input audio
            output_dir: Output directory
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping source names to output paths
        """
        # Use the first model for file-based separation
        # In practice, you might want to combine file outputs too
        return self.separators[0].separate(audio_path, output_dir, **kwargs)

