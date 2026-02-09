"""
Processing Layers Module

Concrete processing layer implementations.
"""

from typing import Any
import logging
import time

logger = logging.getLogger(__name__)

from .base import ProcessingLayer, ProcessingStage


class PreprocessingLayer(ProcessingLayer):
    """Preprocessing layer for audio data."""
    
    def __init__(self):
        super().__init__("preprocessing", ProcessingStage.PREPROCESSING)
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Preprocess audio data."""
        # Normalize, resample, etc.
        return input_data


class FeatureExtractionLayer(ProcessingLayer):
    """Feature extraction layer."""
    
    def __init__(self):
        super().__init__("feature_extraction", ProcessingStage.FEATURE_EXTRACTION)
        try:
            from ..ml_audio_analyzer import AudioFeatureExtractor
            self.extractor = AudioFeatureExtractor()
        except ImportError:
            self.extractor = None
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Extract features from audio."""
        if self.extractor is None:
            raise ImportError("AudioFeatureExtractor not available")
        
        if isinstance(input_data, str):
            # Audio file path
            return self.extractor.extract_features(input_data)
        else:
            # Audio array
            return self.extractor.extract_from_array(input_data, kwargs.get("sr", 22050))


class MLInferenceLayer(ProcessingLayer):
    """ML inference layer."""
    
    def __init__(self, model_name: str = "multi_task"):
        super().__init__("ml_inference", ProcessingStage.ML_INFERENCE)
        self.model_name = model_name
        try:
            from ..models import get_deep_analyzer
            self.analyzer = get_deep_analyzer()
        except ImportError:
            self.analyzer = None
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Run ML inference."""
        if self.analyzer is None:
            raise ImportError("Deep analyzer not available")
        
        # Convert to feature vector if needed
        if hasattr(input_data, "mfcc"):
            # AudioFeatures object
            import numpy as np
            features = np.concatenate([
                input_data.mfcc.mean(axis=1),
                input_data.chroma.mean(axis=1),
                input_data.spectral_contrast.mean(axis=1),
                input_data.tonnetz.mean(axis=1),
                [input_data.tempo]
            ])
        else:
            features = input_data
        
        if self.model_name == "multi_task":
            return self.analyzer.predict_multi_task(features)
        elif self.model_name == "genre":
            return self.analyzer.predict_genre(features)
        else:
            return {"error": f"Unknown model: {self.model_name}"}


class PostprocessingLayer(ProcessingLayer):
    """Postprocessing layer."""
    
    def __init__(self):
        super().__init__("postprocessing", ProcessingStage.POSTPROCESSING)
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Postprocess ML results."""
        # Format, validate, enrich results
        if isinstance(input_data, dict):
            # Add metadata, format output
            result = input_data.copy()
            result["processed_at"] = time.time()
            return result
        return input_data


class ValidationLayer(ProcessingLayer):
    """Validation layer."""
    
    def __init__(self):
        super().__init__("validation", ProcessingStage.VALIDATION)
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Validate processing results."""
        if isinstance(input_data, dict):
            # Check required fields
            required = kwargs.get("required_fields", [])
            for field in required:
                if field not in input_data:
                    raise ValueError(f"Missing required field: {field}")
        
        return input_data

