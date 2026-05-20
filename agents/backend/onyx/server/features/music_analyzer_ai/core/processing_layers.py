"""
Processing Layers - Multi-layer processing pipeline for music analysis
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages in the pipeline"""
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    ML_INFERENCE = "ml_inference"
    POSTPROCESSING = "postprocessing"
    VALIDATION = "validation"


@dataclass
class ProcessingResult:
    """Result from a processing layer"""
    stage: ProcessingStage
    data: Any
    metadata: Dict[str, Any]
    processing_time: float
    success: bool = True
    error: Optional[str] = None


class ProcessingLayer:
    """Base class for processing layers"""
    
    def __init__(self, name: str, stage: ProcessingStage):
        self.name = name
        self.stage = stage
    
    def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """Process input data"""
        start_time = time.time()
        try:
            result = self._process(input_data, **kwargs)
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                stage=self.stage,
                data=result,
                metadata={"layer": self.name},
                processing_time=processing_time,
                success=True
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in layer {self.name}: {str(e)}")
            return ProcessingResult(
                stage=self.stage,
                data=None,
                metadata={"layer": self.name},
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Override this method in subclasses"""
        raise NotImplementedError


class PreprocessingLayer(ProcessingLayer):
    """Preprocessing layer for audio data"""
    
    def __init__(self):
        super().__init__("preprocessing", ProcessingStage.PREPROCESSING)
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Preprocess audio data"""
        # Normalize, resample, etc.
        return input_data


class FeatureExtractionLayer(ProcessingLayer):
    """Feature extraction layer"""
    
    def __init__(self):
        super().__init__("feature_extraction", ProcessingStage.FEATURE_EXTRACTION)
        try:
            from .ml_audio_analyzer import AudioFeatureExtractor
            self.extractor = AudioFeatureExtractor()
        except ImportError:
            self.extractor = None
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Extract features from audio"""
        if self.extractor is None:
            raise ImportError("AudioFeatureExtractor not available")
        
        if isinstance(input_data, str):
            # Audio file path
            return self.extractor.extract_features(input_data)
        else:
            # Audio array
            return self.extractor.extract_from_array(input_data, kwargs.get("sr", 22050))


class MLInferenceLayer(ProcessingLayer):
    """ML inference layer"""
    
    def __init__(self, model_name: str = "multi_task"):
        super().__init__("ml_inference", ProcessingStage.ML_INFERENCE)
        self.model_name = model_name
        try:
            from .deep_models import get_deep_analyzer
            self.analyzer = get_deep_analyzer()
        except ImportError:
            self.analyzer = None
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Run ML inference"""
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
    """Postprocessing layer"""
    
    def __init__(self):
        super().__init__("postprocessing", ProcessingStage.POSTPROCESSING)
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Postprocess ML results"""
        # Format, validate, enrich results
        if isinstance(input_data, dict):
            # Add metadata, format output
            result = input_data.copy()
            result["processed_at"] = time.time()
            return result
        return input_data


class ValidationLayer(ProcessingLayer):
    """Validation layer"""
    
    def __init__(self):
        super().__init__("validation", ProcessingStage.VALIDATION)
    
    def _process(self, input_data: Any, **kwargs) -> Any:
        """Validate processing results"""
        if isinstance(input_data, dict):
            # Check required fields
            required = kwargs.get("required_fields", [])
            for field in required:
                if field not in input_data:
                    raise ValueError(f"Missing required field: {field}")
        
        return input_data


class ProcessingPipeline:
    """
    Multi-layer processing pipeline
    Chains multiple processing layers together
    """
    
    def __init__(self):
        self.layers: List[ProcessingLayer] = []
        self.results: List[ProcessingResult] = []
    
    def add_layer(self, layer: ProcessingLayer):
        """Add a processing layer to the pipeline"""
        self.layers.append(layer)
        logger.info(f"Added layer: {layer.name} ({layer.stage.value})")
    
    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Process data through all layers"""
        start_time = time.time()
        self.results = []
        
        current_data = input_data
        
        for layer in self.layers:
            result = layer.process(current_data, **kwargs)
            self.results.append(result)
            
            if not result.success:
                logger.error(f"Layer {layer.name} failed: {result.error}")
                return {
                    "success": False,
                    "error": result.error,
                    "failed_layer": layer.name,
                    "results": [r.__dict__ for r in self.results]
                }
            
            current_data = result.data
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "data": current_data,
            "results": [r.__dict__ for r in self.results],
            "total_time": total_time,
            "layers_processed": len(self.results)
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline"""
        return {
            "num_layers": len(self.layers),
            "layers": [
                {
                    "name": layer.name,
                    "stage": layer.stage.value
                }
                for layer in self.layers
            ]
        }


def create_default_pipeline() -> ProcessingPipeline:
    """Create default processing pipeline"""
    pipeline = ProcessingPipeline()
    
    # Add layers in order
    pipeline.add_layer(PreprocessingLayer())
    pipeline.add_layer(FeatureExtractionLayer())
    pipeline.add_layer(MLInferenceLayer(model_name="multi_task"))
    pipeline.add_layer(PostprocessingLayer())
    pipeline.add_layer(ValidationLayer())
    
    return pipeline

