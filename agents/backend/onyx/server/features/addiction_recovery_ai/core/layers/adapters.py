"""
Adapters - Bridge between existing models and new layered architecture
Enables seamless integration of legacy code with new modular system
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import logging

from .interfaces import IModel, IPredictor, IService
from .model_layer import ModelRegistry, ModelBuilder
from .inference_layer import InferenceEngine, InferencePipeline
from .service_layer import ServiceContainer

logger = logging.getLogger(__name__)


# ============================================================================
# Model Adapters - Adapt existing models to new architecture
# ============================================================================

class ModelAdapter:
    """Adapter to integrate existing models with new architecture"""
    
    @staticmethod
    def register_existing_model(
        model_class: type,
        model_name: str,
        default_config: Optional[Dict[str, Any]] = None
    ):
        """Register existing model class with new architecture"""
        ModelRegistry.register_model(model_name, model_class)
        
        # Create builder function
        def builder(**kwargs):
            config = {**(default_config or {}), **kwargs}
            return model_class(**config)
        
        ModelRegistry.register_builder(model_name, builder)
        logger.info(f"Registered existing model: {model_name}")
    
    @staticmethod
    def wrap_model(model: nn.Module, name: str = "WrappedModel") -> IModel:
        """Wrap existing model instance to implement IModel protocol"""
        class WrappedModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            
            def forward(self, *args, **kwargs):
                return self.base_model(*args, **kwargs)
        
        return WrappedModel(model)


# ============================================================================
# Predictor Adapters - Adapt existing predictors
# ============================================================================

class PredictorAdapter:
    """Adapter for existing predictor classes"""
    
    @staticmethod
    def create_from_model(
        model: nn.Module,
        preprocess_fn: Optional[callable] = None,
        postprocess_fn: Optional[callable] = None
    ) -> IPredictor:
        """Create predictor from existing model"""
        class ModelPredictor:
            def __init__(self, model, preprocess, postprocess):
                self.model = model
                self.preprocess = preprocess or (lambda x: x)
                self.postprocess = postprocess or (lambda x: x)
                self.model.eval()
            
            def predict(self, inputs: Any, **kwargs) -> Any:
                """Make prediction"""
                processed_inputs = self.preprocess(inputs)
                with torch.no_grad():
                    outputs = self.model(processed_inputs, **kwargs)
                return self.postprocess(outputs)
            
            def predict_batch(self, inputs: List[Any], **kwargs) -> List[Any]:
                """Make batch predictions"""
                return [self.predict(inp, **kwargs) for inp in inputs]
        
        return ModelPredictor(model, preprocess_fn, postprocess_fn)
    
    @staticmethod
    def create_from_analyzer(analyzer: Any) -> IPredictor:
        """Create predictor from existing analyzer (e.g., RecoverySentimentAnalyzer)"""
        class AnalyzerPredictor:
            def __init__(self, analyzer):
                self.analyzer = analyzer
            
            def predict(self, inputs: Any, **kwargs) -> Any:
                """Make prediction using analyzer"""
                if hasattr(self.analyzer, 'analyze'):
                    return self.analyzer.analyze(inputs, **kwargs)
                elif hasattr(self.analyzer, 'predict'):
                    return self.analyzer.predict(inputs, **kwargs)
                else:
                    raise ValueError("Analyzer doesn't have analyze or predict method")
            
            def predict_batch(self, inputs: List[Any], **kwargs) -> List[Any]:
                """Make batch predictions"""
                if hasattr(self.analyzer, 'analyze_batch'):
                    return self.analyzer.analyze_batch(inputs, **kwargs)
                else:
                    return [self.predict(inp, **kwargs) for inp in inputs]
        
        return AnalyzerPredictor(analyzer)


# ============================================================================
# Service Adapters - Adapt existing services
# ============================================================================

class ServiceAdapter:
    """Adapter for existing service classes"""
    
    @staticmethod
    def create_from_function(
        func: callable,
        validate_fn: Optional[callable] = None
    ) -> IService:
        """Create service from function"""
        class FunctionService:
            def __init__(self, func, validate):
                self.func = func
                self.validate = validate or (lambda x: True)
            
            def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
                """Execute service"""
                if not self.validate(request):
                    raise ValueError("Request validation failed")
                return self.func(request)
            
            def validate_request(self, request: Dict[str, Any]) -> bool:
                """Validate request"""
                return self.validate(request)
        
        return FunctionService(func, validate_fn)
    
    @staticmethod
    def create_from_class(service_class: type, **kwargs) -> IService:
        """Create service from class"""
        return service_class(**kwargs)


# ============================================================================
# Integration Utilities
# ============================================================================

class IntegrationHelper:
    """Helper utilities for integrating existing code with new architecture"""
    
    @staticmethod
    def create_inference_pipeline_from_model(
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True
    ) -> InferencePipeline:
        """Create inference pipeline from existing model"""
        engine = InferenceEngine(model, device, use_mixed_precision)
        return InferencePipeline(engine)
    
    @staticmethod
    def register_sentiment_analyzer():
        """Register RecoverySentimentAnalyzer with new architecture"""
        try:
            from ..models.sentiment_analyzer import RecoverySentimentAnalyzer
            
            ModelAdapter.register_existing_model(
                RecoverySentimentAnalyzer,
                "RecoverySentimentAnalyzer",
                default_config={
                    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "use_mixed_precision": True
                }
            )
            logger.info("RecoverySentimentAnalyzer registered")
        except ImportError:
            logger.warning("RecoverySentimentAnalyzer not available")
    
    @staticmethod
    def register_progress_predictor():
        """Register RecoveryProgressPredictor with new architecture"""
        try:
            from ..models.sentiment_analyzer import RecoveryProgressPredictor
            
            ModelAdapter.register_existing_model(
                RecoveryProgressPredictor,
                "RecoveryProgressPredictor",
                default_config={
                    "input_features": 10,
                    "hidden_size": 128
                }
            )
            logger.info("RecoveryProgressPredictor registered")
        except ImportError:
            logger.warning("RecoveryProgressPredictor not available")
    
    @staticmethod
    def register_all_models():
        """Register all existing models with new architecture"""
        IntegrationHelper.register_sentiment_analyzer()
        IntegrationHelper.register_progress_predictor()
        # Add more registrations as needed


# Export main components
__all__ = [
    "ModelAdapter",
    "PredictorAdapter",
    "ServiceAdapter",
    "IntegrationHelper",
]



