"""
Integration Utilities - Seamless integration between layers
Provides high-level utilities for common workflows
"""

from typing import Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

from .data_layer import DataPipeline, DataLoaderFactory
from .model_layer import ModelBuilder, ModelRegistry
from .training_layer import TrainingPipeline, TrainingConfig, OptimizerFactory
from .inference_layer import InferenceEngine, InferencePipeline
from .service_layer import ServiceContainer, ServiceFactory
from .interface_layer import InterfaceFactory
from .adapters import IntegrationHelper

logger = logging.getLogger(__name__)


# ============================================================================
# Complete Workflow Builder
# ============================================================================

class WorkflowBuilder:
    """
    Builder for complete ML workflows
    Combines all layers into a cohesive workflow
    """
    
    def __init__(self, name: str = "Workflow"):
        self.name = name
        self.data_pipeline: Optional[DataPipeline] = None
        self.model: Optional[nn.Module] = None
        self.training_pipeline: Optional[TrainingPipeline] = None
        self.inference_pipeline: Optional[InferencePipeline] = None
        self.service_container: Optional[ServiceContainer] = None
    
    def with_data_pipeline(self, pipeline: DataPipeline) -> 'WorkflowBuilder':
        """Add data processing pipeline"""
        self.data_pipeline = pipeline
        return self
    
    def with_model(
        self,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> 'WorkflowBuilder':
        """Add model"""
        builder = ModelBuilder()
        if config:
            builder.with_config(**config)
        if device:
            builder.with_device(device)
        self.model = builder.build(model_type)
        return self
    
    def with_training(
        self,
        config: Optional[TrainingConfig] = None,
        optimizer_type: str = "adamw",
        scheduler_type: Optional[str] = None
    ) -> 'WorkflowBuilder':
        """Add training pipeline"""
        if not self.model:
            raise ValueError("Model must be set before training pipeline")
        
        config = config or TrainingConfig()
        self.training_pipeline = TrainingPipeline(self.model, config)
        
        # Setup optimizer
        optimizer = OptimizerFactory.create(
            optimizer_type,
            self.model,
            learning_rate=config.learning_rate
        )
        self.training_pipeline.optimizer = optimizer
        
        # Setup scheduler if specified
        if scheduler_type:
            from .training_layer import SchedulerFactory
            scheduler = SchedulerFactory.create(scheduler_type, optimizer)
            self.training_pipeline.set_scheduler(scheduler_type)
        
        return self
    
    def with_inference(
        self,
        use_mixed_precision: bool = True
    ) -> 'WorkflowBuilder':
        """Add inference pipeline"""
        if not self.model:
            raise ValueError("Model must be set before inference pipeline")
        
        engine = InferenceEngine(self.model, use_mixed_precision=use_mixed_precision)
        self.inference_pipeline = InferencePipeline(engine)
        
        # Add data pipeline preprocessing if available
        if self.data_pipeline:
            for processor in self.data_pipeline.processors:
                self.inference_pipeline.add_preprocessor(processor.process)
        
        return self
    
    def with_service(
        self,
        service_name: str,
        service_class: type,
        singleton: bool = True
    ) -> 'WorkflowBuilder':
        """Add service"""
        if not self.service_container:
            self.service_container = ServiceContainer()
        
        # Create service with dependencies
        if self.inference_pipeline:
            service = service_class(engine=self.inference_pipeline.engine)
        else:
            service = service_class()
        
        self.service_container.register(service_name, service, singleton=singleton)
        return self
    
    def build(self) -> 'CompleteWorkflow':
        """Build complete workflow"""
        return CompleteWorkflow(
            name=self.name,
            data_pipeline=self.data_pipeline,
            model=self.model,
            training_pipeline=self.training_pipeline,
            inference_pipeline=self.inference_pipeline,
            service_container=self.service_container
        )


# ============================================================================
# Complete Workflow
# ============================================================================

class CompleteWorkflow:
    """Complete ML workflow combining all layers"""
    
    def __init__(
        self,
        name: str,
        data_pipeline: Optional[DataPipeline] = None,
        model: Optional[nn.Module] = None,
        training_pipeline: Optional[TrainingPipeline] = None,
        inference_pipeline: Optional[InferencePipeline] = None,
        service_container: Optional[ServiceContainer] = None
    ):
        self.name = name
        self.data_pipeline = data_pipeline
        self.model = model
        self.training_pipeline = training_pipeline
        self.inference_pipeline = inference_pipeline
        self.service_container = service_container
    
    def process_data(self, data: Any) -> Any:
        """Process data through pipeline"""
        if not self.data_pipeline:
            return data
        return self.data_pipeline.process(data)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train model"""
        if not self.training_pipeline:
            raise ValueError("Training pipeline not configured")
        
        if num_epochs:
            self.training_pipeline.config.num_epochs = num_epochs
        
        # Training logic would go here
        # This is a simplified version
        return {"status": "training started"}
    
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Make predictions"""
        if not self.inference_pipeline:
            raise ValueError("Inference pipeline not configured")
        
        return self.inference_pipeline.process(inputs, **kwargs)
    
    def get_service(self, service_name: str) -> Any:
        """Get service from container"""
        if not self.service_container:
            raise ValueError("Service container not configured")
        return self.service_container.get(service_name)
    
    def create_api_handler(self, service_name: str) -> Any:
        """Create API handler for service"""
        service = self.get_service(service_name)
        return InterfaceFactory.create_handler(service)


# ============================================================================
# Quick Setup Functions
# ============================================================================

def create_sentiment_workflow(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    device: Optional[torch.device] = None
) -> CompleteWorkflow:
    """Quick setup for sentiment analysis workflow"""
    # Register models
    IntegrationHelper.register_all_models()
    
    # Build workflow
    workflow = (WorkflowBuilder("SentimentAnalysis")
        .with_model("RecoverySentimentAnalyzer", {"model_name": model_name}, device)
        .with_inference(use_mixed_precision=True)
        .build())
    
    return workflow


def create_training_workflow(
    model_type: str,
    model_config: Dict[str, Any],
    training_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None
) -> CompleteWorkflow:
    """Quick setup for training workflow"""
    workflow = (WorkflowBuilder("Training")
        .with_model(model_type, model_config, device)
        .with_training(config=training_config)
        .build())
    
    return workflow


def create_inference_workflow(
    model_type: str,
    model_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> CompleteWorkflow:
    """Quick setup for inference workflow"""
    workflow = (WorkflowBuilder("Inference")
        .with_model(model_type, model_config, device)
        .with_inference(use_mixed_precision=True)
        .build())
    
    return workflow


# Export main components
__all__ = [
    "WorkflowBuilder",
    "CompleteWorkflow",
    "create_sentiment_workflow",
    "create_training_workflow",
    "create_inference_workflow",
]



