"""
Example: Using Ultra-Modular Layered Architecture V3
Demonstrates the new layered architecture with clear separation of concerns
"""

import torch
import torch.nn as nn
from typing import Dict, Any

# Import layered architecture components
from addiction_recovery_ai.core.layers import (
    # Data Layer
    DataPipeline,
    NormalizationProcessor,
    TokenizationProcessor,
    PaddingProcessor,
    DataValidator,
    DatasetFactory,
    DataLoaderFactory,
    # Model Layer
    ModelConfig,
    ModelRegistry,
    ModelBuilder,
    ModelFactory,
    # Training Layer
    TrainingConfig,
    OptimizerFactory,
    SchedulerFactory,
    TrainingPipeline,
    # Inference Layer
    InferenceEngine,
    InferencePipeline,
    BatchProcessor,
    # Service Layer
    ServiceContainer,
    ServiceFactory,
    # Interface Layer
    InterfaceFactory,
    # Dependency Injection
    DependencyContainer,
    inject_dependencies,
    register_service,
)


# ============================================================================
# Example 1: Data Layer - Processing Pipeline
# ============================================================================

def example_data_pipeline():
    """Example of modular data processing"""
    print("\n=== Example 1: Data Pipeline ===")
    
    # Create processing pipeline
    pipeline = DataPipeline("SentimentPipeline")
    
    # Add processors (chain of responsibility pattern)
    pipeline.add_processor(NormalizationProcessor())
    pipeline.add_processor(PaddingProcessor(max_length=128, pad_value=0))
    
    # Process data
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    processed = pipeline.process(data)
    
    print(f"Original: {data}")
    print(f"Processed: {processed}")
    print(f"Pipeline processors: {len(pipeline.processors)}")


# ============================================================================
# Example 2: Model Layer - Model Building
# ============================================================================

def example_model_building():
    """Example of modular model building"""
    print("\n=== Example 2: Model Building ===")
    
    # Define a simple model
    class SimplePredictor(nn.Module):
        def __init__(self, input_size=10, hidden_size=64):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Register model
    ModelRegistry.register_model("SimplePredictor", SimplePredictor)
    
    # Build model using builder pattern
    model = (ModelBuilder()
        .with_config(input_size=10, hidden_size=64)
        .with_device(torch.device("cpu"))
        .with_mixed_precision(False)
        .build("SimplePredictor"))
    
    print(f"Model created: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")


# ============================================================================
# Example 3: Training Layer - Training Pipeline
# ============================================================================

def example_training_pipeline():
    """Example of modular training"""
    print("\n=== Example 3: Training Pipeline ===")
    
    # Create simple model
    model = nn.Linear(10, 1)
    
    # Create training config
    config = TrainingConfig(
        num_epochs=10,
        learning_rate=1e-3,
        batch_size=32,
        use_mixed_precision=False
    )
    
    # Create training pipeline
    pipeline = TrainingPipeline(model, config)
    pipeline.set_criterion(nn.MSELoss())
    pipeline.set_scheduler("reduce_on_plateau", patience=3)
    
    print(f"Training config: {config.to_dict()}")
    print(f"Optimizer: {type(pipeline.optimizer).__name__}")
    print(f"Scheduler: {type(pipeline.scheduler).__name__}")


# ============================================================================
# Example 4: Inference Layer - Inference Engine
# ============================================================================

def example_inference_engine():
    """Example of modular inference"""
    print("\n=== Example 4: Inference Engine ===")
    
    # Create simple model
    model = nn.Linear(10, 1)
    
    # Create inference engine
    engine = InferenceEngine(model, use_mixed_precision=False)
    
    # Create inference pipeline
    pipeline = InferencePipeline(engine)
    
    # Add preprocessing
    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    
    pipeline.add_preprocessor(normalize)
    
    # Process input
    input_data = torch.randn(1, 10)
    output = pipeline.process(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")


# ============================================================================
# Example 5: Service Layer - Service Container
# ============================================================================

def example_service_layer():
    """Example of modular services"""
    print("\n=== Example 5: Service Layer ===")
    
    # Create service container
    container = ServiceContainer()
    
    # Register service as singleton
    class SentimentService:
        def __init__(self):
            self.model = nn.Linear(10, 3)
        
        def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
            text = request.get("text", "")
            return {"sentiment": "positive", "score": 0.95}
    
    container.register("SentimentService", SentimentService, singleton=True)
    
    # Create service factory
    factory = ServiceFactory(container)
    
    # Get service
    service = factory.create("SentimentService")
    
    # Execute
    result = service.execute({"text": "I'm feeling great!"})
    print(f"Service result: {result}")


# ============================================================================
# Example 6: Interface Layer - API Handler
# ============================================================================

def example_interface_layer():
    """Example of modular API handling"""
    print("\n=== Example 6: Interface Layer ===")
    
    # Create simple service
    class SimpleService:
        def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": request.get("value", 0) * 2}
    
    service = SimpleService()
    
    # Create API handler
    handler = InterfaceFactory.create_handler(
        service=service,
        validate_requests=True,
        response_format="json"
    )
    
    # Handle request
    response = handler.handle({"value": 5})
    print(f"API Response: {response}")


# ============================================================================
# Example 7: Dependency Injection
# ============================================================================

def example_dependency_injection():
    """Example of dependency injection"""
    print("\n=== Example 7: Dependency Injection ===")
    
    # Create container
    container = DependencyContainer()
    
    # Register dependencies
    model = nn.Linear(10, 1)
    container.register_singleton("model", model)
    
    container.register_factory("processor", lambda: NormalizationProcessor())
    
    # Use injection decorator
    @inject_dependencies
    def predict(model: nn.Module, processor: NormalizationProcessor, data: torch.Tensor):
        processed = processor.process(data)
        return model(processed)
    
    # Call function (dependencies injected automatically)
    result = predict(torch.randn(1, 10))
    print(f"Prediction result: {result.shape}")


# ============================================================================
# Example 8: Complete Workflow
# ============================================================================

def example_complete_workflow():
    """Example of complete modular workflow"""
    print("\n=== Example 8: Complete Workflow ===")
    
    # 1. Data Layer
    pipeline = DataPipeline()
    pipeline.add_processor(NormalizationProcessor())
    
    # 2. Model Layer
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    ModelRegistry.register_model("SimpleModel", SimpleModel)
    model = ModelBuilder().build("SimpleModel")
    
    # 3. Inference Layer
    engine = InferenceEngine(model)
    inference_pipeline = InferencePipeline(engine)
    
    # 4. Service Layer
    class PredictionService:
        def __init__(self, engine):
            self.engine = engine
        
        def execute(self, request):
            data = torch.tensor(request["data"])
            result = self.engine.process(data)
            return {"prediction": result.item()}
    
    container = ServiceContainer()
    container.register("PredictionService", PredictionService, singleton=True, engine=engine)
    
    # 5. Interface Layer
    service = container.get("PredictionService")
    handler = InterfaceFactory.create_handler(service)
    
    # 6. Use complete workflow
    response = handler.handle({"data": [1.0] * 10})
    print(f"Complete workflow result: {response}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Ultra-Modular Layered Architecture V3 - Examples")
    print("=" * 60)
    
    try:
        example_data_pipeline()
        example_model_building()
        example_training_pipeline()
        example_inference_engine()
        example_service_layer()
        example_interface_layer()
        example_dependency_injection()
        example_complete_workflow()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()



