"""
Example: Improved Integration with Adapters and Utilities
Shows how to integrate existing models with new layered architecture
"""

import torch
import torch.nn as nn

# Import layered architecture with adapters
from addiction_recovery_ai.core.layers import (
    # Core layers
    ModelBuilder,
    InferenceEngine,
    DataPipeline,
    # Adapters
    IntegrationHelper,
    ModelAdapter,
    PredictorAdapter,
    # Integration
    WorkflowBuilder,
    create_sentiment_workflow,
    # Utilities
    quick_model,
    quick_inference_engine,
    get_optimal_device,
)


# ============================================================================
# Example 1: Register Existing Models
# ============================================================================

def example_register_existing_models():
    """Register existing models with new architecture"""
    print("\n=== Example 1: Register Existing Models ===")
    
    # Define a simple existing model
    class ExistingModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=64):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Register with new architecture
    ModelAdapter.register_existing_model(
        ExistingModel,
        "ExistingModel",
        default_config={"input_size": 10, "hidden_size": 64}
    )
    
    # Now can use with ModelBuilder
    model = ModelBuilder().with_config(input_size=10).build("ExistingModel")
    print(f"Model created: {type(model).__name__}")


# ============================================================================
# Example 2: Create Predictor from Existing Model
# ============================================================================

def example_predictor_adapter():
    """Create predictor from existing model"""
    print("\n=== Example 2: Predictor Adapter ===")
    
    # Create existing model
    model = nn.Linear(10, 1)
    
    # Create predictor with preprocessing
    def preprocess(x):
        return torch.tensor(x) if not isinstance(x, torch.Tensor) else x
    
    def postprocess(x):
        return x.item() if x.numel() == 1 else x
    
    predictor = PredictorAdapter.create_from_model(
        model,
        preprocess_fn=preprocess,
        postprocess_fn=postprocess
    )
    
    # Use predictor
    result = predictor.predict([1.0] * 10)
    print(f"Prediction: {result}")


# ============================================================================
# Example 3: Quick Utilities
# ============================================================================

def example_quick_utilities():
    """Use quick utility functions"""
    print("\n=== Example 3: Quick Utilities ===")
    
    # Quick model creation
    device = get_optimal_device()
    print(f"Optimal device: {device}")
    
    # Register a model first
    class QuickModel(nn.Module):
        def __init__(self, size=10):
            super().__init__()
            self.linear = nn.Linear(size, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    from addiction_recovery_ai.core.layers import ModelRegistry
    ModelRegistry.register_model("QuickModel", QuickModel)
    
    # Quick model
    model = quick_model("QuickModel", {"size": 10}, device)
    print(f"Quick model created: {type(model).__name__}")
    
    # Quick inference engine
    engine = quick_inference_engine(model, device)
    print(f"Inference engine created: {type(engine).__name__}")


# ============================================================================
# Example 4: Complete Workflow with Integration
# ============================================================================

def example_complete_workflow():
    """Complete workflow with all layers integrated"""
    print("\n=== Example 4: Complete Workflow ===")
    
    # Register models
    IntegrationHelper.register_all_models()
    
    # Create workflow using builder
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    ModelRegistry.register_model("SimpleModel", SimpleModel)
    
    workflow = (WorkflowBuilder("CompleteWorkflow")
        .with_model("SimpleModel", {}, get_optimal_device())
        .with_inference(use_mixed_precision=True)
        .build())
    
    # Use workflow
    input_data = torch.randn(1, 10)
    prediction = workflow.predict(input_data)
    print(f"Workflow prediction shape: {prediction.shape}")


# ============================================================================
# Example 5: Sentiment Analysis Workflow
# ============================================================================

def example_sentiment_workflow():
    """Quick sentiment analysis workflow"""
    print("\n=== Example 5: Sentiment Workflow ===")
    
    try:
        # Register models first
        IntegrationHelper.register_all_models()
        
        # Create workflow
        workflow = create_sentiment_workflow()
        
        print(f"Workflow created: {workflow.name}")
        print(f"Model available: {workflow.model is not None}")
        print(f"Inference pipeline available: {workflow.inference_pipeline is not None}")
    except Exception as e:
        print(f"Note: Sentiment workflow requires transformers: {e}")


# ============================================================================
# Example 6: Service Integration
# ============================================================================

def example_service_integration():
    """Integrate services with workflow"""
    print("\n=== Example 6: Service Integration ===")
    
    # Create model
    model = nn.Linear(10, 1)
    
    # Create workflow with service
    class PredictionService:
        def __init__(self, engine):
            self.engine = engine
        
        def execute(self, request):
            data = torch.tensor(request["data"])
            result = self.engine.process(data)
            return {"prediction": result.item()}
    
    workflow = (WorkflowBuilder("ServiceWorkflow")
        .with_model("SimpleModel", {}, get_optimal_device())
        .with_inference()
        .with_service("PredictionService", PredictionService)
        .build())
    
    # Get service
    service = workflow.get_service("PredictionService")
    result = service.execute({"data": [1.0] * 10})
    print(f"Service result: {result}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Improved Integration Examples")
    print("=" * 60)
    
    try:
        example_register_existing_models()
        example_predictor_adapter()
        example_quick_utilities()
        example_complete_workflow()
        example_sentiment_workflow()
        example_service_integration()
        
        print("\n" + "=" * 60)
        print("All integration examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()



