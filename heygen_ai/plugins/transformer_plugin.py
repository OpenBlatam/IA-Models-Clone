#!/usr/bin/env python3
"""
Transformer Plugin for HeyGen AI

This plugin provides transformer model capabilities including:
- GPT-2, BERT, T5, RoBERTa models
- Text generation and classification
- Natural language processing tasks
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

try:
    from core.plugin_system import BaseModelPlugin, PluginMetadata
    PLUGIN_SYSTEM_AVAILABLE = True
except ImportError:
    PLUGIN_SYSTEM_AVAILABLE = False
    print("Warning: Plugin system not available")


class TransformerPlugin(BaseModelPlugin):
    """Plugin for transformer models."""
    
    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="transformer_plugin",
            version="1.0.0",
            description="Advanced transformer model plugin for NLP tasks",
            author="HeyGen AI Team",
            plugin_type="model",
            priority="high",
            tags=["transformer", "nlp", "text_generation", "classification"],
            dependencies=["torch", "transformers"],
            requirements={
                "torch": ">=2.0.0",
                "transformers": ">=4.20.0"
            }
        )
    
    def _initialize_impl(self) -> None:
        """Initialize the transformer plugin."""
        self.logger.info("Transformer plugin initialized")
        
        # Available model types
        self.model_types = {
            "gpt2": {
                "name": "GPT-2",
                "parameters": 125000000,
                "max_length": 1024,
                "tasks": ["text_generation", "completion"]
            },
            "bert": {
                "name": "BERT",
                "parameters": 110000000,
                "max_length": 512,
                "tasks": ["classification", "qa", "ner"]
            },
            "t5": {
                "name": "T5",
                "parameters": 220000000,
                "max_length": 512,
                "tasks": ["translation", "summarization", "qa"]
            },
            "roberta": {
                "name": "RoBERTa",
                "parameters": 125000000,
                "max_length": 512,
                "tasks": ["classification", "qa", "ner"]
            }
        }
        
        # Initialize model cache
        self.model_cache = {}
        self.active_models = {}
    
    def _load_model_impl(self, model_config: Dict[str, Any]) -> Any:
        """Load a transformer model."""
        model_type = model_config.get("model_type", "gpt2")
        
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported model type: {model_type}. Available: {list(self.model_types.keys())}")
        
        # Check if model is already loaded
        if model_type in self.active_models:
            self.logger.info(f"Model {model_type} already loaded")
            return self.active_models[model_type]
        
        # Simulate model loading (in practice, this would load actual models)
        model_info = self.model_types[model_type].copy()
        model_info.update({
            "loaded_at": time.time(),
            "device": model_config.get("device", "cpu"),
            "precision": model_config.get("precision", "fp32"),
            "optimization_level": model_config.get("optimization_level", "none")
        })
        
        # Store in active models
        self.active_models[model_type] = model_info
        
        self.logger.info(f"Loaded {model_type} model with {model_info['parameters']:,} parameters")
        return model_info
    
    def _get_model_info_impl(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        if not self.active_models:
            return {"error": "No models loaded"}
        
        return {
            "loaded_models": list(self.active_models.keys()),
            "total_parameters": sum(model["parameters"] for model in self.active_models.values()),
            "model_details": self.active_models,
            "available_types": list(self.model_types.keys()),
            "cache_size": len(self.model_cache)
        }
    
    def _unload_model_impl(self) -> None:
        """Unload all models."""
        for model_type in list(self.active_models.keys()):
            self.logger.info(f"Unloading {model_type} model")
            del self.active_models[model_type]
    
    def _get_model_info_impl(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.active_models:
            return {"error": "No models loaded"}
        
        # Return info about all loaded models
        model_info = {}
        for model_type, model_data in self.active_models.items():
            model_info[model_type] = {
                "type": model_data.get("name", model_type),
                "parameters": model_data.get("parameters", 0),
                "max_length": model_data.get("max_length", 0),
                "tasks": model_data.get("tasks", []),
                "loaded_at": model_data.get("loaded_at", 0),
                "device": model_data.get("device", "cpu"),
                "precision": model_data.get("precision", "fp32"),
                "optimization_level": model_data.get("optimization_level", "none")
            }
        
        return model_info
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities."""
        return [
            "transformer_models",
            "text_generation",
            "text_classification",
            "question_answering",
            "named_entity_recognition",
            "translation",
            "summarization"
        ]
    
    def generate_text(self, model_type: str, prompt: str, max_length: int = 100) -> str:
        """Generate text using a loaded model."""
        if model_type not in self.active_models:
            raise ValueError(f"Model {model_type} not loaded. Please load it first.")
        
        # Simulate text generation
        model_info = self.active_models[model_type]
        self.logger.info(f"Generating text with {model_type} (max_length: {max_length})")
        
        # In practice, this would use the actual model
        generated_text = f"[{model_type.upper()}] Generated text based on: '{prompt[:50]}...'"
        
        return generated_text
    
    def classify_text(self, model_type: str, text: str, labels: List[str]) -> Dict[str, float]:
        """Classify text using a loaded model."""
        if model_type not in self.active_models:
            raise ValueError(f"Model {model_type} not loaded. Please load it first.")
        
        # Simulate classification
        self.logger.info(f"Classifying text with {model_type}")
        
        # In practice, this would use the actual model
        scores = {label: 0.1 + (hash(text + label) % 80) / 100.0 for label in labels}
        
        # Normalize scores
        total = sum(scores.values())
        scores = {label: score / total for label, score in scores.items()}
        
        return scores
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "total_models": len(self.model_types),
            "loaded_models": len(self.active_models),
            "total_parameters": sum(model["parameters"] for model in self.active_models.values()),
            "memory_usage": len(self.active_models) * 100,  # Simulated MB
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }


# Example usage
if __name__ == "__main__":
    if not PLUGIN_SYSTEM_AVAILABLE:
        print("❌ Plugin system not available")
        sys.exit(1)
    
    # Create and test plugin
    plugin = TransformerPlugin()
    
    # Initialize
    if plugin.initialize({}):
        print("✅ Plugin initialized successfully")
        
        # Load a model
        model = plugin.load_model({"model_type": "gpt2", "device": "cpu"})
        print(f"✅ Model loaded: {model}")
        
        # Get capabilities
        capabilities = plugin.get_capabilities()
        print(f"📊 Capabilities: {capabilities}")
        
        # Generate text
        text = plugin.generate_text("gpt2", "Hello, how are you?", max_length=50)
        print(f"📝 Generated text: {text}")
        
        # Classify text
        labels = ["positive", "negative", "neutral"]
        classification = plugin.classify_text("gpt2", "I love this!", labels)
        print(f"🏷️ Classification: {classification}")
        
        # Get stats
        stats = plugin.get_model_stats()
        print(f"📈 Stats: {stats}")
        
        # Cleanup
        plugin.cleanup()
        print("🧹 Plugin cleaned up")
    else:
        print("❌ Failed to initialize plugin")
