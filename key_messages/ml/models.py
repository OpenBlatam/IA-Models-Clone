from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
from transformers import (
from typing import Dict, Any, Optional, Union, List
import structlog
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
    from .config import get_model_config, ConfigManager
from ml.models import create_model, ModelConfig
from ml.models import create_model
from ml.models import create_model
from ml.models import create_ensemble
from ml.models import ModelConfig, CustomTransformerModel
from ml.models import create_model
from ml.models import create_model
from ml.config import get_config
from ml.models import ModelFactory, ModelConfig
from ml.models import ModelFactory, BaseModel
from ml.models import create_model
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Models Module for Key Messages ML Pipeline
Updated to integrate with YAML configuration system
"""

    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    BertForSequenceClassification, BertTokenizer, BertConfig,
    AutoModelForCausalLM, AutoTokenizer
)

# Import configuration system
try:
except ImportError:
    # Fallback for when config module is not available
    def get_model_config(model_name: str, environment: str = None) -> Dict[str, Any]:
        return {}
    
    class ConfigManager:
        def resolve_device(self, device: str) -> str:
            return "cuda" if torch.cuda.is_available() else "cpu"
        
        def resolve_torch_dtype(self, dtype: str) -> torch.dtype:
            match dtype:
    case "auto":
                return torch.float16 if torch.cuda.is_available() else torch.float32
            elif dtype == "float16":
                return torch.float16
            elif dtype == "float32":
                return torch.float32
            else:
                return torch.float32

logger = structlog.get_logger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model initialization and inference."""
    model_name: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    device: str = "auto"
    torch_dtype: str = "auto"
    num_labels: Optional[int] = None
    labels: Optional[List[str]] = None
    vocab_size: Optional[int] = None
    d_model: Optional[int] = None
    nhead: Optional[int] = None
    num_layers: Optional[int] = None
    dim_feedforward: Optional[int] = None
    dropout: Optional[float] = None
    
    def __post_init__(self) -> Any:
        """Post-initialization processing."""
        # Resolve device and dtype
        config_manager = ConfigManager()
        self.device = config_manager.resolve_device(self.device)
        self.torch_dtype = config_manager.resolve_torch_dtype(self.torch_dtype)
        
        # Set default token IDs if not provided
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id

class BaseModel(nn.Module):
    """Base class for all models in the pipeline."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.device = config.device
        self.torch_dtype = config.torch_dtype
        
        logger.info("Initializing base model", 
                   model_name=config.model_name,
                   device=self.device,
                   dtype=str(self.torch_dtype))
    
    def to_device(self) -> Any:
        """Move model to specified device."""
        self.to(self.device, dtype=self.torch_dtype)
        logger.info("Model moved to device", device=self.device)
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), f"{path}_state.pt")
        
        # Save configuration
        config_dict = {
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'do_sample': self.config.do_sample,
            'pad_token_id': self.config.pad_token_id,
            'eos_token_id': self.config.eos_token_id,
            'device': self.config.device,
            'torch_dtype': str(self.config.torch_dtype),
            'num_labels': self.config.num_labels,
            'labels': self.config.labels,
            'vocab_size': self.config.vocab_size,
            'd_model': self.config.d_model,
            'nhead': self.config.nhead,
            'num_layers': self.config.num_layers,
            'dim_feedforward': self.config.dim_feedforward,
            'dropout': self.config.dropout
        }
        
        with open(f"{path}_config.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config_dict, f, indent=2)
        
        logger.info("Model saved", path=path)
    
    @classmethod
    def load(cls, path: str, config: Optional[ModelConfig] = None):
        """Load model from disk."""
        # Load configuration
        config_path = f"{path}_config.json"
        if os.path.exists(config_path) and config is None:
            with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        
        if config is None:
            raise ValueError("ModelConfig must be provided if config file doesn't exist")
        
        # Create model instance
        model = cls(config)
        
        # Load state
        state_path = f"{path}_state.pt"
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, map_location=config.device)
            model.load_state_dict(state_dict)
            logger.info("Model state loaded", path=state_path)
        
        return model

class GPT2MessageModel(BaseModel):
    """GPT-2 model for message generation."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(
            config.model_name,
            torch_dtype=self.torch_dtype
        )
        
        # Set generation parameters
        self.generation_config = {
            'max_length': config.max_length,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'do_sample': config.do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        self.to_device()
        
        logger.info("GPT2MessageModel initialized", 
                   model_name=config.model_name,
                   vocab_size=self.model.config.vocab_size)
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate text from prompt."""
        # Guard clauses for early validation
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if len(prompt) > 10000:
            raise ValueError("Prompt too long (max 10000 characters)")
        
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("Max new tokens must be positive")
        
        if max_new_tokens is not None and max_new_tokens > 2000:
            raise ValueError("Max new tokens too large (max 2000)")
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Set max_new_tokens if provided
            generation_config = self.generation_config.copy()
            if max_new_tokens is not None:
                generation_config['max_new_tokens'] = max_new_tokens
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **generation_config
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove original prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error("Error generating text", error=str(e), prompt_length=len(prompt))
            raise
    
    def generate_batch(self, prompts: List[str], max_new_tokens: Optional[int] = None) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_new_tokens)
            results.append(result)
        return results

class BERTClassifierModel(BaseModel):
    """BERT model for text classification."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels or 2,
            torch_dtype=self.torch_dtype
        )
        
        self.labels = config.labels or [f"label_{i}" for i in range(config.num_labels or 2)]
        
        self.to_device()
        
        logger.info("BERTClassifierModel initialized", 
                   model_name=config.model_name,
                   num_labels=config.num_labels)
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify a single text."""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
        
        # Get results
        predicted_label = self.labels[predicted_class.item()]
        confidence = probabilities[0][predicted_class].item()
        
        result = {
            'text': text,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                label: prob.item() for label, prob in zip(self.labels, probabilities[0])
            }
        }
        
        logger.debug("Text classified", 
                    text_length=len(text),
                    predicted_label=predicted_label,
                    confidence=confidence)
        
        return result
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts."""
        results = []
        for text in texts:
            result = self.classify(text)
            results.append(result)
        return results

class CustomTransformerModel(BaseModel):
    """Custom transformer model for specific tasks."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Validate required parameters
        required_params = ['vocab_size', 'd_model', 'nhead', 'num_layers', 'dim_feedforward']
        for param in required_params:
            if getattr(config, param) is None:
                raise ValueError(f"CustomTransformerModel requires {param} to be set")
        
        # Create custom configuration
        model_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_length,
            n_embd=config.d_model,
            n_head=config.nhead,
            n_layer=config.num_layers,
            n_inner=config.dim_feedforward,
            resid_pdrop=config.dropout or 0.1,
            attn_pdrop=config.dropout or 0.1,
            embd_pdrop=config.dropout or 0.1
        )
        
        # Create model
        self.model = GPT2LMHeadModel(model_config)
        
        # Create tokenizer (using GPT-2 tokenizer as base)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set generation parameters
        self.generation_config = {
            'max_length': config.max_length,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'do_sample': config.do_sample,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        self.to_device()
        
        logger.info("CustomTransformerModel initialized", 
                   vocab_size=config.vocab_size,
                   d_model=config.d_model,
                   nhead=config.nhead,
                   num_layers=config.num_layers)
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate text from prompt."""
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Set max_new_tokens if provided
        generation_config = self.generation_config.copy()
        if max_new_tokens is not None:
            generation_config['max_new_tokens'] = max_new_tokens
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                **generation_config
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        logger.debug("Text generated", 
                    prompt_length=len(prompt),
                    generated_length=len(generated_text),
                    max_new_tokens=max_new_tokens)
        
        return generated_text

class ModelFactory:
    """Factory for creating model instances."""
    
    _models = {
        'gpt2': GPT2MessageModel,
        'gpt2-medium': GPT2MessageModel,
        'gpt2-large': GPT2MessageModel,
        'bert-base-uncased': BERTClassifierModel,
        'bert-large-uncased': BERTClassifierModel,
        'custom': CustomTransformerModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: ModelConfig) -> BaseModel:
        """Create a model instance based on type and configuration."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        return model_class(config)
    
    @classmethod
    def create_model_from_config(cls, model_name: str, environment: str = None) -> BaseModel:
        """Create a model instance from configuration file."""
        # Load model configuration
        model_config_dict = get_model_config(model_name, environment)
        
        if not model_config_dict:
            raise ValueError(f"No configuration found for model: {model_name}")
        
        # Create ModelConfig object
        config = ModelConfig(**model_config_dict)
        
        # Create model
        return cls.create_model(model_name, config)
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a new model type."""
        cls._models[name] = model_class
        logger.info("Model registered", name=name, class_name=model_class.__name__)

class ModelEnsemble:
    """Ensemble of multiple models for improved performance."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        
    """__init__ function."""
self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.models) != len(self.weights):
            raise ValueError("Number of models must match number of weights")
        
        logger.info("ModelEnsemble initialized", 
                   num_models=len(self.models),
                   weights=self.weights)
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate text using ensemble of models."""
        if not hasattr(self.models[0], 'generate'):
            raise ValueError("Ensemble models must support generation")
        
        # Generate from all models
        generations = []
        for model, weight in zip(self.models, self.weights):
            generation = model.generate(prompt, max_new_tokens)
            generations.append((generation, weight))
        
        # Simple weighted combination (for more sophisticated methods, extend this)
        # For now, return the generation from the highest weighted model
        best_generation = max(generations, key=lambda x: x[1])[0]
        
        logger.debug("Ensemble generation completed", 
                    num_models=len(self.models),
                    prompt_length=len(prompt))
        
        return best_generation
    
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text using ensemble of models."""
        if not hasattr(self.models[0], 'classify'):
            raise ValueError("Ensemble models must support classification")
        
        # Classify with all models
        classifications = []
        for model, weight in zip(self.models, self.weights):
            classification = model.classify(text)
            classifications.append((classification, weight))
        
        # Combine predictions (simple weighted voting)
        label_scores = {}
        for classification, weight in classifications:
            predicted_label = classification['predicted_label']
            confidence = classification['confidence']
            
            if predicted_label not in label_scores:
                label_scores[predicted_label] = 0.0
            
            label_scores[predicted_label] += weight * confidence
        
        # Get best label
        best_label = max(label_scores.items(), key=lambda x: x[1])
        
        result = {
            'text': text,
            'predicted_label': best_label[0],
            'confidence': best_label[1],
            'ensemble_scores': label_scores,
            'individual_predictions': classifications
        }
        
        logger.debug("Ensemble classification completed", 
                    num_models=len(self.models),
                    text_length=len(text),
                    predicted_label=best_label[0])
        
        return result

# Convenience functions for easy model creation
def create_model(model_name: str, environment: str = None) -> BaseModel:
    """Create a model instance from configuration."""
    return ModelFactory.create_model_from_config(model_name, environment)

def create_ensemble(model_names: List[str], weights: Optional[List[float]] = None, 
                   environment: str = None) -> ModelEnsemble:
    """Create an ensemble of models from configuration."""
    models = []
    for model_name in model_names:
        model = create_model(model_name, environment)
        models.append(model)
    
    return ModelEnsemble(models, weights)

# Example usage with configuration integration
def example_usage():
    """Example usage of the models with configuration system."""
    print("""
# Models Module Usage Examples

## 1. Basic Model Creation
```python

# Create model with custom configuration
config = ModelConfig(
    model_name="gpt2",
    max_length=512,
    temperature=0.7,
    device="auto"
)
model = GPT2MessageModel(config)

# Generate text
text = model.generate("Generate a key message about:", max_new_tokens=50)
print(text)
```

## 2. Model Creation from Configuration
```python

# Create model from YAML configuration
model = create_model("gpt2", environment="production")

# Generate text
text = model.generate("Create a compelling message:", max_new_tokens=100)
print(text)
```

## 3. Classification Model
```python

# Create BERT classifier from configuration
classifier = create_model("bert-base-uncased", environment="production")

# Classify text
result = classifier.classify("This is a positive message about our product")
print(f"Label: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 4. Model Ensemble
```python

# Create ensemble from configuration
ensemble = create_ensemble(
    model_names=["gpt2", "gpt2-medium"],
    weights=[0.6, 0.4],
    environment="production"
)

# Generate text with ensemble
text = ensemble.generate("Generate a key message:", max_new_tokens=50)
print(text)
```

## 5. Custom Model
```python

# Create custom model configuration
config = ModelConfig(
    model_name="custom",
    max_length=256,
    temperature=0.8,
    vocab_size=50257,
    d_model=768,
    nhead=12,
    num_layers=12,
    dim_feedforward=3072,
    dropout=0.1,
    device="auto"
)

# Create custom model
model = CustomTransformerModel(config)

# Generate text
text = model.generate("Custom model generation:", max_new_tokens=30)
print(text)
```

## 6. Model Saving and Loading
```python

# Create and save model
model = create_model("gpt2", environment="production")
model.save("./models/gpt2_key_messages")

# Load model later
loaded_model = GPT2MessageModel.load("./models/gpt2_key_messages")
text = loaded_model.generate("Load and generate:", max_new_tokens=20)
print(text)
```

## 7. Batch Processing
```python

# Create model
model = create_model("gpt2", environment="production")

# Batch generation
prompts = [
    "Generate a message about innovation:",
    "Create a message about teamwork:",
    "Write a message about success:"
]

results = model.generate_batch(prompts, max_new_tokens=30)
for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    print()
```

## 8. Integration with Configuration System
```python

# Load configuration
config = get_config("production")

# Get model configuration
model_config_dict = config["models"]["gpt2"]
model_config = ModelConfig(**model_config_dict)

# Create model
model = ModelFactory.create_model("gpt2", model_config)

# Use model
text = model.generate("Configuration-driven generation:", max_new_tokens=25)
print(text)
```

## 9. Model Registration
```python

# Define custom model
class CustomMessageModel(BaseModel):
    def __init__(self, config) -> Any:
        super().__init__(config)
        # Custom implementation
    
    def generate(self, prompt, max_new_tokens=None) -> Any:
        # Custom generation logic
        return "Custom generated text"

# Register custom model
ModelFactory.register_model("custom_message", CustomMessageModel)

# Use custom model
model = ModelFactory.create_model("custom_message", config)
```

## 10. Error Handling
```python

try:
    # Try to create model with invalid configuration
    model = create_model("nonexistent_model", environment="production")
except ValueError as e:
    print(f"Error creating model: {e}")

try:
    # Try to generate with invalid input
    model = create_model("gpt2", environment="production")
    text = model.generate("", max_new_tokens=-1)
except Exception as e:
    print(f"Error during generation: {e}")
```
""")

match __name__:
    case "__main__":
    example_usage() 