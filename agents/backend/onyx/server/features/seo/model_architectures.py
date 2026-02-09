from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Object-Oriented Model Architectures for SEO Service
Clean class design with inheritance and composition patterns
"""


logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architectures"""
    model_name: str
    num_classes: int = 2
    dropout_rate: float = 0.1
    hidden_size: Optional[int] = None
    max_length: int = 512
    use_pooler: bool = True
    freeze_backbone: bool = False
    custom_head: bool = False

class BaseModel(ABC, nn.Module):
    """Abstract base class for all SEO models"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass implementation"""
        pass
    
    @abstractmethod
    def get_embeddings(self, *args, **kwargs) -> torch.Tensor:
        """Extract embeddings from the model"""
        pass
    
    def to_device(self, device: torch.device) -> 'BaseModel':
        """Move model to specified device"""
        self.device = device
        return self.to(device)
    
    def freeze_layers(self, layer_names: List[str]) -> None:
        """Freeze specific layers by name"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """Unfreeze specific layers by name"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")

class TransformerBackbone(nn.Module):
    """Reusable transformer backbone component"""
    
    def __init__(self, model_name: str, freeze_backbone: bool = False):
        
    """__init__ function."""
super().__init__()
        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self) -> None:
        """Freeze the transformer backbone"""
        for param in self.transformer.parameters():
            param.requires_grad = False
        logger.info(f"Frozen transformer backbone: {self.model_name}")
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze the transformer backbone"""
        for param in self.transformer.parameters():
            param.requires_grad = True
        logger.info(f"Unfrozen transformer backbone: {self.model_name}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer"""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

class ClassificationHead(nn.Module):
    """Reusable classification head component"""
    
    def __init__(self, input_size: int, num_classes: int, dropout_rate: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(input_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head"""
        x = self.dropout(x)
        return self.classifier(x)

class SEOTextClassifier(BaseModel):
    """SEO text classification model using transformer architecture"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Initialize backbone
        self.backbone = TransformerBackbone(
            config.model_name, 
            freeze_backbone=config.freeze_backbone
        )
        
        # Determine input size for classification head
        if config.hidden_size is None:
            input_size = self.backbone.config.hidden_size
        else:
            input_size = config.hidden_size
        
        # Initialize classification head
        self.classifier = ClassificationHead(
            input_size=input_size,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate
        )
        
        # Optional pooling layer
        self.use_pooler = config.use_pooler
        if config.use_pooler:
            self.pooler = nn.Linear(input_size, input_size)
            self.pooler_activation = nn.Tanh()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete model"""
        # Get transformer outputs
        transformer_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract pooled representation
        if self.use_pooler:
            pooled_output = transformer_outputs.pooler_output
            pooled_output = self.pooler_activation(self.pooler(pooled_output))
        else:
            # Use mean pooling over sequence length
            pooled_output = self._mean_pooling(transformer_outputs.last_hidden_state, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from the model"""
        transformer_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.use_pooler:
            return transformer_outputs.pooler_output
        else:
            return self._mean_pooling(transformer_outputs.last_hidden_state, attention_mask)
    
    def _mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over sequence length"""
        # Mask padded tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class SEOSentimentAnalyzer(BaseModel):
    """SEO sentiment analysis model with multi-label support"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Initialize backbone
        self.backbone = TransformerBackbone(
            config.model_name,
            freeze_backbone=config.freeze_backbone
        )
        
        # Multi-label classification head
        input_size = self.backbone.config.hidden_size
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_size // 2, config.num_classes),
            nn.Sigmoid()  # Multi-label output
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for sentiment analysis"""
        transformer_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = transformer_outputs.pooler_output
        return self.sentiment_classifier(pooled_output)
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract embeddings"""
        transformer_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return transformer_outputs.pooler_output

class SEOKeywordExtractor(BaseModel):
    """SEO keyword extraction model using sequence labeling"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Initialize backbone
        self.backbone = TransformerBackbone(
            config.model_name,
            freeze_backbone=config.freeze_backbone
        )
        
        # Sequence labeling head
        input_size = self.backbone.config.hidden_size
        self.keyword_classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_size // 2, config.num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for keyword extraction"""
        transformer_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = transformer_outputs.last_hidden_state
        return self.keyword_classifier(sequence_output)
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract token-level embeddings"""
        transformer_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return transformer_outputs.last_hidden_state

class SEOMultiTaskModel(BaseModel):
    """Multi-task SEO model for classification, sentiment, and keyword extraction"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Shared backbone
        self.backbone = TransformerBackbone(
            config.model_name,
            freeze_backbone=config.freeze_backbone
        )
        
        input_size = self.backbone.config.hidden_size
        
        # Task-specific heads
        self.classification_head = ClassificationHead(
            input_size, config.num_classes, config.dropout_rate
        )
        
        self.sentiment_head = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 3),  # Positive, Negative, Neutral
            nn.Softmax(dim=-1)
        )
        
        self.keyword_head = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, 2)  # Keyword or not
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task learning"""
        transformer_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Classification task (using pooled output)
        classification_logits = self.classification_head(transformer_outputs.pooler_output)
        
        # Sentiment task (using pooled output)
        sentiment_probs = self.sentiment_head(transformer_outputs.pooler_output)
        
        # Keyword extraction task (using sequence output)
        keyword_logits = self.keyword_head(transformer_outputs.last_hidden_state)
        
        return {
            'classification': classification_logits,
            'sentiment': sentiment_probs,
            'keywords': keyword_logits
        }
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract embeddings"""
        transformer_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return transformer_outputs.pooler_output

class ModelFactory:
    """Factory class for creating model instances"""
    
    @staticmethod
    def create_model(model_type: str, config: ModelConfig) -> BaseModel:
        """Create model instance based on type"""
        model_map = {
            'classifier': SEOTextClassifier,
            'sentiment': SEOSentimentAnalyzer,
            'keywords': SEOKeywordExtractor,
            'multitask': SEOMultiTaskModel
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_map[model_type]
        return model_class(config)
    
    @staticmethod
    def create_config(
        model_name: str,
        model_type: str,
        num_classes: int = 2,
        **kwargs
    ) -> ModelConfig:
        """Create configuration for specific model type"""
        config_kwargs = {
            'model_name': model_name,
            'num_classes': num_classes,
            **kwargs
        }
        
        # Model-specific configurations
        if model_type == 'sentiment':
            config_kwargs['num_classes'] = 3  # Positive, Negative, Neutral
        elif model_type == 'keywords':
            config_kwargs['num_classes'] = 2  # Keyword or not
        elif model_type == 'multitask':
            config_kwargs['num_classes'] = 2  # Default classification classes
        
        return ModelConfig(**config_kwargs)

class ModelManager:
    """Manager class for model lifecycle operations"""
    
    def __init__(self) -> Any:
        self.models: Dict[str, BaseModel] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
    
    def register_model(self, name: str, model: BaseModel, tokenizer: AutoTokenizer) -> None:
        """Register a model and its tokenizer"""
        self.models[name] = model
        self.tokenizers[name] = tokenizer
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get registered model by name"""
        return self.models.get(name)
    
    def get_tokenizer(self, name: str) -> Optional[AutoTokenizer]:
        """Get registered tokenizer by name"""
        return self.tokenizers.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names"""
        return list(self.models.keys())
    
    def remove_model(self, name: str) -> bool:
        """Remove model from registry"""
        if name in self.models:
            del self.models[name]
            del self.tokenizers[name]
            logger.info(f"Removed model: {name}")
            return True
        return False
    
    def save_model(self, name: str, path: str) -> bool:
        """Save model and tokenizer"""
        if name not in self.models:
            logger.error(f"Model {name} not found")
            return False
        
        try:
            model = self.models[name]
            tokenizer = self.tokenizers[name]
            
            # Save model
            torch.save(model.state_dict(), f"{path}/model.pth")
            
            # Save tokenizer
            tokenizer.save_pretrained(path)
            
            # Save config
            config = {
                'model_type': model.__class__.__name__,
                'config': model.config.__dict__,
                'name': name
            }
            torch.save(config, f"{path}/config.pth")
            
            logger.info(f"Saved model {name} to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {name}: {e}")
            return False
    
    def load_model(self, name: str, path: str) -> bool:
        """Load model and tokenizer"""
        try:
            # Load config
            config = torch.load(f"{path}/config.pth")
            model_config = ModelConfig(**config['config'])
            
            # Create model
            model = ModelFactory.create_model(config['model_type'], model_config)
            
            # Load weights
            model.load_state_dict(torch.load(f"{path}/model.pth"))
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
            
            # Register
            self.register_model(name, model, tokenizer)
            
            logger.info(f"Loaded model {name} from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            return False

# Utility functions for model operations
def create_model_with_config(model_type: str, model_name: str, **kwargs) -> BaseModel:
    """Create model with configuration"""
    config = ModelFactory.create_config(model_name, model_type, **kwargs)
    return ModelFactory.create_model(model_type, config)

def get_model_summary(model: BaseModel) -> Dict[str, Any]:
    """Get model summary information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_type': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'device': str(model.device),
        'config': model.config.__dict__
    }

# Example usage
if __name__ == "__main__":
    # Create model manager
    manager = ModelManager()
    
    # Create and register different model types
    classifier = create_model_with_config('classifier', 'bert-base-uncased', num_classes=3)
    sentiment = create_model_with_config('sentiment', 'distilbert-base-uncased')
    
    # Register models
    manager.register_model('seo_classifier', classifier, AutoTokenizer.from_pretrained('bert-base-uncased'))
    manager.register_model('sentiment_analyzer', sentiment, AutoTokenizer.from_pretrained('distilbert-base-uncased'))
    
    # Print model summaries
    for name in manager.list_models():
        model = manager.get_model(name)
        summary = get_model_summary(model)
        print(f"\n{name}:")
        for key, value in summary.items():
            print(f"  {key}: {value}") 