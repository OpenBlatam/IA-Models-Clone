"""
Advanced Transformer Integration
Enhanced transformer models with attention visualization and fine-tuning utilities
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoFeatureExtractor,
        Wav2Vec2Model,
        Wav2Vec2Processor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")


class AttentionVisualizer:
    """
    Visualize attention patterns in transformer models
    """
    
    @staticmethod
    def extract_attention_weights(model, input_ids, layer_idx: Optional[int] = None):
        """Extract attention weights from transformer model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required for attention visualization")
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            attentions = outputs.attentions
        
        if layer_idx is not None:
            return attentions[layer_idx]
        return attentions
    
    @staticmethod
    def visualize_attention_pattern(
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Visualize attention pattern"""
        # Average over heads
        attention_avg = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Get attention scores
        attention_scores = attention_avg[0].cpu().numpy()
        
        return {
            "attention_matrix": attention_scores.tolist(),
            "shape": list(attention_scores.shape),
            "max_attention": float(attention_scores.max()),
            "min_attention": float(attention_scores.min()),
            "mean_attention": float(attention_scores.mean())
        }


class TransformerFineTuner:
    """
    Advanced fine-tuning utilities for transformer models
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "classification",
        num_labels: int = 10
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required")
        
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        
        # Load model and tokenizer
        try:
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            # Try audio model
            try:
                self.model = Wav2Vec2Model.from_pretrained(model_name)
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.tokenizer = None
            except Exception as e:
                logger.error(f"Could not load model {model_name}: {str(e)}")
                raise
    
    def add_classification_head(self, hidden_size: Optional[int] = None):
        """Add classification head to transformer"""
        if hidden_size is None:
            hidden_size = self.model.config.hidden_size
        
        classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, self.num_labels)
        )
        
        return classifier
    
    def freeze_base_model(self, freeze_layers: Optional[List[int]] = None):
        """Freeze base transformer model"""
        if freeze_layers is None:
            # Freeze all
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for layer_idx in freeze_layers:
                if hasattr(self.model, 'encoder'):
                    if layer_idx < len(self.model.encoder.layer):
                        for param in self.model.encoder.layer[layer_idx].parameters():
                            param.requires_grad = False


class MusicTransformerEncoder:
    """
    Advanced transformer encoder for music features
    """
    
    def __init__(
        self,
        input_dim: int = 169,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required")
        
        from .deep_models import TransformerMusicEncoder
        
        self.encoder = TransformerMusicEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout
        )
        self.max_seq_len = max_seq_len
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode music features using transformer"""
        import torch
        
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features_tensor = torch.FloatTensor(features)
        else:
            features_tensor = features
        
        # Add sequence dimension if needed
        if len(features_tensor.shape) == 2:
            features_tensor = features_tensor.unsqueeze(0)
        
        # Encode
        with torch.no_grad():
            encoded = self.encoder(features_tensor)
        
        return encoded.cpu().numpy() if isinstance(encoded, torch.Tensor) else encoded
    
    def get_attention_patterns(self, features: np.ndarray) -> Dict[str, Any]:
        """Get attention patterns from encoding"""
        # This would require modifying the encoder to return attention
        # For now, return placeholder
        return {
            "attention_available": False,
            "message": "Attention patterns require model modification"
        }

