"""
Fast Optimized Models for Quick Inference
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FastTransformerEncoder(nn.Module):
    """Lightweight transformer encoder for fast inference"""
    
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(10000, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=4, batch_first=True),
            num_layers=num_layers
        )
        self.pooler = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pooler(x).squeeze(-1)
        return x


class FastTextClassifier(nn.Module):
    """Fast text classifier"""
    
    def __init__(self, vocab_size=10000, num_classes=3, hidden_size=128):
        super().__init__()
        self.encoder = FastTransformerEncoder(hidden_size, num_layers=2)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


def create_fast_analyzer(device=None):
    """Create fast analyzer with lightweight model"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if TRANSFORMERS_AVAILABLE:
        try:
            # Use smaller model
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
            model.eval()
            
            # Quantize for speed
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            except:
                pass
            
            return tokenizer, model
        except Exception as e:
            logger.warning(f"Fast analyzer creation failed: {e}")
    
    return None, None


def optimize_model_for_inference(model, example_input=None):
    """Optimize model for fast inference"""
    model.eval()
    
    # JIT compilation
    if example_input is not None:
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, example_input)
                traced = torch.jit.optimize_for_inference(traced)
                return traced
        except:
            pass
    
    return model

