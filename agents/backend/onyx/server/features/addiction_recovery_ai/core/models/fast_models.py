"""
Fast Optimized Models for Recovery AI with JIT, Quantization, and Lightweight Architectures
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import torch.compile for PyTorch 2.0+
try:
    if hasattr(torch, 'compile'):
        TORCH_COMPILE_AVAILABLE = True
    else:
        TORCH_COMPILE_AVAILABLE = False
except:
    TORCH_COMPILE_AVAILABLE = False


class FastProgressPredictor(nn.Module):
    """Ultra-lightweight progress predictor with optimizations"""
    
    def __init__(self, input_features: int = 10):
        super().__init__()
        # Reduced layers for speed
        self.model = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        # Fuse operations for speed
        self._fuse_layers()
    
    def _fuse_layers(self):
        """Fuse layers for faster inference"""
        try:
            torch.quantization.fuse_modules(
                self.model,
                [['0', '1']],
                inplace=True
            )
        except:
            pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FastRelapsePredictor(nn.Module):
    """Ultra-lightweight relapse predictor with optimizations"""
    
    def __init__(self, input_size: int = 5):
        super().__init__()
        # Single layer LSTM for speed
        self.lstm = nn.LSTM(input_size, 16, 1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])


class FastSentimentAnalyzer(nn.Module):
    """Lightweight sentiment analyzer using DistilBERT"""
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Use DistilBERT for speed
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)
            self.model.eval()
            self.available = True
        except Exception as e:
            logger.warning(f"Fast sentiment analyzer not available: {e}")
            self.available = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if not self.available:
            return torch.tensor([[0.5, 0.5]])
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return torch.softmax(outputs.logits, dim=-1)


def create_fast_predictors(
    device: Optional[torch.device] = None,
    use_jit: bool = True,
    use_quantization: bool = True,
    use_compile: bool = TORCH_COMPILE_AVAILABLE
) -> Tuple[nn.Module, nn.Module]:
    """
    Create fast predictors with optimizations
    
    Args:
        device: PyTorch device
        use_jit: Use JIT compilation
        use_quantization: Use INT8 quantization
        use_compile: Use torch.compile (PyTorch 2.0+)
    
    Returns:
        Tuple of (progress_predictor, relapse_predictor)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    progress = FastProgressPredictor().to(device)
    relapse = FastRelapsePredictor().to(device)
    
    # Set to eval mode
    progress.eval()
    relapse.eval()
    
    # Apply optimizations
    if use_quantization:
        try:
            # Dynamic quantization for speed
            progress = torch.quantization.quantize_dynamic(
                progress, {nn.Linear}, dtype=torch.qint8
            )
            relapse = torch.quantization.quantize_dynamic(
                relapse, {nn.LSTM, nn.Linear}, dtype=torch.qint8
            )
            logger.info("Models quantized to INT8")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    if use_jit:
        try:
            # JIT compile for faster inference
            progress = torch.jit.script(progress)
            relapse = torch.jit.script(relapse)
            logger.info("Models JIT compiled")
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
    
    if use_compile and TORCH_COMPILE_AVAILABLE:
        try:
            # torch.compile for PyTorch 2.0+
            progress = torch.compile(progress, mode="reduce-overhead")
            relapse = torch.compile(relapse, mode="reduce-overhead")
            logger.info("Models compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    return progress, relapse


def create_fast_sentiment_analyzer(
    device: Optional[torch.device] = None,
    use_quantization: bool = True
) -> Optional[FastSentimentAnalyzer]:
    """Create fast sentiment analyzer"""
    analyzer = FastSentimentAnalyzer(device=device)
    
    if analyzer.available and use_quantization:
        try:
            analyzer.model = torch.quantization.quantize_dynamic(
                analyzer.model, dtype=torch.qint8
            )
            logger.info("Sentiment analyzer quantized")
        except Exception as e:
            logger.warning(f"Sentiment quantization failed: {e}")
    
    return analyzer if analyzer.available else None

