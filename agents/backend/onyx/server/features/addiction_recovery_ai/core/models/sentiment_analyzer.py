"""
Sentiment Analysis for Recovery Tracking using Transformers
Enhanced with mixed precision, batch processing, and GPU optimization
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, AutoModel, AutoConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class RecoverySentimentAnalyzer(nn.Module):
    """
    Transformer-based sentiment analyzer for recovery tracking
    Enhanced with mixed precision, batch processing, and GPU optimization
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        max_length: int = 512
    ):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: HuggingFace model name
            device: PyTorch device
            use_mixed_precision: Use mixed precision (FP16) for faster inference
            max_length: Maximum sequence length
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.max_length = max_length
        
        try:
            # Load model with proper configuration
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.use_mixed_precision else torch.float32
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model.eval()
            
            # Enable torch.compile for PyTorch 2.0+ if available (max speed)
            if hasattr(torch, 'compile') and self.device.type == "cuda":
                try:
                    self.model = torch.compile(
                        self.model,
                        mode="max-autotune",  # Most aggressive optimization
                        fullgraph=True
                    )
                    logger.info("Model compiled with torch.compile (max-autotune)")
                except Exception as e:
                    try:
                        # Fallback to reduce-overhead
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        logger.info("Model compiled with torch.compile (reduce-overhead)")
                    except:
                        logger.warning(f"torch.compile failed: {e}")
            
            # Create pipeline for easy use (fallback)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if self.device.type == "cuda" else -1,
                torch_dtype=torch.float16 if self.use_mixed_precision else torch.float32
            )
            
            logger.info(
                f"RecoverySentimentAnalyzer initialized on {self.device} "
                f"(mixed_precision={self.use_mixed_precision})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise
    
    def analyze(
        self, 
        text: str,
        return_attention: bool = False
    ) -> Dict[str, float]:
        """
        Analyze sentiment of text with optimized inference
        
        Args:
            text: Input text
            return_attention: Return attention weights (for interpretability)
            
        Returns:
            Sentiment scores and optionally attention weights
        """
        try:
            # Tokenize with proper padding and truncation
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Use inference_mode (faster than no_grad) and mixed precision
            with torch.inference_mode():  # Faster than torch.no_grad()
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(**inputs, output_attentions=return_attention)
                else:
                    outputs = self.model(**inputs, output_attentions=return_attention)
            
            # Get predictions
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            
            # Get label from model config
            id2label = self.model.config.id2label
            label = id2label.get(predicted_class, "NEUTRAL")
            score = probs[0][predicted_class].item()
            
            result = {
                "label": label,
                "score": score,
                "probabilities": {
                    id2label.get(i, f"LABEL_{i}"): probs[0][i].item()
                    for i in range(len(id2label))
                }
            }
            
            if return_attention and hasattr(outputs, 'attentions'):
                result["attention_weights"] = outputs.attentions[-1].cpu().numpy().tolist()
            
            return result
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}", exc_info=True)
            return {"label": "NEUTRAL", "score": 0.5, "error": str(e)}
    
    def analyze_batch(
        self, 
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """
        Analyze batch of texts with optimized batch processing
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Process batch with inference_mode (faster)
                with torch.inference_mode():
                    if self.use_mixed_precision:
                        with autocast():
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)
                
                # Get predictions for batch
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probs, dim=-1)
                
                id2label = self.model.config.id2label
                
                for j, pred_class in enumerate(predicted_classes):
                    label = id2label.get(pred_class.item(), "NEUTRAL")
                    score = probs[j][pred_class].item()
                    results.append({
                        "label": label,
                        "score": score,
                        "probabilities": {
                            id2label.get(k, f"LABEL_{k}"): probs[j][k].item()
                            for k in range(len(id2label))
                        }
                    })
            except Exception as e:
                logger.error(f"Batch processing failed for batch {i}: {e}")
                # Fallback to individual processing
                for text in batch_texts:
                    results.append(self.analyze(text))
        
        return results


class RecoveryProgressPredictor(nn.Module):
    """
    Predict recovery progress using deep learning
    Enhanced with residual connections, layer normalization, and better initialization
    """
    
    def __init__(
        self,
        input_features: int = 10,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_residual: bool = True,
        activation: str = "gelu"
    ):
        """
        Initialize progress predictor with improved architecture
        
        Args:
            input_features: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of layers
            dropout: Dropout rate
            use_residual: Use residual connections
            activation: Activation function (relu, gelu, swish)
        """
        super().__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.use_residual = use_residual and input_features == hidden_size
        
        # Activation function selection
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()  # Swish is SiLU
        else:
            self.activation = nn.ReLU()
        
        # Input projection if needed for residual connections
        if self.use_residual and input_features != hidden_size:
            self.input_proj = nn.Linear(input_features, hidden_size)
        else:
            self.input_proj = None
        
        # Build layers with residual connections
        self.layers = nn.ModuleList()
        in_size = input_features
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.LayerNorm(hidden_size),  # LayerNorm instead of BatchNorm for better stability
                self.activation,
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
            in_size = hidden_size
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights with proper initialization
        self._initialize_weights()
        
        logger.info(
            f"RecoveryProgressPredictor initialized "
            f"(layers={num_layers}, residual={self.use_residual}, activation={activation})"
        )
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming for ReLU/GELU, Xavier for others
                if isinstance(self.activation, (nn.ReLU, nn.GELU, nn.SiLU)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Progress prediction tensor of shape (batch_size, 1)
        """
        # Input projection if needed
        if self.input_proj is not None:
            residual = self.input_proj(x)
        else:
            residual = x if self.use_residual else None
        
        # Pass through layers
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # Add residual connection if applicable
            if self.use_residual and residual is not None and out.shape == residual.shape:
                out = out + residual
                residual = out  # Update residual for next layer
        
        # Output layer
        return self.output(out)
    
    def predict_progress(
        self, 
        features: torch.Tensor,
        use_mixed_precision: bool = True
    ) -> float:
        """
        Predict recovery progress (0-1) with optimized inference
        
        Args:
            features: Input features tensor
            use_mixed_precision: Use mixed precision for faster inference
            
        Returns:
            Progress score between 0 and 1
        """
        self.eval()
        
        # Ensure features are on correct device
        if features.device != next(self.parameters()).device:
            features = features.to(next(self.parameters()).device)
        
        with torch.inference_mode():  # Faster than no_grad
            if use_mixed_precision and next(self.parameters()).is_cuda:
                with autocast():
                    output = self.forward(features)
            else:
                output = self.forward(features)
            
            return output.item() if output.numel() == 1 else output.squeeze().item()


class RelapseRiskPredictor(nn.Module):
    """
    Predict relapse risk using bidirectional LSTM with attention
    Enhanced with attention mechanism and better architecture
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        """
        Initialize relapse risk predictor with enhanced architecture
        
        Args:
            input_size: Input feature size
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        if use_attention:
            lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_size // 2, 1)
            )
        
        # Classifier with improved architecture
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.LayerNorm(lstm_output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.LayerNorm(lstm_output_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            f"RelapseRiskPredictor initialized "
            f"(bidirectional={bidirectional}, attention={use_attention})"
        )
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[(n // 4):(n // 2)].fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Risk prediction tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.use_attention:
            # Compute attention weights
            attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Apply attention
            attended_output = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)
        else:
            # Use last output
            attended_output = lstm_out[:, -1, :]
        
        # Classifier
        return self.classifier(attended_output)
    
    def predict_risk(
        self, 
        sequence: torch.Tensor,
        use_mixed_precision: bool = True
    ) -> float:
        """
        Predict relapse risk (0-1) with optimized inference
        
        Args:
            sequence: Input sequence tensor
            use_mixed_precision: Use mixed precision for faster inference
            
        Returns:
            Risk score between 0 and 1
        """
        self.eval()
        
        # Ensure sequence is on correct device
        if sequence.device != next(self.parameters()).device:
            sequence = sequence.to(next(self.parameters()).device)
        
        # Add batch dimension if needed
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)
        
        with torch.inference_mode():  # Faster than no_grad
            if use_mixed_precision and next(self.parameters()).is_cuda:
                with autocast():
                    output = self.forward(sequence)
            else:
                output = self.forward(sequence)
            
            return output.item() if output.numel() == 1 else output.squeeze().item()


def create_sentiment_analyzer(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    device: Optional[torch.device] = None
) -> RecoverySentimentAnalyzer:
    """Factory function for sentiment analyzer"""
    return RecoverySentimentAnalyzer(model_name=model_name, device=device)


def create_progress_predictor(
    input_features: int = 10,
    device: Optional[torch.device] = None
) -> RecoveryProgressPredictor:
    """Factory function for progress predictor"""
    model = RecoveryProgressPredictor(input_features=input_features)
    if device:
        model = model.to(device)
    return model


def create_relapse_predictor(
    input_size: int = 5,
    device: Optional[torch.device] = None
) -> RelapseRiskPredictor:
    """Factory function for relapse predictor"""
    model = RelapseRiskPredictor(input_size=input_size)
    if device:
        model = model.to(device)
    return model

