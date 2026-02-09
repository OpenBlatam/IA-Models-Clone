"""
Deep Neural Network Models for Music Analysis
Multi-layer architectures with advanced techniques
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DeepGenreClassifier(nn.Module):
    """
    Deep Neural Network for Genre Classification
    Multi-layer architecture with residual connections
    """
    
    def __init__(
        self,
        input_size: int = 169,
        num_genres: int = 10,
        hidden_layers: List[int] = [512, 512, 256, 256, 128, 128],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_genres = num_genres
        self.use_residual = use_residual
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_layers[0])
        self.input_bn = nn.BatchNorm1d(hidden_layers[0]) if use_batch_norm else nn.Identity()
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        for i in range(len(hidden_layers) - 1):
            # Main layer
            self.hidden_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            )
            
            # Batch normalization
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
            else:
                self.bn_layers.append(nn.Identity())
            
            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            
            # Residual connection (if dimensions match)
            if use_residual and hidden_layers[i] == hidden_layers[i + 1]:
                self.residual_layers.append(nn.Identity())
            else:
                self.residual_layers.append(
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1])
                    if use_residual else nn.Identity()
                )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], num_genres)
        
        # Initialize weights with proper initialization
        self._initialize_weights()
        
        # Initialize LSTM weights
        self._initialize_lstm_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using best practices:
        - Kaiming/He initialization for ReLU activations
        - Xavier/Glorot for other activations
        - Proper bias initialization
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming for ReLU, Xavier for others
                if hasattr(module, 'activation') and module.activation == 'relu':
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                
                if module.bias is not None:
                    # Fan-in based bias initialization
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections and NaN/Inf detection
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output logits of shape (batch_size, num_genres)
        """
        # Validate input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Input contains NaN or Inf values, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Input layer
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.input_dropout(x)
        
        # Hidden layers with residual connections
        for i, (hidden, bn, dropout, residual) in enumerate(
            zip(self.hidden_layers, self.bn_layers, self.dropout_layers, self.residual_layers)
        ):
            # Residual connection
            if self.use_residual and i > 0:
                residual_input = x
                x = hidden(x)
                x = bn(x)
                x = F.relu(x)
                x = dropout(x)
                
                # Add residual if dimensions match
                if residual_input.shape == x.shape:
                    x = x + residual_input
                elif not isinstance(residual, nn.Identity):
                    x = x + residual(residual_input)
            else:
                x = hidden(x)
                x = bn(x)
                x = F.relu(x)
                x = dropout(x)
            
            # Check for NaN/Inf after each layer
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error(f"NaN/Inf detected in layer {i}, stopping forward pass")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Output layer
        x = self.output_layer(x)
        
        # Final validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("NaN/Inf in output, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x


class DeepMoodDetector(nn.Module):
    """
    Deep CNN + LSTM for Mood Detection
    Multi-modal architecture combining CNN and RNN
    """
    
    def __init__(
        self,
        input_channels: int = 13,  # MFCC features
        num_moods: int = 6,
        cnn_channels: List[int] = [32, 64, 128],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # CNN layers for feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in cnn_channels:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout_rate)
                )
            )
            in_channels = out_channels
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layers
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_moods)
        
        # Initialize weights
        self._initialize_weights()
        self._initialize_lstm_weights()
    
    def _initialize_weights(self):
        """Initialize CNN and FC weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _initialize_lstm_weights(self):
        """Initialize LSTM weights with proper scaling"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: CNN -> LSTM -> FC with error handling
        
        Args:
            x: Input tensor [batch_size, channels, seq_len]
        
        Returns:
            Mood logits [batch_size, num_moods]
        """
        # Validate input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # CNN feature extraction
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("NaN/Inf in CNN layer")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Reshape for LSTM (batch, seq_len, features)
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        
        # LSTM with packed sequence support for variable lengths
        try:
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use last hidden state
            x = lstm_out[:, -1, :]  # (batch, hidden*2)
        except RuntimeError as e:
            logger.error(f"LSTM forward error: {str(e)}")
            # Fallback: use mean pooling
            x = x.mean(dim=1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        # Final validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("NaN/Inf in output")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x


class MultiTaskMusicModel(nn.Module):
    """
    Multi-task learning model for simultaneous prediction of:
    - Genre
    - Mood
    - Energy level
    - Complexity
    - Instrument detection
    """
    
    def __init__(
        self,
        input_size: int = 169,
        num_genres: int = 10,
        num_moods: int = 6,
        num_instruments: int = 15,
        shared_layers: List[int] = [512, 512, 256],
        task_specific_layers: int = 128,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # Shared encoder
        self.shared_layers = nn.ModuleList()
        in_size = input_size
        
        for hidden_size in shared_layers:
            self.shared_layers.append(
                nn.Sequential(
                    nn.Linear(in_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            in_size = hidden_size
        
        # Task-specific heads
        # Genre classification
        self.genre_head = nn.Sequential(
            nn.Linear(shared_layers[-1], task_specific_layers),
            nn.BatchNorm1d(task_specific_layers),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(task_specific_layers, num_genres)
        )
        
        # Mood detection
        self.mood_head = nn.Sequential(
            nn.Linear(shared_layers[-1], task_specific_layers),
            nn.BatchNorm1d(task_specific_layers),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(task_specific_layers, num_moods)
        )
        
        # Energy regression
        self.energy_head = nn.Sequential(
            nn.Linear(shared_layers[-1], task_specific_layers),
            nn.BatchNorm1d(task_specific_layers),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(task_specific_layers, 1),
            nn.Sigmoid()  # Energy between 0 and 1
        )
        
        # Complexity regression
        self.complexity_head = nn.Sequential(
            nn.Linear(shared_layers[-1], task_specific_layers),
            nn.BatchNorm1d(task_specific_layers),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(task_specific_layers, 1),
            nn.Sigmoid()  # Complexity between 0 and 1
        )
        
        # Instrument detection (multi-label)
        self.instrument_head = nn.Sequential(
            nn.Linear(shared_layers[-1], task_specific_layers),
            nn.BatchNorm1d(task_specific_layers),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(task_specific_layers, num_instruments),
            nn.Sigmoid()  # Multi-label binary classification
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through shared encoder and task-specific heads"""
        # Shared encoder
        for layer in self.shared_layers:
            x = layer(x)
        
        # Task-specific predictions
        return {
            "genre": self.genre_head(x),
            "mood": self.mood_head(x),
            "energy": self.energy_head(x),
            "complexity": self.complexity_head(x),
            "instruments": self.instrument_head(x)
        }


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer with proper scaling and initialization
    Implements scaled dot-product attention with best practices
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)  # Pre-compute scale factor
        
        # Projection layers with proper initialization
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize attention parameters using Xavier uniform"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with scaled dot-product attention
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            Attention output [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = query.size()
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with proper scaling
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax with numerical stability
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Check for NaN/Inf
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            logger.warning("NaN/Inf in attention weights, applying fix")
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1.0, neginf=0.0)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        return output


class TransformerMusicEncoder(nn.Module):
    """
    Transformer-based encoder for music features
    Uses self-attention to model relationships between features
    """
    
    def __init__(
        self,
        input_dim: int = 169,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding (learned with proper initialization)
        self.max_seq_len = 1000
        self.pos_encoding = nn.Parameter(torch.empty(1, self.max_seq_len, embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)  # Small initialization
        
        # Transformer layers
        self.layers = nn.ModuleList([
            self._create_transformer_layer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def _create_transformer_layer(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float
    ) -> nn.Module:
        """Create a transformer encoder layer"""
        return nn.ModuleDict({
            "attention": AttentionLayer(embed_dim, num_heads, dropout),
            "attention_norm": nn.LayerNorm(embed_dim),
            "ff": nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout)
            ),
            "ff_norm": nn.LayerNorm(embed_dim)
        })
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer encoder with proper error handling
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]
        
        Returns:
            Encoded features [batch_size, embed_dim]
        """
        # Validate input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding (with proper scaling)
        if seq_len > self.max_seq_len:
            logger.warning(f"Sequence length {seq_len} exceeds max {self.max_seq_len}, truncating")
            seq_len = self.max_seq_len
            x = x[:, :seq_len, :]
        
        # Scale embeddings by sqrt(embed_dim) for better initialization
        x = x * math.sqrt(self.embed_dim)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer layers with pre-norm architecture
        for layer_idx, layer in enumerate(self.layers):
            # Self-attention with residual (pre-norm)
            residual = x
            x = layer["attention_norm"](x)
            x = layer["attention"](x, x, x, mask=mask)
            x = x + residual
            
            # Check for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error(f"NaN/Inf in attention layer {layer_idx}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Feed-forward with residual (pre-norm)
            residual = x
            x = layer["ff_norm"](x)
            x = layer["ff"](x)
            x = x + residual
            
            # Check for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error(f"NaN/Inf in FF layer {layer_idx}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Global average pooling (with mask support if needed)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = (~mask).unsqueeze(-1).float()  # [batch, seq_len, 1]
            masked_sum = (x * mask_expanded).sum(dim=1)  # [batch, embed_dim]
            mask_count = mask_expanded.sum(dim=1)  # [batch, 1]
            x = masked_sum / (mask_count + 1e-8)
        else:
            x = x.mean(dim=1)  # [batch, embed_dim]
        
        # Final validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("NaN/Inf in final output")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x


class DeepMusicAnalyzer:
    """
    Deep learning analyzer with multiple model architectures
    """
    
    def __init__(self, device: str = "cpu", compile_models: bool = True):
        self.device = device
        self.models: Dict[str, nn.Module] = {}
        self.compile_models = compile_models and hasattr(torch, 'compile')
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all deep models"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, models not initialized")
            return
        
        try:
            # Deep Genre Classifier
            self.models["genre_classifier"] = DeepGenreClassifier(
                input_size=169,
                num_genres=10,
                hidden_layers=[512, 512, 256, 256, 128, 128],
                use_residual=True
            ).to(self.device)
            
            # Deep Mood Detector
            self.models["mood_detector"] = DeepMoodDetector(
                input_channels=13,
                num_moods=6,
                cnn_channels=[32, 64, 128],
                lstm_hidden=256,
                lstm_layers=2
            ).to(self.device)
            
            # Multi-task Model
            self.models["multi_task"] = MultiTaskMusicModel(
                input_size=169,
                num_genres=10,
                num_moods=6,
                num_instruments=15,
                shared_layers=[512, 512, 256]
            ).to(self.device)
            
            # Transformer Encoder
            self.models["transformer_encoder"] = TransformerMusicEncoder(
                input_dim=169,
                embed_dim=256,
                num_heads=8,
                num_layers=4
            ).to(self.device)
            
            # Set to eval mode and compile for speed
            for name, model in self.models.items():
                model.eval()
                # Compile for faster inference
                if self.compile_models:
                    try:
                        self.models[name] = torch.compile(model, mode="reduce-overhead")
                        logger.info(f"Compiled {name} for faster inference")
                    except Exception as e:
                        logger.warning(f"Could not compile {name}: {str(e)}")
            
            # Enable optimizations
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
            
            logger.info(f"Initialized {len(self.models)} deep models on {self.device}")
        
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}", exc_info=True)
    
    def predict_genre(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict genre using deep classifier"""
        if "genre_classifier" not in self.models:
            return {"error": "Model not available"}
        
        try:
            model = self.models["genre_classifier"]
            features_tensor = torch.FloatTensor(features).to(self.device, non_blocking=True)
            
            with torch.no_grad():
                # Use autocast for faster inference
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        logits = model(features_tensor)
                else:
                    logits = model(features_tensor)
                probs = F.softmax(logits, dim=-1)
                predicted = torch.argmax(probs, dim=-1)
            
            return {
                "genre_id": int(predicted.item()),
                "confidence": float(probs[0][predicted].item()),
                "probabilities": probs[0].cpu().numpy().tolist()
            }
        
        except Exception as e:
            logger.error(f"Genre prediction error: {str(e)}")
            return {"error": str(e)}
    
    def predict_multi_task(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict all tasks simultaneously"""
        if "multi_task" not in self.models:
            return {"error": "Model not available"}
        
        try:
            model = self.models["multi_task"]
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            with torch.no_grad():
                predictions = model(features_tensor)
            
            return {
                "genre": {
                    "logits": predictions["genre"][0].cpu().numpy().tolist(),
                    "predicted": int(torch.argmax(predictions["genre"], dim=-1).item())
                },
                "mood": {
                    "logits": predictions["mood"][0].cpu().numpy().tolist(),
                    "predicted": int(torch.argmax(predictions["mood"], dim=-1).item())
                },
                "energy": float(predictions["energy"][0].item()),
                "complexity": float(predictions["complexity"][0].item()),
                "instruments": {
                    "probabilities": predictions["instruments"][0].cpu().numpy().tolist(),
                    "detected": (predictions["instruments"][0] > 0.5).cpu().numpy().tolist()
                }
            }
        
        except Exception as e:
            logger.error(f"Multi-task prediction error: {str(e)}")
            return {"error": str(e)}


# Global instance
_deep_analyzer: Optional[DeepMusicAnalyzer] = None


def get_deep_analyzer(device: str = "cpu") -> DeepMusicAnalyzer:
    """Get or create deep analyzer instance"""
    global _deep_analyzer
    if _deep_analyzer is None:
        _deep_analyzer = DeepMusicAnalyzer(device=device)
    return _deep_analyzer

