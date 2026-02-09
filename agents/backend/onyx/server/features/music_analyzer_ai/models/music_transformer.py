"""
Modelo Transformer para análisis musical avanzado
Implementa mejores prácticas de PyTorch, LoRA, y técnicas avanzadas de entrenamiento
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import warnings

logger = logging.getLogger(__name__)

# Try to import PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available, LoRA features disabled")


class MusicFeatureEncoder(nn.Module):
    """Encoder de características musicales usando Transformer con mejores prácticas"""
    
    def __init__(
        self,
        feature_dim: int = 13,  # Audio features de Spotify
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",  # GELU is better than ReLU for transformers
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # Embedding layer with proper initialization
        self.feature_embedding = nn.Linear(feature_dim, d_model)
        self._init_embedding()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder with better configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-norm is more stable
            layer_norm_eps=layer_norm_eps
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False  # Disable for better compatibility
        )
        
        # Output projection with proper initialization
        self.output_proj = nn.Linear(d_model, d_model)
        self._init_output_proj()
    
    def _init_embedding(self):
        """Initialize embedding layer with Xavier uniform"""
        nn.init.xavier_uniform_(self.feature_embedding.weight)
        if self.feature_embedding.bias is not None:
            nn.init.zeros_(self.feature_embedding.bias)
    
    def _init_output_proj(self):
        """Initialize output projection"""
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
            mask: [batch_size, seq_len] optional mask
        Returns:
            [batch_size, seq_len, d_model]
        """
        # Embed features
        x = self.feature_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask if provided
        if mask is not None:
            # Convert to attention mask format
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, x.size(1), x.size(1), -1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        else:
            attn_mask = None
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding para secuencias con mejor inicialización"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) for better initialization
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MusicClassifier(nn.Module):
    """Clasificador musical usando Transformer"""
    
    def __init__(
        self,
        feature_dim: int = 13,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_genres: int = 12,
        num_emotions: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = MusicFeatureEncoder(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Classification heads
        self.genre_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_genres)
        )
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_emotions)
        )
        
        # Regression head for popularity prediction
        self.popularity_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
            mask: [batch_size, seq_len] optional mask
        Returns:
            Dictionary with genre_logits, emotion_logits, popularity_pred
        """
        # Encode features
        encoded = self.encoder(x, mask)  # [batch_size, seq_len, d_model]
        
        # Global pooling (mean over sequence)
        if mask is not None:
            # Masked mean
            mask_expanded = (~mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            masked_sum = (encoded * mask_expanded).sum(dim=1)  # [batch_size, d_model]
            mask_count = mask_expanded.sum(dim=1)  # [batch_size, 1]
            pooled = masked_sum / (mask_count + 1e-8)  # [batch_size, d_model]
        else:
            pooled = encoded.mean(dim=1)  # [batch_size, d_model]
        
        # Classifications
        genre_logits = self.genre_classifier(pooled)  # [batch_size, num_genres]
        emotion_logits = self.emotion_classifier(pooled)  # [batch_size, num_emotions]
        popularity_pred = self.popularity_predictor(pooled)  # [batch_size, 1]
        
        return {
            "genre_logits": genre_logits,
            "emotion_logits": emotion_logits,
            "popularity_pred": popularity_pred.squeeze(-1)  # [batch_size]
        }


class MusicDataset(Dataset):
    """Dataset para entrenamiento de modelos musicales"""
    
    def __init__(
        self,
        features: List[np.ndarray],
        genres: Optional[List[int]] = None,
        emotions: Optional[List[int]] = None,
        popularities: Optional[List[float]] = None
    ):
        self.features = features
        self.genres = genres
        self.emotions = emotions
        self.popularities = popularities
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "features": torch.FloatTensor(self.features[idx])
        }
        
        if self.genres is not None:
            item["genre"] = torch.LongTensor([self.genres[idx]])
        
        if self.emotions is not None:
            item["emotion"] = torch.LongTensor([self.emotions[idx]])
        
        if self.popularities is not None:
            item["popularity"] = torch.FloatTensor([self.popularities[idx] / 100.0])
        
        return item


class MusicModelTrainer:
    """Entrenador para modelos musicales con mejores prácticas"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
        scheduler_type: str = "cosine",  # "cosine", "linear", "plateau"
        compile_model: bool = True,  # Use torch.compile for speed
        enable_tf32: bool = True  # Enable TF32 for faster training
    ):
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Speed optimizations
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile for faster training")
            except Exception as e:
                logger.warning(f"Model compilation failed: {str(e)}")
        
        if enable_tf32 and device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # Initialize mixed precision scaler with better defaults
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=2.**16,  # Initial scale for FP16
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
        else:
            self.scaler = None
        
        # Optimizer with better defaults
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # Standard AdamW betas
            eps=1e-8,
            amsgrad=False
        )
        
        # Learning rate scheduler
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.scheduler = None  # Will be initialized in train method
        
        # Loss functions with label smoothing option
        self.genre_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.emotion_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.popularity_loss_fn = nn.MSELoss()
        
        # Training state
        self.current_step = 0
        self.best_metrics = {}
        
        self.logger = logger
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
        total_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """Entrena una época con mejor manejo de mixed precision y gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        genre_loss_total = 0.0
        emotion_loss_total = 0.0
        popularity_loss_total = 0.0
        
        num_batches = 0
        
        # Initialize scheduler if needed
        if self.scheduler is None and total_steps is not None:
            self._init_scheduler(total_steps)
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Fast batch preparation with non_blocking transfer
                features = batch["features"].to(self.device, non_blocking=True)
                
                # Validate input
                if torch.isnan(features).any() or torch.isinf(features).any():
                    logger.warning(f"NaN/Inf in input batch {batch_idx}, skipping")
                    continue
                
                # Forward pass with mixed precision
                if self.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(features)
                        loss = self._compute_loss(outputs, batch)
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = self.model(features)
                    loss = self._compute_loss(outputs, batch)
                    loss = loss / self.gradient_accumulation_steps
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss at batch {batch_idx}, skipping")
                    self.optimizer.zero_grad()
                    continue
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation: only step optimizer every N steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Check for NaN/Inf gradients before clipping
                    has_nan_grad = False
                    for param in self.model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                logger.warning(f"NaN/Inf gradient detected, zeroing out")
                                param.grad.zero_()
                    
                    if not has_nan_grad:
                        # Gradient clipping
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.max_grad_norm
                            )
                            # Check if gradient norm is valid
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                logger.warning("Invalid gradient norm, skipping step")
                                self.scaler.update()
                                self.optimizer.zero_grad()
                                continue
                            
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.max_grad_norm
                            )
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                logger.warning("Invalid gradient norm, skipping step")
                                self.optimizer.zero_grad()
                                continue
                            self.optimizer.step()
                        
                        # Update learning rate
                        if self.scheduler is not None:
                            if self.scheduler_type == "plateau":
                                # Plateau scheduler needs validation loss
                                pass  # Will be updated in evaluate
                            else:
                                self.scheduler.step()
                        
                        self.optimizer.zero_grad()
                        self.current_step += 1
                
                # Accumulate losses (unscale for accumulation)
                loss_value = loss.item() * self.gradient_accumulation_steps
                if not (math.isnan(loss_value) or math.isinf(loss_value)):
                    total_loss += loss_value
                    genre_loss_total += outputs.get("genre_loss", 0.0)
                    emotion_loss_total += outputs.get("emotion_loss", 0.0)
                    popularity_loss_total += outputs.get("popularity_loss", 0.0)
                num_batches += 1
                
            except RuntimeError as e:
                logger.error(f"Runtime error in batch {batch_idx}: {str(e)}")
                self.optimizer.zero_grad()
                continue
            except Exception as e:
                logger.error(f"Unexpected error in batch {batch_idx}: {str(e)}", exc_info=True)
                self.optimizer.zero_grad()
                continue
        
        return {
            "total_loss": total_loss / num_batches,
            "genre_loss": genre_loss_total / num_batches,
            "emotion_loss": emotion_loss_total / num_batches,
            "popularity_loss": popularity_loss_total / num_batches,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
    
    def _init_scheduler(self, total_steps: int):
        """Initialize learning rate scheduler"""
        if self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
        elif self.scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        elif self.scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calcula la pérdida total"""
        loss = 0.0
        
        if "genre" in batch:
            genre_loss = self.genre_loss_fn(
                outputs["genre_logits"],
                batch["genre"].squeeze().to(self.device)
            )
            loss += genre_loss
            outputs["genre_loss"] = genre_loss.item()
        
        if "emotion" in batch:
            emotion_loss = self.emotion_loss_fn(
                outputs["emotion_logits"],
                batch["emotion"].squeeze().to(self.device)
            )
            loss += emotion_loss
            outputs["emotion_loss"] = emotion_loss.item()
        
        if "popularity" in batch:
            popularity_loss = self.popularity_loss_fn(
                outputs["popularity_pred"],
                batch["popularity"].squeeze().to(self.device)
            )
            loss += popularity_loss
            outputs["popularity_loss"] = popularity_loss.item()
        
        return loss
    
    def evaluate(
        self, 
        dataloader: DataLoader,
        update_scheduler: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the model with improved error handling and NaN detection
        
        Args:
            dataloader: DataLoader for evaluation
            update_scheduler: Whether to update plateau scheduler
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct_genre = 0
        correct_emotion = 0
        total = 0
        num_batches = 0
        
        # For detailed metrics
        genre_predictions = []
        genre_targets = []
        emotion_predictions = []
        emotion_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    features = batch["features"].to(self.device, non_blocking=True)
                    
                    # Validate input
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        logger.warning(f"NaN/Inf in evaluation batch {batch_idx}, skipping")
                        continue
                    
                    # Use autocast for evaluation too (faster)
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(features)
                            loss = self._compute_loss(outputs, batch)
                    else:
                        outputs = self.model(features)
                        loss = self._compute_loss(outputs, batch)
                    
                    # Check for NaN/Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN/Inf loss in evaluation batch {batch_idx}, skipping")
                        continue
                    
                    loss_value = loss.item()
                    if not (math.isnan(loss_value) or math.isinf(loss_value)):
                        total_loss += loss_value
                        num_batches += 1
                    
                    if "genre" in batch:
                        pred_genre = outputs["genre_logits"].argmax(dim=1)
                        targets = batch["genre"].squeeze().to(self.device)
                        correct_genre += (pred_genre == targets).sum().item()
                        genre_predictions.extend(pred_genre.cpu().numpy())
                        genre_targets.extend(targets.cpu().numpy())
                    
                    if "emotion" in batch:
                        pred_emotion = outputs["emotion_logits"].argmax(dim=1)
                        targets = batch["emotion"].squeeze().to(self.device)
                        correct_emotion += (pred_emotion == targets).sum().item()
                        emotion_predictions.extend(pred_emotion.cpu().numpy())
                        emotion_targets.extend(targets.cpu().numpy())
                    
                    total += features.size(0)
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {str(e)}")
                    continue
        
        metrics = {
            "loss": total_loss / num_batches if num_batches > 0 else float('inf'),
            "genre_accuracy": correct_genre / total if total > 0 else 0.0,
            "emotion_accuracy": correct_emotion / total if total > 0 else 0.0
        }
        
        # Update plateau scheduler if needed
        if update_scheduler and self.scheduler is not None and self.scheduler_type == "plateau":
            if not (math.isnan(metrics["loss"]) or math.isinf(metrics["loss"])):
                self.scheduler.step(metrics["loss"])
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Guarda checkpoint del modelo con mejor información"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "current_step": self.current_step,
            "best_metrics": self.best_metrics
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Carga checkpoint del modelo con mejor manejo de errores"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            if "scaler_state_dict" in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            if "current_step" in checkpoint:
                self.current_step = checkpoint["current_step"]
            
            if "best_metrics" in checkpoint:
                self.best_metrics = checkpoint["best_metrics"]
            
            self.logger.info(f"Checkpoint loaded from {path}")
            return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    task_type: str = "FEATURE_EXTRACTION"
) -> nn.Module:
    """
    Apply LoRA (Low-Rank Adaptation) to a model for efficient fine-tuning
    
    Args:
        model: PyTorch model to apply LoRA to
        target_modules: List of module names to apply LoRA (e.g., ["q_proj", "v_proj"])
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout
        task_type: Task type for PEFT
    
    Returns:
        Model with LoRA adapters applied
    """
    if not PEFT_AVAILABLE:
        logger.warning("PEFT not available, returning original model")
        return model
    
    if target_modules is None:
        # Default target modules for transformer models
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    try:
        # Map task type
        peft_task_type = TaskType.FEATURE_EXTRACTION if task_type == "FEATURE_EXTRACTION" else TaskType.CLASSIFICATION
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=peft_task_type
        )
        
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA with r={r}, alpha={lora_alpha} to {len(target_modules)} modules")
        return model
    except Exception as e:
        logger.error(f"Error applying LoRA: {str(e)}")
        return model

