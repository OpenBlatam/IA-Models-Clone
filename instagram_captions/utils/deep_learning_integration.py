from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import time
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .efficient_data_loading import EfficientDataLoaderFactory, DataLoaderConfig
from .model_evaluation import ModelEvaluator, EvaluationConfig, CaptionQualityEvaluator
from .training_optimization import OptimizedTrainer, TrainingConfig, EarlyStopping
from .task_specific_metrics import InstagramCaptionMetrics, MetricsConfig
from .gradient_optimization import GradientOptimizer, GradientConfig
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
            from transformers import get_cosine_schedule_with_warmup
            from transformers import get_linear_schedule_with_warmup
from typing import Any, List, Dict, Optional
import asyncio
# Import our custom modules

logger = logging.getLogger(__name__)


@dataclass
class DeepLearningConfig:
    """Comprehensive configuration for deep learning integration."""
    # Model configuration
    model_name: str = "gpt2"
    model_type: str = "transformer"
    vocab_size: int = 50257
    embedding_dim: int = 768
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 50
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Data configuration
    max_length: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Optimization configuration
    use_amp: bool = True
    gradient_accumulation_steps: int = 4
    early_stopping_patience: int = 7
    scheduler_type: str = "cosine_warmup"
    
    # Evaluation configuration
    eval_batch_size: int = 32
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Output configuration
    output_dir: str = "./deep_learning_output"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


class InstagramCaptionModel(nn.Module):
    """Advanced Instagram caption generation model."""
    
    def __init__(self, config: DeepLearningConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_length, config.embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None) -> Any:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Transformer forward pass
        transformer_output = self.transformer(embeddings, src_key_padding_mask=attention_mask == 0)
        
        # Output projection
        logits = self.output_projection(transformer_output)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': transformer_output
        }
    
    def generate(self, input_ids, max_length=100, temperature=0.7, top_p=0.9, do_sample=True) -> Any:
        """Generate captions."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self.forward(input_ids)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-p sampling
                if do_sample:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if EOS token
                if next_token.item() == 50256:  # GPT-2 EOS token
                    break
        
        return input_ids


class InstagramCaptionDataset(Dataset):
    """Dataset for Instagram caption training."""
    
    def __init__(self, texts: List[str], captions: List[str], tokenizer, max_length: int = 512):
        
    """__init__ function."""
self.texts = texts
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        caption = self.captions[idx]
        
        # Tokenize input text
        text_tokens = self.tokenizer.encode(text, max_length=self.max_length//2, truncation=True)
        
        # Tokenize caption
        caption_tokens = self.tokenizer.encode(caption, max_length=self.max_length//2, truncation=True)
        
        # Combine text and caption
        combined_tokens = text_tokens + [self.tokenizer.sep_token_id] + caption_tokens
        
        # Pad or truncate
        if len(combined_tokens) > self.max_length:
            combined_tokens = combined_tokens[:self.max_length]
        else:
            combined_tokens = combined_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(combined_tokens))
        
        # Create attention mask
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in combined_tokens]
        
        return {
            'input_ids': torch.tensor(combined_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(combined_tokens, dtype=torch.long)
        }


class DeepLearningIntegration:
    """Comprehensive deep learning integration system."""
    
    def __init__(self, config: DeepLearningConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.trainer = None
        self.evaluator = None
        self.metrics_calculator = None
        
        # Training state
        self.training_history = []
        self.best_metrics = {}
        self.current_epoch = 0
    
    def setup_model(self) -> Any:
        """Setup the model and tokenizer."""
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load or create model
        if os.path.exists(os.path.join(self.config.checkpoint_dir, "model.pt")):
            logger.info("Loading existing model...")
            self.model = InstagramCaptionModel(self.config)
            self.model.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, "model.pt")))
        else:
            logger.info("Creating new model...")
            self.model = InstagramCaptionModel(self.config)
        
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")
    
    def setup_optimization(self) -> Any:
        """Setup optimization components."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        num_training_steps = self.config.num_epochs * 1000  # Approximate
        if self.config.scheduler_type == "cosine_warmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            )
        
        # Gradient optimizer
        gradient_config = GradientConfig(
            max_grad_norm=self.config.max_grad_norm,
            use_amp=self.config.use_amp,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )
        self.gradient_optimizer = GradientOptimizer(self.model, self.optimizer, gradient_config)
        
        # Training monitor
        training_config = TrainingConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            patience=self.config.early_stopping_patience,
            scheduler_type=self.config.scheduler_type
        )
        self.trainer = OptimizedTrainer(self.model, training_config)
        
        # Evaluator
        eval_config = EvaluationConfig(
            batch_size=self.config.eval_batch_size,
            device=str(self.device)
        )
        self.evaluator = ModelEvaluator(self.model, eval_config)
        
        # Metrics calculator
        metrics_config = MetricsConfig(task_type="instagram")
        self.metrics_calculator = InstagramCaptionMetrics(metrics_config)
    
    def create_data_loaders(self, train_texts: List[str], train_captions: List[str],
                           val_texts: List[str], val_captions: List[str]):
        """Create optimized data loaders."""
        # Create datasets
        train_dataset = InstagramCaptionDataset(train_texts, train_captions, self.tokenizer, self.config.max_length)
        val_dataset = InstagramCaptionDataset(val_texts, val_captions, self.tokenizer, self.config.max_length)
        
        # Data loader configuration
        train_loader_config = DataLoaderConfig(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            shuffle=True
        )
        
        val_loader_config = DataLoaderConfig(
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            shuffle=False
        )
        
        # Create optimized loaders
        self.train_loader = EfficientDataLoaderFactory.create_loader(
            train_dataset, train_loader_config,
            enable_caching=True,
            enable_prefetching=True,
            enable_memory_optimization=True
        )
        
        self.val_loader = EfficientDataLoaderFactory.create_loader(
            val_dataset, val_loader_config,
            enable_caching=True,
            enable_prefetching=True,
            enable_memory_optimization=True
        )
        
        logger.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Training step with gradient optimization
            step_stats = self.gradient_optimizer.train_step(
                lambda: self.model(**batch)['loss'],
                batch
            )
            
            total_loss += step_stats['loss']
            num_batches += 1
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log progress
            if num_batches % 10 == 0:
                logger.info(f"Batch {num_batches}: Loss = {step_stats['loss']:.4f}, "
                           f"Grad Norm = {step_stats['grad_norm']:.4f}")
        
        return {'train_loss': total_loss / num_batches}
    
    def evaluate_epoch(self) -> Dict[str, float]:
        """Evaluate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        generated_captions = []
        reference_captions = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()
                num_batches += 1
                
                # Generate captions for evaluation
                if len(generated_captions) < 100:  # Limit for efficiency
                    generated = self.model.generate(
                        batch['input_ids'][:5],  # Generate for first 5 examples
                        max_length=100,
                        temperature=0.7
                    )
                    
                    for i, gen_ids in enumerate(generated):
                        gen_caption = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                        ref_caption = self.tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
                        generated_captions.append(gen_caption)
                        reference_captions.append(ref_caption)
        
        # Calculate metrics
        eval_metrics = {
            'val_loss': total_loss / num_batches
        }
        
        if generated_captions:
            caption_metrics = self.metrics_calculator.compute_instagram_metrics(
                generated_captions, reference_captions
            )
            eval_metrics.update(caption_metrics)
        
        return eval_metrics
    
    def train(self, train_texts: List[str], train_captions: List[str],
              val_texts: List[str], val_captions: List[str]) -> Dict[str, Any]:
        """Complete training pipeline."""
        logger.info("Starting deep learning training pipeline...")
        
        # Setup components
        self.setup_model()
        self.setup_optimization()
        self.create_data_loaders(train_texts, train_captions, val_texts, val_captions)
        
        # Training loop
        best_val_loss = float('inf')
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Evaluation
            val_metrics = self.evaluate_epoch()
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}, "
                       f"Val Loss = {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics)
                self.best_metrics = val_metrics.copy()
            
            # Record history
            epoch_stats = {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_stats)
            
            # Early stopping
            if early_stopping(val_metrics['val_loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Final evaluation
        final_metrics = self.evaluate_epoch()
        final_stats = self.gradient_optimizer.get_training_statistics()
        
        # Save final results
        self.save_training_results(final_metrics, final_stats)
        
        return {
            'training_history': self.training_history,
            'best_metrics': self.best_metrics,
            'final_metrics': final_metrics,
            'gradient_stats': final_stats
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        torch.save(self.model.state_dict(), best_model_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_training_results(self, final_metrics: Dict[str, float], gradient_stats: Dict[str, Any]):
        """Save training results and statistics."""
        results = {
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_metrics': self.best_metrics,
            'final_metrics': final_metrics,
            'gradient_stats': gradient_stats
        }
        
        results_path = os.path.join(self.config.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved: {results_path}")
    
    def generate_captions(self, texts: List[str], max_length: int = 100) -> List[str]:
        """Generate captions for given texts."""
        self.model.eval()
        captions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize input text
                input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
                
                # Generate caption
                generated_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                # Decode caption
                caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                captions.append(caption)
        
        return captions


# Example usage functions
def create_deep_learning_integration(
    model_name: str = "gpt2",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 50
) -> DeepLearningIntegration:
    """Create a deep learning integration system with best practices."""
    
    config = DeepLearningConfig(
        model_name=model_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        use_amp=True,
        gradient_accumulation_steps=4,
        early_stopping_patience=7
    )
    
    return DeepLearningIntegration(config)


def train_instagram_caption_model(
    train_texts: List[str],
    train_captions: List[str],
    val_texts: List[str],
    val_captions: List[str],
    output_dir: str = "./instagram_caption_model"
) -> Dict[str, Any]:
    """Complete Instagram caption model training pipeline."""
    
    config = DeepLearningConfig(
        output_dir=output_dir,
        checkpoint_dir=f"{output_dir}/checkpoints",
        log_dir=f"{output_dir}/logs"
    )
    
    integration = DeepLearningIntegration(config)
    results = integration.train(train_texts, train_captions, val_texts, val_captions)
    
    return results


def generate_instagram_captions(
    model_path: str,
    texts: List[str],
    max_length: int = 100
) -> List[str]:
    """Generate Instagram captions using a trained model."""
    
    config = DeepLearningConfig()
    integration = DeepLearningIntegration(config)
    
    # Load model
    integration.setup_model()
    integration.model.load_state_dict(torch.load(model_path))
    
    # Generate captions
    captions = integration.generate_captions(texts, max_length)
    
    return captions 