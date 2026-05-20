"""
Deep Learning Model Development System
Comprehensive PyTorch-based model development with best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import time
import numpy as np
import math


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture parameters."""
    
    # Model dimensions
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Initialization
    initializer_range: float = 0.02
    activation_type: str = "gelu"
    
    # Training specific
    use_bias: bool = True
    tie_word_embeddings: bool = True


class CustomTransformerBlock(nn.Module):
    """Custom transformer block with proper weight initialization."""
    
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout_rate,
            bias=config.use_bias,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias),
            nn.Dropout(config.dropout_rate)
        )
        
        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using proper initialization techniques."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Initialize layer norm
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections and layer normalization."""
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.layer_norm_1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm_2(x + ff_output)
        
        return x


class CustomTransformerModel(nn.Module):
    """Custom transformer model with proper architecture."""
    
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embeddings.weight, std=self.config.initializer_range)
        nn.init.normal_(self.position_embeddings.weight, std=self.config.initializer_range)
        
        # Initialize transformer blocks
        for block in self.transformer_blocks:
            block._init_weights()
        
        # Initialize final layer norm
        nn.init.ones_(self.final_layer_norm.weight)
        nn.init.zeros_(self.final_layer_norm.bias)
        
        # Initialize language model head
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=self.config.initializer_range)
    
    def get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate position IDs for input sequence."""
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        return position_ids
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with proper loss calculation."""
        batch_size, seq_length = input_ids.shape
        
        # Get position IDs
        position_ids = self.get_position_ids(input_ids)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self(current_ids)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
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
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and (current_ids == eos_token_id).any():
                    break
            
            return current_ids


class AdvancedDataset(Dataset):
    """Advanced dataset with proper data handling."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess and validate data."""
        if not self.texts:
            raise ValueError("Dataset cannot be empty")
        
        # Filter out empty texts
        self.texts = [text.strip() for text in self.texts if text.strip()]
        
        if not self.texts:
            raise ValueError("No valid texts found after preprocessing")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize with proper error handling
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception as e:
            logging.warning(f"Error tokenizing text at index {idx}: {e}")
            # Return a default encoding
            encoding = {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long)
            }
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }


class AdvancedTrainer:
    """Advanced trainer with comprehensive training features."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Setup training
        self._setup_training()
    
    def _setup_training(self):
        """Setup training components with proper initialization."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=self.config.get('adam_betas', (0.9, 0.999)),
            eps=self.config.get('adam_epsilon', 1e-8)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('warmup_steps', 1000),
            T_mult=2,
            eta_min=1e-7
        )
        
        # Mixed precision scaler
        if self.config.get('use_fp16', True):
            self.scaler = torch.cuda.amp.GradScaler()
        
        logging.info("Training components initialized")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with comprehensive metrics."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            device_batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Mixed precision training
            if self.config.get('use_fp16', True) and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**device_batch)
                    loss = outputs['loss']
                
                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**device_batch)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # Optimizer step
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logging.info(
                    f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                    f"LR: {current_lr:.2e}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss, 'num_batches': num_batches}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                device_batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**device_batch)
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss, 'num_batches': num_batches}
    
    def save_checkpoint(self, filepath: str):
        """Save comprehensive checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load comprehensive checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        logging.info(f"Checkpoint loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Model configuration
    model_config = ModelArchitectureConfig(
        vocab_size=50257,
        hidden_size=768,
        num_layers=6,  # Smaller for demo
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    # Training configuration
    train_config = {
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'adam_betas': (0.9, 0.999),
        'adam_epsilon': 1e-8,
        'warmup_steps': 100,
        'use_fp16': True,
        'max_grad_norm': 1.0,
        'log_interval': 5
    }
    
    # Initialize model
    model = CustomTransformerModel(model_config)
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, train_config)
    
    # Sample data
    sample_texts = [
        "The future of artificial intelligence is promising.",
        "Machine learning algorithms can solve complex problems.",
        "Deep learning models are transforming technology.",
        "Neural networks are powerful computational models.",
        "Transformers have revolutionized natural language processing."
    ]
    
    # Create dataset and dataloader
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = AdvancedDataset(sample_texts, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(dataloader)
        trainer.train_losses.append(train_metrics['train_loss'])
        
        # Evaluate
        val_metrics = trainer.evaluate(dataloader)
        trainer.val_losses.append(val_metrics['val_loss'])
        
        logging.info(
            f"Epoch {epoch + 1} completed. "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}"
        )
    
    # Generate sample text
    test_input = tokenizer("The future of AI is", return_tensors="pt")
    generated_ids = model.generate(
        test_input['input_ids'],
        max_length=50,
        temperature=0.7,
        do_sample=True
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Generated text: {generated_text}")
    
    # Save checkpoint
    trainer.save_checkpoint("advanced_model_checkpoint.pt")





