"""
PEP 8 Compliant System - Python Style Guidelines
Follows PEP 8 style guidelines for clean, readable code
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import time
import numpy as np


@dataclass
class ModelConfig:
    """Model configuration following PEP 8 naming conventions."""
    
    model_name: str = "gpt2"
    max_sequence_length: int = 512
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout_rate: float = 0.1
    activation_type: str = "gelu"
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02


@dataclass
class TrainingConfig:
    """Training configuration following PEP 8 naming conventions."""
    
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    scheduler_type: str = "cosine"
    optimizer_type: str = "adamw"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8


@dataclass
class OptimizationConfig:
    """Optimization configuration following PEP 8 naming conventions."""
    
    use_fp16: bool = True
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True


class PEP8CompliantModel(nn.Module):
    """Transformer model following PEP 8 style guidelines."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model with PEP 8 compliant parameters."""
        super().__init__()
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_components()
    
    def _init_components(self):
        """Initialize model components following PEP 8."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Set padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with proper settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Apply optimizations
            self._apply_optimizations()
            
            # Move to device
            self.model.to(self.device)
            
        except Exception as init_error:
            logging.error(f"Error initializing model: {init_error}")
            raise
    
    def _apply_optimizations(self):
        """Apply model optimizations following PEP 8."""
        if self.config.use_fp16:
            self.model = self.model.half()
            logging.info("Model converted to FP16")
        
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing enabled")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with PEP 8 compliant parameters."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, **kwargs):
        """Generate text with PEP 8 compliant parameters."""
        return self.model.generate(input_ids, **kwargs)


class PEP8CompliantDataset(Dataset):
    """Dataset following PEP 8 style guidelines."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """Initialize dataset with PEP 8 compliant parameters."""
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return dataset length."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get item with PEP 8 compliant indexing."""
        text = self.texts[idx]
        
        # Tokenize with proper parameters
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }


class PEP8CompliantTrainer:
    """Training system following PEP 8 style guidelines."""
    
    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig):
        """Initialize trainer with PEP 8 compliant parameters."""
        self.model_config = model_config
        self.train_config = train_config
        self.model = PEP8CompliantModel(model_config)
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self._setup_training()
    
    def _setup_training(self):
        """Setup training components following PEP 8."""
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=self.train_config.adam_betas,
            eps=self.train_config.adam_epsilon
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.train_config.warmup_steps,
            T_mult=2,
            eta_min=1e-7
        )
        
        logging.info("Training components initialized")
    
    def create_dataloader(self, texts: List[str], shuffle: bool = True) -> DataLoader:
        """Create dataloader with PEP 8 compliant parameters."""
        dataset = PEP8CompliantDataset(
            texts=texts,
            tokenizer=self.model.tokenizer,
            max_length=self.model_config.max_sequence_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform training step with PEP 8 compliant parameters."""
        self.model.train()
        
        # Move batch to device
        device_batch = {
            key: tensor.to(self.model.device) 
            for key, tensor in batch.items()
        }
        
        # Forward pass
        outputs = self.model(**device_batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_config.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update scheduler
        self.scheduler.step()
        
        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0]
        
        return {
            'loss': loss.item(),
            'learning_rate': current_lr
        }
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text with PEP 8 compliant parameters."""
        try:
            # Tokenize input
            inputs = self.model.tokenizer(
                prompt, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.model.tokenizer.eos_token_id
                )
            
            # Decode text
            generated_text = self.model.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
            
            return generated_text
            
        except Exception as gen_error:
            logging.error(f"Error generating text: {gen_error}")
            return prompt
    
    def evaluate_model(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model with PEP 8 compliant parameters."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                device_batch = {
                    key: tensor.to(self.model.device) 
                    for key, tensor in batch.items()
                }
                
                outputs = self.model(**device_batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint with PEP 8 compliant parameters."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'model_config': self.model_config,
                'train_config': self.train_config,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
            
            torch.save(checkpoint, filepath)
            logging.info(f"Checkpoint saved to {filepath}")
            
        except Exception as save_error:
            logging.error(f"Error saving checkpoint: {save_error}")
            raise
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint with PEP 8 compliant parameters."""
        try:
            checkpoint = torch.load(filepath, map_location=self.model.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            logging.info(f"Checkpoint loaded from {filepath}")
            
        except Exception as load_error:
            logging.error(f"Error loading checkpoint: {load_error}")
            raise


def setup_logging(level: int = logging.INFO):
    """Setup logging with PEP 8 compliant parameters."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def calculate_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics with PEP 8 compliant parameters."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    # Calculate basic metrics
    num_samples = len(predictions)
    exact_matches = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    
    # Calculate accuracy
    accuracy = exact_matches / num_samples if num_samples > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'num_samples': num_samples,
        'exact_matches': exact_matches
    }


# Main execution block following PEP 8
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Create configurations
    model_config = ModelConfig(
        model_name="gpt2",
        max_sequence_length=512,
        use_fp16=True
    )
    
    train_config = TrainingConfig(
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3
    )
    
    # Initialize trainer
    trainer = PEP8CompliantTrainer(
        model_config=model_config,
        train_config=train_config
    )
    
    # Sample training data
    sample_texts = [
        "The future of artificial intelligence is promising.",
        "Machine learning algorithms can solve complex problems.",
        "Deep learning models are transforming technology."
    ]
    
    # Create dataloader
    dataloader = trainer.create_dataloader(texts=sample_texts)
    
    # Training loop
    for epoch in range(train_config.num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            metrics = trainer.train_step(batch)
            epoch_loss += metrics['loss']
            
            # Log progress
            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"LR: {metrics['learning_rate']:.2e}"
                )
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        trainer.train_losses.append(avg_epoch_loss)
        
        logging.info(f"Epoch {epoch} completed. Avg loss: {avg_epoch_loss:.4f}")
    
    # Generate sample text
    test_prompt = "The future of AI is"
    generated_text = trainer.generate_text(
        prompt=test_prompt,
        max_length=50
    )
    
    print(f"Generated text: {generated_text}")
    
    # Save checkpoint
    trainer.save_checkpoint("model_checkpoint.pt")





