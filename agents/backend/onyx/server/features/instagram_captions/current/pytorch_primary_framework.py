"""
PyTorch Primary Framework System
Comprehensive PyTorch-based deep learning implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import time
import numpy as np
import os
import json


@dataclass
class PyTorchConfig:
    """Configuration for PyTorch-based training."""
    
    # Model parameters
    model_name: str = "gpt2"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    
    # Training parameters
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    
    # PyTorch specific
    use_amp: bool = True
    use_data_parallel: bool = False
    use_distributed: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Device configuration
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Logging and monitoring
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"
    
    # Checkpointing
    save_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    
    # Profiling
    enable_profiling: bool = False
    profile_interval: int = 100


class PyTorchDataset(Dataset):
    """PyTorch-native dataset implementation."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._validate_data()
    
    def _validate_data(self):
        """Validate dataset integrity."""
        if not self.texts:
            raise ValueError("Dataset cannot be empty")
        
        # Filter and clean texts
        self.texts = [text.strip() for text in self.texts if text.strip()]
        
        if not self.texts:
            raise ValueError("No valid texts found after preprocessing")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # PyTorch-native tokenization
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Convert to PyTorch tensors
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": encoding["input_ids"].squeeze(0)
            }
        except Exception as e:
            logging.warning(f"Error tokenizing text at index {idx}: {e}")
            # Return zero tensors as fallback
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.zeros(self.max_length, dtype=torch.long)
            }


class PyTorchModel(nn.Module):
    """PyTorch-native model implementation."""
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load pretrained model using PyTorch best practices."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with PyTorch optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use FP16 for efficiency
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logging.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """PyTorch forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, **kwargs):
        """PyTorch-native text generation."""
        return self.model.generate(input_ids, **kwargs)


class PyTorchTrainer:
    """PyTorch-native trainer with comprehensive features."""
    
    def __init__(self, model: nn.Module, config: PyTorchConfig):
        self.model = model
        self.config = config
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # PyTorch training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.writer = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.global_step = 0
        
        # Setup PyTorch components
        self._setup_training()
        self._setup_logging()
    
    def _setup_device(self) -> torch.device:
        """Setup PyTorch device configuration."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        # PyTorch device optimizations
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        logging.info(f"Using device: {device}")
        return device
    
    def _setup_training(self):
        """Setup PyTorch training components."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.warmup_steps,
            T_mult=2,
            eta_min=1e-7
        )
        
        # Mixed precision scaler
        if self.config.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler()
        
        # DataParallel if specified
        if self.config.use_data_parallel and torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)
            logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        
        logging.info("PyTorch training components initialized")
    
    def _setup_logging(self):
        """Setup PyTorch logging and monitoring."""
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(self.config.tensorboard_dir)
        
        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)
    
    def create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader with optimizations."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            drop_last=False
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single PyTorch training step."""
        self.model.train()
        
        # Move batch to device
        device_batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # PyTorch mixed precision training
        if self.config.use_amp and self.scaler is not None:
            with autocast():
                outputs = self.model(**device_batch)
                loss = outputs.loss
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_val
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard PyTorch training
            outputs = self.model(**device_batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_val
            )
            
            # Optimizer step
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Update scheduler
        self.scheduler.step()
        
        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0]
        
        return {
            'loss': loss.item(),
            'learning_rate': current_lr
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """PyTorch evaluation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                device_batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**device_batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss, 'num_batches': num_batches}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch using PyTorch."""
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # PyTorch profiling
            if self.config.enable_profiling and batch_idx % self.config.profile_interval == 0:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True
                ) as prof:
                    with record_function("training_step"):
                        metrics = self.train_step(batch)
                prof.export_chrome_trace(f"trace_{self.global_step}.json")
            else:
                metrics = self.train_step(batch)
            
            # Record metrics
            epoch_loss += metrics['loss']
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                logging.info(
                    f"Step {self.global_step}, Batch {batch_idx}, "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"LR: {metrics['learning_rate']:.2e}"
                )
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Loss/train', metrics['loss'], self.global_step)
                    self.writer.add_scalar('Learning_rate', metrics['learning_rate'], self.global_step)
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss, 'num_batches': num_batches}
    
    def save_checkpoint(self, filename: str):
        """Save PyTorch checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        filepath = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load PyTorch checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        logging.info(f"Checkpoint loaded from {filepath}")
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using PyTorch model."""
        try:
            # Tokenize input
            inputs = self.model.tokenizer(
                prompt, 
                return_tensors="pt"
            ).to(self.device)
            
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
            
        except Exception as e:
            logging.error(f"Error generating text: {e}")
            return prompt
    
    def close(self):
        """Cleanup PyTorch resources."""
        if self.writer:
            self.writer.close()


class PyTorchDataManager:
    """PyTorch data management utilities."""
    
    @staticmethod
    def split_dataset(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset using PyTorch utilities."""
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for PyTorch DataLoader."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # PyTorch configuration
    config = PyTorchConfig(
        model_name="gpt2",
        max_length=128,
        batch_size=4,
        learning_rate=2e-5,
        num_epochs=3,
        use_amp=True,
        use_tensorboard=True,
        enable_profiling=False
    )
    
    # Initialize PyTorch model
    model = PyTorchModel(config.model_name)
    
    # Initialize PyTorch trainer
    trainer = PyTorchTrainer(model, config)
    
    # Sample data
    sample_texts = [
        "The future of artificial intelligence is promising.",
        "Machine learning algorithms can solve complex problems.",
        "Deep learning models are transforming technology.",
        "Neural networks are powerful computational models.",
        "Transformers have revolutionized natural language processing.",
        "PyTorch is a powerful deep learning framework.",
        "Automatic differentiation is a key feature of PyTorch.",
        "GPU acceleration makes training much faster."
    ]
    
    # Create PyTorch dataset
    dataset = PyTorchDataset(sample_texts, model.tokenizer, config.max_length)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = PyTorchDataManager.split_dataset(dataset)
    
    # Create PyTorch dataloaders
    train_dataloader = trainer.create_dataloader(train_dataset, shuffle=True)
    val_dataloader = trainer.create_dataloader(val_dataset, shuffle=False)
    
    # Training loop
    for epoch in range(config.num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_dataloader)
        trainer.train_losses.append(train_metrics['train_loss'])
        
        # Evaluate
        val_metrics = trainer.evaluate(val_dataloader)
        trainer.val_losses.append(val_metrics['val_loss'])
        
        logging.info(
            f"Epoch {epoch + 1} completed. "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}"
        )
    
    # Generate sample text
    test_prompt = "PyTorch is"
    generated_text = trainer.generate_text(test_prompt, max_length=50)
    print(f"Generated text: {generated_text}")
    
    # Save final checkpoint
    trainer.save_checkpoint("final_checkpoint.pt")
    
    # Cleanup
    trainer.close()





