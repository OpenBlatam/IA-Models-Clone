"""
Optimized NLP System v15.0 - Production Ready
Advanced deep learning with PyTorch, Transformers, and optimization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    pipeline, TrainingArguments, Trainer
)
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from enum import Enum
import json
import os
from torch.profiler import profile, record_function, ProfilerActivity

# GPU optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class ModelType(Enum):
    BERT = "bert"
    GPT2 = "gpt2"
    T5 = "t5"
    CUSTOM = "custom"

@dataclass
class NLPSystemConfig:
    model_name: str = "gpt2"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    mixed_precision: bool = True
    # Multi-GPU settings
    use_data_parallel: bool = False
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    # Performance settings
    gradient_clip_val: float = 1.0
    enable_profiling: bool = False
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001

class OptimizedNLPSystem:
    def __init__(self, config: NLPSystemConfig):
        self.config = config
        self.device = device
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.fp16 else None
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_type: ModelType = ModelType.GPT2):
        """Load optimized model with mixed precision and multi-GPU support"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if self.config.mixed_precision:
                self.model = self.model.half()
                
            # Multi-GPU setup
            if self.config.use_data_parallel and torch.cuda.device_count() > 1:
                self.model = DataParallel(self.model)
                self.logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            elif self.config.use_distributed:
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.config.rank],
                    output_device=self.config.rank
                )
                self.logger.info(f"Using DistributedDataParallel on rank {self.config.rank}")
                
            self.model.to(self.device)
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
            
    def setup_training(self):
        """Setup optimizer and scheduler with advanced techniques"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-7
        )
        
    def train_with_optimization(self, train_dataloader, val_dataloader=None):
        """Train with comprehensive optimization features"""
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            
            # Enable autograd anomaly detection for debugging
            if self.config.enable_profiling:
                torch.autograd.set_detect_anomaly(True)
            
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    # Move batch to device
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Mixed precision training
                    if self.config.fp16:
                        with autocast():
                            outputs = self.model(**inputs)
                            loss = outputs.loss / self.config.gradient_accumulation_steps
                        
                        self.scaler.scale(loss).backward()
                    else:
                        outputs = self.model(**inputs)
                        loss = outputs.loss / self.config.gradient_accumulation_steps
                        loss.backward()
                    
                    epoch_loss += loss.item()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if self.config.fp16:
                            self.scaler.unscale_(self.optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.gradient_clip_val
                        )
                        
                        # Check for NaN/Inf gradients
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any():
                                    self.logger.warning(f"NaN gradient detected in {name}")
                                    param.grad.data.zero_()
                                if torch.isinf(param.grad).any():
                                    self.logger.warning(f"Inf gradient detected in {name}")
                                    param.grad.data.zero_()
                        
                        if self.config.fp16:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        
                        # Logging
                        if batch_idx % 100 == 0:
                            self.logger.info(
                                f"Epoch {epoch}, Batch {batch_idx}, "
                                f"Loss: {loss.item():.4f}, "
                                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                            )
                
                except Exception as e:
                    self.logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            # Validation and early stopping
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                self.logger.info(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss - self.config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model("best_model.pt")
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break
            else:
                self.logger.info(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}")
    
    def evaluate(self, dataloader) -> float:
        """Evaluate model with proper metrics"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    total_loss += outputs.loss.item()
                    num_batches += 1
                except Exception as e:
                    self.logger.error(f"Error in evaluation: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def profile_performance(self, dataloader, num_batches=10):
        """Profile model performance using PyTorch profiler"""
        if not self.config.enable_profiling:
            return
            
        self.model.eval()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                for i, batch in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    try:
                        inputs = {k: v.to(self.device) for k, v in batch.items()}
                        with autocast() if self.config.fp16 else torch.no_grad():
                            _ = self.model(**inputs)
                    except Exception as e:
                        self.logger.error(f"Error in profiling: {e}")
                        continue
        
        # Save profiling results
        prof.export_chrome_trace("nlp_system_trace.json")
        self.logger.info("Performance profiling completed. Check nlp_system_trace.json")
        
    def save_model(self, path: str):
        """Save model with proper error handling"""
        try:
            if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                torch.save(self.model.module.state_dict(), path)
            else:
                torch.save(self.model.state_dict(), path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
            
    def load_model_weights(self, path: str):
        """Load model weights with proper error handling"""
        try:
            state_dict = torch.load(path, map_location=self.device)
            if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.logger.info(f"Model weights loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model weights: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text with optimized inference and error handling"""
        try:
            self.logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.logger.info("Text generation completed successfully")
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error in text generation: {e}")
            return prompt
    
    def batch_generate(self, prompts: List[str], max_length: int = 100) -> List[str]:
        """Batch text generation for efficiency"""
        try:
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            return generated_texts
            
        except Exception as e:
            self.logger.error(f"Error in batch generation: {e}")
            return prompts

class CustomNLPDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
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

class AdvancedNLPTrainer:
    def __init__(self, nlp_system: OptimizedNLPSystem):
        self.nlp_system = nlp_system
        self.logger = logging.getLogger(__name__)
        
    def create_data_loaders(self, texts: List[str], train_ratio: float = 0.8, 
                           val_ratio: float = 0.1, test_ratio: float = 0.1,
                           shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test data loaders with proper splits"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle data
        if shuffle:
            import random
            random.shuffle(texts)
        
        # Calculate split indices
        total_size = len(texts)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Split data
        train_texts = texts[:train_size]
        val_texts = texts[train_size:train_size + val_size]
        test_texts = texts[train_size + val_size:]
        
        self.logger.info(f"Data split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Create datasets
        train_dataset = CustomNLPDataset(train_texts, self.nlp_system.tokenizer, self.nlp_system.config.max_length)
        val_dataset = CustomNLPDataset(val_texts, self.nlp_system.tokenizer, self.nlp_system.config.max_length)
        test_dataset = CustomNLPDataset(test_texts, self.nlp_system.tokenizer, self.nlp_system.config.max_length)
        
        # Create data loaders with proper settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.nlp_system.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.nlp_system.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.nlp_system.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train_with_validation(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train model with validation and comprehensive logging"""
        self.nlp_system.setup_training()
        self.nlp_system.train_with_optimization(train_loader, val_loader)
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics"""
        self.nlp_system.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    inputs = {k: v.to(self.nlp_system.device) for k, v in batch.items()}
                    outputs = self.nlp_system.model(**inputs)
                    
                    total_loss += outputs.loss.item()
                    num_batches += 1
                    
                    # Collect predictions for additional metrics
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_labels.extend(inputs['labels'].cpu().numpy().flatten())
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation: {e}")
                    continue
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Calculate accuracy (for language modeling, this is token-level accuracy)
        if all_predictions and all_labels:
            correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
            accuracy = correct / len(all_predictions) if all_predictions else 0.0
        else:
            accuracy = 0.0
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'num_batches': num_batches
        }
        
        self.logger.info(f"Evaluation Metrics: {metrics}")
        return metrics
    
    def cross_validate(self, texts: List[str], n_folds: int = 5) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
            self.logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Split data for this fold
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            
            # Create datasets
            train_dataset = CustomNLPDataset(train_texts, self.nlp_system.tokenizer, self.nlp_system.config.max_length)
            val_dataset = CustomNLPDataset(val_texts, self.nlp_system.tokenizer, self.nlp_system.config.max_length)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.nlp_system.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.nlp_system.config.batch_size, shuffle=False)
            
            # Train and evaluate
            self.nlp_system.setup_training()
            self.nlp_system.train_with_optimization(train_loader, val_loader)
            
            # Evaluate on validation set
            metrics = self.evaluate_model(val_loader)
            fold_metrics.append(metrics)
            
            self.logger.info(f"Fold {fold + 1} metrics: {metrics}")
        
        # Aggregate results
        aggregated_metrics = {}
        for metric in fold_metrics[0].keys():
            if metric != 'num_batches':
                values = [fold[metric] for fold in fold_metrics]
                aggregated_metrics[metric] = values
        
        self.logger.info(f"Cross-validation results: {aggregated_metrics}")
        return aggregated_metrics

class NLPAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentiment_pipeline = None
        self.classification_pipeline = None
        
    def setup_pipelines(self):
        """Setup analysis pipelines"""
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            self.classification_pipeline = pipeline("text-classification")
            self.logger.info("Analysis pipelines setup complete")
        except Exception as e:
            self.logger.error(f"Error setting up pipelines: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment with error handling"""
        try:
            if self.sentiment_pipeline is None:
                self.setup_pipelines()
            
            result = self.sentiment_pipeline(text)[0]
            return {
                'label': result['label'],
                'score': result['score'],
                'text': text
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {'label': 'ERROR', 'score': 0.0, 'text': text}
    
    def classify_text(self, text: str, candidate_labels: List[str]) -> Dict[str, Union[str, float]]:
        """Classify text with error handling"""
        try:
            if self.classification_pipeline is None:
                self.setup_pipelines()
            
            result = self.classification_pipeline(text, candidate_labels=candidate_labels)[0]
            return {
                'label': result['label'],
                'score': result['score'],
                'text': text
            }
        except Exception as e:
            self.logger.error(f"Error in text classification: {e}")
            return {'label': 'ERROR', 'score': 0.0, 'text': text}

def create_optimized_nlp_system(config: NLPSystemConfig) -> OptimizedNLPSystem:
    """Factory function for creating optimized NLP system"""
    nlp_system = OptimizedNLPSystem(config)
    nlp_system.load_model()
    nlp_system.setup_training()
    return nlp_system

# Example usage
if __name__ == "__main__":
    # Configuration
    config = NLPSystemConfig(
        model_name="gpt2",
        batch_size=8,
        learning_rate=1e-4,
        num_epochs=3,
        fp16=True,
        use_data_parallel=True,
        enable_profiling=True
    )
    
    # Initialize system
    nlp_system = OptimizedNLPSystem(config)
    nlp_system.load_model()
    
    # Example training
    trainer = AdvancedNLPTrainer(nlp_system)
    
    # Sample data
    sample_texts = [
        "This is a sample text for training.",
        "Another example text for the model.",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating."
    ]
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(sample_texts)
    
    # Train model
    trainer.train_with_validation(train_loader, val_loader)
    
    # Evaluate model
    metrics = trainer.evaluate_model(test_loader)
    print(f"Final metrics: {metrics}")
    
    # Generate text
    generated_text = nlp_system.generate_text("The future of AI is")
    print(f"Generated text: {generated_text}")
    
    # Setup analyzer
    analyzer = NLPAnalyzer()
    analyzer.setup_pipelines()
    
    # Analyze sentiment
    sentiment_result = analyzer.analyze_sentiment("I love this NLP system!")
    print(f"Sentiment analysis: {sentiment_result}")
    
    # Classify text
    classification_result = analyzer.classify_text(
        "This is about technology", 
        ["technology", "sports", "politics"]
    )
    print(f"Text classification: {classification_result}")
