"""
Best Practices Implementation - Deep Learning Workflows
Object-oriented programming for models, functional programming for data pipelines
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from functools import partial, reduce
import numpy as np
import logging

# Object-Oriented Programming for Model Architectures
class OptimizedTransformerModel(nn.Module):
    """Object-oriented transformer model with best practices"""
    
    def __init__(self, config: Dict[str, any]):
        super().__init__()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16 if config.get('fp16', True) else torch.float32
        )
        self._setup_optimization()
    
    def _setup_optimization(self):
        """Setup model optimizations"""
        if self.config.get('compile_model', False):
            self.model = torch.compile(self.model)
        if self.config.get('fp16', True):
            self.model = self.model.half()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)

class AdvancedDataset(Dataset):
    """Object-oriented dataset with efficient data handling"""
    
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

# Functional Programming for Data Processing Pipelines
def create_data_pipeline(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    batch_size: int = 16,
    shuffle: bool = True
) -> DataLoader:
    """Functional data processing pipeline"""
    
    # Pure function for text preprocessing
    def preprocess_text(text: str) -> str:
        return text.strip().lower()
    
    # Pure function for tokenization
    def tokenize_text(text: str) -> Dict[str, torch.Tensor]:
        return tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Functional pipeline: map -> filter -> create dataset
    processed_texts = list(map(preprocess_text, texts))
    tokenized_data = list(map(tokenize_text, processed_texts))
    
    # Create dataset and dataloader
    dataset = AdvancedDataset(processed_texts, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

def create_training_pipeline(
    model: nn.Module,
    optimizer_fn: Callable,
    scheduler_fn: Callable,
    loss_fn: Callable
) -> Callable:
    """Functional training pipeline factory"""
    
    def training_step(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Pure training step function"""
        model.train()
        outputs = model(**batch)
        loss = loss_fn(outputs.loss)
        
        return {
            'loss': loss.item(),
            'logits': outputs.logits
        }
    
    def validation_step(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Pure validation step function"""
        model.eval()
        with torch.no_grad():
            outputs = model(**batch)
            loss = loss_fn(outputs.loss)
        
        return {
            'val_loss': loss.item(),
            'val_logits': outputs.logits
        }
    
    return training_step, validation_step

# Configuration with dataclasses
@dataclass
class TrainingConfig:
    """Immutable configuration using dataclass"""
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    fp16: bool = True
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

# Functional utilities
def compose(*functions):
    """Function composition utility"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions)

def pipeline(*functions):
    """Pipeline utility for data processing"""
    def apply_pipeline(data):
        return reduce(lambda x, f: f(x), functions, data)
    return apply_pipeline

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'model_name': 'gpt2',
        'fp16': True,
        'compile_model': False
    }
    
    # Object-oriented model
    model = OptimizedTransformerModel(config)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Functional data pipeline
    sample_texts = ["Hello world", "Deep learning", "Transformers"]
    dataloader = create_data_pipeline(sample_texts, tokenizer)
    
    # Functional training pipeline
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    loss_fn = lambda x: x
    
    training_step, validation_step = create_training_pipeline(
        model, optimizer, scheduler, loss_fn
    )
    
    # Training loop with functional approach
    for batch in dataloader:
        metrics = training_step(batch)
        print(f"Loss: {metrics['loss']:.4f}")





