from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import logging
from deep_learning_framework import (
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Deep Learning Framework Examples
Comprehensive examples demonstrating the framework usage for various tasks.
"""

    DeepLearningFramework, FrameworkConfig, TaskType,
    BaseModel, BaseDataset, BaseTrainer
)


class TransformerClassificationModel(BaseModel):
    """Transformer-based classification model."""
    
    def __init__(self, config: FrameworkConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Model parameters
        self.vocab_size = config.model_config.get('vocab_size', 50000)
        self.embedding_dim = config.model_config.get('embedding_dim', 768)
        self.num_heads = config.model_config.get('num_heads', 12)
        self.num_layers = config.model_config.get('num_layers', 6)
        self.num_classes = config.model_config.get('num_classes', 2)
        self.max_length = config.model_config.get('max_length', 512)
        self.dropout = config.model_config.get('dropout', 0.1)
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embedding_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, self.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass."""
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings
        
        # Transformer encoding
        if attention_mask is not None:
            # Convert attention mask to transformer format
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        encoded = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # Pooling: use [CLS] token (first token)
        pooled_output = encoded[:, 0, :]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Compute classification loss."""
        return F.cross_entropy(outputs, targets)


class TextClassificationDataset(BaseDataset):
    """Dataset for text classification tasks."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        
    """__init__ function."""
super().__init__("text_data")
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def _load_data(self) -> Any:
        """Load and tokenize data."""
        return list(zip(self.texts, self.labels))
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get tokenized text and label."""
        text, label = self.data[index]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def __len__(self) -> int:
        return len(self.data)


class TransformerClassificationTrainer(BaseTrainer):
    """Trainer for transformer classification models."""
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step for transformer classification."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        if self.config.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, attention_mask)
                loss = self.model.compute_loss(outputs, labels)
            
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
        else:
            outputs = self.model(input_ids, attention_mask)
            loss = self.model.compute_loss(outputs, labels)
            
            loss.backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
        
        self.global_step += 1
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        
        return {'loss': loss.item(), 'metric': accuracy}
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validation step for transformer classification."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            loss = self.model.compute_loss(outputs, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == labels).float().mean().item()
        
        return {'loss': loss.item(), 'metric': accuracy}


class DiffusionModel(BaseModel):
    """Diffusion model for image generation."""
    
    def __init__(self, config: FrameworkConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Model parameters
        self.image_size = config.model_config.get('image_size', 64)
        self.in_channels = config.model_config.get('in_channels', 3)
        self.model_channels = config.model_config.get('model_channels', 128)
        self.num_res_blocks = config.model_config.get('num_res_blocks', 2)
        self.dropout = config.model_config.get('dropout', 0.1)
        
        # Time embedding
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # UNet architecture
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling
        ch = self.model_channels
        input_block_chans = [ch]
        for level, mult in enumerate([1, 2, 4, 8]):
            for _ in range(self.num_res_blocks):
                layers = [ResNetBlock(ch, time_embed_dim, self.dropout)]
                ch = mult * self.model_channels
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != 3:
                self.input_blocks.append(
                    nn.ModuleList([ResNetBlock(ch, time_embed_dim, self.dropout)])
                )
                input_block_chans.append(ch)
                self.input_blocks.append(
                    nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)])
                )
                input_block_chans.append(ch)
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ResNetBlock(ch, time_embed_dim, self.dropout),
            ResNetBlock(ch, time_embed_dim, self.dropout)
        ])
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate([1, 2, 4, 8]))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResNetBlock(ch + ich, time_embed_dim, self.dropout)]
                ch = mult * self.model_channels
                if level and i == self.num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 3, stride=2, padding=1, output_padding=1))
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, self.in_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass through diffusion model."""
        # Time embedding
        emb = self.time_embed(timesteps)
        
        # Downsampling
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResNetBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle block
        for module in self.middle_block:
            h = module(h, emb)
        
        # Upsampling
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResNetBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
        
        return self.out(h)
    
    def compute_loss(self, noise_pred: torch.Tensor, noise: torch.Tensor):
        """Compute diffusion loss."""
        return F.mse_loss(noise_pred, noise, reduction="mean")


class ResNetBlock(nn.Module):
    """ResNet block for diffusion models."""
    
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, channels)
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.skip_connection = nn.Conv2d(channels, channels, 1) if channels != channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out.unsqueeze(-1).unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class DiffusionTrainer(BaseTrainer):
    """Trainer for diffusion models."""
    
    def __init__(self, model: BaseModel, config: FrameworkConfig):
        
    """__init__ function."""
super().__init__(model, config)
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
    
    def training_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Training step for diffusion model."""
        clean_images = batch.to(self.device)
        batch_size = clean_images.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to images
        noise = torch.randn_like(clean_images)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        if self.config.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                noise_pred = self.model(noisy_images, timesteps)
                loss = self.model.compute_loss(noise_pred, noise)
            
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
        else:
            noise_pred = self.model(noisy_images, timesteps)
            loss = self.model.compute_loss(noise_pred, noise)
            
            loss.backward()
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
        
        self.global_step += 1
        
        return {'loss': loss.item(), 'metric': loss.item()}
    
    def validation_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Validation step for diffusion model."""
        clean_images = batch.to(self.device)
        batch_size = clean_images.shape[0]
        
        with torch.no_grad():
            timesteps = torch.randint(
                0, self.noise_scheduler.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()
            
            noise = torch.randn_like(clean_images)
            noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            noise_pred = self.model(noisy_images, timesteps)
            loss = self.model.compute_loss(noise_pred, noise)
        
        return {'loss': loss.item(), 'metric': loss.item()}


def run_text_classification_experiment():
    """Run text classification experiment."""
    # Setup configuration
    config = FrameworkConfig(
        task_type=TaskType.CLASSIFICATION,
        model_name="transformer_classification",
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=10,
        experiment_name="text_classification_experiment",
        model_config={
            'vocab_size': 10000,
            'embedding_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'num_classes': 2,
            'max_length': 128,
            'dropout': 0.1
        }
    )
    
    # Create framework
    framework = DeepLearningFramework(config)
    
    # Setup model
    model = framework.setup_model(TransformerClassificationModel)
    
    # Create dummy data
    sample_texts = [
        "This is a positive review. I love this product!",
        "This is terrible. Worst purchase ever.",
        "Amazing service and quality. Highly recommended!",
        "Poor quality and bad customer service.",
        "Great experience, will buy again.",
        "Disappointed with the product quality."
    ]
    sample_labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
    
    # Mock tokenizer
    class MockTokenizer:
        def __call__(self, text, **kwargs) -> Any:
            # Simple tokenization
            tokens = text.split()[:kwargs.get('max_length', 128)]
            input_ids = [hash(token) % 10000 for token in tokens]
            attention_mask = [1] * len(input_ids)
            
            # Padding
            while len(input_ids) < kwargs.get('max_length', 128):
                input_ids.append(0)
                attention_mask.append(0)
            
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
    
    tokenizer = MockTokenizer()
    
    # Create datasets
    train_dataset = TextClassificationDataset(sample_texts, sample_labels, tokenizer, max_length=128)
    val_dataset = TextClassificationDataset(sample_texts, sample_labels, tokenizer, max_length=128)
    
    framework.setup_data(train_dataset, val_dataset)
    
    # Setup trainer
    trainer = framework.setup_trainer(TransformerClassificationTrainer)
    
    # Train
    trainer = framework.train()
    
    return framework, trainer


def run_diffusion_experiment():
    """Run diffusion model experiment."""
    # Setup configuration
    config = FrameworkConfig(
        task_type=TaskType.DIFFUSION,
        model_name="diffusion_model",
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=50,
        experiment_name="diffusion_experiment",
        model_config={
            'image_size': 32,
            'in_channels': 3,
            'model_channels': 64,
            'num_res_blocks': 2,
            'dropout': 0.1
        }
    )
    
    # Create framework
    framework = DeepLearningFramework(config)
    
    # Setup model
    model = framework.setup_model(DiffusionModel)
    
    # Create dummy image data
    class DummyImageDataset(BaseDataset):
        def _load_data(self) -> Any:
            return np.random.randn(100, 3, 32, 32)
        
        def __getitem__(self, index) -> Optional[Dict[str, Any]]:
            return torch.randn(3, 32, 32)
        
        def __len__(self) -> Any:
            return 100
    
    train_dataset = DummyImageDataset("dummy_images")
    val_dataset = DummyImageDataset("dummy_images")
    
    framework.setup_data(train_dataset, val_dataset)
    
    # Setup trainer
    trainer = framework.setup_trainer(DiffusionTrainer)
    
    # Train
    trainer = framework.train()
    
    return framework, trainer


def run_comprehensive_experiment():
    """Run comprehensive experiment with multiple models."""
    experiments = [
        {
            'name': 'text_classification',
            'task': TaskType.CLASSIFICATION,
            'model_class': TransformerClassificationModel,
            'trainer_class': TransformerClassificationTrainer,
            'config': {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'num_epochs': 5,
                'model_config': {
                    'vocab_size': 5000,
                    'embedding_dim': 128,
                    'num_heads': 4,
                    'num_layers': 2,
                    'num_classes': 2,
                    'max_length': 64,
                    'dropout': 0.1
                }
            }
        },
        {
            'name': 'diffusion_generation',
            'task': TaskType.DIFFUSION,
            'model_class': DiffusionModel,
            'trainer_class': DiffusionTrainer,
            'config': {
                'learning_rate': 1e-4,
                'batch_size': 4,
                'num_epochs': 10,
                'model_config': {
                    'image_size': 16,
                    'in_channels': 3,
                    'model_channels': 32,
                    'num_res_blocks': 1,
                    'dropout': 0.1
                }
            }
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        
        # Setup configuration
        config = FrameworkConfig(
            task_type=exp['task'],
            model_name=exp['name'],
            experiment_name=f"{exp['name']}_experiment",
            **exp['config']
        )
        
        # Create framework
        framework = DeepLearningFramework(config)
        
        # Setup model
        model = framework.setup_model(exp['model_class'])
        
        # Create dummy data based on task type
        if exp['task'] == TaskType.CLASSIFICATION:
            # Text classification data
            sample_texts = ["positive text", "negative text"] * 50
            sample_labels = [1, 0] * 50
            
            class MockTokenizer:
                def __call__(self, text, **kwargs) -> Any:
                    tokens = text.split()[:kwargs.get('max_length', 64)]
                    input_ids = [hash(token) % 5000 for token in tokens]
                    attention_mask = [1] * len(input_ids)
                    
                    while len(input_ids) < kwargs.get('max_length', 64):
                        input_ids.append(0)
                        attention_mask.append(0)
                    
                    return {
                        'input_ids': torch.tensor([input_ids]),
                        'attention_mask': torch.tensor([attention_mask])
                    }
            
            tokenizer = MockTokenizer()
            train_dataset = TextClassificationDataset(sample_texts, sample_labels, tokenizer, max_length=64)
            val_dataset = TextClassificationDataset(sample_texts, sample_labels, tokenizer, max_length=64)
            
        elif exp['task'] == TaskType.DIFFUSION:
            # Image generation data
            class DummyImageDataset(BaseDataset):
                def _load_data(self) -> Any:
                    return np.random.randn(50, 3, 16, 16)
                
                def __getitem__(self, index) -> Optional[Dict[str, Any]]:
                    return torch.randn(3, 16, 16)
                
                def __len__(self) -> Any:
                    return 50
            
            train_dataset = DummyImageDataset("dummy_images")
            val_dataset = DummyImageDataset("dummy_images")
        
        framework.setup_data(train_dataset, val_dataset)
        
        # Setup trainer
        trainer = framework.setup_trainer(exp['trainer_class'])
        
        # Train
        trainer = framework.train()
        
        results[exp['name']] = {
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
            'best_metric': trainer.best_metric
        }
    
    return results


if __name__ == "__main__":
    # Run comprehensive experiments
    print("Running comprehensive deep learning framework experiments...")
    results = run_comprehensive_experiment()
    
    print("\nExperiment Results:")
    for exp_name, exp_results in results.items():
        print(f"\n{exp_name}:")
        print(f"  Final Train Loss: {exp_results['final_train_loss']:.6f}")
        print(f"  Final Val Loss: {exp_results['final_val_loss']:.6f}")
        print(f"  Best Metric: {exp_results['best_metric']:.6f}") 