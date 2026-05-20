from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torchvision.transforms as transforms
from PIL import Image
import math
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import gc
from pathlib import Path
from tqdm import tqdm
import re
import unicodedata
from collections import Counter
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Diffusion System with Text Tokenization and Sequence Handling
Production-ready diffusion models with proper text processing, tokenization, and sequence management.
"""


logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for advanced diffusion models."""
    # Model configuration
    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 128
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.1
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    conv_resample: bool = True
    num_heads: int = 4
    use_spatial_transformer: bool = True
    transformer_depth: int = 1
    context_dim: Optional[int] = None
    legacy: bool = False
    
    # Text processing configuration
    max_text_length: int = 77
    text_encoder_dim: int = 768
    text_encoder_layers: int = 12
    text_encoder_heads: int = 12
    text_encoder_dropout: float = 0.1
    vocab_size: int = 49408  # CLIP vocabulary size
    pad_token_id: int = 0
    eos_token_id: int = 49407
    unk_token_id: int = 49408
    
    # Diffusion configuration
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_diffusion_timesteps: int = 1000
    schedule: str = "linear"  # linear, cosine
    
    # Training configuration
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Mixed precision and optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # Output configuration
    output_dir: str = "./diffusion_outputs"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Generation configuration
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0


class TextTokenizer:
    """Advanced text tokenizer with proper sequence handling."""
    
    def __init__(self, config: DiffusionConfig):
        
    """__init__ function."""
self.config = config
        self.vocab_size = config.vocab_size
        self.max_length = config.max_text_length
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.unk_token_id = config.unk_token_id
        
        # Simple vocabulary (in practice, you'd use a proper tokenizer like BPE)
        self.vocab = self._create_vocabulary()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Special tokens
        self.special_tokens = {
            '<pad>': self.pad_token_id,
            '<eos>': self.eos_token_id,
            '<unk>': self.unk_token_id,
            '<start>': 1,
            '<end>': 2
        }
    
    def _create_vocabulary(self) -> List[str]:
        """Create a simple vocabulary for demonstration."""
        # Basic vocabulary with common words and characters
        vocab = ['<pad>', '<start>', '<end>', '<unk>', '<eos>']
        
        # Add common words
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        ]
        
        vocab.extend(common_words)
        
        # Add individual characters
        for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            vocab.append(char)
        
        # Add punctuation
        vocab.extend(['.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}'])
        
        # Pad to vocab_size
        while len(vocab) < self.vocab_size:
            vocab.append(f'<extra_{len(vocab)}>')
        
        return vocab[:self.vocab_size]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent tokenization."""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        text = self._normalize_text(text)
        
        # Simple word-based tokenization
        tokens = []
        words = text.split()
        
        for word in words:
            # Handle punctuation
            if re.match(r'^[^\w\s]+$', word):
                tokens.append(word)
            else:
                # Split word into characters if not in vocabulary
                if word in self.token_to_id:
                    tokens.append(word)
                else:
                    tokens.extend(list(word))
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize_text(text)
        
        if add_special_tokens:
            tokens = ['<start>'] + tokens + ['<eos>']
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.unk_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append('<unk>')
        
        # Join tokens
        text = ' '.join(tokens)
        
        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def __call__(self, texts: Union[str, List[str]], 
                padding: bool = True, 
                truncation: bool = True,
                max_length: Optional[int] = None,
                return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize texts with proper padding and truncation."""
        if isinstance(texts, str):
            texts = [texts]
        
        max_length = max_length or self.max_length
        
        # Encode all texts
        encoded_texts = []
        for text in texts:
            token_ids = self.encode(text)
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            encoded_texts.append(token_ids)
        
        # Pad sequences
        if padding:
            max_len = max(len(seq) for seq in encoded_texts)
            max_len = min(max_len, max_length)
            
            padded_texts = []
            attention_masks = []
            
            for token_ids in encoded_texts:
                if len(token_ids) < max_len:
                    padding_length = max_len - len(token_ids)
                    padded_ids = token_ids + [self.pad_token_id] * padding_length
                    attention_mask = [1] * len(token_ids) + [0] * padding_length
                else:
                    padded_ids = token_ids[:max_len]
                    attention_mask = [1] * max_len
                
                padded_texts.append(padded_ids)
                attention_masks.append(attention_mask)
            
            encoded_texts = padded_texts
        else:
            attention_masks = [[1] * len(seq) for seq in encoded_texts]
        
        # Convert to tensors
        if return_tensors == "pt":
            input_ids = torch.tensor(encoded_texts, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        else:
            input_ids = encoded_texts
            attention_mask = attention_masks
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class TextEncoder(nn.Module):
    """Text encoder for diffusion models with proper sequence handling."""
    
    def __init__(self, config: DiffusionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_encoder_dim)
        
        # Position embedding
        self.position_embedding = nn.Embedding(config.max_text_length, config.text_encoder_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.text_encoder_dim,
            nhead=config.text_encoder_heads,
            dim_feedforward=config.text_encoder_dim * 4,
            dropout=config.text_encoder_dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.text_encoder_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.text_encoder_dim)
        
        # Projection to context dimension
        if config.context_dim:
            self.context_projection = nn.Linear(config.text_encoder_dim, config.context_dim)
        else:
            self.context_projection = nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using proper techniques."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through text encoder."""
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Create attention mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)
        
        # Convert to transformer format
        transformer_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        transformer_mask = (1.0 - transformer_mask) * -10000.0
        
        # Pass through transformer
        encoded_text = self.transformer(embeddings, src_key_padding_mask=attention_mask == 0)
        
        # Layer normalization
        encoded_text = self.layer_norm(encoded_text)
        
        # Project to context dimension
        context = self.context_projection(encoded_text)
        
        return context


class AdvancedDiffusionDataset(Dataset):
    """Dataset for diffusion models with text conditioning."""
    
    def __init__(self, image_paths: List[str], texts: List[str], 
                 tokenizer: TextTokenizer, config: DiffusionConfig):
        
    """__init__ function."""
self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = tokenizer
        self.config = config
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Validate data
        assert len(image_paths) == len(texts), "Number of images and texts must match"
    
    def __len__(self) -> Any:
        return len(self.image_paths)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        image = self.transform(image)
        
        # Tokenize text
        text = self.texts[idx]
        text_encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt"
        )
        
        return {
            'image': image,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'text': text
        }


class AdvancedDiffusionSystem:
    """Advanced diffusion system with text conditioning and proper sequence handling."""
    
    def __init__(self, config: DiffusionConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = TextTokenizer(config)
        self.text_encoder = TextEncoder(config)
        
        # Initialize model (you would implement the actual diffusion model here)
        self.model = self._create_diffusion_model()
        
        # Move to device
        self.text_encoder.to(self.device)
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.text_encoder.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        logger.info(f"Advanced diffusion system initialized on device: {self.device}")
        logger.info(f"Text encoder parameters: {sum(p.numel() for p in self.text_encoder.parameters()):,}")
    
    def _create_diffusion_model(self) -> nn.Module:
        """Create the diffusion model (placeholder for actual implementation)."""
        # This would be your actual diffusion model implementation
        # For now, we'll create a simple placeholder
        class PlaceholderDiffusionModel(nn.Module):
            def __init__(self, config) -> Any:
                super().__init__()
                self.config = config
                # Add your diffusion model layers here
                self.conv1 = nn.Conv2d(config.in_channels, config.model_channels, 3, padding=1)
                self.conv2 = nn.Conv2d(config.model_channels, config.out_channels, 3, padding=1)
            
            def forward(self, x, timesteps, context=None) -> Any:
                # Placeholder forward pass
                x = F.relu(self.conv1(x))
                x = self.conv2(x)
                return x
        
        return PlaceholderDiffusionModel(self.config)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with text conditioning."""
        images = batch['image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        batch_size = images.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.config.num_diffusion_timesteps, (batch_size,), device=self.device)
        
        # Encode text
        with torch.cuda.amp.autocast() if self.config.fp16 else torch.no_grad():
            text_context = self.text_encoder(input_ids, attention_mask)
        
        # Add noise to images (simplified)
        noise = torch.randn_like(images)
        noisy_images = images + noise * 0.1  # Simplified noise addition
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.config.fp16 else torch.no_grad():
            noise_pred = self.model(noisy_images, timesteps, text_context)
            loss = F.mse_loss(noise_pred, noise, reduction='mean')
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def generate_image(self, prompt: str, batch_size: int = 1, 
                      guidance_scale: Optional[float] = None) -> torch.Tensor:
        """Generate image from text prompt."""
        self.model.eval()
        self.text_encoder.eval()
        
        guidance_scale = guidance_scale or self.config.guidance_scale
        
        # Tokenize prompt
        text_encoding = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt"
        )
        
        input_ids = text_encoding['input_ids'].to(self.device)
        attention_mask = text_encoding['attention_mask'].to(self.device)
        
        # Encode text
        with torch.no_grad():
            text_context = self.text_encoder(input_ids, attention_mask)
        
        # Start from pure noise
        x = torch.randn(batch_size, self.config.in_channels, self.config.image_size, 
                       self.config.image_size, device=self.device)
        
        # Denoising loop (simplified)
        with torch.no_grad():
            for t in tqdm(reversed(range(self.config.num_inference_steps)), desc="Generating"):
                timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self.model(x, timesteps, text_context)
                
                # Apply classifier-free guidance if scale > 1
                if guidance_scale > 1.0:
                    # Unconditional prediction (you'd need to implement this)
                    uncond_noise_pred = noise_pred  # Placeholder
                    noise_pred = uncond_noise_pred + guidance_scale * (noise_pred - uncond_noise_pred)
                
                # Denoising step (simplified)
                x = x - 0.1 * noise_pred
        
        # Denormalize
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        
        return x
    
    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(path, exist_ok=True)
        
        # Save model states
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config.__dict__
        }, os.path.join(path, 'diffusion_model.pth'))
        
        # Save tokenizer vocabulary
        tokenizer_path = os.path.join(path, 'tokenizer.json')
        with open(tokenizer_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump({
                'vocab': self.tokenizer.vocab,
                'token_to_id': self.tokenizer.token_to_id,
                'special_tokens': self.tokenizer.special_tokens
            }, f, indent=2)
        
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(os.path.join(path, 'diffusion_model.pth'), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load tokenizer vocabulary
        tokenizer_path = os.path.join(path, 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                tokenizer_data = json.load(f)
                self.tokenizer.vocab = tokenizer_data['vocab']
                self.tokenizer.token_to_id = tokenizer_data['token_to_id']
                self.tokenizer.special_tokens = tokenizer_data['special_tokens']
        
        logger.info(f"Model loaded from: {path}")


def create_advanced_diffusion_system(image_size: int = 256, 
                                   text_encoder_dim: int = 768,
                                   use_fp16: bool = True) -> AdvancedDiffusionSystem:
    """Create an advanced diffusion system with text conditioning."""
    config = DiffusionConfig(
        image_size=image_size,
        text_encoder_dim=text_encoder_dim,
        fp16=use_fp16,
        batch_size=2 if use_fp16 else 4,
        num_epochs=10
    )
    return AdvancedDiffusionSystem(config)


# Example usage
if __name__ == "__main__":
    # Create advanced diffusion system
    diffusion_system = create_advanced_diffusion_system()
    
    # Sample training data (placeholder)
    sample_images = torch.randn(2, 3, 256, 256)
    sample_texts = ["A beautiful sunset over the mountains", "A cat sitting on a windowsill"]
    
    # Tokenize texts
    tokenizer = diffusion_system.tokenizer
    text_encoding = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    batch = {
        'image': sample_images,
        'input_ids': text_encoding['input_ids'],
        'attention_mask': text_encoding['attention_mask'],
        'text': sample_texts
    }
    
    # Training step
    loss_info = diffusion_system.train_step(batch)
    print(f"Training loss: {loss_info['loss']:.4f}")
    
    # Generate image from text
    prompt = "A beautiful landscape with mountains and trees"
    generated_image = diffusion_system.generate_image(prompt, batch_size=1)
    print(f"Generated image shape: {generated_image.shape}")
    
    # Save model
    diffusion_system.save_model("./advanced_diffusion_checkpoint") 