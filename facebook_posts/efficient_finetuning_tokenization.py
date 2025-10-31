#!/usr/bin/env python3
"""
Efficient Fine-tuning and Tokenization System
Advanced LoRA, P-tuning, and tokenization techniques for optimal model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding,
    PreTrainedTokenizer, PreTrainedModel
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ===== EFFICIENT FINE-TUNING TECHNIQUES =====

class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for efficient fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, 
                 alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize B with zeros for stable training
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        return self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self, linear_layer: nn.Linear, rank: int = 16, 
                 alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features, 
            rank, alpha, dropout
        )
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original and LoRA weights."""
        return self.linear(x) + self.lora(x)

class LoRAModelWrapper(nn.Module):
    """Wrapper for applying LoRA to pre-trained models."""
    
    def __init__(self, model: PreTrainedModel, rank: int = 16, 
                 alpha: float = 32.0, target_modules: List[str] = None):
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        
        # Apply LoRA to target modules
        self._apply_lora_to_model()
    
    def _apply_lora_to_model(self):
        """Apply LoRA to specified modules in the model."""
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA version
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name, 
                           LoRALinear(module, self.rank, self.alpha))
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model with LoRA."""
        return self.model(*args, **kwargs)

class PromptEmbedding(nn.Module):
    """Learnable prompt embeddings for P-tuning."""
    
    def __init__(self, num_tokens: int, hidden_size: int, 
                 prompt_length: int = 20, dropout: float = 0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.prompt_length = prompt_length
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, hidden_size) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        
        # LSTM for prompt encoding
        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2, 
            num_layers=2, bidirectional=True, 
            batch_first=True
        )
        
        # MLP for prompt transformation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate prompt embeddings."""
        # Expand prompt embeddings to batch size
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Apply LSTM encoding
        lstm_out, _ = self.lstm(prompt_embeds)
        
        # Apply MLP transformation
        transformed = self.mlp(lstm_out)
        
        return self.dropout(transformed)

class P_TuningModelWrapper(nn.Module):
    """Wrapper for P-tuning with pre-trained models."""
    
    def __init__(self, model: PreTrainedModel, prompt_length: int = 20,
                 hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.model = model
        self.prompt_embedding = PromptEmbedding(
            model.config.vocab_size,
            hidden_size,
            prompt_length,
            dropout
        )
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """Forward pass with P-tuning prompt embeddings."""
        batch_size = input_ids.shape[0]
        
        # Generate prompt embeddings
        prompt_embeds = self.prompt_embedding(batch_size)
        
        # Get model embeddings
        model_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Concatenate prompt and input embeddings
        combined_embeds = torch.cat([prompt_embeds, model_embeds], dim=1)
        
        # Update attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, prompt_embeds.shape[1], 
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # Forward through model
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            **kwargs
        )
        
        return outputs

class AdaLoRALayer(nn.Module):
    """Adaptive LoRA layer with dynamic rank adjustment."""
    
    def __init__(self, in_features: int, out_features: int, 
                 max_rank: int = 64, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.max_rank = max_rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        
        # Adaptive rank parameters
        self.rank_importance = nn.Parameter(torch.ones(max_rank))
        self.current_rank = max_rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(max_rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))
        
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive rank."""
        # Get current rank based on importance
        importance_scores = F.softmax(self.rank_importance, dim=0)
        active_rank = min(self.current_rank, self.max_rank)
        
        # Use only top-k components
        top_k_indices = torch.topk(importance_scores, active_rank).indices
        
        lora_A_active = self.lora_A[top_k_indices]
        lora_B_active = self.lora_B[:, top_k_indices]
        
        scaling = self.alpha / active_rank
        return self.dropout(x) @ lora_A_active.T @ lora_B_active.T * scaling
    
    def update_rank(self, new_rank: int):
        """Update the current rank dynamically."""
        self.current_rank = min(new_rank, self.max_rank)

class AdaLoRAModelWrapper(nn.Module):
    """Wrapper for Adaptive LoRA with dynamic rank adjustment."""
    
    def __init__(self, model: PreTrainedModel, max_rank: int = 64,
                 alpha: float = 32.0, target_modules: List[str] = None):
        super().__init__()
        self.model = model
        self.max_rank = max_rank
        self.alpha = alpha
        self.target_modules = target_modules or ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        
        # Apply AdaLoRA to target modules
        self._apply_adalora_to_model()
    
    def _apply_adalora_to_model(self):
        """Apply AdaLoRA to specified modules."""
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name,
                           AdaLoRALayer(module.in_features, module.out_features,
                                       self.max_rank, self.alpha))
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model with AdaLoRA."""
        return self.model(*args, **kwargs)
    
    def update_ranks(self, new_rank: int):
        """Update ranks for all AdaLoRA layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, AdaLoRALayer):
                module.update_rank(new_rank)

# ===== ADVANCED TOKENIZATION AND SEQUENCE HANDLING =====

class LinkedInTokenizer:
    """Advanced tokenizer for LinkedIn content with special handling."""
    
    def __init__(self, base_tokenizer_name: str = "distilbert-base-uncased",
                 max_length: int = 512, special_tokens: Dict[str, str] = None):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        self.max_length = max_length
        
        # LinkedIn-specific special tokens
        self.special_tokens = special_tokens or {
            'hashtag': '[HASHTAG]',
            'mention': '[MENTION]',
            'link': '[LINK]',
            'emoji': '[EMOJI]',
            'industry': '[INDUSTRY]',
            'company': '[COMPANY]',
            'position': '[POSITION]'
        }
        
        # Add special tokens to vocabulary
        self._add_special_tokens()
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _add_special_tokens(self):
        """Add LinkedIn-specific special tokens to tokenizer."""
        special_tokens_list = list(self.special_tokens.values())
        self.base_tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens_list
        })
    
    def _compile_patterns(self):
        """Compile regex patterns for content extraction."""
        self.patterns = {
            'hashtag': re.compile(r'#\w+'),
            'mention': re.compile(r'@\w+'),
            'link': re.compile(r'https?://[^\s]+'),
            'emoji': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]'),
            'industry_keywords': re.compile(r'\b(tech|finance|healthcare|education|manufacturing|retail|consulting|non-profit)\b', re.IGNORECASE),
            'company_pattern': re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Group)\b'),
            'position_pattern': re.compile(r'\b(CEO|CTO|CFO|VP|Director|Manager|Lead|Senior|Junior|Associate)\b')
        }
    
    def preprocess_content(self, content: str) -> str:
        """Preprocess LinkedIn content for tokenization."""
        processed_content = content
        
        # Replace patterns with special tokens
        processed_content = self.patterns['hashtag'].sub(
            lambda m: f" {self.special_tokens['hashtag']} {m.group()} ", processed_content
        )
        
        processed_content = self.patterns['mention'].sub(
            lambda m: f" {self.special_tokens['mention']} {m.group()} ", processed_content
        )
        
        processed_content = self.patterns['link'].sub(
            lambda m: f" {self.special_tokens['link']} ", processed_content
        )
        
        processed_content = self.patterns['emoji'].sub(
            lambda m: f" {self.special_tokens['emoji']} ", processed_content
        )
        
        # Clean up whitespace
        processed_content = re.sub(r'\s+', ' ', processed_content).strip()
        
        return processed_content
    
    def tokenize(self, content: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize LinkedIn content with special handling."""
        # Preprocess content
        processed_content = self.preprocess_content(content)
        
        # Tokenize
        tokenized = self.base_tokenizer(
            processed_content,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
        
        return tokenized
    
    def extract_content_elements(self, content: str) -> Dict[str, List[str]]:
        """Extract various content elements from LinkedIn posts."""
        elements = {
            'hashtags': self.patterns['hashtag'].findall(content),
            'mentions': self.patterns['mention'].findall(content),
            'links': self.patterns['link'].findall(content),
            'emojis': self.patterns['emoji'].findall(content),
            'industries': self.patterns['industry_keywords'].findall(content),
            'companies': self.patterns['company_pattern'].findall(content),
            'positions': self.patterns['position_pattern'].findall(content)
        }
        
        return elements

class LinkedInSequenceHandler:
    """Advanced sequence handling for LinkedIn content."""
    
    def __init__(self, tokenizer: LinkedInTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for input sequences."""
        return (input_ids != self.tokenizer.base_tokenizer.pad_token_id).long()
    
    def pad_sequences(self, sequences: List[torch.Tensor], 
                     padding: str = "max_length") -> torch.Tensor:
        """Pad sequences to uniform length."""
        if padding == "max_length":
            padded = torch.zeros(len(sequences), self.max_length, dtype=torch.long)
            for i, seq in enumerate(sequences):
                length = min(len(seq), self.max_length)
                padded[i, :length] = seq[:length]
            return padded
        else:
            # Dynamic padding
            max_len = max(len(seq) for seq in sequences)
            padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
            for i, seq in enumerate(sequences):
                padded[i, :len(seq)] = seq
            return padded
    
    def truncate_sequences(self, sequences: List[torch.Tensor], 
                          strategy: str = "longest_first") -> List[torch.Tensor]:
        """Truncate sequences to fit within max length."""
        if strategy == "longest_first":
            # Sort by length and truncate longest first
            sorted_indices = sorted(range(len(sequences)), 
                                  key=lambda i: len(sequences[i]), reverse=True)
            
            truncated = []
            for idx in sorted_indices:
                seq = sequences[idx]
                if len(seq) > self.max_length:
                    # Keep start and end, truncate middle
                    half_length = self.max_length // 2
                    truncated_seq = torch.cat([
                        seq[:half_length],
                        seq[-half_length:]
                    ])
                else:
                    truncated_seq = seq
                truncated.append(truncated_seq)
            
            # Restore original order
            result = [None] * len(sequences)
            for i, idx in enumerate(sorted_indices):
                result[idx] = truncated[i]
            return result
        
        else:
            # Simple truncation
            return [seq[:self.max_length] if len(seq) > self.max_length else seq 
                   for seq in sequences]
    
    def create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create position IDs for sequences."""
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        return position_ids
    
    def create_token_type_ids(self, input_ids: torch.Tensor, 
                            segment_boundaries: List[int] = None) -> torch.Tensor:
        """Create token type IDs for multi-segment sequences."""
        batch_size, seq_len = input_ids.shape
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        if segment_boundaries:
            for i, boundary in enumerate(segment_boundaries):
                if boundary < seq_len:
                    token_type_ids[i, boundary:] = 1
        
        return token_type_ids

class LinkedInDataCollator:
    """Custom data collator for LinkedIn content."""
    
    def __init__(self, tokenizer: LinkedInTokenizer, 
                 sequence_handler: LinkedInSequenceHandler):
        self.tokenizer = tokenizer
        self.sequence_handler = sequence_handler
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of LinkedIn content samples."""
        # Extract content from batch
        contents = [sample['content'] for sample in batch]
        labels = [sample.get('label', 0) for sample in batch]
        
        # Tokenize all contents
        tokenized_batch = [self.tokenizer.tokenize(content) for content in contents]
        
        # Extract input IDs
        input_ids = [item['input_ids'].squeeze() for item in tokenized_batch]
        
        # Pad sequences
        padded_input_ids = self.sequence_handler.pad_sequences(input_ids)
        
        # Create attention masks
        attention_mask = self.sequence_handler.create_attention_mask(padded_input_ids)
        
        # Create position IDs
        position_ids = self.sequence_handler.create_position_ids(padded_input_ids)
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels_tensor
        }

# ===== TRAINING UTILITIES =====

class FineTuningTrainer:
    """Advanced trainer for fine-tuning with LoRA and P-tuning."""
    
    def __init__(self, model: nn.Module, tokenizer: LinkedInTokenizer,
                 training_args: TrainingArguments):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.trainer = None
    
    def setup_trainer(self, train_dataset, eval_dataset=None):
        """Setup the trainer with custom data collator."""
        sequence_handler = LinkedInSequenceHandler(self.tokenizer)
        data_collator = LinkedInDataCollator(self.tokenizer, sequence_handler)
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer.base_tokenizer
        )
    
    def train(self):
        """Train the model with fine-tuning techniques."""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        return self.trainer.train()
    
    def evaluate(self):
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        return self.trainer.evaluate()

# ===== USAGE EXAMPLES =====

def create_lora_model(model_name: str = "distilbert-base-uncased", 
                     rank: int = 16, alpha: float = 32.0) -> LoRAModelWrapper:
    """Create a model with LoRA fine-tuning."""
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return LoRAModelWrapper(base_model, rank, alpha)

def create_p_tuning_model(model_name: str = "distilbert-base-uncased",
                         prompt_length: int = 20) -> P_TuningModelWrapper:
    """Create a model with P-tuning."""
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return P_TuningModelWrapper(base_model, prompt_length)

def create_adalora_model(model_name: str = "distilbert-base-uncased",
                        max_rank: int = 64) -> AdaLoRAModelWrapper:
    """Create a model with AdaLoRA fine-tuning."""
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return AdaLoRAModelWrapper(base_model, max_rank)

# Example usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = LinkedInTokenizer()
    
    # Create LoRA model
    lora_model = create_lora_model()
    
    # Create P-tuning model
    p_tuning_model = create_p_tuning_model()
    
    # Create AdaLoRA model
    adalora_model = create_adalora_model()
    
    # Example content
    content = "Excited to share that I've joined #TechCompany as Senior Software Engineer! 🚀 #AI #MachineLearning"
    
    # Tokenize content
    tokenized = tokenizer.tokenize(content)
    
    # Extract elements
    elements = tokenizer.extract_content_elements(content)
    
    print("Tokenized content:", tokenized)
    print("Extracted elements:", elements)
