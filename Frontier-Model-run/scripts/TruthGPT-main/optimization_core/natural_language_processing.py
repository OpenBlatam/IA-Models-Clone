"""
Advanced Natural Language Processing System for TruthGPT Optimization Core
Text generation, understanding, and language modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import re
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class NLPTask(Enum):
    """Natural Language Processing tasks"""
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "text_summarization"
    MACHINE_TRANSLATION = "machine_translation"
    TEXT_SIMILARITY = "text_similarity"

class ModelType(Enum):
    """NLP model types"""
    TRANSFORMER = "transformer"
    BERT = "bert"
    GPT = "gpt"
    T5 = "t5"
    CUSTOM = "custom"

@dataclass
class NLPConfig:
    """Configuration for NLP tasks"""
    # Task settings
    task: NLPTask = NLPTask.TEXT_CLASSIFICATION
    model_type: ModelType = ModelType.TRANSFORMER
    num_classes: int = 2
    max_length: int = 512
    
    # Model settings
    vocab_size: int = 30000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 1000
    
    # Advanced features
    enable_pretrained: bool = True
    enable_fine_tuning: bool = True
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    
    def __post_init__(self):
        """Validate NLP configuration"""
        if self.num_classes < 2:
            raise ValueError("Number of classes must be at least 2")
        if self.max_length < 1:
            raise ValueError("Max length must be at least 1")

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.tokenizer = None
        
        # Initialize tokenizer
        if config.enable_pretrained:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            except:
                logger.warning("Could not load pretrained tokenizer, using custom")
                self.tokenizer = None
        
        logger.info("✅ Text Preprocessor initialized")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        else:
            # Simple word tokenization
            return text.split()
    
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text to tensors"""
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            }
        else:
            # Simple encoding
            tokens = self.tokenize(text)
            input_ids = torch.zeros(self.config.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.config.max_length, dtype=torch.long)
            
            for i, token in enumerate(tokens[:self.config.max_length]):
                input_ids[i] = hash(token) % self.config.vocab_size
                attention_mask[i] = 1
            
            return {
                'input_ids': input_ids.unsqueeze(0),
                'attention_mask': attention_mask.unsqueeze(0)
            }

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out(attended)
        
        return output

class TransformerBlock(nn.Module):
    """Transformer block"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class TransformerModel(nn.Module):
    """Transformer model for NLP tasks"""
    
    def __init__(self, config: NLPConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.num_heads, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        
        # Task-specific head
        if config.task == NLPTask.TEXT_CLASSIFICATION:
            self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        elif config.task == NLPTask.TEXT_GENERATION:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        elif config.task == NLPTask.QUESTION_ANSWERING:
            self.qa_head = nn.Linear(config.hidden_size, 2)  # start and end positions
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        
        logger.info(f"✅ Transformer Model initialized for {config.task.value}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        
        # Transformer blocks
        hidden_states = embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, attention_mask)
        
        # Task-specific output
        if self.config.task == NLPTask.TEXT_CLASSIFICATION:
            # Use [CLS] token (first token) for classification
            cls_output = hidden_states[:, 0, :]
            logits = self.classifier(cls_output)
            return {'logits': logits}
        
        elif self.config.task == NLPTask.TEXT_GENERATION:
            # Language modeling head
            logits = self.lm_head(hidden_states)
            return {'logits': logits}
        
        elif self.config.task == NLPTask.QUESTION_ANSWERING:
            # QA head for start and end positions
            qa_logits = self.qa_head(hidden_states)
            start_logits = qa_logits[:, :, 0]
            end_logits = qa_logits[:, :, 1]
            return {'start_logits': start_logits, 'end_logits': end_logits}
        
        else:
            # Default classification
            cls_output = hidden_states[:, 0, :]
            logits = self.classifier(cls_output)
            return {'logits': logits}

class TextClassifier(nn.Module):
    """Text classification model"""
    
    def __init__(self, config: NLPConfig):
        super().__init__()
        self.config = config
        
        # Use transformer backbone
        self.transformer = TransformerModel(config)
        
        logger.info("✅ Text Classifier initialized")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        outputs = self.transformer(input_ids, attention_mask)
        return outputs['logits']

class TextGenerator(nn.Module):
    """Text generation model"""
    
    def __init__(self, config: NLPConfig):
        super().__init__()
        self.config = config
        
        # Use transformer backbone
        self.transformer = TransformerModel(config)
        
        logger.info("✅ Text Generator initialized")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        outputs = self.transformer(input_ids, attention_mask)
        return outputs['logits']
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate text from prompt"""
        self.eval()
        
        # Tokenize prompt
        preprocessor = TextPreprocessor(self.config)
        encoding = preprocessor.encode(prompt)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(input_ids, attention_mask)
                
                # Get next token probabilities
                next_token_logits = logits[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, 1)
                generated_tokens.append(next_token.item())
                
                # Update input for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=1)
                
                # Truncate if too long
                if input_ids.shape[1] > self.config.max_length:
                    input_ids = input_ids[:, -self.config.max_length:]
                    attention_mask = attention_mask[:, -self.config.max_length:]
        
        # Convert tokens back to text
        if preprocessor.tokenizer:
            generated_text = preprocessor.tokenizer.decode(generated_tokens)
        else:
            generated_text = ' '.join([str(token) for token in generated_tokens])
        
        return generated_text

class QuestionAnsweringModel(nn.Module):
    """Question answering model"""
    
    def __init__(self, config: NLPConfig):
        super().__init__()
        self.config = config
        
        # Use transformer backbone
        self.transformer = TransformerModel(config)
        
        logger.info("✅ Question Answering Model initialized")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        outputs = self.transformer(input_ids, attention_mask)
        return {
            'start_logits': outputs['start_logits'],
            'end_logits': outputs['end_logits']
        }
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer question given context"""
        self.eval()
        
        # Combine question and context
        text = f"{question} [SEP] {context}"
        
        # Tokenize
        preprocessor = TextPreprocessor(self.config)
        encoding = preprocessor.encode(text)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']
            
            # Find best start and end positions
            start_pos = start_logits.argmax(dim=1).item()
            end_pos = end_logits.argmax(dim=1).item()
            
            # Extract answer
            if start_pos <= end_pos:
                answer_tokens = input_ids[0, start_pos:end_pos+1]
                if preprocessor.tokenizer:
                    answer = preprocessor.tokenizer.decode(answer_tokens)
                else:
                    answer = ' '.join([str(token.item()) for token in answer_tokens])
            else:
                answer = "No answer found"
        
        return answer

class NLPTrainer:
    """NLP model trainer"""
    
    def __init__(self, model: nn.Module, config: NLPConfig):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Loss function
        if config.task == NLPTask.TEXT_CLASSIFICATION:
            self.criterion = nn.CrossEntropyLoss()
        elif config.task == NLPTask.TEXT_GENERATION:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        elif config.task == NLPTask.QUESTION_ANSWERING:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.training_history = []
        self.best_accuracy = 0.0
        
        logger.info("✅ NLP Trainer initialized")
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Extract batch data
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                labels = batch['labels']
            else:
                input_ids, labels = batch
                attention_mask = None
            
            # Forward pass
            if self.config.task == NLPTask.QUESTION_ANSWERING:
                outputs = self.model(input_ids, attention_mask)
                start_loss = self.criterion(outputs['start_logits'], labels['start_positions'])
                end_loss = self.criterion(outputs['end_logits'], labels['end_positions'])
                loss = start_loss + end_loss
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            if self.config.task != NLPTask.QUESTION_ANSWERING:
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Extract batch data
                if isinstance(batch, dict):
                    input_ids = batch['input_ids']
                    attention_mask = batch.get('attention_mask', None)
                    labels = batch['labels']
                else:
                    input_ids, labels = batch
                    attention_mask = None
                
                # Forward pass
                if self.config.task == NLPTask.QUESTION_ANSWERING:
                    outputs = self.model(input_ids, attention_mask)
                    start_loss = self.criterion(outputs['start_logits'], labels['start_positions'])
                    end_loss = self.criterion(outputs['end_logits'], labels['end_positions'])
                    loss = start_loss + end_loss
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                if self.config.task != NLPTask.QUESTION_ANSWERING:
                    pred = logits.argmax(dim=1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader, num_epochs: int = None) -> Dict[str, Any]:
        """Train model"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        logger.info(f"🚀 Starting NLP training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_stats = self.train_epoch(train_loader)
            
            # Validate
            val_stats = self.validate(val_loader)
            
            # Update best accuracy
            if val_stats['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_stats['accuracy']
            
            # Record history
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'train_accuracy': train_stats['accuracy'],
                'val_loss': val_stats['loss'],
                'val_accuracy': val_stats['accuracy'],
                'best_accuracy': self.best_accuracy
            }
            self.training_history.append(epoch_stats)
            
            # Log progress
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Train Acc = {train_stats['accuracy']:.2f}%, "
                          f"Val Acc = {val_stats['accuracy']:.2f}%")
        
        final_stats = {
            'total_epochs': num_epochs,
            'best_accuracy': self.best_accuracy,
            'final_train_accuracy': self.training_history[-1]['train_accuracy'],
            'final_val_accuracy': self.training_history[-1]['val_accuracy'],
            'training_history': self.training_history
        }
        
        logger.info(f"✅ NLP training completed. Best accuracy: {self.best_accuracy:.2f}%")
        return final_stats

# Factory functions
def create_nlp_config(**kwargs) -> NLPConfig:
    """Create NLP configuration"""
    return NLPConfig(**kwargs)

def create_text_classifier(config: NLPConfig) -> TextClassifier:
    """Create text classifier"""
    return TextClassifier(config)

def create_text_generator(config: NLPConfig) -> TextGenerator:
    """Create text generator"""
    return TextGenerator(config)

def create_question_answering_model(config: NLPConfig) -> QuestionAnsweringModel:
    """Create question answering model"""
    return QuestionAnsweringModel(config)

def create_nlp_trainer(model: nn.Module, config: NLPConfig) -> NLPTrainer:
    """Create NLP trainer"""
    return NLPTrainer(model, config)

# Example usage
def example_natural_language_processing():
    """Example of natural language processing"""
    # Create configuration
    config = create_nlp_config(
        task=NLPTask.TEXT_CLASSIFICATION,
        model_type=ModelType.TRANSFORMER,
        num_classes=3,
        max_length=128,
        hidden_size=256,
        num_layers=6,
        num_heads=8
    )
    
    # Create model
    model = create_text_classifier(config)
    
    # Create trainer
    trainer = create_nlp_trainer(model, config)
    
    # Simulate training data
    dummy_input_ids = torch.randint(0, config.vocab_size, (100, config.max_length))
    dummy_labels = torch.randint(0, config.num_classes, (100,))
    
    # Create dummy dataloader
    dataset = torch.utils.data.TensorDataset(dummy_input_ids, dummy_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train model
    training_stats = trainer.train(dataloader, dataloader, num_epochs=5)
    
    # Test text generation
    generator_config = create_nlp_config(
        task=NLPTask.TEXT_GENERATION,
        model_type=ModelType.TRANSFORMER,
        max_length=64,
        hidden_size=256,
        num_layers=6,
        num_heads=8
    )
    
    generator = create_text_generator(generator_config)
    generated_text = generator.generate("The future of AI is", max_length=20)
    
    print(f"✅ Natural Language Processing Example Complete!")
    print(f"📝 NLP Statistics:")
    print(f"   Task: {config.task.value}")
    print(f"   Model Type: {config.model_type.value}")
    print(f"   Number of Classes: {config.num_classes}")
    print(f"   Best Accuracy: {training_stats['best_accuracy']:.2f}%")
    print(f"   Final Train Accuracy: {training_stats['final_train_accuracy']:.2f}%")
    print(f"   Final Val Accuracy: {training_stats['final_val_accuracy']:.2f}%")
    print(f"🤖 Generated Text: {generated_text}")
    
    return model

# Export utilities
__all__ = [
    'NLPTask',
    'ModelType',
    'NLPConfig',
    'TextPreprocessor',
    'MultiHeadAttention',
    'TransformerBlock',
    'TransformerModel',
    'TextClassifier',
    'TextGenerator',
    'QuestionAnsweringModel',
    'NLPTrainer',
    'create_nlp_config',
    'create_text_classifier',
    'create_text_generator',
    'create_question_answering_model',
    'create_nlp_trainer',
    'example_natural_language_processing'
]

if __name__ == "__main__":
    example_natural_language_processing()
    print("✅ Natural language processing example completed successfully!")

