"""
Advanced Transformer Models for Document Classification
=====================================================

State-of-the-art transformer models for document classification, including
custom architectures, fine-tuning techniques, and advanced training methods.

Features:
- Custom transformer architectures
- Advanced fine-tuning (LoRA, P-tuning, AdaLoRA)
- Multi-task learning
- Knowledge distillation
- Model compression and quantization
- Gradient checkpointing and mixed precision training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, RobertaModel, DebertaModel,
    GPT2Model, T5Model, BartModel,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, AutoModelForCausalLM
)
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model, TaskType, AdaLoraConfig
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import math
import warnings
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for transformer models"""
    model_name: str = "bert-base-uncased"
    num_classes: int = 100
    max_length: int = 512
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    gradient_checkpointing: bool = True
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with relative positional encoding"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Relative positional encoding
        self.relative_position_bias = nn.Parameter(
            torch.randn(2 * 512 - 1, num_heads)
        )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add relative positional bias
        relative_bias = self._get_relative_position_bias(seq_len)
        scores = scores + relative_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        return output
    
    def _get_relative_position_bias(self, seq_len: int) -> torch.Tensor:
        """Get relative position bias"""
        relative_positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        relative_positions = relative_positions + seq_len - 1
        relative_positions = torch.clamp(relative_positions, 0, 2 * seq_len - 2)
        
        bias = self.relative_position_bias[relative_positions]
        return bias.permute(2, 0, 1).unsqueeze(0)

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CustomTransformerClassifier(nn.Module):
    """Custom transformer-based document classifier"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = 768  # BERT hidden size
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.max_length, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, config.max_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, 12, 3072, config.dropout)
            for _ in range(12)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.d_model // 2, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
        
        # Global average pooling
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        return logits

class MultiTaskTransformer(nn.Module):
    """Multi-task transformer for document classification and related tasks"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base transformer model
        self.base_model = AutoModel.from_pretrained(config.model_name)
        
        # Task-specific heads
        self.classification_head = nn.Linear(self.base_model.config.hidden_size, config.num_classes)
        self.sentiment_head = nn.Linear(self.base_model.config.hidden_size, 3)  # positive, negative, neutral
        self.language_head = nn.Linear(self.base_model.config.hidden_size, 50)  # 50 languages
        self.complexity_head = nn.Linear(self.base_model.config.hidden_size, 1)  # readability score
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                task: str = "classification") -> Dict[str, torch.Tensor]:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Shared representation
        shared_repr = self.shared_layers(pooled_output)
        
        results = {}
        if task == "classification" or task == "all":
            results["classification"] = self.classification_head(shared_repr)
        if task == "sentiment" or task == "all":
            results["sentiment"] = self.sentiment_head(shared_repr)
        if task == "language" or task == "all":
            results["language"] = self.language_head(shared_repr)
        if task == "complexity" or task == "all":
            results["complexity"] = self.complexity_head(shared_repr)
        
        return results

class DocumentTransformer(nn.Module):
    """Advanced document transformer with hierarchical attention"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Sentence-level transformer
        self.sentence_transformer = AutoModel.from_pretrained(config.model_name)
        
        # Document-level transformer
        self.document_transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.sentence_transformer.config.hidden_size,
                nhead=12,
                dim_feedforward=3072,
                dropout=config.dropout,
                activation='gelu'
            ),
            num_layers=6
        )
        
        # Hierarchical attention
        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=self.sentence_transformer.config.hidden_size,
            num_heads=12,
            dropout=config.dropout
        )
        
        self.document_attention = nn.MultiheadAttention(
            embed_dim=self.sentence_transformer.config.hidden_size,
            num_heads=12,
            dropout=config.dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.sentence_transformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                sentence_boundaries: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        # Sentence-level encoding
        sentence_outputs = self.sentence_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sentence_embeddings = sentence_outputs.last_hidden_state
        
        # Hierarchical attention
        # Sentence-level attention
        sentence_attended, _ = self.sentence_attention(
            sentence_embeddings, sentence_embeddings, sentence_embeddings,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Document-level encoding
        document_embeddings = self.document_transformer(sentence_attended.transpose(0, 1))
        document_embeddings = document_embeddings.transpose(0, 1)
        
        # Document-level attention
        document_attended, _ = self.document_attention(
            document_embeddings, document_embeddings, document_embeddings,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Global pooling
        if attention_mask is not None:
            pooled_output = (document_attended * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_output = document_attended.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits

class LoRATransformer(nn.Module):
    """Transformer with LoRA (Low-Rank Adaptation) for efficient fine-tuning"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base model
        self.base_model = AutoModel.from_pretrained(config.model_name)
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "key", "value", "dense"]
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Classification head
        self.classifier = nn.Linear(
            self.base_model.config.hidden_size, 
            config.num_classes
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class AdaLoRATransformer(nn.Module):
    """Transformer with AdaLoRA for adaptive low-rank adaptation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base model
        self.base_model = AutoModel.from_pretrained(config.model_name)
        
        # AdaLoRA configuration
        adalora_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            orth_reg_weight=0.5
        )
        
        # Apply AdaLoRA
        self.model = get_peft_model(self.base_model, adalora_config)
        
        # Classification head
        self.classifier = nn.Linear(
            self.base_model.config.hidden_size, 
            config.num_classes
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class QuantizedTransformer(nn.Module):
    """Quantized transformer for efficient inference"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load quantized model
        self.model = AutoModel.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # Classification head
        self.classifier = nn.Linear(
            self.model.config.hidden_size, 
            config.num_classes
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class TransformerTrainer:
    """Advanced trainer for transformer models"""
    
    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.scaler = GradScaler() if config.fp16 else None
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.warmup_steps, eta_min=1e-7
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        if self.config.fp16:
            with autocast():
                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
        else:
            logits = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                
                if self.config.fp16:
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                        loss = F.cross_entropy(logits, labels)
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = F.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {"loss": avg_loss, "accuracy": accuracy}

class ModelFactory:
    """Factory for creating different transformer models"""
    
    @staticmethod
    def create_model(model_type: str, config: ModelConfig) -> nn.Module:
        """Create model based on type"""
        if model_type == "custom":
            return CustomTransformerClassifier(config)
        elif model_type == "multitask":
            return MultiTaskTransformer(config)
        elif model_type == "hierarchical":
            return DocumentTransformer(config)
        elif model_type == "lora":
            return LoRATransformer(config)
        elif model_type == "adalora":
            return AdaLoRATransformer(config)
        elif model_type == "quantized":
            return QuantizedTransformer(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_tokenizer(model_name: str):
        """Get tokenizer for model"""
        return AutoTokenizer.from_pretrained(model_name)

# Example usage
if __name__ == "__main__":
    # Configuration
    config = ModelConfig(
        model_name="bert-base-uncased",
        num_classes=100,
        max_length=512,
        use_lora=True,
        lora_rank=16
    )
    
    # Create model
    model = ModelFactory.create_model("lora", config)
    tokenizer = ModelFactory.get_tokenizer(config.model_name)
    
    # Example input
    text = "This is a sample document for classification."
    inputs = tokenizer(text, return_tensors="pt", max_length=512, padding=True, truncation=True)
    
    # Forward pass
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        predictions = torch.softmax(logits, dim=-1)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Predictions: {predictions}")
























