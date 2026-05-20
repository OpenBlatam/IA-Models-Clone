"""
Transformer and Large Language Model (LLM) System

This module provides comprehensive Transformer architectures and LLM capabilities
for deep learning models. It includes:

1. Various Transformer architectures (Encoder, Decoder, Encoder-Decoder)
2. Attention mechanisms (Multi-Head, Scaled Dot-Product, Relative, etc.)
3. LLM-specific features (positional encoding, tokenization, generation)
4. Pre-trained model integration and fine-tuning utilities
5. Advanced LLM techniques (prompt engineering, few-shot learning)
6. Hugging Face Transformers library integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings

# Import Hugging Face Transformers
try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification, AutoModelForTokenClassification,
        AutoModelForQuestionAnswering, AutoModelForMaskedLM,
        TrainingArguments, Trainer, DataCollatorWithPadding,
        pipeline, GenerationConfig, PreTrainedModel, PreTrainedTokenizer
    )
    from transformers.utils import logging as transformers_logging
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not found. Pre-trained model features will be limited.")
    TRANSFORMERS_AVAILABLE = False

# Import additional utilities
try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets library not found. Dataset features will be limited.")
    DATASETS_AVAILABLE = False


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
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
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder architecture."""
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder architecture."""
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder block with self-attention and cross-attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x


class TransformerModel(nn.Module):
    """Complete Transformer model (Encoder-Decoder)."""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 n_layers: int = 6, n_heads: int = 8, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        return output


class PreTrainedModelManager:
    """Manager for working with pre-trained models from Hugging Face."""
    
    def __init__(self, model_name: str, task_type: str = "auto", device: str = "auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for pre-trained models.")
        
        self.model_name = model_name
        self.task_type = task_type
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == "auto" else device)
        
        # Suppress transformers logging
        transformers_logging.set_verbosity_error()
        
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model based on task type
            if self.task_type == "auto":
                self.model = AutoModel.from_pretrained(self.model_name)
            elif self.task_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            elif self.task_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            elif self.task_type == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            elif self.task_type == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            elif self.task_type == "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            elif self.task_type == "masked_lm":
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            else:
                self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded {self.model_name} ({self.task_type}) on {self.device}")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        info = {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "vocab_size": getattr(self.tokenizer, 'vocab_size', None),
            "max_length": getattr(self.tokenizer, 'model_max_length', None)
        }
        
        return info
    
    def tokenize_text(self, text: Union[str, List[str]], 
                     max_length: Optional[int] = None,
                     padding: bool = True,
                     truncation: bool = True,
                     return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize text using the model's tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        
        if max_length is None:
            max_length = getattr(self.tokenizer, 'model_max_length', 512)
        
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 1.0, top_k: int = 50, 
                     top_p: float = 0.9, do_sample: bool = True,
                     num_return_sequences: int = 1) -> List[str]:
        """Generate text using the pre-trained model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")
        
        # Tokenize input
        inputs = self.tokenize_text(prompt)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def get_embeddings(self, text: Union[str, List[str]], 
                      pooling: str = "mean") -> torch.Tensor:
        """Extract embeddings from the model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")
        
        # Tokenize
        inputs = self.tokenize_text(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
            else:
                embeddings = outputs.hidden_states[-1]
            
            # Apply pooling
            if pooling == "mean":
                # Mean pooling (excluding padding tokens)
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                else:
                    embeddings = embeddings.mean(dim=1)
            elif pooling == "cls":
                embeddings = embeddings[:, 0, :]  # CLS token
            elif pooling == "max":
                embeddings = embeddings.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
        
        return embeddings
    
    def fine_tune(self, train_dataset, eval_dataset=None, 
                  output_dir: str = "./fine_tuned_model",
                  num_epochs: int = 3,
                  batch_size: int = 8,
                  learning_rate: float = 2e-5,
                  warmup_steps: int = 500):
        """Fine-tune the pre-trained model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for fine-tuning.")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000 if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False if eval_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model fine-tuned and saved to {output_dir}")


class LLMGenerator:
    """Large Language Model text generation utilities."""
    
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0,
                top_k: int = 50, top_p: float = 0.9, do_sample: bool = True) -> str:
        """Generate text using the language model."""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            # Generate tokens
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class PromptEngineer:
    """Prompt engineering utilities for LLMs."""
    
    def __init__(self):
        self.templates = {
            'zero_shot': "{instruction}",
            'few_shot': "{examples}\n{instruction}",
            'chain_of_thought': "Let's approach this step by step:\n{instruction}",
            'role_playing': "You are {role}. {instruction}",
            'formatting': "Please format your response as {format}.\n{instruction}"
        }
    
    def create_prompt(self, instruction: str, template: str = 'zero_shot',
                     examples: Optional[List[str]] = None, role: Optional[str] = None,
                     format_type: Optional[str] = None) -> str:
        """Create a formatted prompt."""
        if template == 'few_shot' and examples:
            examples_text = '\n'.join(examples)
            return self.templates[template].format(examples=examples_text, instruction=instruction)
        elif template == 'role_playing' and role:
            return self.templates[template].format(role=role, instruction=instruction)
        elif template == 'formatting' and format_type:
            return self.templates[template].format(format=format_type, instruction=instruction)
        else:
            return self.templates[template].format(instruction=instruction)
    
    def create_few_shot_prompt(self, examples: List[Tuple[str, str]], instruction: str) -> str:
        """Create a few-shot learning prompt."""
        examples_text = '\n'.join([f"Input: {ex[0]}\nOutput: {ex[1]}" for ex in examples])
        return f"{examples_text}\n\nInput: {instruction}\nOutput:"


class LLMAnalyzer:
    """Analysis utilities for LLM performance and behavior."""
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
    
    def analyze_attention_weights(self, input_text: str, layer_idx: int = 0) -> torch.Tensor:
        """Analyze attention weights for a given input."""
        # Implementation for attention weight analysis
        pass
    
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of the model on given text."""
        # Implementation for perplexity computation
        pass
    
    def analyze_generation_diversity(self, prompts: List[str], n_samples: int = 10) -> Dict[str, float]:
        """Analyze diversity of generated outputs."""
        # Implementation for diversity analysis
        pass


class TransformersPipeline:
    """High-level interface for common NLP tasks using Transformers."""
    
    def __init__(self, task: str, model_name: str = None, device: str = "auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for pipelines.")
        
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == "auto" else device)
        
        # Create pipeline
        self.pipeline = pipeline(
            task=task,
            model=model_name,
            device=0 if self.device.type == 'cuda' else -1
        )
    
    def process(self, inputs: Union[str, List[str]], **kwargs) -> Any:
        """Process inputs using the pipeline."""
        return self.pipeline(inputs, **kwargs)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            "task": self.task,
            "model": self.pipeline.model.name_or_path,
            "device": str(self.device),
            "framework": self.pipeline.framework
        }


def create_transformer_model(config: Dict[str, Any]) -> nn.Module:
    """Create a Transformer model from configuration."""
    model_type = config.get('model_type', 'encoder_decoder')
    
    if model_type == 'encoder_decoder':
        return TransformerModel(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 2048),
            max_len=config.get('max_len', 5000),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'encoder_only':
        return TransformerEncoder(
            vocab_size=config['vocab_size'],
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 2048),
            max_len=config.get('max_len', 5000),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_available_models() -> Dict[str, List[str]]:
    """Get a list of popular pre-trained models by task."""
    if not TRANSFORMERS_AVAILABLE:
        return {"error": "Transformers library not available"}
    
    models = {
        "text_generation": [
            "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
            "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B",
            "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"
        ],
        "text_classification": [
            "bert-base-uncased", "bert-base-cased",
            "distilbert-base-uncased", "roberta-base",
            "albert-base-v2", "xlnet-base-cased"
        ],
        "question_answering": [
            "bert-base-uncased", "distilbert-base-uncased",
            "deepset/roberta-base-squad2", "microsoft/DialoGPT-medium"
        ],
        "summarization": [
            "facebook/bart-base", "facebook/bart-large",
            "t5-base", "t5-small", "google/pegasus-large"
        ],
        "translation": [
            "Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en",
            "t5-base", "facebook/mbart-large-50-many-to-many-mmt"
        ],
        "token_classification": [
            "bert-base-uncased", "distilbert-base-uncased",
            "dbmdz/bert-large-cased-finetuned-conll03-english"
        ]
    }
    
    return models


def demonstrate_transformer_llm():
    """Demonstrate Transformer and LLM capabilities."""
    print("=== Transformer and LLM System Demonstration ===\n")
    
    # Create a simple Transformer model
    config = {
        'model_type': 'encoder_decoder',
        'src_vocab_size': 1000,
        'tgt_vocab_size': 1000,
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 8,
        'd_ff': 1024
    }
    
    model = create_transformer_model(config)
    print(f"Created Transformer model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample data
    batch_size, seq_len = 2, 10
    src = torch.randint(0, 1000, (batch_size, seq_len))
    tgt = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    # Demonstrate positional encoding
    pos_encoding = PositionalEncoding(d_model=256, max_len=100)
    x = torch.randn(50, 256)  # 50 tokens, 256 dimensions
    encoded = pos_encoding(x)
    print(f"Positional encoding output shape: {encoded.shape}")
    
    # Demonstrate attention mechanism
    attention = MultiHeadAttention(d_model=256, n_heads=8)
    query = torch.randn(2, 10, 256)
    key = torch.randn(2, 10, 256)
    value = torch.randn(2, 10, 256)
    
    attn_output, attn_weights = attention(query, key, value)
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Demonstrate pre-trained models if available
    if TRANSFORMERS_AVAILABLE:
        print("\n--- Pre-trained Models Demo ---")
        available_models = get_available_models()
        print("Available model categories:")
        for category, models in available_models.items():
            print(f"  {category}: {len(models)} models")
        
        # Try to load a simple model
        try:
            print("\nLoading GPT-2 for text generation...")
            gpt2_manager = PreTrainedModelManager("gpt2", "causal_lm")
            model_info = gpt2_manager.get_model_info()
            print(f"Model info: {model_info}")
            
            # Generate some text
            generated = gpt2_manager.generate_text(
                "The future of artificial intelligence",
                max_length=50,
                temperature=0.8
            )
            print(f"Generated text: {generated[0]}")
            
        except Exception as e:
            print(f"Could not load GPT-2: {e}")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_transformer_llm()
