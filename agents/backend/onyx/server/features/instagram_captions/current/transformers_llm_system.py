"""
Transformers and LLM System
Comprehensive implementation demonstrating Transformers library and LLM development
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    DataCollatorWithPadding, pipeline, PreTrainedModel, PreTrainedTokenizer
)
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os


@dataclass
class TransformerConfig:
    """Configuration for transformer models and LLM development."""
    
    # Model parameters
    model_name: str = "gpt2"  # gpt2, bert-base-uncased, t5-base, etc.
    model_type: str = "causal_lm"  # causal_lm, sequence_classification, token_classification, qa
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Training parameters
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4
    
    # Generation parameters
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Fine-tuning parameters
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Evaluation parameters
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    logging_steps: int = 100


class AttentionMechanism(nn.Module):
    """Custom attention mechanism implementation."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "Hidden size must be divisible by num_heads"
        
        # Linear projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through attention mechanism."""
        batch_size, seq_len, hidden_size = x.shape
        
        # Linear projections and reshape
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        output = self.output(context)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, hidden_size: int, max_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(np.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = AttentionMechanism(hidden_size, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.attention_norm(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + ff_output)
        
        return x


class CustomTransformerModel(nn.Module):
    """Custom transformer model implementation."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(50257, 768)  # GPT-2 vocab size
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(768, config.max_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(768, 12, 3072) for _ in range(12)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(768)
        
        # Language model head
        self.lm_head = nn.Linear(768, 50257, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embeddings.weight
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass through the model."""
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        embeddings = self.token_embeddings(input_ids)
        embeddings = self.positional_encoding(embeddings)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits}


class TransformerDataset(Dataset):
    """Dataset for transformer models."""
    
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
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


class TransformerTrainer:
    """Advanced trainer for transformer models."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Setup training components
        self._setup_training()
        
        logging.info(f"Transformer trainer initialized with {config.model_name}")
    
    def _load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load pre-trained model and tokenizer."""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Set padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model based on type
            if self.config.model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32
                )
            elif self.config.model_type == "sequence_classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=2  # Binary classification
                )
            elif self.config.model_type == "token_classification":
                model = AutoModelForTokenClassification.from_pretrained(
                    self.config.model_name,
                    num_labels=9  # NER labels
                )
            elif self.config.model_type == "qa":
                model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_name)
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
            
            # Apply LoRA if specified
            if self.config.use_lora:
                from peft import LoraConfig, get_peft_model
                
                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, lora_config)
            
            return model, tokenizer
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def _setup_training(self):
        """Setup training arguments and components."""
        self.training_args = TrainingArguments(
            output_dir="./transformer_output",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=self.config.dataloader_num_workers,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False
        )
    
    def train(self, train_texts: List[str], eval_texts: List[str] = None):
        """Train the transformer model."""
        # Create datasets
        train_dataset = TransformerDataset(train_texts, self.tokenizer, self.config.max_length)
        
        if eval_texts:
            eval_dataset = TransformerDataset(eval_texts, self.tokenizer, self.config.max_length)
        else:
            eval_dataset = None
        
        # Create data collator
        if self.config.model_type == "causal_lm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        logging.info("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model("./final_model")
        self.tokenizer.save_pretrained("./final_model")
        
        logging.info("Training completed successfully!")
        return trainer
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the trained model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            logging.error(f"Error generating text: {e}")
            return prompt
    
    def batch_generate(self, prompts: List[str], max_length: int = 100) -> List[str]:
        """Generate text for multiple prompts."""
        generated_texts = []
        
        for prompt in prompts:
            generated_text = self.generate_text(prompt, max_length)
            generated_texts.append(generated_text)
        
        return generated_texts


class PipelineManager:
    """Manager for transformer pipelines."""
    
    def __init__(self):
        self.pipelines = {}
    
    def create_pipeline(self, task: str, model_name: str, **kwargs):
        """Create a transformer pipeline."""
        try:
            pipeline_obj = pipeline(task, model=model_name, **kwargs)
            self.pipelines[task] = pipeline_obj
            logging.info(f"Pipeline created for task: {task}")
            return pipeline_obj
        except Exception as e:
            logging.error(f"Error creating pipeline for {task}: {e}")
            raise
    
    def text_generation(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using text-generation pipeline."""
        if "text-generation" not in self.pipelines:
            self.create_pipeline("text-generation", "gpt2")
        
        result = self.pipelines["text-generation"](prompt, max_length=max_length)
        return result[0]["generated_text"]
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using sentiment-analysis pipeline."""
        if "sentiment-analysis" not in self.pipelines:
            self.create_pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
        
        result = self.pipelines["sentiment-analysis"](text)
        return result[0]
    
    def question_answering(self, question: str, context: str) -> Dict[str, Any]:
        """Answer questions using question-answering pipeline."""
        if "question-answering" not in self.pipelines:
            self.create_pipeline("question-answering", "distilbert-base-cased-distilled-squad")
        
        result = self.pipelines["question-answering"](question=question, context=context)
        return result
    
    def named_entity_recognition(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using NER pipeline."""
        if "ner" not in self.pipelines:
            self.create_pipeline("ner", "dbmdz/bert-large-cased-finetuned-conll03-english")
        
        result = self.pipelines["ner"](text)
        return result


class TransformerAnalyzer:
    """Analyze transformer model performance and behavior."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def analyze_attention_weights(self, text: str, layer_idx: int = 0) -> torch.Tensor:
        """Extract attention weights for analysis."""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Forward pass with output_attentions=True
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention weights for specified layer
        attention_weights = outputs.attentions[layer_idx][0]  # Remove batch dimension
        
        return attention_weights
    
    def visualize_attention(self, text: str, layer_idx: int = 0, head_idx: int = 0):
        """Visualize attention weights."""
        attention_weights = self.analyze_attention_weights(text, layer_idx)
        
        # Get tokens for visualization
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        
        # Plot attention heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_weights[head_idx].cpu().numpy(), cmap='Blues')
        plt.colorbar()
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
        plt.tight_layout()
        plt.show()
    
    def analyze_model_complexity(self) -> Dict[str, Any]:
        """Analyze model complexity and parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'parameter_ratio': trainable_params / total_params if total_params > 0 else 0
        }
    
    def benchmark_inference_speed(self, texts: List[str], num_runs: int = 10) -> Dict[str, float]:
        """Benchmark inference speed."""
        self.model.eval()
        
        # Warmup
        for _ in range(5):
            _ = self.model.generate(
                torch.randint(0, 1000, (1, 10)).to(self.device),
                max_length=20
            )
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    _ = self.model.generate(**inputs, max_length=50)
        
        total_time = time.time() - start_time
        avg_time_per_text = total_time / (num_runs * len(texts))
        
        return {
            'total_time': total_time,
            'avg_time_per_text': avg_time_per_text,
            'texts_per_second': (num_runs * len(texts)) / total_time
        }


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = TransformerConfig(
        model_name="gpt2",
        model_type="causal_lm",
        max_length=128,
        batch_size=4,
        learning_rate=2e-5,
        num_epochs=1,
        use_lora=True
    )
    
    # Sample training data
    sample_texts = [
        "The future of artificial intelligence is promising.",
        "Machine learning algorithms can solve complex problems.",
        "Deep learning models are transforming technology.",
        "Natural language processing enables human-computer interaction.",
        "Transformers have revolutionized the field of NLP."
    ]
    
    # Initialize trainer
    trainer = TransformerTrainer(config)
    
    # Train the model
    trainer_obj = trainer.train(sample_texts)
    
    # Generate text
    prompt = "The future of AI is"
    generated_text = trainer.generate_text(prompt, max_length=50)
    print(f"Generated text: {generated_text}")
    
    # Initialize pipeline manager
    pipeline_manager = PipelineManager()
    
    # Test different pipelines
    sentiment_result = pipeline_manager.sentiment_analysis("I love this technology!")
    print(f"Sentiment analysis: {sentiment_result}")
    
    qa_result = pipeline_manager.question_answering(
        "What is artificial intelligence?",
        "Artificial intelligence is a branch of computer science that aims to create intelligent machines."
    )
    print(f"Question answering: {qa_result}")
    
    ner_result = pipeline_manager.named_entity_recognition(
        "Apple Inc. is headquartered in Cupertino, California."
    )
    print(f"Named entity recognition: {ner_result}")
    
    # Analyze model
    analyzer = TransformerAnalyzer(trainer.model, trainer.tokenizer)
    
    # Analyze complexity
    complexity = analyzer.analyze_model_complexity()
    print(f"Model complexity: {complexity}")
    
    # Benchmark speed
    speed_benchmark = analyzer.benchmark_inference_speed(sample_texts[:3])
    print(f"Speed benchmark: {speed_benchmark}")
    
    # Visualize attention
    analyzer.visualize_attention("The transformer model uses attention mechanisms.")
    
    logging.info("Transformer and LLM demonstration completed successfully!")





