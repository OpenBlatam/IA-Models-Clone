from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
from transformers.modeling_outputs import (
from transformers.trainer import Trainer, TrainingArguments
from transformers.data import DataCollatorWithPadding, DataCollatorForTokenClassification
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging as transformers_logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
import logging
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from tqdm import tqdm
import time
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Transformers Library Integration for SEO Service
Comprehensive integration with Hugging Face Transformers library
"""

    AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM,
    BertModel, BertTokenizer, BertConfig, BertForSequenceClassification,
    RobertaModel, RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification,
    DistilBertModel, DistilBertTokenizer, DistilBertConfig,
    GPT2Model, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel,
    T5Model, T5Tokenizer, T5Config, T5ForConditionalGeneration,
    XLNetModel, XLNetTokenizer, XLNetConfig,
    AlbertModel, AlbertTokenizer, AlbertConfig,
    DebertaModel, DebertaTokenizer, DebertaConfig,
    PreTrainedModel, PretrainedConfig, PreTrainedTokenizer,
    pipeline, Pipeline, TextGenerationPipeline, TextClassificationPipeline,
    TokenClassificationPipeline, QuestionAnsweringPipeline, SummarizationPipeline,
    TranslationPipeline, FillMaskPipeline, FeatureExtractionPipeline
)
    BaseModelOutput, SequenceClassifierOutput, TokenClassifierOutput,
    QuestionAnsweringModelOutput, CausalLMOutput, Seq2SeqLMOutput
)

# Configure transformers logging
transformers_logging.set_verbosity_info()
logger = logging.getLogger(__name__)

@dataclass
class TransformersConfig:
    """Configuration for Transformers library integration"""
    model_name: str = "bert-base-uncased"
    model_type: str = "bert"  # bert, roberta, distilbert, gpt2, t5, xlnet, albert, deberta
    task_type: str = "sequence_classification"  # sequence_classification, token_classification, question_answering, text_generation, summarization, translation, fill_mask, feature_extraction
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"  # max_length, longest, do_not_pad
    return_tensors: str = "pt"
    device: str = "auto"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    use_auth_token: Optional[str] = None
    revision: Optional[str] = None
    mirror: Optional[str] = None
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    torch_dtype: Optional[str] = None  # float16, bfloat16, float32

@dataclass
class TokenizerConfig:
    """Configuration for tokenizer setup"""
    model_name: str = "bert-base-uncased"
    use_fast: bool = True
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_tensors: str = "pt"
    padding: str = "max_length"
    truncation: bool = True
    max_length: int = 512
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False

@dataclass
class PipelineConfig:
    """Configuration for Transformers pipelines"""
    task: str = "text-classification"
    model: str = "bert-base-uncased"
    tokenizer: Optional[str] = None
    device: int = -1  # -1 for CPU, 0+ for GPU
    batch_size: int = 1
    top_k: int = 5
    temperature: float = 1.0
    do_sample: bool = False
    max_length: int = 50
    num_return_sequences: int = 1
    return_all_scores: bool = False
    function_to_apply: Optional[str] = None

class TransformersModelManager:
    """Advanced manager for Transformers library models and tokenizers"""
    
    def __init__(self, config: TransformersConfig):
        
    """__init__ function."""
self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.config_model = None
        self.pipeline = None
        
        # Model registry
        self.model_registry = {
            'bert': {
                'model_class': BertModel,
                'tokenizer_class': BertTokenizer,
                'config_class': BertConfig,
                'sequence_classification': BertForSequenceClassification
            },
            'roberta': {
                'model_class': RobertaModel,
                'tokenizer_class': RobertaTokenizer,
                'config_class': RobertaConfig,
                'sequence_classification': RobertaForSequenceClassification
            },
            'distilbert': {
                'model_class': DistilBertModel,
                'tokenizer_class': DistilBertTokenizer,
                'config_class': DistilBertConfig
            },
            'gpt2': {
                'model_class': GPT2Model,
                'tokenizer_class': GPT2Tokenizer,
                'config_class': GPT2Config,
                'causal_lm': GPT2LMHeadModel
            },
            't5': {
                'model_class': T5Model,
                'tokenizer_class': T5Tokenizer,
                'config_class': T5Config,
                'seq2seq_lm': T5ForConditionalGeneration
            },
            'xlnet': {
                'model_class': XLNetModel,
                'tokenizer_class': XLNetTokenizer,
                'config_class': XLNetConfig
            },
            'albert': {
                'model_class': AlbertModel,
                'tokenizer_class': AlbertTokenizer,
                'config_class': AlbertConfig
            },
            'deberta': {
                'model_class': DebertaModel,
                'tokenizer_class': DebertaTokenizer,
                'config_class': DebertaConfig
            }
        }
        
        logger.info(f"Transformers Model Manager initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup device for model and tokenizer"""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.config.device == "cpu":
            return torch.device("cpu")
        elif self.config.device.startswith("cuda"):
            return torch.device(self.config.device)
        else:
            return torch.device("cpu")
    
    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer based on configuration"""
        try:
            logger.info(f"Loading model and tokenizer: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=self.config.use_fast_tokenizer,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token = "[PAD]"
            
            # Load model configuration
            self.config_model = AutoConfig.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
            
            # Load model based on task type
            self.model = self._load_model_for_task()
            
            # Move model to device
            self.model.to(self.device)
            
            # Enable mixed precision if specified
            if self.config.use_mixed_precision:
                if self.config.torch_dtype == "float16":
                    self.model = self.model.half()
                elif self.config.torch_dtype == "bfloat16":
                    self.model = self.model.to(torch.bfloat16)
            
            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            logger.info(f"Model and tokenizer loaded successfully")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise
    
    def _load_model_for_task(self) -> PreTrainedModel:
        """Load model based on task type"""
        if self.config.task_type == "sequence_classification":
            return AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
        elif self.config.task_type == "token_classification":
            return AutoModelForTokenClassification.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
        elif self.config.task_type == "question_answering":
            return AutoModelForQuestionAnswering.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
        elif self.config.task_type == "text_generation":
            return AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
        elif self.config.task_type == "summarization":
            return AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
        elif self.config.task_type == "fill_mask":
            return AutoModelForMaskedLM.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
        else:
            # Default to base model
            return AutoModel.from_pretrained(
                self.config.model_name,
                config=self.config_model,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
    
    def create_pipeline(self, pipeline_config: PipelineConfig) -> Pipeline:
        """Create a Transformers pipeline"""
        try:
            logger.info(f"Creating pipeline for task: {pipeline_config.task}")
            
            self.pipeline = pipeline(
                task=pipeline_config.task,
                model=pipeline_config.model,
                tokenizer=pipeline_config.tokenizer,
                device=pipeline_config.device,
                batch_size=pipeline_config.batch_size,
                top_k=pipeline_config.top_k,
                temperature=pipeline_config.temperature,
                do_sample=pipeline_config.do_sample,
                max_length=pipeline_config.max_length,
                num_return_sequences=pipeline_config.num_return_sequences,
                return_all_scores=pipeline_config.return_all_scores,
                function_to_apply=pipeline_config.function_to_apply
            )
            
            logger.info(f"Pipeline created successfully")
            return self.pipeline
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            raise
    
    def tokenize_text(self, text: Union[str, List[str]], **kwargs) -> BatchEncoding:
        """Tokenize text using the loaded tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        # Merge default config with kwargs
        tokenizer_kwargs = {
            'add_special_tokens': True,
            'return_attention_mask': True,
            'return_tensors': 'pt',
            'padding': 'max_length',
            'truncation': True,
            'max_length': self.config.max_length
        }
        tokenizer_kwargs.update(kwargs)
        
        return self.tokenizer(text, **tokenizer_kwargs)
    
    def encode_text(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Encode text and return model inputs"""
        tokenized = self.tokenize_text(text, **kwargs)
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}
        
        return inputs
    
    def predict(self, text: Union[str, List[str]], **kwargs) -> Any:
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        self.model.eval()
        inputs = self.encode_text(text, **kwargs)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs
    
    def get_embeddings(self, text: Union[str, List[str]], pooling_strategy: str = "mean") -> torch.Tensor:
        """Extract embeddings from the model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        self.model.eval()
        inputs = self.encode_text(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1]
            else:
                raise ValueError("Model output does not contain hidden states")
            
            # Apply pooling strategy
            if pooling_strategy == "mean":
                # Mean pooling (excluding padding tokens)
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    # Create mask for non-padding tokens
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    masked_hidden_states = hidden_states * mask
                    sum_hidden_states = masked_hidden_states.sum(dim=1)
                    count_mask = mask.sum(dim=1)
                    embeddings = sum_hidden_states / count_mask
                else:
                    embeddings = hidden_states.mean(dim=1)
            
            elif pooling_strategy == "cls":
                # Use [CLS] token embedding
                embeddings = hidden_states[:, 0, :]
            
            elif pooling_strategy == "max":
                # Max pooling
                embeddings = hidden_states.max(dim=1)[0]
            
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
        
        return embeddings
    
    def fine_tune(self, train_dataset, eval_dataset=None, training_args: TrainingArguments = None) -> Trainer:
        """Fine-tune the model using Transformers Trainer"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        # Default training arguments
        if training_args is None:
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                save_steps=1000,
                eval_steps=1000,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False if eval_dataset else None,
            )
        
        # Create data collator
        if self.config.task_type == "token_classification":
            data_collator = DataCollatorForTokenClassification(self.tokenizer)
        else:
            data_collator = DataCollatorWithPadding(self.tokenizer)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        return trainer
    
    def save_model(self, save_path: str):
        """Save the model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        config_save_path = os.path.join(save_path, "transformers_config.json")
        with open(config_save_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model and tokenizer saved to: {save_path}")
    
    def load_saved_model(self, load_path: str):
        """Load a saved model and tokenizer"""
        # Load configuration
        config_load_path = os.path.join(load_path, "transformers_config.json")
        if os.path.exists(config_load_path):
            with open(config_load_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = json.load(f)
            self.config = TransformersConfig(**config_dict)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.load_model_and_tokenizer()
        
        logger.info(f"Model and tokenizer loaded from: {load_path}")

class SEOSpecificTransformers:
    """SEO-specific wrapper for Transformers library"""
    
    def __init__(self, config: TransformersConfig):
        
    """__init__ function."""
self.config = config
        self.manager = TransformersModelManager(config)
        self.model = None
        self.tokenizer = None
        
        # SEO-specific configurations
        self.seo_labels = {
            'content_quality': ['poor', 'fair', 'good', 'excellent'],
            'seo_score': list(range(1, 11)),  # 1-10 scale
            'readability': ['very_difficult', 'difficult', 'moderate', 'easy', 'very_easy'],
            'keyword_density': ['low', 'medium', 'high', 'optimal']
        }
    
    def setup_seo_model(self, task: str = "sequence_classification", num_labels: int = 2):
        """Setup model for SEO-specific tasks"""
        # Update config for SEO task
        self.config.task_type = task
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.manager.load_model_and_tokenizer()
        
        # Update model head for classification if needed
        if task == "sequence_classification" and hasattr(self.model, 'classifier'):
            # Update classifier head for number of labels
            if hasattr(self.model.config, 'num_labels'):
                self.model.config.num_labels = num_labels
                # Reinitialize classifier head
                if hasattr(self.model, 'classifier'):
                    self.model.classifier = nn.Linear(
                        self.model.config.hidden_size, 
                        num_labels
                    )
        
        logger.info(f"SEO model setup for task: {task} with {num_labels} labels")
    
    def analyze_seo_content(self, content: str) -> Dict[str, Any]:
        """Analyze SEO content using the model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_seo_model() first.")
        
        # Tokenize content
        inputs = self.manager.encode_text(content)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Extract logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            else:
                # For regression tasks
                predictions = outputs.logits.squeeze()
                probabilities = None
        
        # Get embeddings
        embeddings = self.manager.get_embeddings(content)
        
        return {
            'content': content,
            'predictions': predictions.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy() if probabilities is not None else None,
            'embeddings': embeddings.cpu().numpy(),
            'model_outputs': outputs
        }
    
    def batch_analyze_seo_content(self, contents: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple pieces of SEO content"""
        results = []
        
        for content in tqdm(contents, desc="Analyzing SEO content"):
            try:
                result = self.analyze_seo_content(content)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error analyzing content: {e}")
                results.append({
                    'content': content,
                    'error': str(e)
                })
        
        return results
    
    def generate_seo_content(self, prompt: str, max_length: int = 200) -> str:
        """Generate SEO content using the model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_seo_model() first.")
        
        # Check if model supports text generation
        if not hasattr(self.model, 'generate'):
            raise ValueError("Model does not support text generation")
        
        # Tokenize prompt
        inputs = self.manager.tokenize_text(prompt, return_tensors="pt")
        inputs = {k: v.to(self.manager.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def get_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content"""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_seo_model() first.")
        
        # Get embeddings
        embeddings1 = self.manager.get_embeddings(content1)
        embeddings2 = self.manager.get_embeddings(content2)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(embeddings1, embeddings2, dim=1)
        
        return similarity.item()
    
    def create_seo_pipeline(self, task: str = "text-classification") -> Pipeline:
        """Create SEO-specific pipeline"""
        pipeline_config = PipelineConfig(
            task=task,
            model=self.config.model_name,
            device=0 if torch.cuda.is_available() else -1,
            batch_size=8
        )
        
        return self.manager.create_pipeline(pipeline_config)

class TransformersUtilities:
    """Utility functions for Transformers library"""
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get list of available models by type"""
        return {
            'bert': [
                'bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased',
                'bert-base-multilingual-uncased', 'bert-base-multilingual-cased'
            ],
            'roberta': [
                'roberta-base', 'roberta-large', 'roberta-large-mnli'
            ],
            'distilbert': [
                'distilbert-base-uncased', 'distilbert-base-cased'
            ],
            'gpt2': [
                'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            ],
            't5': [
                't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'
            ],
            'xlnet': [
                'xlnet-base-cased', 'xlnet-large-cased'
            ],
            'albert': [
                'albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2'
            ],
            'deberta': [
                'microsoft/deberta-base', 'microsoft/deberta-large'
            ]
        }
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            config = AutoConfig.from_pretrained(model_name)
            return {
                'model_name': model_name,
                'model_type': config.model_type,
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_layers': getattr(config, 'num_hidden_layers', None),
                'num_attention_heads': getattr(config, 'num_attention_heads', None),
                'vocab_size': getattr(config, 'vocab_size', None),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', None),
                'architectures': getattr(config, 'architectures', None)
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {'model_name': model_name, 'error': str(e)}
    
    @staticmethod
    def estimate_model_size(model_name: str) -> Dict[str, Any]:
        """Estimate model size and memory requirements"""
        try:
            config = AutoConfig.from_pretrained(model_name)
            
            # Calculate parameters
            hidden_size = getattr(config, 'hidden_size', 768)
            num_layers = getattr(config, 'num_hidden_layers', 12)
            vocab_size = getattr(config, 'vocab_size', 30000)
            intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
            
            # Estimate parameters
            embedding_params = vocab_size * hidden_size
            transformer_params = num_layers * (
                4 * hidden_size * hidden_size +  # Self-attention
                2 * hidden_size * intermediate_size +  # Feed-forward
                4 * hidden_size  # Layer norms
            )
            
            total_params = embedding_params + transformer_params
            
            # Estimate memory (in MB)
            memory_fp32 = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
            memory_fp16 = total_params * 2 / (1024 * 1024)  # 2 bytes per parameter
            
            return {
                'model_name': model_name,
                'total_parameters': total_params,
                'memory_fp32_mb': memory_fp32,
                'memory_fp16_mb': memory_fp16,
                'memory_fp32_gb': memory_fp32 / 1024,
                'memory_fp16_gb': memory_fp16 / 1024
            }
        except Exception as e:
            logger.error(f"Error estimating model size for {model_name}: {e}")
            return {'model_name': model_name, 'error': str(e)}
    
    @staticmethod
    def create_optimized_config(model_name: str, **kwargs) -> TransformersConfig:
        """Create optimized configuration for a model"""
        # Get model info
        model_info = TransformersUtilities.get_model_info(model_name)
        size_info = TransformersUtilities.estimate_model_size(model_name)
        
        # Determine optimal settings based on model size
        if size_info.get('memory_fp32_gb', 0) > 8:
            # Large model - use optimizations
            config = TransformersConfig(
                model_name=model_name,
                use_mixed_precision=True,
                gradient_checkpointing=True,
                torch_dtype="float16",
                max_length=512  # Reduce sequence length
            )
        else:
            # Smaller model - standard settings
            config = TransformersConfig(
                model_name=model_name,
                use_mixed_precision=True,
                gradient_checkpointing=False,
                max_length=512
            )
        
        # Update with custom settings
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = TransformersConfig(
        model_name="bert-base-uncased",
        task_type="sequence_classification",
        max_length=512,
        use_mixed_precision=True
    )
    
    # Create SEO-specific transformers
    seo_transformers = SEOSpecificTransformers(config)
    
    # Setup model for SEO analysis
    seo_transformers.setup_seo_model(task="sequence_classification", num_labels=4)
    
    # Analyze SEO content
    content = "This is a sample SEO content about digital marketing strategies."
    analysis = seo_transformers.analyze_seo_content(content)
    print(f"SEO Analysis: {analysis}")
    
    # Get model information
    model_info = TransformersUtilities.get_model_info("bert-base-uncased")
    print(f"Model Info: {model_info}")
    
    # Estimate model size
    size_info = TransformersUtilities.estimate_model_size("bert-base-uncased")
    print(f"Size Info: {size_info}") 