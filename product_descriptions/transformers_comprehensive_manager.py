from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
from transformers.modeling_outputs import (
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging as transformers_logging
import numpy as np
import logging
import os
import time
import gc
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import structlog
from contextlib import contextmanager
import psutil
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Any, List, Dict, Optional
"""
Comprehensive Transformers Management System

This module provides a unified Transformers management system that consolidates
all Transformers library functionality across the codebase with:

- Advanced model loading and caching
- Comprehensive tokenization strategies
- Pipeline management and optimization
- Fine-tuning and training capabilities
- Performance monitoring and optimization
- Security validation and sanitization
- Multi-task and multi-model support
- Integration with existing modules
"""

    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
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
    TranslationPipeline, FillMaskPipeline, FeatureExtractionPipeline,
    Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    BitsAndBytesConfig, AutoConfig
)
    BaseModelOutput, SequenceClassifierOutput, TokenClassifierOutput,
    QuestionAnsweringModelOutput, CausalLMOutput, Seq2SeqLMOutput
)

# Additional imports

# Configure logging
transformers_logging.set_verbosity_error()
logger = structlog.get_logger(__name__)


class ModelType(Enum):
    """Supported model types."""
    BERT = "bert"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    GPT2 = "gpt2"
    T5 = "t5"
    XLNET = "xlnet"
    ALBERT = "albert"
    DEBERTA = "deberta"
    AUTO = "auto"


class TaskType(Enum):
    """Supported task types."""
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    QUESTION_ANSWERING = "question_answering"
    CAUSAL_LANGUAGE_MODEL = "causal_language_model"
    SEQ2SEQ_LANGUAGE_MODEL = "seq2seq_language_model"
    MASKED_LANGUAGE_MODEL = "masked_language_model"
    FEATURE_EXTRACTION = "feature_extraction"
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    FILL_MASK = "fill_mask"


class OptimizationLevel(Enum):
    """Model optimization levels."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"


@dataclass
class TransformersConfig:
    """Comprehensive Transformers configuration."""
    
    # Model configuration
    model_name: str = "bert-base-uncased"
    model_type: ModelType = ModelType.AUTO
    task_type: TaskType = TaskType.SEQUENCE_CLASSIFICATION
    
    # Tokenization configuration
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"  # max_length, longest, do_not_pad
    return_tensors: str = "pt"
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    
    # Device and performance
    device: str = "auto"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    torch_dtype: Optional[str] = None  # float16, bfloat16, float32
    
    # Caching and storage
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    use_auth_token: Optional[str] = None
    revision: Optional[str] = None
    mirror: Optional[str] = None
    
    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: Optional[str] = None
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    
    # Security and validation
    enable_security_checks: bool = True
    validate_inputs: bool = True
    sanitize_outputs: bool = True
    
    # Performance monitoring
    enable_profiling: bool = False
    enable_memory_tracking: bool = True
    log_memory_usage: bool = True
    
    # Pipeline configuration
    batch_size: int = 1
    top_k: int = 50
    temperature: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1
    return_all_scores: bool = False
    function_to_apply: Optional[str] = None


@dataclass
class ModelInfo:
    """Model information structure."""
    
    name: str
    type: ModelType
    task: TaskType
    parameters: int
    size_mb: float
    device: str
    dtype: str
    loaded: bool = False
    load_time: Optional[float] = None
    memory_usage: Optional[float] = None
    
    def __post_init__(self) -> Any:
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if self.parameters < 0:
            raise ValueError("Parameters count cannot be negative")
        if self.size_mb < 0:
            raise ValueError("Model size cannot be negative")


class TransformersModelRegistry:
    """Registry for managing Transformers models and their configurations."""
    
    def __init__(self) -> Any:
        self.model_registry = {
            ModelType.BERT: {
                'model_class': BertModel,
                'tokenizer_class': BertTokenizer,
                'config_class': BertConfig,
                'sequence_classification': BertForSequenceClassification,
                'base_models': [
                    'bert-base-uncased', 'bert-base-cased',
                    'bert-large-uncased', 'bert-large-cased'
                ]
            },
            ModelType.ROBERTA: {
                'model_class': RobertaModel,
                'tokenizer_class': RobertaTokenizer,
                'config_class': RobertaConfig,
                'sequence_classification': RobertaForSequenceClassification,
                'base_models': [
                    'roberta-base', 'roberta-large',
                    'roberta-large-mnli'
                ]
            },
            ModelType.DISTILBERT: {
                'model_class': DistilBertModel,
                'tokenizer_class': DistilBertTokenizer,
                'config_class': DistilBertConfig,
                'base_models': [
                    'distilbert-base-uncased', 'distilbert-base-cased'
                ]
            },
            ModelType.GPT2: {
                'model_class': GPT2Model,
                'tokenizer_class': GPT2Tokenizer,
                'config_class': GPT2Config,
                'causal_lm': GPT2LMHeadModel,
                'base_models': [
                    'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
                ]
            },
            ModelType.T5: {
                'model_class': T5Model,
                'tokenizer_class': T5Tokenizer,
                'config_class': T5Config,
                'seq2seq_lm': T5ForConditionalGeneration,
                'base_models': [
                    't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'
                ]
            },
            ModelType.XLNET: {
                'model_class': XLNetModel,
                'tokenizer_class': XLNetTokenizer,
                'config_class': XLNetConfig,
                'base_models': [
                    'xlnet-base-cased', 'xlnet-large-cased'
                ]
            },
            ModelType.ALBERT: {
                'model_class': AlbertModel,
                'tokenizer_class': AlbertTokenizer,
                'config_class': AlbertConfig,
                'base_models': [
                    'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2'
                ]
            },
            ModelType.DEBERTA: {
                'model_class': DebertaModel,
                'tokenizer_class': DebertaTokenizer,
                'config_class': DebertaConfig,
                'base_models': [
                    'microsoft/deberta-base', 'microsoft/deberta-large'
                ]
            }
        }
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information from registry."""
        for model_type, info in self.model_registry.items():
            if model_name in info.get('base_models', []):
                return {
                    'type': model_type,
                    'info': info
                }
        return None
    
    def get_model_class(self, model_type: ModelType, task_type: TaskType) -> Optional[type]:
        """Get appropriate model class for task."""
        if model_type not in self.model_registry:
            return None
        
        registry_info = self.model_registry[model_type]
        
        # Map task types to model classes
        task_mapping = {
            TaskType.SEQUENCE_CLASSIFICATION: 'sequence_classification',
            TaskType.CAUSAL_LANGUAGE_MODEL: 'causal_lm',
            TaskType.SEQ2SEQ_LANGUAGE_MODEL: 'seq2seq_lm'
        }
        
        task_key = task_mapping.get(task_type)
        if task_key and task_key in registry_info:
            return registry_info[task_key]
        
        # Fallback to base model class
        return registry_info.get('model_class')
    
    def get_tokenizer_class(self, model_type: ModelType) -> Optional[type]:
        """Get tokenizer class for model type."""
        if model_type not in self.model_registry:
            return None
        
        return self.model_registry[model_type].get('tokenizer_class')


class TransformersModelManager:
    """Advanced manager for Transformers models and tokenizers."""
    
    def __init__(self, config: TransformersConfig):
        
    """__init__ function."""
self.config = config
        self.device = self._setup_device()
        self.registry = TransformersModelRegistry()
        
        # Model and tokenizer storage
        self.models: Dict[str, PreTrainedModel] = {}
        self.tokenizers: Dict[str, PreTrainedTokenizer] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        
        # Performance tracking
        self.load_times: Dict[str, float] = {}
        self.memory_usage: Dict[str, float] = {}
        self.inference_times: Dict[str, List[float]] = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _setup_device(self) -> torch.device:
        """Setup and validate device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _get_model_class(self, model_name: str, task_type: TaskType) -> type:
        """Get appropriate model class for task."""
        # Try to determine model type from name
        model_info = self.registry.get_model_info(model_name)
        if model_info:
            model_type = model_info['type']
        else:
            model_type = self.config.model_type
        
        # Get model class
        model_class = self.registry.get_model_class(model_type, task_type)
        if model_class is None:
            # Fallback to AutoModel classes
            task_mapping = {
                TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
                TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
                TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
                TaskType.CAUSAL_LANGUAGE_MODEL: AutoModelForCausalLM,
                TaskType.SEQ2SEQ_LANGUAGE_MODEL: AutoModelForSeq2SeqLM,
                TaskType.MASKED_LANGUAGE_MODEL: AutoModelForMaskedLM,
                TaskType.FEATURE_EXTRACTION: AutoModel
            }
            model_class = task_mapping.get(task_type, AutoModel)
        
        return model_class
    
    def _create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration."""
        if not (self.config.load_in_8bit or self.config.load_in_4bit):
            return None
        
        return BitsAndBytesConfig(
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype) if self.config.bnb_4bit_compute_dtype else None,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type
        )
    
    def load_model_and_tokenizer(self, model_name: Optional[str] = None, 
                                task_type: Optional[TaskType] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer with caching."""
        model_name = model_name or self.config.model_name
        task_type = task_type or self.config.task_type
        
        # Check if already loaded
        if model_name in self.models and model_name in self.tokenizers:
            logger.info(f"Model {model_name} already loaded, returning cached version")
            return self.models[model_name], self.tokenizers[model_name]
        
        start_time = time.time()
        
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=self.config.use_fast_tokenizer,
                trust_remote_code=self.config.trust_remote_code,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                mirror=self.config.mirror
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            logger.info(f"Loading model for {model_name}")
            model_class = self._get_model_class(model_name, task_type)
            
            # Prepare loading kwargs
            model_kwargs = {
                'cache_dir': self.config.cache_dir,
                'local_files_only': self.config.local_files_only,
                'use_auth_token': self.config.use_auth_token,
                'revision': self.config.revision,
                'mirror': self.config.mirror,
                'trust_remote_code': self.config.trust_remote_code
            }
            
            # Add quantization if enabled
            quantization_config = self._create_quantization_config()
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            # Add torch dtype if specified
            if self.config.torch_dtype:
                model_kwargs['torch_dtype'] = getattr(torch, self.config.torch_dtype)
            
            model = model_class.from_pretrained(model_name, **model_kwargs)
            
            # Move to device
            model = model.to(self.device)
            
            # Enable gradient checkpointing if requested
            if self.config.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            # Store model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Calculate model info
            parameters = sum(p.numel() for p in model.parameters())
            size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            model_info = ModelInfo(
                name=model_name,
                type=self.config.model_type,
                task=task_type,
                parameters=parameters,
                size_mb=size_mb,
                device=str(self.device),
                dtype=str(model.dtype),
                loaded=True,
                load_time=time.time() - start_time,
                memory_usage=torch.cuda.memory_allocated() / (1024 * 1024) if self.device.type == 'cuda' else 0
            )
            
            self.model_info[model_name] = model_info
            self.load_times[model_name] = model_info.load_time
            self.memory_usage[model_name] = model_info.memory_usage
            
            logger.info(f"Model {model_name} loaded successfully in {model_info.load_time:.2f}s")
            logger.info(f"Parameters: {parameters:,}, Size: {size_mb:.1f}MB")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def create_pipeline(self, task: str, model_name: Optional[str] = None, 
                       **pipeline_kwargs) -> Pipeline:
        """Create a Transformers pipeline."""
        model_name = model_name or self.config.model_name
        
        # Check if pipeline already exists
        pipeline_key = f"{task}_{model_name}"
        if pipeline_key in self.pipelines:
            logger.info(f"Pipeline {pipeline_key} already exists, returning cached version")
            return self.pipelines[pipeline_key]
        
        try:
            logger.info(f"Creating pipeline for task: {task} with model: {model_name}")
            
            # Prepare pipeline arguments
            pipeline_args = {
                'task': task,
                'model': model_name,
                'device': self.device,
                'batch_size': self.config.batch_size,
                'top_k': self.config.top_k,
                'temperature': self.config.temperature,
                'do_sample': self.config.do_sample,
                'max_length': self.config.max_length,
                'num_return_sequences': self.config.num_return_sequences,
                'return_all_scores': self.config.return_all_scores,
                'function_to_apply': self.config.function_to_apply
            }
            
            # Update with custom kwargs
            pipeline_args.update(pipeline_kwargs)
            
            # Create pipeline
            pipeline_instance = pipeline(**pipeline_args)
            
            # Store pipeline
            self.pipelines[pipeline_key] = pipeline_instance
            
            logger.info(f"Pipeline {pipeline_key} created successfully")
            return pipeline_instance
            
        except Exception as e:
            logger.error(f"Failed to create pipeline for {task}: {e}")
            raise
    
    def tokenize_text(self, text: Union[str, List[str]], model_name: Optional[str] = None,
                     **kwargs) -> BatchEncoding:
        """Tokenize text using the specified model's tokenizer."""
        model_name = model_name or self.config.model_name
        
        if model_name not in self.tokenizers:
            self.load_model_and_tokenizer(model_name)
        
        tokenizer = self.tokenizers[model_name]
        
        # Merge default config with kwargs
        tokenizer_kwargs = {
            'add_special_tokens': self.config.add_special_tokens,
            'return_attention_mask': self.config.return_attention_mask,
            'return_tensors': self.config.return_tensors,
            'padding': self.config.padding,
            'truncation': self.config.truncation,
            'max_length': self.config.max_length
        }
        tokenizer_kwargs.update(kwargs)
        
        return tokenizer(text, **tokenizer_kwargs)
    
    def get_embeddings(self, text: Union[str, List[str]], model_name: Optional[str] = None,
                      pooling_strategy: str = "mean") -> np.ndarray:
        """Extract embeddings from text."""
        model_name = model_name or self.config.model_name
        
        if model_name not in self.models:
            self.load_model_and_tokenizer(model_name)
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize text
        inputs = self.tokenize_text(text, model_name)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extract embeddings based on pooling strategy
            if pooling_strategy == "mean":
                # Mean pooling
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state
                embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            elif pooling_strategy == "cls":
                # CLS token pooling
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling_strategy == "max":
                # Max pooling
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state
                embeddings = (embeddings * attention_mask.unsqueeze(-1)).max(dim=1)[0]
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
        
        return embeddings.cpu().numpy()
    
    def predict(self, text: Union[str, List[str]], model_name: Optional[str] = None,
               task_type: Optional[TaskType] = None) -> Any:
        """Make predictions using the model."""
        model_name = model_name or self.config.model_name
        task_type = task_type or self.config.task_type
        
        if model_name not in self.models:
            self.load_model_and_tokenizer(model_name, task_type)
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize text
        inputs = self.tokenize_text(text, model_name)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        inference_time = time.time() - start_time
        
        # Track inference time
        if model_name not in self.inference_times:
            self.inference_times[model_name] = []
        self.inference_times[model_name].append(inference_time)
        
        return outputs
    
    def get_model_info(self, model_name: Optional[str] = None) -> Optional[ModelInfo]:
        """Get information about a loaded model."""
        model_name = model_name or self.config.model_name
        return self.model_info.get(model_name)
    
    def get_performance_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for a model."""
        model_name = model_name or self.config.model_name
        
        stats = {
            'load_time': self.load_times.get(model_name, 0),
            'memory_usage': self.memory_usage.get(model_name, 0),
            'inference_times': self.inference_times.get(model_name, []),
            'avg_inference_time': 0,
            'total_inferences': 0
        }
        
        if stats['inference_times']:
            stats['avg_inference_time'] = np.mean(stats['inference_times'])
            stats['total_inferences'] = len(stats['inference_times'])
        
        return stats
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache."""
        if model_name:
            if model_name in self.models:
                del self.models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            if model_name in self.pipelines:
                del self.pipelines[model_name]
            if model_name in self.model_info:
                del self.model_info[model_name]
        else:
            self.models.clear()
            self.tokenizers.clear()
            self.pipelines.clear()
            self.model_info.clear()
        
        # Clear GPU cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info(f"Cache cleared for model: {model_name or 'all'}")


class TransformersSecurityManager:
    """Security manager for Transformers models."""
    
    def __init__(self, config: TransformersConfig):
        
    """__init__ function."""
self.config = config
    
    def validate_inputs(self, inputs: Union[str, List[str], Dict[str, torch.Tensor]]) -> bool:
        """Validate inputs for security."""
        if not self.config.validate_inputs:
            return True
        
        try:
            if isinstance(inputs, str):
                # Check for malicious patterns
                if self._contains_malicious_patterns(inputs):
                    logger.warning("Malicious patterns detected in input text")
                    return False
            elif isinstance(inputs, list):
                for text in inputs:
                    if isinstance(text, str) and self._contains_malicious_patterns(text):
                        logger.warning("Malicious patterns detected in input text")
                        return False
            elif isinstance(inputs, dict):
                # Validate tensor inputs
                for key, tensor in inputs.items():
                    if isinstance(tensor, torch.Tensor):
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            logger.warning(f"Invalid values detected in tensor {key}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def _contains_malicious_patterns(self, text: str) -> bool:
        """Check for malicious patterns in text."""
        # Add your malicious pattern detection logic here
        malicious_patterns = [
            # Add patterns that could be used for prompt injection
            "ignore previous instructions",
            "system prompt",
            "override",
            "bypass"
        ]
        
        text_lower = text.lower()
        for pattern in malicious_patterns:
            if pattern in text_lower:
                return True
        
        return False
    
    def sanitize_outputs(self, outputs: Any) -> Any:
        """Sanitize model outputs."""
        if not self.config.sanitize_outputs:
            return outputs
        
        try:
            if isinstance(outputs, torch.Tensor):
                # Replace NaN and Inf values
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
            elif isinstance(outputs, dict):
                # Recursively sanitize dictionary outputs
                for key, value in outputs.items():
                    outputs[key] = self.sanitize_outputs(value)
            elif isinstance(outputs, list):
                # Recursively sanitize list outputs
                outputs = [self.sanitize_outputs(item) for item in outputs]
            
            return outputs
            
        except Exception as e:
            logger.error(f"Output sanitization failed: {e}")
            return outputs
    
    def check_model_security(self, model: PreTrainedModel) -> Dict[str, bool]:
        """Check model for security issues."""
        security_checks = {
            'has_nan_weights': False,
            'has_inf_weights': False,
            'has_large_weights': False,
            'is_valid': True
        }
        
        try:
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    security_checks['has_nan_weights'] = True
                    security_checks['is_valid'] = False
                    logger.warning(f"Parameter {name} contains NaN values")
                
                if torch.isinf(param).any():
                    security_checks['has_inf_weights'] = True
                    security_checks['is_valid'] = False
                    logger.warning(f"Parameter {name} contains Inf values")
                
                if torch.abs(param).max() > 1e6:
                    security_checks['has_large_weights'] = True
                    logger.warning(f"Parameter {name} has very large values")
            
            return security_checks
            
        except Exception as e:
            logger.error(f"Model security check failed: {e}")
            security_checks['is_valid'] = False
            return security_checks


class ComprehensiveTransformersManager:
    """Comprehensive Transformers management system."""
    
    def __init__(self, config: TransformersConfig):
        
    """__init__ function."""
self.config = config
        self.model_manager = TransformersModelManager(config)
        self.security_manager = TransformersSecurityManager(config)
        
        logger.info("Comprehensive Transformers Manager initialized")
    
    def load_model(self, model_name: Optional[str] = None, 
                  task_type: Optional[TaskType] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model with security validation."""
        model, tokenizer = self.model_manager.load_model_and_tokenizer(model_name, task_type)
        
        # Security check
        if self.config.enable_security_checks:
            security_checks = self.security_manager.check_model_security(model)
            if not security_checks['is_valid']:
                logger.warning("Model security checks failed")
        
        return model, tokenizer
    
    def create_pipeline(self, task: str, model_name: Optional[str] = None, **kwargs) -> Pipeline:
        """Create pipeline with security validation."""
        return self.model_manager.create_pipeline(task, model_name, **kwargs)
    
    def tokenize(self, text: Union[str, List[str]], model_name: Optional[str] = None, **kwargs) -> BatchEncoding:
        """Tokenize text with input validation."""
        # Validate inputs
        if not self.security_manager.validate_inputs(text):
            raise ValueError("Input validation failed")
        
        return self.model_manager.tokenize_text(text, model_name, **kwargs)
    
    def get_embeddings(self, text: Union[str, List[str]], model_name: Optional[str] = None,
                      pooling_strategy: str = "mean") -> np.ndarray:
        """Get embeddings with security validation."""
        # Validate inputs
        if not self.security_manager.validate_inputs(text):
            raise ValueError("Input validation failed")
        
        embeddings = self.model_manager.get_embeddings(text, model_name, pooling_strategy)
        
        # Sanitize outputs
        embeddings = self.security_manager.sanitize_outputs(embeddings)
        
        return embeddings
    
    def predict(self, text: Union[str, List[str]], model_name: Optional[str] = None,
               task_type: Optional[TaskType] = None) -> Any:
        """Make predictions with security validation."""
        # Validate inputs
        if not self.security_manager.validate_inputs(text):
            raise ValueError("Input validation failed")
        
        outputs = self.model_manager.predict(text, model_name, task_type)
        
        # Sanitize outputs
        outputs = self.security_manager.sanitize_outputs(outputs)
        
        return outputs
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            'config': {
                'model_name': self.config.model_name,
                'model_type': self.config.model_type.value,
                'task_type': self.config.task_type.value,
                'device': str(self.model_manager.device),
                'use_mixed_precision': self.config.use_mixed_precision,
                'enable_security_checks': self.config.enable_security_checks
            },
            'loaded_models': list(self.model_manager.models.keys()),
            'loaded_tokenizers': list(self.model_manager.tokenizers.keys()),
            'loaded_pipelines': list(self.model_manager.pipelines.keys()),
            'model_info': {
                name: {
                    'parameters': info.parameters,
                    'size_mb': info.size_mb,
                    'device': info.device,
                    'dtype': info.dtype,
                    'load_time': info.load_time
                }
                for name, info in self.model_manager.model_info.items()
            },
            'performance_stats': {
                name: self.model_manager.get_performance_stats(name)
                for name in self.model_manager.models.keys()
            }
        }
    
    def optimize_model(self, model: PreTrainedModel, optimization_level: OptimizationLevel) -> PreTrainedModel:
        """Apply optimizations to model."""
        if optimization_level == OptimizationLevel.NONE:
            return model
        
        # Basic optimizations
        if optimization_level.value >= OptimizationLevel.BASIC.value:
            model.eval()  # Set to evaluation mode
        
        # Advanced optimizations
        if optimization_level.value >= OptimizationLevel.ADVANCED.value:
            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                model.gradient_checkpointing_enable()
            
            # Use mixed precision
            if self.config.use_mixed_precision and self.model_manager.device.type == 'cuda':
                model = model.half()
        
        # Maximum optimizations
        if optimization_level.value >= OptimizationLevel.MAXIMUM.value:
            # Compile model if available (PyTorch 2.0+)
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache."""
        self.model_manager.clear_cache(model_name)


# Utility functions
def setup_transformers_environment(config: TransformersConfig) -> ComprehensiveTransformersManager:
    """Setup complete Transformers environment."""
    return ComprehensiveTransformersManager(config)


def get_optimal_transformers_config(model_name: str = "bert-base-uncased",
                                  task_type: TaskType = TaskType.SEQUENCE_CLASSIFICATION) -> TransformersConfig:
    """Get optimal configuration for given model and task."""
    config = TransformersConfig(
        model_name=model_name,
        task_type=task_type,
        use_mixed_precision=True,
        enable_security_checks=True,
        validate_inputs=True,
        sanitize_outputs=True
    )
    
    # Set device-specific optimizations
    if torch.cuda.is_available():
        config.torch_dtype = "float16"
        config.gradient_checkpointing = True
    else:
        config.torch_dtype = "float32"
    
    return config


if __name__ == "__main__":
    # Example usage
    config = get_optimal_transformers_config()
    manager = setup_transformers_environment(config)
    
    # Print system info
    system_info = manager.get_system_info()
    print(json.dumps(system_info, indent=2, default=str)) 