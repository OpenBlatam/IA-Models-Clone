from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from typing_extensions import TypedDict
import torch
import numpy as np
from transformers import (
from transformers.utils import logging as transformers_logging
import psutil
import gc
                from transformers import BitsAndBytesConfig
                from transformers import BitsAndBytesConfig
from typing import Any, List, Dict, Optional
"""
Transformers Library Manager for Pre-trained Models and Tokenizers
================================================================

This module provides a comprehensive interface for working with Hugging Face
Transformers library, including model loading, tokenization, inference,
and optimization for cybersecurity applications.

Features:
- Lazy loading of models and tokenizers
- Memory-efficient model management
- Batch processing capabilities
- GPU/CPU optimization
- Model caching and persistence
- Error handling and recovery
- Performance monitoring
- Security-focused model validation

Author: AI Assistant
License: MIT
"""


    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline,
    PreTrainedTokenizer, PreTrainedModel, Pipeline
)

# Configure logging
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()


class ModelType(Enum):
    """Supported model types for different tasks."""
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    QUESTION_ANSWERING = "question_answering"
    CAUSAL_LANGUAGE_MODEL = "causal_language_model"
    SEQ2SEQ_LANGUAGE_MODEL = "seq2seq_language_model"
    GENERIC = "generic"
    PIPELINE = "pipeline"


class DeviceType(Enum):
    """Available device types for model execution."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


@dataclass
class ModelConfig:
    """Configuration for model loading and management."""
    model_name: str
    model_type: ModelType
    device: DeviceType = DeviceType.AUTO
    max_length: int = 512
    batch_size: int = 1
    use_cache: bool = True
    low_cpu_mem_usage: bool = True
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class TokenizationResult:
    """Result of tokenization operation."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
    tokens: List[str] = field(default_factory=list)
    token_count: int = 0
    processing_time: float = 0.0


@dataclass
class InferenceResult:
    """Result of model inference."""
    predictions: Union[torch.Tensor, List, Dict]
    confidence_scores: Optional[List[float]] = None
    processing_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    model_outputs: Optional[Dict[str, Any]] = None


class ModelMetrics(TypedDict):
    """Metrics for model performance monitoring."""
    load_time: float
    inference_time: float
    memory_usage: float
    throughput: float
    error_count: int
    success_count: int


class TransformersManager:
    """
    Comprehensive manager for Hugging Face Transformers library.
    
    Provides lazy loading, memory management, and optimization for
    pre-trained models and tokenizers in cybersecurity applications.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the Transformers Manager.
        
        Args:
            cache_dir: Directory for caching models and tokenizers
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "transformers")
        self._models: Dict[str, PreTrainedModel] = {}
        self._tokenizers: Dict[str, PreTrainedTokenizer] = {}
        self._pipelines: Dict[str, Pipeline] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        self._metrics: Dict[str, ModelMetrics] = {}
        self._device = self._detect_device()
        self._lock = asyncio.Lock()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"TransformersManager initialized with device: {self._device}")
    
    def _detect_device(self) -> torch.device:
        """Detect the best available device for model execution."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    async def load_model(
        self,
        config: ModelConfig,
        force_reload: bool = False
    ) -> PreTrainedModel:
        """
        Load a pre-trained model with lazy loading and caching.
        
        Args:
            config: Model configuration
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded model
        """
        model_key = f"{config.model_name}_{config.model_type.value}"
        
        async with self._lock:
            if model_key in self._models and not force_reload:
                logger.info(f"Model {model_key} already loaded, returning cached version")
                return self._models[model_key]
            
            start_time = time.time()
            
            try:
                logger.info(f"Loading model: {config.model_name} ({config.model_type.value})")
                
                # Determine device
                device = self._get_device(config.device)
                
                # Model loading parameters
                model_kwargs = {
                    "trust_remote_code": config.trust_remote_code,
                    "low_cpu_mem_usage": config.low_cpu_mem_usage,
                    "cache_dir": config.cache_dir or self.cache_dir,
                    "local_files_only": config.local_files_only,
                }
                
                if config.revision:
                    model_kwargs["revision"] = config.revision
                
                if config.torch_dtype:
                    model_kwargs["torch_dtype"] = config.torch_dtype
                
                # Load model based on type
                model = await self._load_model_by_type(config, model_kwargs)
                
                # Apply quantization if specified
                if config.load_in_8bit or config.load_in_4bit:
                    model = await self._apply_quantization(model, config)
                
                # Move to device
                model = model.to(device)
                model.eval()
                
                # Cache the model
                self._models[model_key] = model
                self._model_configs[model_key] = config
                
                # Update metrics
                load_time = time.time() - start_time
                self._metrics[model_key] = {
                    "load_time": load_time,
                    "inference_time": 0.0,
                    "memory_usage": self._get_memory_usage(),
                    "throughput": 0.0,
                    "error_count": 0,
                    "success_count": 0
                }
                
                logger.info(f"Model {model_key} loaded successfully in {load_time:.2f}s")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {config.model_name}: {str(e)}")
                raise
    
    async def _load_model_by_type(
        self,
        config: ModelConfig,
        model_kwargs: Dict[str, Any]
    ) -> PreTrainedModel:
        """Load model based on its type."""
        model_class_map = {
            ModelType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
            ModelType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
            ModelType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
            ModelType.CAUSAL_LANGUAGE_MODEL: AutoModelForCausalLM,
            ModelType.SEQ2SEQ_LANGUAGE_MODEL: AutoModelForSeq2SeqLM,
            ModelType.GENERIC: AutoModel,
        }
        
        model_class = model_class_map.get(config.model_type, AutoModel)
        
        # Run model loading in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: model_class.from_pretrained(config.model_name, **model_kwargs)
        )
    
    async def _apply_quantization(
        self,
        model: PreTrainedModel,
        config: ModelConfig
    ) -> PreTrainedModel:
        """Apply quantization to the model."""
        try:
            if config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                return model.quantize(quantization_config)
            
            elif config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
                    bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type
                )
                return model.quantize(quantization_config)
            
            return model
            
        except ImportError:
            logger.warning("BitsAndBytes not available, skipping quantization")
            return model
    
    def _get_device(self, device_type: DeviceType) -> torch.device:
        """Get the appropriate device for model execution."""
        if device_type == DeviceType.AUTO:
            return self._device
        elif device_type == DeviceType.CUDA and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == DeviceType.MPS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    async def load_tokenizer(
        self,
        model_name: str,
        force_reload: bool = False
    ) -> PreTrainedTokenizer:
        """
        Load a tokenizer for the specified model.
        
        Args:
            model_name: Name of the model
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded tokenizer
        """
        async with self._lock:
            if model_name in self._tokenizers and not force_reload:
                return self._tokenizers[model_name]
            
            try:
                logger.info(f"Loading tokenizer for model: {model_name}")
                
                loop = asyncio.get_event_loop()
                tokenizer = await loop.run_in_executor(
                    None,
                    lambda: AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )
                )
                
                # Set padding token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self._tokenizers[model_name] = tokenizer
                logger.info(f"Tokenizer for {model_name} loaded successfully")
                return tokenizer
                
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {model_name}: {str(e)}")
                raise
    
    async def tokenize_text(
        self,
        text: Union[str, List[str]],
        model_name: str,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = True,
        return_tensors: str = "pt"
    ) -> TokenizationResult:
        """
        Tokenize text using the specified model's tokenizer.
        
        Args:
            text: Text or list of texts to tokenize
            model_name: Name of the model
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences
            padding: Whether to pad sequences
            return_tensors: Type of tensors to return
            
        Returns:
            Tokenization result
        """
        start_time = time.time()
        
        try:
            tokenizer = await self.load_tokenizer(model_name)
            
            # Get max_length from config if not provided
            if max_length is None:
                model_key = next((k for k in self._model_configs.keys() if model_name in k), None)
                if model_key:
                    max_length = self._model_configs[model_key].max_length
                else:
                    max_length = 512
            
            # Tokenize
            loop = asyncio.get_event_loop()
            tokenized = await loop.run_in_executor(
                None,
                lambda: tokenizer(
                    text,
                    max_length=max_length,
                    truncation=truncation,
                    padding=padding,
                    return_tensors=return_tensors,
                    return_token_type_ids=True
                )
            )
            
            # Get token count
            if isinstance(text, str):
                token_count = len(tokenizer.encode(text))
            else:
                token_count = sum(len(tokenizer.encode(t)) for t in text)
            
            processing_time = time.time() - start_time
            
            return TokenizationResult(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                token_type_ids=tokenized.get("token_type_ids"),
                tokens=tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0]) if return_tensors == "pt" else [],
                token_count=token_count,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise
    
    async def run_inference(
        self,
        model_key: str,
        inputs: Union[str, List[str], torch.Tensor, Dict[str, torch.Tensor]],
        batch_size: Optional[int] = None
    ) -> InferenceResult:
        """
        Run inference using the specified model.
        
        Args:
            model_key: Key of the loaded model
            inputs: Input data for inference
            batch_size: Batch size for processing
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        if model_key not in self._models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model = self._models[model_key]
        config = self._model_configs[model_key]
        
        try:
            # Use model's batch size if not specified
            if batch_size is None:
                batch_size = config.batch_size
            
            # Prepare inputs
            if isinstance(inputs, str):
                # Tokenize text input
                tokenized = await self.tokenize_text(
                    inputs,
                    config.model_name,
                    max_length=config.max_length
                )
                model_inputs = {
                    "input_ids": tokenized.input_ids,
                    "attention_mask": tokenized.attention_mask
                }
                if tokenized.token_type_ids is not None:
                    model_inputs["token_type_ids"] = tokenized.token_type_ids
            elif isinstance(inputs, list):
                # Tokenize multiple texts
                tokenized = await self.tokenize_text(
                    inputs,
                    config.model_name,
                    max_length=config.max_length
                )
                model_inputs = {
                    "input_ids": tokenized.input_ids,
                    "attention_mask": tokenized.attention_mask
                }
                if tokenized.token_type_ids is not None:
                    model_inputs["token_type_ids"] = tokenized.token_type_ids
            else:
                model_inputs = inputs
            
            # Run inference
            with torch.no_grad():
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(
                    None,
                    lambda: model(**model_inputs)
                )
            
            # Process outputs based on model type
            predictions, confidence_scores = self._process_model_outputs(
                outputs, config.model_type
            )
            
            processing_time = time.time() - start_time
            
            # Update metrics
            if model_key in self._metrics:
                self._metrics[model_key]["inference_time"] += processing_time
                self._metrics[model_key]["success_count"] += 1
                self._metrics[model_key]["throughput"] = (
                    self._metrics[model_key]["success_count"] / 
                    self._metrics[model_key]["inference_time"]
                )
            
            return InferenceResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                memory_usage=self._get_memory_usage(),
                model_outputs=outputs
            )
            
        except Exception as e:
            logger.error(f"Inference failed for {model_key}: {str(e)}")
            
            # Update error metrics
            if model_key in self._metrics:
                self._metrics[model_key]["error_count"] += 1
            
            raise
    
    def _process_model_outputs(
        self,
        outputs: Any,
        model_type: ModelType
    ) -> Tuple[Any, Optional[List[float]]]:
        """Process model outputs based on model type."""
        if model_type == ModelType.SEQUENCE_CLASSIFICATION:
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0].tolist()
            return predictions.tolist(), confidence_scores
        
        elif model_type == ModelType.TOKEN_CLASSIFICATION:
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0].tolist()
            return predictions.tolist(), confidence_scores
        
        elif model_type == ModelType.QUESTION_ANSWERING:
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)
            
            start_predictions = torch.argmax(start_logits, dim=-1)
            end_predictions = torch.argmax(end_logits, dim=-1)
            
            confidence_scores = [
                start_probs[i, start_predictions[i]].item() * 
                end_probs[i, end_predictions[i]].item()
                for i in range(len(start_predictions))
            ]
            
            return {
                "start_positions": start_predictions.tolist(),
                "end_positions": end_predictions.tolist()
            }, confidence_scores
        
        elif model_type in [ModelType.CAUSAL_LANGUAGE_MODEL, ModelType.SEQ2SEQ_LANGUAGE_MODEL]:
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            return predictions.tolist(), None
        
        else:
            # Generic model - return raw outputs
            return outputs, None
    
    async def create_pipeline(
        self,
        task: str,
        model_name: str,
        device: Optional[DeviceType] = None
    ) -> Pipeline:
        """
        Create a Hugging Face pipeline for the specified task.
        
        Args:
            task: Pipeline task (e.g., 'text-classification', 'token-classification')
            model_name: Name of the model
            device: Device to run the pipeline on
            
        Returns:
            Created pipeline
        """
        pipeline_key = f"{task}_{model_name}"
        
        async with self._lock:
            if pipeline_key in self._pipelines:
                return self._pipelines[pipeline_key]
            
            try:
                logger.info(f"Creating pipeline for task: {task}, model: {model_name}")
                
                device_id = self._get_device(device or DeviceType.AUTO)
                
                loop = asyncio.get_event_loop()
                pipeline_obj = await loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        task,
                        model=model_name,
                        device=device_id,
                        cache_dir=self.cache_dir
                    )
                )
                
                self._pipelines[pipeline_key] = pipeline_obj
                logger.info(f"Pipeline {pipeline_key} created successfully")
                return pipeline_obj
                
            except Exception as e:
                logger.error(f"Failed to create pipeline {pipeline_key}: {str(e)}")
                raise
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    
    def get_metrics(self, model_key: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for models."""
        if model_key:
            return self._metrics.get(model_key, {})
        return self._metrics
    
    def clear_cache(self, model_key: Optional[str] = None):
        """Clear model cache to free memory."""
        if model_key:
            if model_key in self._models:
                del self._models[model_key]
            if model_key in self._tokenizers:
                del self._tokenizers[model_key]
            if model_key in self._pipelines:
                del self._pipelines[model_key]
            if model_key in self._model_configs:
                del self._model_configs[model_key]
            if model_key in self._metrics:
                del self._metrics[model_key]
        else:
            self._models.clear()
            self._tokenizers.clear()
            self._pipelines.clear()
            self._model_configs.clear()
            self._metrics.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")
    
    @asynccontextmanager
    async def model_context(self, config: ModelConfig):
        """Context manager for model loading and cleanup."""
        model = None
        try:
            model = await self.load_model(config)
            yield model
        finally:
            if model is not None:
                # Optionally clear cache after use
                pass
    
    def list_loaded_models(self) -> List[str]:
        """List all currently loaded models."""
        return list(self._models.keys())
    
    def list_loaded_tokenizers(self) -> List[str]:
        """List all currently loaded tokenizers."""
        return list(self._tokenizers.keys())
    
    def list_loaded_pipelines(self) -> List[str]:
        """List all currently loaded pipelines."""
        return list(self._pipelines.keys())


# Global instance for easy access
transformers_manager = TransformersManager() 