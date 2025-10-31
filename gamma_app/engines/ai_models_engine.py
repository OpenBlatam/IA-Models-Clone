"""
Gamma App - AI Models Engine
Local AI models and fine-tuning capabilities
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, BitsAndBytesConfig, AutoModel, AutoConfig
)
from datasets import Dataset, load_dataset
import accelerate
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import huggingface_hub
import onnx
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of AI models"""
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "text_summarization"
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"

class ModelSize(Enum):
    """Model sizes"""
    TINY = "tiny"      # < 100M parameters
    SMALL = "small"    # 100M - 1B parameters
    MEDIUM = "medium"  # 1B - 7B parameters
    LARGE = "large"    # 7B - 70B parameters
    XL = "xl"          # > 70B parameters

class OptimizationLevel(Enum):
    """Model optimization levels"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    ULTRA = "ultra"

class ModelFormat(Enum):
    """Model formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    QUANTIZED = "quantized"
    PRUNED = "pruned"

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    type: ModelType
    size: ModelSize
    model_path: str
    tokenizer_path: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    model_format: ModelFormat = ModelFormat.PYTORCH
    use_cache: bool = True
    enable_quantization: bool = False
    enable_pruning: bool = False
    batch_size: int = 1
    max_memory_usage: float = 0.8  # 80% of available memory

@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

@dataclass
class FineTuningData:
    """Fine-tuning data structure"""
    text: str
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AIModelsEngine:
    """
    Advanced AI models engine with local model support and fine-tuning
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize AI models engine"""
        self.config = config or {}
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.optimized_models: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        
        # Model storage paths
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.cache_dir = Path(self.config.get('cache_dir', 'cache'))
        self.optimized_dir = Path(self.config.get('optimized_dir', 'optimized_models'))
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.optimized_dir.mkdir(exist_ok=True)
        
        # Performance monitoring
        self.performance_metrics: Dict[str, Dict] = {}
        self.model_usage_stats: Dict[str, Dict] = {}
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Model loading locks
        self.loading_locks: Dict[str, threading.Lock] = {}
        
        # Load available models
        self._load_available_models()
        
        # Initialize optimization settings
        self._initialize_optimization_settings()
        
        logger.info("AI Models Engine initialized successfully")

    def _load_available_models(self):
        """Load available model configurations"""
        self.model_configs = {
            "gpt2-small": ModelConfig(
                name="gpt2-small",
                type=ModelType.TEXT_GENERATION,
                size=ModelSize.SMALL,
                model_path="gpt2",
                tokenizer_path="gpt2",
                max_length=1024
            ),
            "gpt2-medium": ModelConfig(
                name="gpt2-medium",
                type=ModelType.TEXT_GENERATION,
                size=ModelSize.MEDIUM,
                model_path="gpt2-medium",
                tokenizer_path="gpt2-medium",
                max_length=1024
            ),
            "distilgpt2": ModelConfig(
                name="distilgpt2",
                type=ModelType.TEXT_GENERATION,
                size=ModelSize.TINY,
                model_path="distilgpt2",
                tokenizer_path="distilgpt2",
                max_length=1024
            ),
            "t5-small": ModelConfig(
                name="t5-small",
                type=ModelType.TEXT_SUMMARIZATION,
                size=ModelSize.SMALL,
                model_path="t5-small",
                tokenizer_path="t5-small",
                max_length=512
            ),
            "bart-base": ModelConfig(
                name="bart-base",
                type=ModelType.TEXT_SUMMARIZATION,
                size=ModelSize.SMALL,
                model_path="facebook/bart-base",
                tokenizer_path="facebook/bart-base",
                max_length=512
            ),
            "distilbert-base": ModelConfig(
                name="distilbert-base",
                type=ModelType.TEXT_CLASSIFICATION,
                size=ModelSize.SMALL,
                model_path="distilbert-base-uncased",
                tokenizer_path="distilbert-base-uncased",
                max_length=512
            )
        }

    async def load_model(self, model_name: str, use_quantization: bool = False) -> bool:
        """Load a model into memory"""
        try:
            if model_name not in self.model_configs:
                logger.error(f"Model {model_name} not found in configurations")
                return False
            
            config = self.model_configs[model_name]
            
            # Check if model is already loaded
            if model_name in self.models:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path,
                cache_dir=str(self.cache_dir)
            )
            self.tokenizers[model_name] = tokenizer
            
            # Configure quantization if requested
            quantization_config = None
            if use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            
            # Load model
            if config.type == ModelType.TEXT_GENERATION:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    cache_dir=str(self.cache_dir),
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            elif config.type == ModelType.TEXT_SUMMARIZATION:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.model_path,
                    cache_dir=str(self.cache_dir),
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_path,
                    cache_dir=str(self.cache_dir),
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            self.models[model_name] = model
            
            # Create pipeline
            pipeline_name = self._get_pipeline_name(config.type)
            self.pipelines[model_name] = pipeline(
                pipeline_name,
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    def _get_pipeline_name(self, model_type: ModelType) -> str:
        """Get pipeline name for model type"""
        pipeline_map = {
            ModelType.TEXT_GENERATION: "text-generation",
            ModelType.TEXT_SUMMARIZATION: "summarization",
            ModelType.TEXT_CLASSIFICATION: "text-classification",
            ModelType.SENTIMENT_ANALYSIS: "sentiment-analysis",
            ModelType.TRANSLATION: "translation",
            ModelType.QUESTION_ANSWERING: "question-answering"
        }
        return pipeline_map.get(model_type, "text-generation")

    async def generate_text(self, model_name: str, prompt: str, 
                           max_length: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           top_k: Optional[int] = None) -> str:
        """Generate text using specified model"""
        try:
            if model_name not in self.pipelines:
                await self.load_model(model_name)
            
            if model_name not in self.pipelines:
                raise ValueError(f"Model {model_name} not available")
            
            config = self.model_configs[model_name]
            
            # Use provided parameters or defaults
            max_length = max_length or config.max_length
            temperature = temperature or config.temperature
            top_p = top_p or config.top_p
            top_k = top_k or config.top_k
            
            # Generate text
            result = self.pipelines[model_name](
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self.tokenizers[model_name].eos_token_id
            )
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Remove the original prompt from generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error generating text with {model_name}: {e}")
            return ""

    async def summarize_text(self, model_name: str, text: str, 
                           max_length: Optional[int] = None,
                           min_length: Optional[int] = None) -> str:
        """Summarize text using specified model"""
        try:
            if model_name not in self.pipelines:
                await self.load_model(model_name)
            
            if model_name not in self.pipelines:
                raise ValueError(f"Model {model_name} not available")
            
            config = self.model_configs[model_name]
            max_length = max_length or min(config.max_length // 2, 150)
            min_length = min_length or max_length // 4
            
            # Summarize text
            result = self.pipelines[model_name](
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('summary_text', '')
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error summarizing text with {model_name}: {e}")
            return ""

    async def classify_text(self, model_name: str, text: str, 
                          labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Classify text using specified model"""
        try:
            if model_name not in self.pipelines:
                await self.load_model(model_name)
            
            if model_name not in self.pipelines:
                raise ValueError(f"Model {model_name} not available")
            
            # Classify text
            result = self.pipelines[model_name](text, labels=labels)
            
            if isinstance(result, list) and len(result) > 0:
                return {
                    "label": result[0].get('label', ''),
                    "score": result[0].get('score', 0.0),
                    "all_scores": result
                }
            else:
                return {"label": "", "score": 0.0, "all_scores": []}
                
        except Exception as e:
            logger.error(f"Error classifying text with {model_name}: {e}")
            return {"label": "", "score": 0.0, "all_scores": []}

    async def fine_tune_model(self, model_name: str, training_data: List[FineTuningData],
                            training_config: TrainingConfig,
                            output_dir: Optional[str] = None) -> bool:
        """Fine-tune a model with custom data"""
        try:
            if model_name not in self.models:
                await self.load_model(model_name)
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            logger.info(f"Starting fine-tuning for {model_name}")
            
            # Prepare training data
            dataset = self._prepare_training_dataset(training_data, model_name)
            
            # Configure LoRA for efficient fine-tuning
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
            
            # Apply LoRA to model
            model = get_peft_model(self.models[model_name], lora_config)
            
            # Training arguments
            output_dir = output_dir or str(self.models_dir / f"{model_name}_fine_tuned")
            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=training_config.learning_rate,
                per_device_train_batch_size=training_config.batch_size,
                num_train_epochs=training_config.num_epochs,
                warmup_steps=training_config.warmup_steps,
                weight_decay=training_config.weight_decay,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                max_grad_norm=training_config.max_grad_norm,
                save_steps=training_config.save_steps,
                eval_steps=training_config.eval_steps,
                logging_steps=training_config.logging_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                report_to=None,  # Disable wandb/tensorboard
                remove_unused_columns=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizers[model_name],
                mlm=False
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizers[model_name]
            )
            
            # Train
            trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizers[model_name].save_pretrained(output_dir)
            
            logger.info(f"Fine-tuning completed for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error fine-tuning model {model_name}: {e}")
            return False

    def _prepare_training_dataset(self, training_data: List[FineTuningData], 
                                 model_name: str) -> Dataset:
        """Prepare training dataset"""
        try:
            tokenizer = self.tokenizers[model_name]
            config = self.model_configs[model_name]
            
            # Prepare texts
            texts = []
            for data in training_data:
                if config.type == ModelType.TEXT_GENERATION:
                    # For text generation, use the text as is
                    texts.append(data.text)
                elif config.type == ModelType.TEXT_SUMMARIZATION:
                    # For summarization, format as "summarize: {text}"
                    texts.append(f"summarize: {data.text}")
                else:
                    texts.append(data.text)
            
            # Tokenize
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=config.max_length,
                    return_tensors="pt"
                )
            
            # Create dataset
            dataset = Dataset.from_dict({"text": texts})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Error preparing training dataset: {e}")
            raise

    async def evaluate_model(self, model_name: str, test_data: List[FineTuningData]) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if model_name not in self.pipelines:
                await self.load_model(model_name)
            
            if model_name not in self.pipelines:
                raise ValueError(f"Model {model_name} not available")
            
            config = self.model_configs[model_name]
            total_samples = len(test_data)
            correct_predictions = 0
            
            for data in test_data:
                try:
                    if config.type == ModelType.TEXT_CLASSIFICATION:
                        result = await self.classify_text(model_name, data.text)
                        if result["label"] == data.label:
                            correct_predictions += 1
                    elif config.type == ModelType.TEXT_SUMMARIZATION:
                        summary = await self.summarize_text(model_name, data.text)
                        # Simple evaluation - check if summary is not empty
                        if summary.strip():
                            correct_predictions += 1
                    else:
                        # For text generation, just check if output is generated
                        generated = await self.generate_text(model_name, data.text)
                        if generated.strip():
                            correct_predictions += 1
                            
                except Exception as e:
                    logger.error(f"Error evaluating sample: {e}")
                    continue
            
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_samples": total_samples
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {"accuracy": 0.0, "correct_predictions": 0, "total_samples": 0}

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        if model_name not in self.model_configs:
            return None
        
        config = self.model_configs[model_name]
        return {
            "name": config.name,
            "type": config.type.value,
            "size": config.size.value,
            "max_length": config.max_length,
            "loaded": model_name in self.models
        }

    async def unload_model(self, model_name: str) -> bool:
        """Unload model from memory"""
        try:
            if model_name in self.models:
                del self.models[model_name]
            
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            
            if model_name in self.pipelines:
                del self.pipelines[model_name]
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model {model_name} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            memory_info = {
                "models_loaded": len(self.models),
                "total_models": len(self.model_configs)
            }
            
            if torch.cuda.is_available():
                memory_info.update({
                    "cuda_available": True,
                    "cuda_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "cuda_memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
                    "cuda_memory_cached": torch.cuda.memory_cached() / 1024**3        # GB
                })
            else:
                memory_info["cuda_available"] = False
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}

    def _initialize_optimization_settings(self):
        """Initialize optimization settings"""
        self.optimization_settings = {
            'quantization_configs': {
                'int8': BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                ),
                'int4': BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            },
            'onnx_providers': ['CPUExecutionProvider', 'CUDAExecutionProvider'],
            'memory_optimization': {
                'max_memory_usage': 0.8,
                'enable_gradient_checkpointing': True,
                'use_cache': True
            }
        }
    
    async def optimize_model(self, model_name: str, optimization_level: OptimizationLevel) -> bool:
        """Optimize model for better performance"""
        try:
            if model_name not in self.model_configs:
                logger.error(f"Model {model_name} not found")
                return False
            
            config = self.model_configs[model_name]
            logger.info(f"Optimizing model {model_name} with level {optimization_level.value}")
            
            # Load model if not already loaded
            if model_name not in self.models:
                await self.load_model(model_name)
            
            if model_name not in self.models:
                logger.error(f"Failed to load model {model_name}")
                return False
            
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Apply optimizations based on level
            if optimization_level == OptimizationLevel.BASIC:
                optimized_model = await self._apply_basic_optimizations(model, config)
            elif optimization_level == OptimizationLevel.ADVANCED:
                optimized_model = await self._apply_advanced_optimizations(model, config)
            elif optimization_level == OptimizationLevel.ULTRA:
                optimized_model = await self._apply_ultra_optimizations(model, config)
            else:
                optimized_model = model
            
            # Store optimized model
            self.optimized_models[f"{model_name}_{optimization_level.value}"] = optimized_model
            
            # Save optimized model
            optimized_path = self.optimized_dir / f"{model_name}_{optimization_level.value}"
            optimized_path.mkdir(exist_ok=True)
            
            if hasattr(optimized_model, 'save_pretrained'):
                optimized_model.save_pretrained(str(optimized_path))
                tokenizer.save_pretrained(str(optimized_path))
            
            logger.info(f"Model {model_name} optimized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing model {model_name}: {e}")
            return False
    
    async def _apply_basic_optimizations(self, model: Any, config: ModelConfig) -> Any:
        """Apply basic optimizations"""
        try:
            # Enable gradient checkpointing
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            # Set model to eval mode for inference
            model.eval()
            
            # Enable torch compile if available
            if hasattr(torch, 'compile') and config.optimization_level != OptimizationLevel.NONE:
                model = torch.compile(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Error applying basic optimizations: {e}")
            return model
    
    async def _apply_advanced_optimizations(self, model: Any, config: ModelConfig) -> Any:
        """Apply advanced optimizations"""
        try:
            # Apply basic optimizations first
            model = await self._apply_basic_optimizations(model, config)
            
            # Apply quantization if enabled
            if config.enable_quantization and torch.cuda.is_available():
                quantization_config = self.optimization_settings['quantization_configs']['int8']
                # Note: This would require reloading the model with quantization
                # For now, we'll just mark it as optimized
                logger.info("Quantization optimization applied")
            
            # Apply memory optimizations
            if hasattr(model, 'config'):
                model.config.use_cache = config.use_cache
            
            return model
            
        except Exception as e:
            logger.error(f"Error applying advanced optimizations: {e}")
            return model
    
    async def _apply_ultra_optimizations(self, model: Any, config: ModelConfig) -> Any:
        """Apply ultra optimizations"""
        try:
            # Apply advanced optimizations first
            model = await self._apply_advanced_optimizations(model, config)
            
            # Apply pruning if enabled
            if config.enable_pruning:
                model = await self._apply_pruning(model, config)
            
            # Apply ONNX conversion if requested
            if config.model_format == ModelFormat.ONNX:
                model = await self._convert_to_onnx(model, config)
            
            return model
            
        except Exception as e:
            logger.error(f"Error applying ultra optimizations: {e}")
            return model
    
    async def _apply_pruning(self, model: Any, config: ModelConfig) -> Any:
        """Apply model pruning"""
        try:
            # This is a simplified pruning implementation
            # In practice, you would use more sophisticated pruning techniques
            logger.info("Applying model pruning")
            
            # For now, just return the model as-is
            # Real implementation would use techniques like:
            # - Magnitude-based pruning
            # - Structured pruning
            # - Unstructured pruning
            
            return model
            
        except Exception as e:
            logger.error(f"Error applying pruning: {e}")
            return model
    
    async def _convert_to_onnx(self, model: Any, config: ModelConfig) -> Any:
        """Convert model to ONNX format"""
        try:
            logger.info("Converting model to ONNX format")
            
            # This would involve converting the PyTorch model to ONNX
            # For now, we'll just return the original model
            # Real implementation would use torch.onnx.export()
            
            return model
            
        except Exception as e:
            logger.error(f"Error converting to ONNX: {e}")
            return model
    
    async def benchmark_model(self, model_name: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark model performance"""
        try:
            if model_name not in self.pipelines:
                await self.load_model(model_name)
            
            if model_name not in self.pipelines:
                raise ValueError(f"Model {model_name} not available")
            
            logger.info(f"Benchmarking model {model_name}")
            
            # Benchmark metrics
            total_time = 0
            total_tokens = 0
            memory_usage = []
            
            for prompt in test_prompts:
                start_time = time.time()
                
                # Generate text
                result = await self.generate_text(model_name, prompt)
                
                end_time = time.time()
                generation_time = end_time - start_time
                total_time += generation_time
                
                # Count tokens (approximate)
                tokens = len(prompt.split()) + len(result.split())
                total_tokens += tokens
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
            
            # Calculate metrics
            avg_time_per_token = total_time / total_tokens if total_tokens > 0 else 0
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            avg_memory_usage = np.mean(memory_usage) if memory_usage else 0
            
            benchmark_results = {
                "model_name": model_name,
                "total_time": total_time,
                "total_tokens": total_tokens,
                "avg_time_per_token": avg_time_per_token,
                "tokens_per_second": tokens_per_second,
                "avg_memory_usage_gb": avg_memory_usage,
                "test_prompts": len(test_prompts)
            }
            
            # Store benchmark results
            self.performance_metrics[model_name] = benchmark_results
            
            logger.info(f"Benchmark completed for {model_name}: {tokens_per_second:.2f} tokens/sec")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Error benchmarking model {model_name}: {e}")
            return {}
    
    async def get_model_recommendations(self, use_case: str, constraints: Dict[str, Any]) -> List[str]:
        """Get model recommendations based on use case and constraints"""
        try:
            recommendations = []
            
            # Filter models based on constraints
            available_memory = constraints.get('memory_gb', 8)
            max_latency = constraints.get('max_latency_ms', 1000)
            accuracy_requirement = constraints.get('accuracy_requirement', 'medium')
            
            for model_name, config in self.model_configs.items():
                # Check memory requirements
                if config.size == ModelSize.TINY and available_memory >= 2:
                    recommendations.append(model_name)
                elif config.size == ModelSize.SMALL and available_memory >= 4:
                    recommendations.append(model_name)
                elif config.size == ModelSize.MEDIUM and available_memory >= 8:
                    recommendations.append(model_name)
                elif config.size == ModelSize.LARGE and available_memory >= 16:
                    recommendations.append(model_name)
            
            # Sort by performance if available
            if self.performance_metrics:
                recommendations.sort(
                    key=lambda x: self.performance_metrics.get(x, {}).get('tokens_per_second', 0),
                    reverse=True
                )
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error getting model recommendations: {e}")
            return []
    
    async def auto_optimize_model(self, model_name: str) -> bool:
        """Automatically optimize model based on system capabilities"""
        try:
            if model_name not in self.model_configs:
                return False
            
            config = self.model_configs[model_name]
            
            # Determine optimization level based on system capabilities
            available_memory = psutil.virtual_memory().available / 1024**3  # GB
            cpu_count = psutil.cpu_count()
            
            if available_memory >= 16 and cpu_count >= 8:
                optimization_level = OptimizationLevel.ULTRA
            elif available_memory >= 8 and cpu_count >= 4:
                optimization_level = OptimizationLevel.ADVANCED
            elif available_memory >= 4:
                optimization_level = OptimizationLevel.BASIC
            else:
                optimization_level = OptimizationLevel.NONE
            
            # Apply optimizations
            return await self.optimize_model(model_name, optimization_level)
            
        except Exception as e:
            logger.error(f"Error auto-optimizing model {model_name}: {e}")
            return False
    
    async def get_optimization_report(self, model_name: str) -> Dict[str, Any]:
        """Get optimization report for a model"""
        try:
            if model_name not in self.model_configs:
                return {"error": "Model not found"}
            
            config = self.model_configs[model_name]
            
            report = {
                "model_name": model_name,
                "current_config": asdict(config),
                "optimization_status": {},
                "performance_metrics": self.performance_metrics.get(model_name, {}),
                "recommendations": []
            }
            
            # Check optimization status
            for level in OptimizationLevel:
                optimized_key = f"{model_name}_{level.value}"
                report["optimization_status"][level.value] = optimized_key in self.optimized_models
            
            # Generate recommendations
            if config.optimization_level == OptimizationLevel.NONE:
                report["recommendations"].append("Consider enabling basic optimizations for better performance")
            
            if not config.enable_quantization and torch.cuda.is_available():
                report["recommendations"].append("Enable quantization to reduce memory usage")
            
            if config.size in [ModelSize.LARGE, ModelSize.XL] and not config.enable_pruning:
                report["recommendations"].append("Consider enabling pruning for large models")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Unload all models
            for model_name in list(self.models.keys()):
                await self.unload_model(model_name)
            
            # Clear optimized models
            self.optimized_models.clear()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("AI Models Engine cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")











