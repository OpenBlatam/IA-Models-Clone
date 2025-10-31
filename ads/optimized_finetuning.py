from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
from peft import (
import numpy as np
from datasets import Dataset as HFDataset
import json
import os
from datetime import datetime
import asyncio
from functools import lru_cache
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
import hashlib
import time
import psutil
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_config import settings
from onyx.server.features.ads.optimized_db_service import OptimizedAdsDBService
from onyx.server.features.ads.tokenization_service import TokenizationService, OptimizedAdsDataset
from onyx.server.features.ads.training_logger import TrainingLogger, TrainingPhase, AsyncTrainingLogger
from onyx.server.features.ads.performance_optimizer import (
from onyx.server.features.ads.multi_gpu_training import (
from onyx.server.features.ads.gradient_accumulation import (
from onyx.server.features.ads.mixed_precision_training import (
from onyx.server.features.ads.profiling_optimizer import (
from onyx.server.features.ads.data_optimization import (
            import glob
            from datetime import datetime, timedelta
                            import shutil
from typing import Any, List, Dict, Optional
import logging
"""
Optimized fine-tuning service for ads generation using LoRA and P-tuning techniques.
"""
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2SeqLM
)
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training
)

    performance_monitor, 
    cache_result, 
    performance_context, 
    memory_context,
    optimizer
)

    MultiGPUTrainingManager,
    GPUConfig,
    DataParallelTrainer,
    DistributedDataParallelTrainer,
    gpu_monitoring_context,
    cleanup_gpu_memory
)

    GradientAccumulationConfig,
    GradientAccumulator,
    AdaptiveGradientAccumulator,
    GradientAccumulationTrainer,
    calculate_effective_batch_size,
    calculate_accumulation_steps,
    adjust_learning_rate,
    gradient_accumulation_context
)

    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    AdaptiveMixedPrecisionTrainer,
    MixedPrecisionGradientAccumulator,
    create_mixed_precision_config,
    should_use_mixed_precision,
    optimize_mixed_precision_settings,
    mixed_precision_context
)

    ProfilingConfig,
    ProfilingOptimizer,
    profile_function,
    profiling_context
)
    DataOptimizationConfig,
    DataLoadingOptimizer,
    PreprocessingOptimizer,
    MemoryOptimizer,
    IOOptimizer,
    optimize_dataset,
    optimize_dataloader,
    optimize_preprocessing,
    memory_optimization_context
)

logger = setup_logger()

# Removed AdsDataset class - now using OptimizedAdsDataset from tokenization_service

class OptimizedFineTuningService:
    """Optimized fine-tuning service with LoRA and P-tuning techniques."""
    
    def __init__(self) -> Any:
        """Initialize the fine-tuning service."""
        self._redis_client = None
        self.db_service = OptimizedAdsDBService()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models = {}
        self._tokenizers = {}
        self.tokenization_service = TokenizationService()
        self.training_logger = None
        
        # Initialize performance optimizer
        self.optimizer = optimizer
        
        # Initialize multi-GPU training manager
        self.multi_gpu_manager = MultiGPUTrainingManager()
        
        # Initialize gradient accumulation API
        self.gradient_accumulation_api = GradientAccumulationAPI()
        
        # Initialize mixed precision trainer
        self.mp_trainer = None
        self.mp_config = None
        
        # Initialize profiling and optimization systems
        self.profiling_config = ProfilingConfig(
            enabled=True,
            profile_cpu=True,
            profile_memory=True,
            profile_gpu=True,
            profile_data_loading=True,
            profile_preprocessing=True,
            real_time_monitoring=True
        )
        self.profiler = ProfilingOptimizer(self.profiling_config)
        
        # Initialize data optimization
        self.data_optimization_config = DataOptimizationConfig(
            optimize_loading=True,
            memory_efficient=True,
            enable_caching=True,
            parallel_preprocessing=True,
            optimize_io=True
        )
        self.data_loading_optimizer = DataLoadingOptimizer(self.data_optimization_config)
        self.preprocessing_optimizer = PreprocessingOptimizer(self.data_optimization_config)
        self.memory_optimizer = MemoryOptimizer(self.data_optimization_config)
        self.io_optimizer = IOOptimizer(self.data_optimization_config)
        
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    def _get_model_key(self, model_name: str, task: str) -> str:
        """Generate cache key for model."""
        return f"finetuning:{model_name}:{task}"
    
    @lru_cache(maxsize=10)
    def get_tokenizer(self, model_name: str):
        """Get cached tokenizer."""
        if model_name not in self._tokenizers:
            self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            if self._tokenizers[model_name].pad_token is None:
                self._tokenizers[model_name].pad_token = self._tokenizers[model_name].eos_token
        return self._tokenizers[model_name]
    
    def get_advanced_tokenizer(self, model_name: str):
        """Get advanced tokenizer with preprocessing capabilities."""
        return TokenizationService(model_name)
    
    def get_lora_config(self, task_type: str = "CAUSAL_LM") -> LoraConfig:
        """Get optimized LoRA configuration."""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,  # Alpha scaling
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
    
    def get_p_tuning_config(self, num_virtual_tokens: int = 20) -> Dict[str, Any]:
        """Get P-tuning configuration."""
        return {
            "num_virtual_tokens": num_virtual_tokens,
            "encoder_hidden_size": 128,
            "encoder_num_layers": 2,
            "encoder_dropout": 0.1
        }
    
    async def prepare_training_data(
        self,
        user_id: int,
        model_name: str = "microsoft/DialoGPT-medium",
        max_samples: int = 1000
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Prepare training data from user's ads generations.
        
        Args:
            user_id: User ID to get data for
            model_name: Base model name
            max_samples: Maximum number of samples to use
            
        Returns:
            Tuple of (training_data, total_samples)
        """
        # Initialize training logger for this session with autograd debugging
        self.training_logger = AsyncTrainingLogger(
            user_id=user_id,
            model_name=model_name,
            log_dir=f"logs/training/user_{user_id}",
            enable_autograd_debug=True  # Enable PyTorch debugging
        )
        
        try:
            self.training_logger.log_info("Starting data preparation phase", TrainingPhase.DATA_PREPARATION)
            
            # Get user's ads generations
            ads_generations = await self.db_service.list_ads_generations(
                user_id=user_id,
                limit=max_samples
            )
            
            self.training_logger.log_info(f"Retrieved {len(ads_generations)} ads generations from database")
            
            training_data = []
            for i, ad in enumerate(ads_generations):
                if ad.content and ad.prompt:
                    training_data.append({
                        "prompt": ad.prompt,
                        "content": ad.content,
                        "type": ad.type,
                        "target_audience": ad.metadata.get("target_audience"),
                        "keywords": ad.metadata.get("keywords", []),
                        "created_at": ad.created_at.isoformat()
                    })
                
                # Log progress every 100 samples
                if (i + 1) % 100 == 0:
                    self.training_logger.log_info(f"Processed {i + 1}/{len(ads_generations)} samples")
            
            self.training_logger.log_info(f"Successfully prepared {len(training_data)} training samples for user {user_id}")
            return training_data, len(training_data)
            
        except Exception as e:
            self.training_logger.log_error(e, TrainingPhase.DATA_PREPARATION, {
                "user_id": user_id,
                "model_name": model_name,
                "max_samples": max_samples
            })
            raise
    
    @performance_monitor("prepare_dataset")
    @cache_result(ttl=1800, cache_type="l1")
    async def prepare_dataset(
        self,
        texts: List[str],
        model_name: str,
        max_length: int = 512,
        batch_size: int = 32
    ) -> OptimizedAdsDataset:
        """Prepare dataset with performance optimization."""
        async with performance_context("dataset_preparation"):
            with memory_context():
                # Use tokenization service with caching
                tokenized_data = await self.tokenization_service.tokenize_batch(
                    texts, model_name, max_length
                )
                
                # Create dataset with optimized processing
                dataset = OptimizedAdsDataset(
                    tokenized_data,
                    max_length=max_length,
                    batch_size=batch_size
                )
                
                return dataset
    
    @performance_monitor("load_model")
    @cache_result(ttl=3600, cache_type="l2")
    async def load_model(self, model_name: str, task: str = "text-generation"):
        """Load model with performance optimization."""
        model_key = self._get_model_key(model_name, task)
        
        # Check cache first
        cached_model = self.optimizer.cache.get(model_key, "l2")
        if cached_model is not None:
            return cached_model
        
        async with performance_context("model_loading"):
            with memory_context():
                # Load model with memory optimization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                # Cache model
                self.optimizer.cache.set(model_key, model, cache_type="l2")
                return model
    
    @performance_monitor("finetune_model")
    async def finetune_model(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int
    ) -> Dict[str, Any]:
        """Fine-tune model with comprehensive performance optimization."""
        async with performance_context("model_finetuning"):
            with memory_context():
                start_time = time.time()
                
                # Initialize training logger with performance monitoring
                self.training_logger = TrainingLogger(
                    user_id=user_id,
                    model_name=model_name,
                    task_type="finetuning",
                    performance_optimizer=self.optimizer
                )
                
                # Load model with optimization
                model = await self.load_model(model_name)
                
                # Prepare training data
                train_dataloader = DataLoader(
                    dataset,
                    batch_size=training_config.get("batch_size", 8),
                    shuffle=True,
                    num_workers=min(4, os.cpu_count() or 1)
                )
                
                # Setup optimizer with gradient clipping
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=training_config.get("learning_rate", 5e-5),
                    weight_decay=training_config.get("weight_decay", 0.01)
                )
                
                # Learning rate scheduler
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=len(train_dataloader) * training_config.get("epochs", 3)
                )
                
                # Training loop with performance monitoring
                model.train()
                total_loss = 0
                num_batches = 0
                
                for epoch in range(training_config.get("epochs", 3)):
                    epoch_loss = 0
                    epoch_batches = 0
                    
                    for batch_idx, batch in enumerate(train_dataloader):
                        # Memory cleanup if needed
                        if self.optimizer.memory_manager.should_cleanup_memory():
                            await self.optimizer.task_manager.submit_task(
                                self.optimizer.memory_manager.cleanup_memory
                            )
                        
                        # Move batch to device
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        
                        # Forward pass
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        
                        # Backward pass with gradient clipping
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            training_config.get("max_grad_norm", 1.0)
                        )
                        
                        optimizer.step()
                        scheduler.step()
                        
                        # Update metrics
                        total_loss += loss.item()
                        epoch_loss += loss.item()
                        num_batches += 1
                        epoch_batches += 1
                        
                        # Log training progress
                        if self.training_logger:
                            await self.training_logger.log_training_step(
                                epoch=epoch,
                                batch=batch_idx,
                                loss=loss.item(),
                                learning_rate=scheduler.get_last_lr()[0]
                            )
                        
                        # Memory cleanup every N batches
                        if batch_idx % 10 == 0:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                # Calculate final metrics
                avg_loss = total_loss / num_batches
                training_time = time.time() - start_time
                
                # Save model with optimization
                model_path = f"models/{model_name}_finetuned_{int(time.time())}"
                model.save_pretrained(model_path)
                
                # Log final results
                if self.training_logger:
                    await self.training_logger.log_training_completion(
                        total_loss=avg_loss,
                        training_time=training_time,
                        model_path=model_path
                    )
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "avg_loss": avg_loss,
                    "training_time": training_time,
                    "epochs": training_config.get("epochs", 3),
                    "total_batches": num_batches
                }
    
    @performance_monitor("generate_text")
    @cache_result(ttl=3600, cache_type="l1")
    async def generate_text(
        self,
        model_path: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        user_id: int = None
    ) -> Dict[str, Any]:
        """Generate text with performance optimization."""
        async with performance_context("text_generation"):
            with memory_context():
                start_time = time.time()
                
                # Load model with caching
                model = await self.load_model(model_path, "text-generation")
                tokenizer = self.get_tokenizer(model_path)
                
                # Tokenize input with optimization
                inputs = await self.tokenization_service.tokenize_text(
                    prompt, model_path, max_length
                )
                
                # Generate text
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"].to(self.device),
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Calculate metrics
                generation_time = time.time() - start_time
                token_count = len(outputs[0])
                
                # Log generation
                if user_id and self.training_logger:
                    await self.training_logger.log_generation(
                        user_id=user_id,
                        model_path=model_path,
                        prompt=prompt,
                        generated_text=generated_text,
                        generation_time=generation_time,
                        token_count=token_count
                    )
                
                return {
                    "generated_text": generated_text,
                    "generation_time": generation_time,
                    "token_count": token_count,
                    "model_path": model_path
                }
    
    @performance_monitor("evaluate_model")
    async def evaluate_model(
        self,
        model_path: str,
        test_dataset: OptimizedAdsDataset,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model with performance optimization."""
        async with performance_context("model_evaluation"):
            with memory_context():
                start_time = time.time()
                
                # Load model
                model = await self.load_model(model_path, "evaluation")
                model.eval()
                
                # Setup evaluation
                metrics = metrics or ["perplexity", "accuracy"]
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=16,
                    shuffle=False,
                    num_workers=min(4, os.cpu_count() or 1)
                )
                
                total_loss = 0
                total_samples = 0
                correct_predictions = 0
                
                # Evaluation loop
                with torch.no_grad():
                    for batch in test_dataloader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        total_loss += loss.item() * input_ids.size(0)
                        total_samples += input_ids.size(0)
                        
                        # Calculate accuracy if needed
                        if "accuracy" in metrics:
                            predictions = torch.argmax(outputs.logits, dim=-1)
                            correct_predictions += (predictions == labels).sum().item()
                
                # Calculate final metrics
                avg_loss = total_loss / total_samples
                perplexity = torch.exp(torch.tensor(avg_loss))
                accuracy = correct_predictions / total_samples if "accuracy" in metrics else None
                
                evaluation_time = time.time() - start_time
                
                results = {
                    "perplexity": perplexity.item(),
                    "avg_loss": avg_loss,
                    "evaluation_time": evaluation_time,
                    "total_samples": total_samples
                }
                
                if accuracy is not None:
                    results["accuracy"] = accuracy
                
                return results
    
    @performance_monitor("cleanup_resources")
    async def cleanup_resources(self) -> Any:
        """Cleanup resources with performance optimization."""
        async with performance_context("resource_cleanup"):
            # Clear model cache
            self._models.clear()
            self._tokenizers.clear()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force memory cleanup
            await self.optimizer.task_manager.submit_task(
                self.optimizer.memory_manager.cleanup_memory,
                force=True
            )
            
            # Clear training logger
            if self.training_logger:
                await self.training_logger.cleanup()
                self.training_logger = None
            
            return {"success": True, "message": "Resources cleaned up successfully"}
    
    async def generate_with_finetuned_model(
        self,
        user_id: int,
        prompt: str,
        base_model_name: str = "microsoft/DialoGPT-medium",
        max_length: int = 200,
        temperature: float = 0.7
    ) -> str:
        """
        Generate ads using fine-tuned model.
        
        Args:
            user_id: User ID for personalized model
            prompt: Input prompt for generation
            base_model_name: Base model name
            max_length: Maximum generation length
            temperature: Generation temperature
            
        Returns:
            Generated ad content
        """
        try:
            # Check cache for recent generation
            cache_key = f"finetuned_generation:{user_id}:{hashlib.md5(prompt.encode()).hexdigest()}"
            redis = await self.redis_client
            cached_result = await redis.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for fine-tuned generation: {cache_key}")
                return json.loads(cached_result)
            
            # Load fine-tuned model
            model, tokenizer = await self.create_lora_model(base_model_name, "ads_generation", user_id)
            
            # Prepare input using advanced tokenization
            tokenization_service = self.get_advanced_tokenizer(base_model_name)
            inputs = await tokenization_service.tokenize_for_inference(
                prompt, max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Cache result
            await redis.setex(cache_key, 3600, json.dumps(generated_text))  # 1 hour
            
            return generated_text
            
        except Exception as e:
            logger.exception("Error generating with fine-tuned model")
            raise
    
    async def evaluate_model_performance(
        self,
        user_id: int,
        base_model_name: str = "microsoft/DialoGPT-medium"
    ) -> Dict[str, Any]:
        """
        Evaluate fine-tuned model performance.
        
        Args:
            user_id: User ID for model evaluation
            base_model_name: Base model name
            
        Returns:
            Evaluation metrics
        """
        try:
            # Get test data (recent ads generations)
            test_data, _ = await self.prepare_training_data(
                user_id=user_id,
                model_name=base_model_name,
                max_samples=100
            )
            
            if not test_data:
                return {"error": "No test data available"}
            
            # Load model
            model, tokenizer = await self.create_lora_model(base_model_name, "ads_generation", user_id)
            
            # Evaluation metrics
            total_loss = 0
            total_samples = 0
            generated_samples = []
            
            model.eval()
            with torch.no_grad():
                for sample in test_data[:10]:  # Evaluate on 10 samples
                    prompt = sample["prompt"]
                    target = sample["content"].get("content", "")
                    
                    # Generate with fine-tuned model
                    generated = await self.generate_with_finetuned_model(
                        user_id=user_id,
                        prompt=prompt,
                        base_model_name=base_model_name
                    )
                    
                    generated_samples.append({
                        "prompt": prompt,
                        "target": target,
                        "generated": generated
                    })
                    
                    # Calculate loss
                    inputs = tokenizer(
                        prompt + " " + target,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    outputs = model(**inputs)
                    loss = outputs.loss
                    total_loss += loss.item()
                    total_samples += 1
            
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            
            evaluation_metrics = {
                "average_loss": avg_loss,
                "total_samples": total_samples,
                "generated_samples": generated_samples,
                "model_performance": "good" if avg_loss < 2.0 else "needs_improvement"
            }
            
            # Cache evaluation results
            redis = await self.redis_client
            await redis.setex(
                f"finetuning:evaluation:{user_id}",
                3600,  # 1 hour
                json.dumps(evaluation_metrics)
            )
            
            return evaluation_metrics
            
        except Exception as e:
            logger.exception("Error evaluating model performance")
            raise
    
    async def get_training_status(self, user_id: int) -> Dict[str, Any]:
        """
        Get fine-tuning training status for user.
        
        Args:
            user_id: User ID
            
        Returns:
            Training status and metrics
        """
        try:
            redis = await self.redis_client
            
            # Get cached results
            results_key = f"finetuning:results:{user_id}"
            evaluation_key = f"finetuning:evaluation:{user_id}"
            
            results = await redis.get(results_key)
            evaluation = await redis.get(evaluation_key)
            
            status = {
                "user_id": user_id,
                "has_trained_model": results is not None,
                "last_training": None,
                "model_performance": None,
                "recommendations": []
            }
            
            if results:
                results_data = json.loads(results)
                status["last_training"] = results_data
                status["recommendations"].append("Model is ready for personalized generation")
            
            if evaluation:
                eval_data = json.loads(evaluation)
                status["model_performance"] = eval_data.get("model_performance")
                
                if eval_data.get("model_performance") == "needs_improvement":
                    status["recommendations"].append("Consider retraining with more data")
            
            if not results:
                status["recommendations"].append("Start fine-tuning for personalized ads")
            
            return status
            
        except Exception as e:
            logger.exception("Error getting training status")
            raise
    
    async def cleanup_old_models(self, days: int = 30) -> Dict[str, int]:
        """
        Clean up old fine-tuned models.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Cleanup statistics
        """
        try:
            
            cutoff_date = datetime.now() - timedelta(days=days)
            cleaned_count = 0
            
            # Clean up model directories
            model_patterns = [
                "models/lora/*",
                "checkpoints/lora_*",
                "logs/lora_*"
            ]
            
            for pattern in model_patterns:
                for path in glob.glob(pattern):
                    try:
                        # Check if directory is older than threshold
                        dir_time = datetime.fromtimestamp(os.path.getctime(path))
                        if dir_time < cutoff_date:
                            shutil.rmtree(path)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old model: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {path}: {e}")
            
            return {
                "cleaned_models": cleaned_count,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.exception("Error cleaning up old models")
            raise
    
    async def close(self) -> Any:
        """Cleanup resources."""
        if self._redis_client:
            await self._redis_client.close() 

    @performance_monitor("setup_multi_gpu_training")
    async def setup_multi_gpu_training(
        self,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        gpu_config: Optional[GPUConfig] = None
    ) -> MultiGPUTrainingManager:
        """Setup multi-GPU training environment."""
        # Detect GPU configuration
        if gpu_config is None:
            gpu_config = self.multi_gpu_manager.detect_gpu_configuration()
        
        # Setup trainer
        trainer = self.multi_gpu_manager.setup_trainer(
            distributed=distributed,
            world_size=world_size,
            rank=rank
        )
        
        logger.info(f"Multi-GPU training setup: distributed={distributed}, "
                   f"world_size={world_size}, rank={rank}")
        
        return self.multi_gpu_manager
    
    @performance_monitor("finetune_model_multi_gpu")
    async def finetune_model_multi_gpu(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
    ) -> Dict[str, Any]:
        """Fine-tune model with multi-GPU support."""
        async with performance_context("multi_gpu_finetuning"):
            with gpu_monitoring_context([]):  # Will be populated by GPU detection
                start_time = time.time()
                
                # Setup multi-GPU training
                multi_gpu_manager = await self.setup_multi_gpu_training(
                    distributed=distributed,
                    world_size=world_size,
                    rank=rank
                )
                
                # Initialize training logger with performance monitoring
                self.training_logger = TrainingLogger(
                    user_id=user_id,
                    model_name=model_name,
                    task_type="multi_gpu_finetuning",
                    performance_optimizer=self.optimizer
                )
                
                # Load model with optimization
                model = await self.load_model(model_name)
                
                # Setup optimizer with gradient clipping
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=training_config.get("learning_rate", 5e-5),
                    weight_decay=training_config.get("weight_decay", 0.01)
                )
                
                # Learning rate scheduler
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=len(dataset) // training_config.get("batch_size_per_gpu", 8) * training_config.get("epochs", 3)
                )
                
                # Setup criterion
                criterion = nn.CrossEntropyLoss()
                
                # Train model using multi-GPU
                training_result = await multi_gpu_manager.train_model(
                    model=model,
                    train_dataset=dataset,
                    epochs=training_config.get("epochs", 3),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion
                )
                
                # Calculate final metrics
                training_time = time.time() - start_time
                
                # Save model with optimization
                model_path = f"models/{model_name}_multigpu_finetuned_{int(time.time())}"
                if hasattr(model, 'module'):
                    model.module.save_pretrained(model_path)
                else:
                    model.save_pretrained(model_path)
                
                # Log final results
                if self.training_logger:
                    await self.training_logger.log_training_completion(
                        total_loss=training_result["best_loss"],
                        training_time=training_time,
                        model_path=model_path,
                        gpu_stats=multi_gpu_manager.get_gpu_stats()
                    )
                
                # Cleanup
                multi_gpu_manager.cleanup()
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "best_loss": training_result["best_loss"],
                    "training_time": training_time,
                    "epochs": training_config.get("epochs", 3),
                    "gpu_stats": multi_gpu_manager.get_gpu_stats(),
                    "training_history": training_result["training_history"]
                }
    
    @performance_monitor("finetune_model_dataparallel")
    async def finetune_model_dataparallel(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int
    ) -> Dict[str, Any]:
        """Fine-tune model using DataParallel for single-node multi-GPU."""
        return await self.finetune_model_multi_gpu(
            model_name=model_name,
            dataset=dataset,
            training_config=training_config,
            user_id=user_id,
            distributed=False
        )
    
    @performance_monitor("finetune_model_distributed")
    async def finetune_model_distributed(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int,
        world_size: int = 4
    ) -> Dict[str, Any]:
        """Fine-tune model using DistributedDataParallel for multi-node training."""
        return await self.finetune_model_multi_gpu(
            model_name=model_name,
            dataset=dataset,
            training_config=training_config,
            user_id=user_id,
            distributed=True,
            world_size=world_size
        )
    
    @performance_monitor("get_gpu_stats")
    async def get_gpu_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU statistics."""
        return self.multi_gpu_manager.get_gpu_stats()
    
    @performance_monitor("cleanup_gpu_resources")
    async def cleanup_gpu_resources(self) -> Any:
        """Cleanup GPU resources."""
        # Cleanup multi-GPU training
        self.multi_gpu_manager.cleanup()
        
        # Cleanup PyTorch cache
        cleanup_gpu_memory()
        
        # Clear model cache
        self._models.clear()
        self._tokenizers.clear()
        
        # Clear training logger
        if self.training_logger:
            await self.training_logger.cleanup()
            self.training_logger = None
        
        return {"success": True, "message": "GPU resources cleaned up successfully"} 

    @performance_monitor("setup_gradient_accumulation")
    async def setup_gradient_accumulation(
        self,
        target_effective_batch_size: Optional[int] = None,
        accumulation_steps: Optional[int] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Setup gradient accumulation for fine-tuning."""
        # Create accumulation config
        config = GradientAccumulationConfig(
            accumulation_steps=accumulation_steps or 4,
            target_batch_size=target_effective_batch_size,
            auto_adjust_batch_size=True,
            mixed_precision=True,
            gradient_clipping=1.0,
            log_accumulation=True,
            log_memory_usage=True
        )
        
        # Generate training ID
        training_id = f"accumulation_{int(time.time())}"
        
        # Create accumulator
        self.gradient_accumulation_api.create_accumulator(config, training_id)
        
        # Update multi-GPU manager if available
        if self.multi_gpu_manager.current_trainer:
            self.multi_gpu_manager.setup_gradient_accumulation(
                target_effective_batch_size=target_effective_batch_size,
                accumulation_steps=accumulation_steps
            )
        
        return {
            "training_id": training_id,
            "config": config.__dict__,
            "target_effective_batch_size": target_effective_batch_size,
            "accumulation_steps": accumulation_steps
        }
    
    @performance_monitor("setup_mixed_precision")
    async def setup_mixed_precision(
        self,
        model_name: str,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        init_scale: float = 2**16,
        memory_efficient: bool = True
    ) -> Dict[str, Any]:
        """Setup mixed precision training for fine-tuning."""
        # Create mixed precision config
        self.mp_config = create_mixed_precision_config(
            enabled=enabled,
            dtype=dtype,
            init_scale=init_scale,
            memory_efficient=memory_efficient
        )
        
        # Create mixed precision trainer
        self.mp_trainer = AdaptiveMixedPrecisionTrainer(self.mp_config)
        
        # Load model to check if it supports mixed precision
        model = await self.load_model(model_name)
        
        # Update mixed precision settings based on model
        if self.mp_trainer:
            self.mp_trainer.update_config_adaptive(model)
        
        # Calculate memory savings
        memory_savings = self.mp_trainer.get_memory_savings(model) if self.mp_trainer else 0.0
        
        return {
            "enabled": enabled,
            "dtype": str(dtype),
            "init_scale": init_scale,
            "memory_efficient": memory_efficient,
            "memory_savings_gb": memory_savings,
            "mp_stats": self.mp_trainer.get_training_stats() if self.mp_trainer else {}
        }
    
    @performance_monitor("finetune_model_with_mixed_precision")
    async def finetune_model_with_mixed_precision(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int,
        mixed_precision_config: Optional[Dict[str, Any]] = None,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
    ) -> Dict[str, Any]:
        """Fine-tune model with mixed precision training."""
        async with performance_context("mixed_precision_finetuning"):
            with gpu_monitoring_context([]):  # Will be populated by GPU detection
                start_time = time.time()
                
                # Setup mixed precision if not already configured
                if not self.mp_trainer:
                    mp_config = mixed_precision_config or {
                        "enabled": True,
                        "dtype": torch.float16,
                        "init_scale": 2**16,
                        "memory_efficient": True
                    }
                    await self.setup_mixed_precision(
                        model_name=model_name,
                        **mp_config
                    )
                
                # Setup multi-GPU training with mixed precision
                multi_gpu_manager = await self.setup_multi_gpu_training(
                    distributed=distributed,
                    world_size=world_size,
                    rank=rank
                )
                
                # Enable mixed precision in multi-GPU config
                multi_gpu_manager.config.mixed_precision = True
                multi_gpu_manager.config.amp_enabled = True
                multi_gpu_manager.config.amp_dtype = self.mp_config.dtype if self.mp_config else torch.float16
                multi_gpu_manager.config.amp_init_scale = self.mp_config.init_scale if self.mp_config else 2**16
                
                # Initialize training logger
                self.training_logger = TrainingLogger(
                    user_id=user_id,
                    model_name=model_name,
                    task_type="mixed_precision_finetuning",
                    performance_optimizer=self.optimizer
                )
                
                # Load model with optimization
                model = await self.load_model(model_name)
                
                # Update mixed precision settings based on model
                if self.mp_trainer:
                    self.mp_trainer.update_config_adaptive(model)
                
                # Setup optimizer with mixed precision
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=training_config.get("learning_rate", 5e-5),
                    weight_decay=training_config.get("weight_decay", 0.01)
                )
                
                # Learning rate scheduler
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=len(dataset) // training_config.get("batch_size", 8) * training_config.get("epochs", 3)
                )
                
                # Setup criterion
                criterion = nn.CrossEntropyLoss()
                
                # Train model using multi-GPU with mixed precision
                training_result = await multi_gpu_manager.train_model(
                    model=model,
                    train_dataset=dataset,
                    epochs=training_config.get("epochs", 3),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion
                )
                
                # Calculate final metrics
                training_time = time.time() - start_time
                
                # Save model with optimization
                model_path = f"models/{model_name}_mixed_precision_finetuned_{int(time.time())}"
                if hasattr(model, 'module'):
                    model.module.save_pretrained(model_path)
                else:
                    model.save_pretrained(model_path)
                
                # Get mixed precision statistics
                mp_stats = self.mp_trainer.get_training_stats() if self.mp_trainer else {}
                
                # Log final results
                if self.training_logger:
                    await self.training_logger.log_training_completion(
                        total_loss=training_result["best_loss"],
                        training_time=training_time,
                        model_path=model_path,
                        gpu_stats=multi_gpu_manager.get_gpu_stats(),
                        mp_stats=mp_stats
                    )
                
                # Cleanup
                multi_gpu_manager.cleanup()
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "best_loss": training_result["best_loss"],
                    "training_time": training_time,
                    "epochs": training_config.get("epochs", 3),
                    "gpu_stats": multi_gpu_manager.get_gpu_stats(),
                    "training_history": training_result["training_history"],
                    "mixed_precision_enabled": True,
                    "mp_stats": mp_stats,
                    "memory_savings_gb": mp_stats.get("memory_saved", 0.0),
                    "scaler_scale": mp_stats.get("scaler_scale", 1.0)
                }
    
    @performance_monitor("finetune_model_optimized")
    async def finetune_model_optimized(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int,
        use_mixed_precision: bool = True,
        use_gradient_accumulation: bool = True,
        target_effective_batch_size: int = 32
    ) -> Dict[str, Any]:
        """Fine-tune model with optimized settings (mixed precision + gradient accumulation)."""
        async with performance_context("optimized_finetuning"):
            with gpu_monitoring_context([]):
                start_time = time.time()
                
                # Setup mixed precision if enabled
                if use_mixed_precision:
                    await self.setup_mixed_precision(
                        model_name=model_name,
                        enabled=True,
                        dtype=torch.float16,
                        init_scale=2**16,
                        memory_efficient=True
                    )
                
                # Setup gradient accumulation if enabled
                if use_gradient_accumulation:
                    accumulation_setup = await self.setup_gradient_accumulation(
                        target_effective_batch_size=target_effective_batch_size,
                        accumulation_steps=4
                    )
                
                # Setup multi-GPU training
                multi_gpu_manager = await self.setup_multi_gpu_training(
                    distributed=False,
                    world_size=1,
                    rank=0
                )
                
                # Configure multi-GPU settings
                if use_mixed_precision:
                    multi_gpu_manager.config.mixed_precision = True
                    multi_gpu_manager.config.amp_enabled = True
                    multi_gpu_manager.config.amp_dtype = torch.float16
                    multi_gpu_manager.config.amp_init_scale = 2**16
                
                if use_gradient_accumulation:
                    multi_gpu_manager.config.use_gradient_accumulation = True
                    multi_gpu_manager.config.target_effective_batch_size = target_effective_batch_size
                
                # Initialize training logger
                self.training_logger = TrainingLogger(
                    user_id=user_id,
                    model_name=model_name,
                    task_type="optimized_finetuning",
                    performance_optimizer=self.optimizer
                )
                
                # Load model
                model = await self.load_model(model_name)
                
                # Setup optimizer
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=training_config.get("learning_rate", 5e-5),
                    weight_decay=training_config.get("weight_decay", 0.01)
                )
                
                # Learning rate scheduler
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=len(dataset) // training_config.get("batch_size", 8) * training_config.get("epochs", 3)
                )
                
                # Setup criterion
                criterion = nn.CrossEntropyLoss()
                
                # Train model
                training_result = await multi_gpu_manager.train_model(
                    model=model,
                    train_dataset=dataset,
                    epochs=training_config.get("epochs", 3),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion
                )
                
                # Calculate final metrics
                training_time = time.time() - start_time
                
                # Save model
                model_path = f"models/{model_name}_optimized_finetuned_{int(time.time())}"
                if hasattr(model, 'module'):
                    model.module.save_pretrained(model_path)
                else:
                    model.save_pretrained(model_path)
                
                # Get statistics
                mp_stats = self.mp_trainer.get_training_stats() if self.mp_trainer else {}
                accumulation_stats = {}
                if use_gradient_accumulation:
                    accumulation_stats = self.gradient_accumulation_api.get_accumulation_stats(
                        accumulation_setup["training_id"]
                    )
                
                # Log final results
                if self.training_logger:
                    await self.training_logger.log_training_completion(
                        total_loss=training_result["best_loss"],
                        training_time=training_time,
                        model_path=model_path,
                        gpu_stats=multi_gpu_manager.get_gpu_stats(),
                        mp_stats=mp_stats,
                        accumulation_stats=accumulation_stats
                    )
                
                # Cleanup
                multi_gpu_manager.cleanup()
                if use_gradient_accumulation:
                    self.gradient_accumulation_api.cleanup(accumulation_setup["training_id"])
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "best_loss": training_result["best_loss"],
                    "training_time": training_time,
                    "epochs": training_config.get("epochs", 3),
                    "gpu_stats": multi_gpu_manager.get_gpu_stats(),
                    "training_history": training_result["training_history"],
                    "mixed_precision_enabled": use_mixed_precision,
                    "gradient_accumulation_enabled": use_gradient_accumulation,
                    "mp_stats": mp_stats,
                    "accumulation_stats": accumulation_stats,
                    "effective_batch_size": target_effective_batch_size if use_gradient_accumulation else training_config.get("batch_size", 8),
                    "memory_savings_gb": mp_stats.get("memory_saved", 0.0),
                    "scaler_scale": mp_stats.get("scaler_scale", 1.0)
                }
    
    @performance_monitor("get_mixed_precision_stats")
    async def get_mixed_precision_stats(self) -> Dict[str, Any]:
        """Get mixed precision training statistics."""
        try:
            if not self.mp_trainer:
                return {
                    "success": False,
                    "error": "Mixed precision not initialized"
                }
            
            stats = self.mp_trainer.get_training_stats()
            return {
                "success": True,
                "mp_stats": stats,
                "enabled": stats.get("amp_enabled", False),
                "scaler_scale": stats.get("scaler_scale", 1.0),
                "memory_savings_gb": stats.get("memory_saved", 0.0),
                "overflow_count": stats.get("overflow_count", 0),
                "underflow_count": stats.get("underflow_count", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get mixed precision stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @performance_monitor("optimize_mixed_precision_settings")
    async def optimize_mixed_precision_settings(
        self,
        model_name: str,
        gpu_memory_gb: float,
        batch_size: int
    ) -> Dict[str, Any]:
        """Optimize mixed precision settings for a specific model and hardware."""
        try:
            # Load model
            model = await self.load_model(model_name)
            
            # Get optimization recommendations
            optimization = optimize_mixed_precision_settings(model, gpu_memory_gb, batch_size)
            
            # Check if mixed precision should be used
            should_use_mp = should_use_mixed_precision(model, gpu_memory_gb)
            
            return {
                "success": True,
                "should_use_mixed_precision": should_use_mp,
                "recommendations": optimization,
                "model_params": sum(p.numel() for p in model.parameters()),
                "gpu_memory_gb": gpu_memory_gb,
                "batch_size": batch_size
            }
        except Exception as e:
            logger.error(f"Failed to optimize mixed precision settings: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @performance_monitor("finetune_model_with_accumulation")
    async def finetune_model_with_accumulation(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int,
        target_effective_batch_size: int = 32,
        accumulation_steps: Optional[int] = None,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
    ) -> Dict[str, Any]:
        """Fine-tune model with gradient accumulation for large batch sizes."""
        async with performance_context("gradient_accumulation_finetuning"):
            with gpu_monitoring_context([]):  # Will be populated by GPU detection
                start_time = time.time()
                
                # Setup mixed precision for better performance
                await self.setup_mixed_precision(
                    model_name=model_name,
                    enabled=True,
                    dtype=torch.float16,
                    init_scale=2**16,
                    memory_efficient=True
                )
                
                # Setup gradient accumulation
                accumulation_setup = await self.setup_gradient_accumulation(
                    target_effective_batch_size=target_effective_batch_size,
                    accumulation_steps=accumulation_steps
                )
                
                # Setup multi-GPU training with accumulation and mixed precision
                multi_gpu_manager = await self.setup_multi_gpu_training(
                    distributed=distributed,
                    world_size=world_size,
                    rank=rank
                )
                
                # Enable mixed precision in multi-GPU config
                multi_gpu_manager.config.mixed_precision = True
                multi_gpu_manager.config.amp_enabled = True
                multi_gpu_manager.config.amp_dtype = torch.float16
                multi_gpu_manager.config.amp_init_scale = 2**16
                
                # Initialize training logger
                self.training_logger = TrainingLogger(
                    user_id=user_id,
                    model_name=model_name,
                    task_type="gradient_accumulation_finetuning",
                    performance_optimizer=self.optimizer
                )
                
                # Load model with optimization
                model = await self.load_model(model_name)
                
                # Calculate actual batch size per GPU
                gpu_count = len(multi_gpu_manager.config.device_ids) if not distributed else world_size
                actual_batch_size = target_effective_batch_size // (gpu_count * (accumulation_steps or 4))
                actual_batch_size = max(1, actual_batch_size)  # Ensure minimum batch size
                
                # Update training config
                training_config["batch_size_per_gpu"] = actual_batch_size
                training_config["gradient_accumulation_steps"] = accumulation_steps or 4
                
                # Setup optimizer with adjusted learning rate
                base_lr = training_config.get("learning_rate", 5e-5)
                adjusted_lr = adjust_learning_rate(base_lr, accumulation_steps or 4)
                
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=adjusted_lr,
                    weight_decay=training_config.get("weight_decay", 0.01)
                )
                
                # Learning rate scheduler
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=len(dataset) // actual_batch_size * training_config.get("epochs", 3)
                )
                
                # Setup criterion
                criterion = nn.CrossEntropyLoss()
                
                # Train model using multi-GPU with accumulation and mixed precision
                training_result = await multi_gpu_manager.train_model(
                    model=model,
                    train_dataset=dataset,
                    epochs=training_config.get("epochs", 3),
                    target_effective_batch_size=target_effective_batch_size,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion
                )
                
                # Calculate final metrics
                training_time = time.time() - start_time
                
                # Save model with optimization
                model_path = f"models/{model_name}_accumulation_finetuned_{int(time.time())}"
                if hasattr(model, 'module'):
                    model.module.save_pretrained(model_path)
                else:
                    model.save_pretrained(model_path)
                
                # Get accumulation statistics
                accumulation_stats = self.gradient_accumulation_api.get_accumulation_stats(
                    accumulation_setup["training_id"]
                )
                
                # Get mixed precision statistics
                mp_stats = self.mp_trainer.get_training_stats() if self.mp_trainer else {}
                
                # Log final results
                if self.training_logger:
                    await self.training_logger.log_training_completion(
                        total_loss=training_result["best_loss"],
                        training_time=training_time,
                        model_path=model_path,
                        gpu_stats=multi_gpu_manager.get_gpu_stats(),
                        accumulation_stats=accumulation_stats,
                        mp_stats=mp_stats
                    )
                
                # Cleanup
                multi_gpu_manager.cleanup()
                self.gradient_accumulation_api.cleanup(accumulation_setup["training_id"])
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "best_loss": training_result["best_loss"],
                    "training_time": training_time,
                    "epochs": training_config.get("epochs", 3),
                    "gpu_stats": multi_gpu_manager.get_gpu_stats(),
                    "training_history": training_result["training_history"],
                    "accumulation_stats": accumulation_stats,
                    "mp_stats": mp_stats,
                    "effective_batch_size": target_effective_batch_size,
                    "actual_batch_size_per_gpu": actual_batch_size,
                    "accumulation_steps": accumulation_steps or 4,
                    "adjusted_learning_rate": adjusted_lr,
                    "mixed_precision_enabled": True,
                    "memory_savings_gb": mp_stats.get("memory_saved", 0.0),
                    "scaler_scale": mp_stats.get("scaler_scale", 1.0)
                }
    
    @performance_monitor("finetune_model_large_batch")
    async def finetune_model_large_batch(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int,
        target_batch_size: int = 64,
        max_memory_usage: float = 0.9
    ) -> Dict[str, Any]:
        """Fine-tune model with automatic large batch size optimization."""
        # Calculate optimal accumulation steps based on available GPU memory
        gpu_monitor = GPUMonitor(GPUConfig())
        gpu_info = gpu_monitor.get_gpu_info()
        available_gpus = gpu_monitor.get_available_gpus()
        
        if not available_gpus:
            logger.warning("No GPUs available, falling back to CPU training")
            return await self.finetune_model(model_name, dataset, training_config, user_id)
        
        # Estimate optimal batch size per GPU
        total_memory = 0
        for gpu_id in available_gpus:
            if f"gpu_{gpu_id}" in gpu_info:
                total_memory += gpu_info[f"gpu_{gpu_id}"]["memory_total"]
        
        avg_memory_per_gpu = total_memory / len(available_gpus) / 1024  # GB
        safe_memory = avg_memory_per_gpu * max_memory_usage
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample = 0.1  # GB per sample (adjust based on model)
        max_batch_size_per_gpu = int(safe_memory / memory_per_sample)
        max_batch_size_per_gpu = min(max_batch_size_per_gpu, 16)  # Reasonable upper limit
        
        # Calculate required accumulation steps
        total_gpus = len(available_gpus)
        actual_batch_size = max_batch_size_per_gpu * total_gpus
        accumulation_steps = calculate_accumulation_steps(target_batch_size, actual_batch_size)
        
        logger.info(f"Large batch optimization: "
                   f"target_batch_size={target_batch_size}, "
                   f"actual_batch_size={actual_batch_size}, "
                   f"accumulation_steps={accumulation_steps}, "
                   f"gpus={total_gpus}")
        
        # Use gradient accumulation training
        return await self.finetune_model_with_accumulation(
            model_name=model_name,
            dataset=dataset,
            training_config=training_config,
            user_id=user_id,
            target_effective_batch_size=target_batch_size,
            accumulation_steps=accumulation_steps,
            distributed=total_gpus >= 4  # Use distributed for 4+ GPUs
        )
    
    @performance_monitor("get_accumulation_stats")
    async def get_accumulation_stats(self, training_id: str) -> Dict[str, Any]:
        """Get gradient accumulation statistics."""
        try:
            stats = self.gradient_accumulation_api.get_accumulation_stats(training_id)
            return {
                "success": True,
                "training_id": training_id,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Failed to get accumulation stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @performance_monitor("update_accumulation_config")
    async def update_accumulation_config(
        self,
        training_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update gradient accumulation configuration."""
        try:
            accumulation_config = GradientAccumulationConfig(**config)
            self.gradient_accumulation_api.update_config(training_id, accumulation_config)
            
            return {
                "success": True,
                "training_id": training_id,
                "message": "Accumulation config updated successfully"
            }
        except Exception as e:
            logger.error(f"Failed to update accumulation config: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @performance_monitor("calculate_optimal_batch_size")
    async def calculate_optimal_batch_size(
        self,
        model_name: str,
        target_effective_batch_size: int,
        gpu_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Calculate optimal batch size and accumulation steps."""
        try:
            # Load model to estimate memory usage
            model = await self.load_model(model_name)
            
            # Get GPU information
            gpu_monitor = GPUMonitor(GPUConfig())
            if gpu_ids is None:
                gpu_ids = gpu_monitor.get_available_gpus()
            
            gpu_info = gpu_monitor.get_gpu_info()
            
            # Calculate optimal batch size per GPU
            optimal_batch_sizes = []
            for gpu_id in gpu_ids:
                if f"gpu_{gpu_id}" in gpu_info:
                    gpu = gpu_info[f"gpu_{gpu_id}"]
                    total_memory = gpu["memory_total"] / 1024  # GB
                    used_memory = gpu["memory_used"] / 1024  # GB
                    free_memory = total_memory - used_memory
                    
                    # Estimate model memory usage
                    model_params = sum(p.numel() for p in model.parameters())
                    model_memory_gb = model_params * 4 / 1024**3  # Assuming float32
                    
                    # Calculate safe batch size
                    safe_memory = free_memory * 0.8  # 80% safety margin
                    remaining_memory = safe_memory - model_memory_gb
                    
                    # Estimate memory per sample
                    memory_per_sample = 0.1  # GB per sample
                    max_batch_size = int(remaining_memory / memory_per_sample)
                    max_batch_size = max(1, min(max_batch_size, 16))  # Clamp to reasonable range
                    
                    optimal_batch_sizes.append(max_batch_size)
            
            if not optimal_batch_sizes:
                raise ValueError("No valid GPUs found")
            
            # Calculate average optimal batch size
            avg_batch_size = sum(optimal_batch_sizes) / len(optimal_batch_sizes)
            total_gpus = len(gpu_ids)
            actual_batch_size = int(avg_batch_size * total_gpus)
            
            # Calculate accumulation steps
            accumulation_steps = calculate_accumulation_steps(target_effective_batch_size, actual_batch_size)
            
            # Calculate effective batch size
            effective_batch_size = calculate_effective_batch_size(actual_batch_size, accumulation_steps)
            
            return {
                "success": True,
                "target_effective_batch_size": target_effective_batch_size,
                "actual_batch_size_per_gpu": int(avg_batch_size),
                "total_gpus": total_gpus,
                "accumulation_steps": accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "gpu_info": {f"gpu_{gpu_id}": gpu_info.get(f"gpu_{gpu_id}", {}) for gpu_id in gpu_ids}
            }
        except Exception as e:
            logger.error(f"Failed to calculate optimal batch size: {e}")
            return {
                "success": False,
                "error": str(e)
            } 

    @performance_monitor("profile_and_optimize_training")
    async def profile_and_optimize_training(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int,
        profile_data_loading: bool = True,
        profile_preprocessing: bool = True,
        profile_training: bool = True
    ) -> Dict[str, Any]:
        """Profile and optimize the entire training pipeline."""
        async with profiling_context(self.profiling_config):
            with memory_optimization_context(self.data_optimization_config):
                start_time = time.time()
                
                # Profile data loading
                data_loading_results = {}
                if profile_data_loading:
                    data_loading_results = await self._profile_data_loading(dataset)
                
                # Profile preprocessing
                preprocessing_results = {}
                if profile_preprocessing:
                    preprocessing_results = await self._profile_preprocessing(dataset)
                
                # Profile training
                training_results = {}
                if profile_training:
                    training_results = await self._profile_training(
                        model_name, dataset, training_config, user_id
                    )
                
                # Generate optimization recommendations
                optimization_plan = self._generate_optimization_plan(
                    data_loading_results,
                    preprocessing_results,
                    training_results
                )
                
                # Apply optimizations
                optimization_results = await self._apply_optimizations(optimization_plan)
                
                total_time = time.time() - start_time
                
                return {
                    "success": True,
                    "profiling_time": total_time,
                    "data_loading_results": data_loading_results,
                    "preprocessing_results": preprocessing_results,
                    "training_results": training_results,
                    "optimization_plan": optimization_plan,
                    "optimization_results": optimization_results,
                    "recommendations": optimization_plan.get("recommendations", [])
                }
    
    @performance_monitor("profile_data_loading")
    async def _profile_data_loading(self, dataset: OptimizedAdsDataset) -> Dict[str, Any]:
        """Profile data loading performance."""
        
        # Create sample dataloader for profiling
        sample_dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2
        )
        
        # Profile data loading
        def data_loading_operation():
            
    """data_loading_operation function."""
total_time = 0
            num_batches = 0
            
            for batch in sample_dataloader:
                batch_start = time.time()
                _ = batch  # Consume batch
                batch_time = time.time() - batch_start
                total_time += batch_time
                num_batches += 1
                
                if num_batches >= 10:  # Profile first 10 batches
                    break
            
            return total_time, num_batches
        
        # Profile with detailed analysis
        profiling_result = self.profiler.profile_code(data_loading_operation)
        
        # Identify bottlenecks
        bottlenecks = self.data_loading_optimizer.identify_bottlenecks()
        
        # Calculate metrics
        avg_batch_time = profiling_result.function_times.get("data_loading_operation", 0) / 10
        memory_usage = profiling_result.memory_usage.get("peak", 0)
        
        return {
            "avg_batch_time": avg_batch_time,
            "total_time": profiling_result.function_times.get("data_loading_operation", 0),
            "memory_usage_mb": memory_usage / (1024 * 1024),
            "bottlenecks": bottlenecks,
            "bottleneck_functions": profiling_result.bottleneck_functions,
            "recommendations": profiling_result.recommendations
        }
    
    @performance_monitor("profile_preprocessing")
    async def _profile_preprocessing(self, dataset: OptimizedAdsDataset) -> Dict[str, Any]:
        """Profile preprocessing performance."""
        
        # Sample preprocessing functions
        def tokenize_text(text) -> Any:
            return self.tokenization_service.tokenize(text)
        
        def normalize_text(text) -> Any:
            return text.lower().strip()
        
        def augment_text(text) -> Any:
            # Simple augmentation
            return text + " [AUGMENTED]"
        
        preprocessing_funcs = [normalize_text, tokenize_text, augment_text]
        
        # Create sample data
        sample_texts = ["Sample text for preprocessing analysis"] * 100
        
        # Profile preprocessing pipeline
        def preprocessing_operation():
            
    """preprocessing_operation function."""
pipeline = self.preprocessing_optimizer.optimize_preprocessing_pipeline(
                preprocessing_funcs, sample_texts
            )
            return pipeline(sample_texts)
        
        # Profile with detailed analysis
        profiling_result = self.profiler.profile_code(preprocessing_operation)
        
        # Calculate metrics
        total_time = profiling_result.function_times.get("preprocessing_operation", 0)
        avg_time_per_sample = total_time / len(sample_texts)
        
        return {
            "total_time": total_time,
            "avg_time_per_sample": avg_time_per_sample,
            "samples_processed": len(sample_texts),
            "bottleneck_functions": profiling_result.bottleneck_functions,
            "memory_usage_mb": profiling_result.memory_usage.get("peak", 0) / (1024 * 1024),
            "recommendations": profiling_result.recommendations
        }
    
    @performance_monitor("profile_training")
    async def _profile_training(
        self,
        model_name: str,
        dataset: OptimizedAdsDataset,
        training_config: Dict[str, Any],
        user_id: int
    ) -> Dict[str, Any]:
        """Profile training performance."""
        
        # Load model for profiling
        model = await self.load_model(model_name)
        
        # Create small dataset for profiling
        small_dataset = OptimizedAdsDataset(
            dataset.data[:100],  # Use first 100 samples
            self.data_optimization_config
        )
        
        # Profile training loop
        def training_operation():
            
    """training_operation function."""
# Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_config.get("learning_rate", 5e-5)
            )
            
            # Setup criterion
            criterion = nn.CrossEntropyLoss()
            
            # Create dataloader
            dataloader = DataLoader(small_dataset, batch_size=8, shuffle=True)
            
            # Training loop
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = torch.randint(0, 10, (inputs.size(0),))
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 5:  # Profile first 5 batches
                    break
            
            return total_loss / num_batches
        
        # Profile with detailed analysis
        profiling_result = self.profiler.profile_code(training_operation)
        
        # Calculate metrics
        avg_loss = profiling_result.function_times.get("training_operation", 0)
        gpu_utilization = profiling_result.gpu_utilization
        gpu_memory_usage = profiling_result.gpu_memory_usage
        
        return {
            "avg_loss": avg_loss,
            "gpu_utilization": gpu_utilization,
            "gpu_memory_usage_mb": gpu_memory_usage / (1024 * 1024),
            "bottleneck_functions": profiling_result.bottleneck_functions,
            "gpu_bottlenecks": profiling_result.gpu_bottlenecks,
            "recommendations": profiling_result.recommendations
        }
    
    def _generate_optimization_plan(
        self,
        data_loading_results: Dict[str, Any],
        preprocessing_results: Dict[str, Any],
        training_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization plan based on profiling results."""
        
        plan = {
            "data_loading_optimizations": [],
            "preprocessing_optimizations": [],
            "training_optimizations": [],
            "recommendations": []
        }
        
        # Data loading optimizations
        if data_loading_results.get("avg_batch_time", 0) > 0.1:  # 100ms threshold
            plan["data_loading_optimizations"].append({
                "type": "increase_workers",
                "current": 2,
                "recommended": 4,
                "expected_improvement": "50% faster loading"
            })
            plan["recommendations"].append("Increase DataLoader num_workers")
        
        if data_loading_results.get("memory_usage_mb", 0) > 500:  # 500MB threshold
            plan["data_loading_optimizations"].append({
                "type": "memory_optimization",
                "current_memory": data_loading_results["memory_usage_mb"],
                "recommended": "Use memory-efficient dataset",
                "expected_improvement": "30% memory reduction"
            })
            plan["recommendations"].append("Optimize dataset memory usage")
        
        # Preprocessing optimizations
        if preprocessing_results.get("avg_time_per_sample", 0) > 0.001:  # 1ms threshold
            plan["preprocessing_optimizations"].append({
                "type": "parallel_processing",
                "current": "sequential",
                "recommended": "parallel",
                "expected_improvement": "2-4x faster preprocessing"
            })
            plan["recommendations"].append("Enable parallel preprocessing")
        
        # Training optimizations
        if training_results.get("gpu_utilization", 0) < 50:
            plan["training_optimizations"].append({
                "type": "batch_size_optimization",
                "current": "small batches",
                "recommended": "increase batch size",
                "expected_improvement": "higher GPU utilization"
            })
            plan["recommendations"].append("Increase batch size for better GPU utilization")
        
        if training_results.get("gpu_memory_usage_mb", 0) > 4000:  # 4GB threshold
            plan["training_optimizations"].append({
                "type": "memory_optimization",
                "current_memory": training_results["gpu_memory_usage_mb"],
                "recommended": "gradient checkpointing",
                "expected_improvement": "50% memory reduction"
            })
            plan["recommendations"].append("Enable gradient checkpointing")
        
        return plan
    
    @performance_monitor("apply_optimizations")
    async def _apply_optimizations(self, optimization_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations based on the plan."""
        
        results = {
            "applied_optimizations": [],
            "performance_improvements": {},
            "errors": []
        }
        
        # Apply data loading optimizations
        for opt in optimization_plan.get("data_loading_optimizations", []):
            try:
                if opt["type"] == "increase_workers":
                    # Update configuration
                    self.data_optimization_config.num_workers = opt["recommended"]
                    results["applied_optimizations"].append("Increased DataLoader workers")
                
                elif opt["type"] == "memory_optimization":
                    # Enable memory optimization
                    self.data_optimization_config.memory_efficient = True
                    results["applied_optimizations"].append("Enabled memory optimization")
                
            except Exception as e:
                results["errors"].append(f"Failed to apply {opt['type']}: {str(e)}")
        
        # Apply preprocessing optimizations
        for opt in optimization_plan.get("preprocessing_optimizations", []):
            try:
                if opt["type"] == "parallel_processing":
                    # Enable parallel preprocessing
                    self.data_optimization_config.parallel_preprocessing = True
                    results["applied_optimizations"].append("Enabled parallel preprocessing")
                
            except Exception as e:
                results["errors"].append(f"Failed to apply {opt['type']}: {str(e)}")
        
        # Apply training optimizations
        for opt in optimization_plan.get("training_optimizations", []):
            try:
                if opt["type"] == "batch_size_optimization":
                    # This would be applied during training
                    results["applied_optimizations"].append("Batch size optimization recommended")
                
                elif opt["type"] == "memory_optimization":
                    # Enable gradient checkpointing
                    results["applied_optimizations"].append("Gradient checkpointing recommended")
                
            except Exception as e:
                results["errors"].append(f"Failed to apply {opt['type']}: {str(e)}")
        
        return results
    
    @performance_monitor("optimize_dataset_loading")
    async def optimize_dataset_loading(
        self,
        dataset: OptimizedAdsDataset,
        batch_size: int = 32
    ) -> DataLoader:
        """Optimize dataset loading with profiling and automatic configuration."""
        
        # Profile current dataset
        loading_results = await self._profile_data_loading(dataset)
        
        # Create optimized dataset
        optimized_dataset = optimize_dataset(dataset, self.data_optimization_config)
        
        # Create optimized dataloader
        optimized_dataloader = self.data_loading_optimizer.optimize_dataloader(
            optimized_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Profile optimized dataloader
        optimized_results = await self._profile_data_loading(optimized_dataset)
        
        # Calculate improvement
        improvement = {
            "original_avg_time": loading_results.get("avg_batch_time", 0),
            "optimized_avg_time": optimized_results.get("avg_batch_time", 0),
            "memory_reduction": loading_results.get("memory_usage_mb", 0) - optimized_results.get("memory_usage_mb", 0)
        }
        
        if improvement["original_avg_time"] > 0:
            improvement["speedup"] = improvement["original_avg_time"] / improvement["optimized_avg_time"]
        else:
            improvement["speedup"] = 1.0
        
        logger.info(f"Dataset loading optimization: "
                   f"Speedup: {improvement['speedup']:.2f}x, "
                   f"Memory reduction: {improvement['memory_reduction']:.2f}MB")
        
        return optimized_dataloader
    
    @performance_monitor("optimize_preprocessing_pipeline")
    async def optimize_preprocessing_pipeline(
        self,
        preprocessing_funcs: List[Callable],
        sample_data: List[Any]
    ) -> Callable:
        """Optimize preprocessing pipeline with profiling and automatic configuration."""
        
        # Profile current preprocessing
        preprocessing_results = await self._profile_preprocessing(
            OptimizedAdsDataset(sample_data, self.data_optimization_config)
        )
        
        # Create optimized pipeline
        optimized_pipeline = self.preprocessing_optimizer.optimize_preprocessing_pipeline(
            preprocessing_funcs, sample_data
        )
        
        # Profile optimized pipeline
        def optimized_operation():
            
    """optimized_operation function."""
return optimized_pipeline(sample_data)
        
        optimized_results = self.profiler.profile_code(optimized_operation)
        
        # Calculate improvement
        improvement = {
            "original_time": preprocessing_results.get("total_time", 0),
            "optimized_time": optimized_results.function_times.get("optimized_operation", 0),
            "samples_processed": preprocessing_results.get("samples_processed", 0)
        }
        
        if improvement["original_time"] > 0:
            improvement["speedup"] = improvement["original_time"] / improvement["optimized_time"]
        else:
            improvement["speedup"] = 1.0
        
        logger.info(f"Preprocessing optimization: "
                   f"Speedup: {improvement['speedup']:.2f}x, "
                   f"Samples processed: {improvement['samples_processed']}")
        
        return optimized_pipeline
    
    @performance_monitor("get_performance_report")
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        
        # Get profiling results
        profiling_summary = {
            "total_profiles": len(self.profiler.profiling_results),
            "optimization_history": len(self.profiler.optimization_history)
        }
        
        # Get data optimization stats
        data_optimization_stats = {
            "dataloader_optimizations": self.data_loading_optimizer.optimization_stats,
            "bottlenecks_identified": self.data_loading_optimizer.bottleneck_analysis
        }
        
        # Get system metrics
        system_metrics = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "gpu_memory_usage": 0
        }
        
        if torch.cuda.is_available():
            system_metrics["gpu_memory_usage"] = (
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            ) * 100
        
        return {
            "profiling_summary": profiling_summary,
            "data_optimization_stats": data_optimization_stats,
            "system_metrics": system_metrics,
            "recommendations": self._get_current_recommendations()
        }
    
    def _get_current_recommendations(self) -> List[str]:
        """Get current optimization recommendations."""
        recommendations = []
        
        # System-based recommendations
        if psutil.cpu_percent() > 80:
            recommendations.append("High CPU usage - consider reducing parallel workers")
        
        if psutil.virtual_memory().percent > 80:
            recommendations.append("High memory usage - consider memory optimization")
        
        if torch.cuda.is_available():
            gpu_memory_usage = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            if gpu_memory_usage > 80:
                recommendations.append("High GPU memory usage - consider gradient checkpointing")
        
        return recommendations 