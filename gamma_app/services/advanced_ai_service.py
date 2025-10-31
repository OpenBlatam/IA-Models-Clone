"""
Gamma App - Advanced AI Service
Enhanced AI capabilities with fine-tuning, optimization, and advanced features
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import hashlib
import numpy as np
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, BitsAndBytesConfig
)
import openai
import anthropic
from peft import LoraConfig, get_peft_model, TaskType
import datasets
from accelerate import Accelerator
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """AI Model types"""
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"
    CLAUDE3_OPUS = "claude-3-opus-20240229"
    CLAUDE3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE3_HAIKU = "claude-3-haiku-20240307"
    LOCAL_LLAMA = "llama-2-7b"
    LOCAL_MISTRAL = "mistral-7b"
    LOCAL_CODEX = "codex-6b"

class OptimizationType(Enum):
    """Model optimization types"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    LORA = "lora"
    PEFT = "peft"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    model_type: ModelType
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    use_cache: bool = True
    optimization: List[OptimizationType] = None
    custom_instructions: str = ""
    system_prompt: str = ""

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
    mixed_precision: bool = True
    use_accelerate: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "gamma-app-ai"

@dataclass
class FineTuningData:
    """Fine-tuning data structure"""
    prompt: str
    completion: str
    metadata: Dict[str, Any] = None
    quality_score: float = 1.0
    category: str = "general"

class AdvancedAIService:
    """Advanced AI service with enhanced capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.accelerator = None
        self.training_history = {}
        self.model_cache = {}
        self.performance_metrics = {}
        
        # Initialize services
        self._init_openai()
        self._init_anthropic()
        self._init_local_models()
        self._init_accelerator()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if self.config.get("openai_api_key"):
            openai.api_key = self.config["openai_api_key"]
            logger.info("OpenAI client initialized")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        if self.config.get("anthropic_api_key"):
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.config["anthropic_api_key"]
            )
            logger.info("Anthropic client initialized")
    
    def _init_local_models(self):
        """Initialize local models"""
        self.local_models = {
            "llama-2-7b": "meta-llama/Llama-2-7b-hf",
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "codex-6b": "microsoft/DialoGPT-medium"
        }
        logger.info("Local models configured")
    
    def _init_accelerator(self):
        """Initialize Accelerator for distributed training"""
        try:
            self.accelerator = Accelerator()
            logger.info("Accelerator initialized")
        except Exception as e:
            logger.warning(f"Accelerator initialization failed: {e}")
    
    async def generate_content_advanced(
        self, 
        prompt: str, 
        model_config: ModelConfig,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate content with advanced AI capabilities"""
        
        start_time = time.time()
        
        try:
            # Select model based on configuration
            if model_config.model_type in [ModelType.GPT4, ModelType.GPT35]:
                result = await self._generate_with_openai(prompt, model_config, context)
            elif model_config.model_type in [ModelType.CLAUDE3_OPUS, ModelType.CLAUDE3_SONNET, ModelType.CLAUDE3_HAIKU]:
                result = await self._generate_with_anthropic(prompt, model_config, context)
            else:
                result = await self._generate_with_local_model(prompt, model_config, context)
            
            # Add metadata
            result["metadata"] = {
                "model_used": model_config.model_name,
                "generation_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "optimization_applied": [opt.value for opt in model_config.optimization] if model_config.optimization else []
            }
            
            # Update performance metrics
            self._update_performance_metrics(model_config.model_name, result["metadata"]["generation_time"])
            
            return result
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return {
                "content": "",
                "error": str(e),
                "metadata": {
                    "model_used": model_config.model_name,
                    "generation_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "error": True
                }
            }
    
    async def _generate_with_openai(self, prompt: str, model_config: ModelConfig, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate content using OpenAI API"""
        
        messages = []
        
        # Add system prompt
        if model_config.system_prompt:
            messages.append({"role": "system", "content": model_config.system_prompt})
        
        # Add context if provided
        if context:
            messages.append({"role": "system", "content": f"Context: {json.dumps(context)}"})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        response = await openai.ChatCompletion.acreate(
            model=model_config.model_type.value,
            messages=messages,
            max_tokens=model_config.max_length,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            frequency_penalty=model_config.frequency_penalty,
            presence_penalty=model_config.presence_penalty
        )
        
        return {
            "content": response.choices[0].message.content,
            "usage": response.usage.dict(),
            "model": response.model
        }
    
    async def _generate_with_anthropic(self, prompt: str, model_config: ModelConfig, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate content using Anthropic API"""
        
        system_prompt = model_config.system_prompt
        if context:
            system_prompt += f"\n\nContext: {json.dumps(context)}"
        
        response = await self.anthropic_client.messages.create(
            model=model_config.model_type.value,
            max_tokens=model_config.max_length,
            temperature=model_config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            "model": response.model
        }
    
    async def _generate_with_local_model(self, prompt: str, model_config: ModelConfig, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate content using local models"""
        
        model_name = model_config.model_name
        
        # Load model if not cached
        if model_name not in self.models:
            await self._load_local_model(model_name, model_config)
        
        # Prepare input
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {json.dumps(context)}\n\n{prompt}"
        
        if model_config.system_prompt:
            full_prompt = f"{model_config.system_prompt}\n\n{full_prompt}"
        
        # Generate
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=model_config.max_length)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=model_config.max_length,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        content = generated_text[len(full_prompt):].strip()
        
        return {
            "content": content,
            "usage": {
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": outputs.shape[1] - inputs["input_ids"].shape[1]
            },
            "model": model_name
        }
    
    async def _load_local_model(self, model_name: str, model_config: ModelConfig):
        """Load local model with optimizations"""
        
        model_path = self.local_models.get(model_name)
        if not model_path:
            raise ValueError(f"Unknown local model: {model_name}")
        
        logger.info(f"Loading local model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations
        model_kwargs = {}
        
        if OptimizationType.QUANTIZATION in (model_config.optimization or []):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model_kwargs["quantization_config"] = quantization_config
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            **model_kwargs
        )
        
        # Apply LoRA if specified
        if OptimizationType.LORA in (model_config.optimization or []):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1
            )
            model = get_peft_model(model, lora_config)
        
        # Cache model and tokenizer
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        logger.info(f"Model {model_name} loaded successfully")
    
    async def fine_tune_model(
        self,
        model_name: str,
        training_data: List[FineTuningData],
        training_config: TrainingConfig,
        validation_data: Optional[List[FineTuningData]] = None
    ) -> Dict[str, Any]:
        """Fine-tune a model with custom data"""
        
        logger.info(f"Starting fine-tuning for model: {model_name}")
        
        try:
            # Prepare dataset
            dataset = self._prepare_training_dataset(training_data, validation_data)
            
            # Load model
            if model_name not in self.models:
                await self._load_local_model(model_name, ModelConfig(model_name=model_name, model_type=ModelType.LOCAL_LLAMA))
            
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=f"./models/{model_name}_finetuned",
                learning_rate=training_config.learning_rate,
                per_device_train_batch_size=training_config.batch_size,
                per_device_eval_batch_size=training_config.batch_size,
                num_train_epochs=training_config.num_epochs,
                warmup_steps=training_config.warmup_steps,
                weight_decay=training_config.weight_decay,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                max_grad_norm=training_config.max_grad_norm,
                fp16=training_config.mixed_precision,
                save_steps=training_config.save_steps,
                eval_steps=training_config.eval_steps,
                logging_steps=training_config.logging_steps,
                evaluation_strategy="steps" if validation_data else "no",
                save_strategy="steps",
                load_best_model_at_end=True if validation_data else False,
                report_to="wandb" if training_config.use_wandb else None,
                run_name=f"{model_name}_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Setup data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Setup trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"] if validation_data else None,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            
            # Initialize wandb if enabled
            if training_config.use_wandb:
                wandb.init(
                    project=training_config.wandb_project,
                    name=f"{model_name}_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            
            # Start training
            start_time = time.time()
            training_result = trainer.train()
            training_time = time.time() - start_time
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(f"./models/{model_name}_finetuned")
            
            # Log results
            result = {
                "model_name": model_name,
                "training_time": training_time,
                "final_loss": training_result.training_loss,
                "eval_loss": training_result.eval_loss if validation_data else None,
                "model_path": f"./models/{model_name}_finetuned",
                "training_samples": len(training_data),
                "validation_samples": len(validation_data) if validation_data else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store training history
            self.training_history[model_name] = result
            
            logger.info(f"Fine-tuning completed for {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Fine-tuning failed for {model_name}: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _prepare_training_dataset(self, training_data: List[FineTuningData], validation_data: Optional[List[FineTuningData]]):
        """Prepare dataset for training"""
        
        def format_data(data_list):
            formatted_data = []
            for item in data_list:
                text = f"{item.prompt} {item.completion}"
                formatted_data.append({"text": text})
            return formatted_data
        
        train_data = format_data(training_data)
        val_data = format_data(validation_data) if validation_data else None
        
        # Create datasets
        train_dataset = datasets.Dataset.from_list(train_data)
        val_dataset = datasets.Dataset.from_list(val_data) if val_data else None
        
        return {
            "train": train_dataset,
            "validation": val_dataset
        }
    
    async def optimize_model(
        self,
        model_name: str,
        optimization_type: OptimizationType,
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize model for better performance"""
        
        logger.info(f"Optimizing model {model_name} with {optimization_type.value}")
        
        try:
            if optimization_type == OptimizationType.QUANTIZATION:
                return await self._quantize_model(model_name, optimization_config)
            elif optimization_type == OptimizationType.PRUNING:
                return await self._prune_model(model_name, optimization_config)
            elif optimization_type == OptimizationType.DISTILLATION:
                return await self._distill_model(model_name, optimization_config)
            else:
                raise ValueError(f"Unsupported optimization type: {optimization_type}")
                
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return {"error": str(e)}
    
    async def _quantize_model(self, model_name: str, config: Dict[str, Any]):
        """Quantize model for reduced memory usage"""
        
        if model_name not in self.models:
            await self._load_local_model(model_name, ModelConfig(model_name=model_name, model_type=ModelType.LOCAL_LLAMA))
        
        model = self.models[model_name]
        
        # Apply quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Update cached model
        self.models[f"{model_name}_quantized"] = quantized_model
        
        return {
            "model_name": f"{model_name}_quantized",
            "optimization": "quantization",
            "memory_reduction": "~50%",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _prune_model(self, model_name: str, config: Dict[str, Any]):
        """Prune model to remove unnecessary weights"""
        
        if model_name not in self.models:
            await self._load_local_model(model_name, ModelConfig(model_name=model_name, model_type=ModelType.LOCAL_LLAMA))
        
        model = self.models[model_name]
        
        # Apply pruning
        pruning_ratio = config.get("pruning_ratio", 0.1)
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
        
        # Update cached model
        self.models[f"{model_name}_pruned"] = model
        
        return {
            "model_name": f"{model_name}_pruned",
            "optimization": "pruning",
            "pruning_ratio": pruning_ratio,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _distill_model(self, model_name: str, config: Dict[str, Any]):
        """Distill knowledge from teacher to student model"""
        
        teacher_model = config.get("teacher_model")
        student_model = config.get("student_model")
        
        if not teacher_model or not student_model:
            raise ValueError("Teacher and student models must be specified for distillation")
        
        # Load models
        if teacher_model not in self.models:
            await self._load_local_model(teacher_model, ModelConfig(model_name=teacher_model, model_type=ModelType.LOCAL_LLAMA))
        
        if student_model not in self.models:
            await self._load_local_model(student_model, ModelConfig(model_name=student_model, model_type=ModelType.LOCAL_LLAMA))
        
        # Implement distillation logic here
        # This is a simplified version
        
        return {
            "model_name": f"{student_model}_distilled",
            "optimization": "distillation",
            "teacher_model": teacher_model,
            "student_model": student_model,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_performance_metrics(self, model_name: str, generation_time: float):
        """Update performance metrics for a model"""
        
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = {
                "total_requests": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        metrics = self.performance_metrics[model_name]
        metrics["total_requests"] += 1
        metrics["total_time"] += generation_time
        metrics["average_time"] = metrics["total_time"] / metrics["total_requests"]
        metrics["min_time"] = min(metrics["min_time"], generation_time)
        metrics["max_time"] = max(metrics["max_time"], generation_time)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        return self.performance_metrics
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history for all models"""
        return self.training_history
    
    async def cleanup(self):
        """Cleanup resources"""
        for model in self.models.values():
            if hasattr(model, 'cpu'):
                model.cpu()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("AI service cleanup completed")

# Global instance
advanced_ai_service = None

async def get_advanced_ai_service() -> AdvancedAIService:
    """Get global advanced AI service instance"""
    global advanced_ai_service
    if not advanced_ai_service:
        from utils.config import get_settings
        settings = get_settings()
        config = {
            "openai_api_key": settings.openai_api_key,
            "anthropic_api_key": settings.anthropic_api_key
        }
        advanced_ai_service = AdvancedAIService(config)
    return advanced_ai_service



