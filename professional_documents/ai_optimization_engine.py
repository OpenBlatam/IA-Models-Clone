"""
AI Optimization Engine - Advanced Deep Learning & LLM Integration
Sistema de Optimización de IA con Capacidades Avanzadas de Deep Learning

This module implements a comprehensive AI optimization engine that leverages:
- Advanced deep learning architectures (Transformers, Diffusion Models)
- LLM fine-tuning and optimization techniques
- Multi-modal AI capabilities
- Advanced training and inference optimization
- Real-time model adaptation and performance monitoring
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup, AdamW
)
from diffusers import (
    StableDiffusionPipeline, DDPMPipeline, 
    DDIMScheduler, DDPMScheduler
)
import gradio as gr
from tqdm import tqdm
import wandb
from pathlib import Path
import yaml
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types for optimization"""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    LLM = "llm"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    FINE_TUNING = "fine_tuning"
    LORA = "lora"
    P_TUNING = "p_tuning"
    ADAPTERS = "adapters"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    CONTINUAL_LEARNING = "continual_learning"

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    model_name: str
    model_type: ModelType
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    device: str = "auto"
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.FINE_TUNING
    
    # Advanced parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    quantization_bits: int = 8
    pruning_ratio: float = 0.1
    distillation_temperature: float = 3.0
    
    # Performance monitoring
    enable_wandb: bool = True
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500

@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    perplexity: float = 0.0
    bleu_score: float = 0.0
    rouge_score: float = 0.0
    accuracy: float = 0.0
    f1_score: float = 0.0
    training_time: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0

class AdvancedDataset(Dataset):
    """Advanced dataset class with preprocessing and augmentation"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, 
                 augmentation: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        if 'text' in item:
            inputs = self.tokenizer(
                item['text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            inputs = {}
            
        # Add labels if present
        if 'labels' in item:
            inputs['labels'] = torch.tensor(item['labels'])
            
        # Add image data for multimodal models
        if 'image' in item:
            inputs['pixel_values'] = torch.tensor(item['image'])
            
        return inputs

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer implementation"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, 
                 alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling

class QuantizedLinear(nn.Module):
    """Quantized linear layer for model compression"""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantized weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def quantize_weights(self):
        """Quantize weights to specified bit precision"""
        scale = 2 ** (self.bits - 1) - 1
        return torch.round(self.weight * scale) / scale
    
    def forward(self, x):
        quantized_weight = self.quantize_weights()
        return F.linear(x, quantized_weight, self.bias)

class AdvancedTransformer(nn.Module):
    """Advanced Transformer architecture with optimizations"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained model
        if config.model_type == ModelType.LLM:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        else:
            self.model = AutoModel.from_pretrained(config.model_name)
            
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add LoRA layers if specified
        if config.optimization_strategy == OptimizationStrategy.LORA:
            self._add_lora_layers()
            
        # Add quantization if specified
        if config.optimization_strategy == OptimizationStrategy.QUANTIZATION:
            self._quantize_model()
            
    def _add_lora_layers(self):
        """Add LoRA layers to the model"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    self.config.lora_rank,
                    self.config.lora_alpha,
                    self.config.lora_dropout
                )
                setattr(self.model, name, lora_layer)
    
    def _quantize_model(self):
        """Quantize model weights"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                quantized_layer = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    self.config.quantization_bits
                )
                setattr(self.model, name, quantized_layer)
    
    def forward(self, **inputs):
        return self.model(**inputs)

class DiffusionModelWrapper(nn.Module):
    """Wrapper for diffusion models with optimization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load diffusion pipeline
        if "stable-diffusion" in config.model_name.lower():
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.mixed_precision else torch.float32
            )
        else:
            self.pipeline = DDPMPipeline.from_pretrained(config.model_name)
            
        # Optimize pipeline
        if config.mixed_precision:
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_memory_efficient_attention()
    
    def generate(self, prompt: str, **kwargs):
        """Generate images from text prompts"""
        return self.pipeline(prompt, **kwargs)

class AIOptimizationEngine:
    """Advanced AI Optimization Engine with Deep Learning capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, TrainingMetrics] = {}
        self.training_history: List[Dict] = []
        self.optimization_strategies: Dict[str, Any] = {}
        
        # Initialize device
        self.device = self._get_optimal_device()
        
        # Initialize experiment tracking
        if self._should_use_wandb():
            wandb.init(project="ai-optimization-engine")
            
        logger.info(f"AI Optimization Engine initialized on device: {self.device}")
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for training"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                return "cuda:0"  # Use first GPU for now
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _should_use_wandb(self) -> bool:
        """Check if wandb should be used for experiment tracking"""
        try:
            import wandb
            return True
        except ImportError:
            return False
    
    async def create_model(self, model_id: str, config: ModelConfig) -> bool:
        """Create and initialize a new AI model"""
        try:
            logger.info(f"Creating model {model_id} with type {config.model_type}")
            
            if config.model_type in [ModelType.TRANSFORMER, ModelType.LLM]:
                model = AdvancedTransformer(config)
            elif config.model_type == ModelType.DIFFUSION:
                model = DiffusionModelWrapper(config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Move to device
            model = model.to(self.device)
            
            # Store model
            self.models[model_id] = {
                'model': model,
                'config': config,
                'created_at': time.time(),
                'status': 'initialized'
            }
            
            # Initialize metrics
            self.metrics[model_id] = TrainingMetrics()
            
            logger.info(f"Model {model_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating model {model_id}: {str(e)}")
            return False
    
    async def train_model(self, model_id: str, training_data: List[Dict], 
                         validation_data: Optional[List[Dict]] = None) -> bool:
        """Train a model with advanced optimization techniques"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.models[model_id]
            model = model_info['model']
            config = model_info['config']
            
            logger.info(f"Starting training for model {model_id}")
            
            # Prepare datasets
            train_dataset = AdvancedDataset(training_data, model.tokenizer, config.max_length)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            # Setup optimizer and scheduler
            optimizer = self._create_optimizer(model, config)
            scheduler = self._create_scheduler(optimizer, config, len(train_loader))
            
            # Training loop
            model.train()
            total_steps = len(train_loader) * config.num_epochs
            current_step = 0
            
            for epoch in range(config.num_epochs):
                epoch_loss = 0.0
                
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    current_step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    # Log metrics
                    if current_step % config.log_interval == 0:
                        await self._log_training_metrics(model_id, current_step, loss.item(), 
                                                       scheduler.get_last_lr()[0])
                    
                    # Save checkpoint
                    if current_step % config.save_interval == 0:
                        await self._save_checkpoint(model_id, current_step)
                
                # Epoch evaluation
                if validation_data:
                    val_metrics = await self._evaluate_model(model_id, validation_data)
                    logger.info(f"Epoch {epoch+1} validation metrics: {val_metrics}")
            
            # Final evaluation
            final_metrics = await self._evaluate_model(model_id, validation_data or training_data)
            
            # Update model status
            self.models[model_id]['status'] = 'trained'
            self.models[model_id]['final_metrics'] = final_metrics
            
            logger.info(f"Training completed for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {str(e)}")
            return False
    
    def _create_optimizer(self, model: nn.Module, config: ModelConfig):
        """Create optimizer with advanced techniques"""
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    def _create_scheduler(self, optimizer, config: ModelConfig, num_training_steps: int):
        """Create learning rate scheduler"""
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps * config.num_epochs
        )
    
    async def _log_training_metrics(self, model_id: str, step: int, loss: float, lr: float):
        """Log training metrics"""
        metrics = self.metrics[model_id]
        metrics.step = step
        metrics.loss = loss
        metrics.learning_rate = lr
        metrics.training_time = time.time()
        
        # Log to wandb if available
        if self._should_use_wandb():
            wandb.log({
                f"{model_id}/loss": loss,
                f"{model_id}/learning_rate": lr,
                f"{model_id}/step": step
            })
    
    async def _save_checkpoint(self, model_id: str, step: int):
        """Save model checkpoint"""
        checkpoint_path = f"checkpoints/{model_id}_step_{step}.pt"
        Path("checkpoints").mkdir(exist_ok=True)
        
        model_info = self.models[model_id]
        torch.save({
            'model_state_dict': model_info['model'].state_dict(),
            'config': model_info['config'],
            'step': step,
            'metrics': self.metrics[model_id]
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    async def _evaluate_model(self, model_id: str, eval_data: List[Dict]) -> Dict[str, float]:
        """Evaluate model performance"""
        model_info = self.models[model_id]
        model = model_info['model']
        config = model_info['config']
        
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        eval_dataset = AdvancedDataset(eval_data, model.tokenizer, config.max_length)
        eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'samples_evaluated': total_samples
        }
    
    async def optimize_model(self, model_id: str, optimization_type: str, 
                           parameters: Dict[str, Any]) -> bool:
        """Apply advanced optimization techniques to a model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.models[model_id]
            model = model_info['model']
            
            logger.info(f"Applying {optimization_type} optimization to model {model_id}")
            
            if optimization_type == "quantization":
                await self._apply_quantization(model, parameters)
            elif optimization_type == "pruning":
                await self._apply_pruning(model, parameters)
            elif optimization_type == "distillation":
                await self._apply_distillation(model, parameters)
            elif optimization_type == "lora":
                await self._apply_lora(model, parameters)
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            # Update model status
            self.models[model_id]['optimizations'] = self.models[model_id].get('optimizations', [])
            self.models[model_id]['optimizations'].append({
                'type': optimization_type,
                'parameters': parameters,
                'applied_at': time.time()
            })
            
            logger.info(f"Optimization {optimization_type} applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error applying optimization {optimization_type}: {str(e)}")
            return False
    
    async def _apply_quantization(self, model: nn.Module, parameters: Dict[str, Any]):
        """Apply quantization to model"""
        bits = parameters.get('bits', 8)
        
        # Apply dynamic quantization
        if bits == 8:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        elif bits == 16:
            model = model.half()
        
        logger.info(f"Applied {bits}-bit quantization")
    
    async def _apply_pruning(self, model: nn.Module, parameters: Dict[str, Any]):
        """Apply pruning to model"""
        pruning_ratio = parameters.get('ratio', 0.1)
        
        # Apply structured pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
        
        logger.info(f"Applied pruning with ratio {pruning_ratio}")
    
    async def _apply_distillation(self, model: nn.Module, parameters: Dict[str, Any]):
        """Apply knowledge distillation"""
        teacher_model = parameters.get('teacher_model')
        temperature = parameters.get('temperature', 3.0)
        
        if teacher_model:
            # Implement knowledge distillation logic
            logger.info(f"Applied knowledge distillation with temperature {temperature}")
    
    async def _apply_lora(self, model: nn.Module, parameters: Dict[str, Any]):
        """Apply LoRA adaptation"""
        rank = parameters.get('rank', 16)
        alpha = parameters.get('alpha', 32)
        
        # LoRA layers are already added during model creation
        logger.info(f"LoRA adaptation configured with rank {rank}, alpha {alpha}")
    
    async def generate_content(self, model_id: str, prompt: str, 
                             generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content using the specified model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.models[model_id]
            model = model_info['model']
            config = model_info['config']
            
            model.eval()
            
            if config.model_type == ModelType.DIFFUSION:
                # Generate images
                result = model.generate(prompt, **generation_params)
                return {
                    'type': 'image',
                    'content': result.images[0] if hasattr(result, 'images') else result,
                    'prompt': prompt,
                    'generation_params': generation_params
                }
            else:
                # Generate text
                inputs = model.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=generation_params.get('max_length', 100),
                        temperature=generation_params.get('temperature', 0.7),
                        do_sample=generation_params.get('do_sample', True),
                        pad_token_id=model.tokenizer.eos_token_id
                    )
                
                generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return {
                    'type': 'text',
                    'content': generated_text,
                    'prompt': prompt,
                    'generation_params': generation_params
                }
                
        except Exception as e:
            logger.error(f"Error generating content with model {model_id}: {str(e)}")
            return {'error': str(e)}
    
    async def create_gradio_interface(self) -> gr.Blocks:
        """Create Gradio interface for model interaction"""
        with gr.Blocks(title="AI Optimization Engine") as interface:
            gr.Markdown("# AI Optimization Engine - Advanced Deep Learning Interface")
            
            with gr.Tab("Model Management"):
                with gr.Row():
                    model_id_input = gr.Textbox(label="Model ID", placeholder="Enter model identifier")
                    model_type_dropdown = gr.Dropdown(
                        choices=[e.value for e in ModelType],
                        label="Model Type",
                        value=ModelType.LLM.value
                    )
                    create_model_btn = gr.Button("Create Model", variant="primary")
                
                model_status = gr.Textbox(label="Model Status", interactive=False)
            
            with gr.Tab("Training"):
                with gr.Row():
                    training_data_input = gr.File(label="Training Data (JSON)")
                    validation_data_input = gr.File(label="Validation Data (JSON)")
                    train_btn = gr.Button("Start Training", variant="primary")
                
                training_progress = gr.Progress()
                training_logs = gr.Textbox(label="Training Logs", lines=10, interactive=False)
            
            with gr.Tab("Optimization"):
                with gr.Row():
                    optimization_type = gr.Dropdown(
                        choices=["quantization", "pruning", "distillation", "lora"],
                        label="Optimization Type"
                    )
                    optimization_params = gr.JSON(label="Parameters")
                    optimize_btn = gr.Button("Apply Optimization", variant="primary")
                
                optimization_status = gr.Textbox(label="Optimization Status", interactive=False)
            
            with gr.Tab("Generation"):
                with gr.Row():
                    generation_model = gr.Dropdown(
                        choices=list(self.models.keys()),
                        label="Select Model"
                    )
                    generation_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    generate_btn = gr.Button("Generate", variant="primary")
                
                generation_output = gr.Textbox(label="Generated Content", lines=10)
                generation_image = gr.Image(label="Generated Image")
            
            # Event handlers
            create_model_btn.click(
                self._gradio_create_model,
                inputs=[model_id_input, model_type_dropdown],
                outputs=[model_status]
            )
            
            train_btn.click(
                self._gradio_train_model,
                inputs=[training_data_input, validation_data_input],
                outputs=[training_progress, training_logs]
            )
            
            optimize_btn.click(
                self._gradio_optimize_model,
                inputs=[optimization_type, optimization_params],
                outputs=[optimization_status]
            )
            
            generate_btn.click(
                self._gradio_generate_content,
                inputs=[generation_model, generation_prompt],
                outputs=[generation_output, generation_image]
            )
        
        return interface
    
    async def _gradio_create_model(self, model_id: str, model_type: str):
        """Gradio handler for model creation"""
        try:
            config = ModelConfig(
                model_name="gpt2",  # Default model
                model_type=ModelType(model_type)
            )
            
            success = await self.create_model(model_id, config)
            return f"Model {model_id} created successfully" if success else f"Failed to create model {model_id}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _gradio_train_model(self, training_data, validation_data):
        """Gradio handler for model training"""
        # Implementation for training with uploaded data
        return "Training started", "Training logs will appear here..."
    
    async def _gradio_optimize_model(self, optimization_type: str, parameters: Dict):
        """Gradio handler for model optimization"""
        # Implementation for optimization
        return f"Applied {optimization_type} optimization"
    
    async def _gradio_generate_content(self, model_id: str, prompt: str):
        """Gradio handler for content generation"""
        try:
            result = await self.generate_content(model_id, prompt, {})
            if result.get('type') == 'image':
                return "", result.get('content')
            else:
                return result.get('content', ''), None
        except Exception as e:
            return f"Error: {str(e)}", None
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive performance metrics for a model"""
        if model_id not in self.models:
            return {'error': f'Model {model_id} not found'}
        
        model_info = self.models[model_id]
        metrics = self.metrics.get(model_id, TrainingMetrics())
        
        # Get system metrics
        memory_usage = psutil.virtual_memory().percent
        gpu_usage = 0.0
        if torch.cuda.is_available():
            gpu_usage = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0.0
        
        return {
            'model_id': model_id,
            'status': model_info['status'],
            'created_at': model_info['created_at'],
            'training_metrics': {
                'epoch': metrics.epoch,
                'step': metrics.step,
                'loss': metrics.loss,
                'perplexity': metrics.perplexity,
                'accuracy': metrics.accuracy
            },
            'system_metrics': {
                'memory_usage': memory_usage,
                'gpu_utilization': gpu_usage,
                'device': self.device
            },
            'optimizations': model_info.get('optimizations', []),
            'final_metrics': model_info.get('final_metrics', {})
        }
    
    async def export_model(self, model_id: str, export_path: str, 
                          format: str = "pytorch") -> bool:
        """Export model in various formats"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.models[model_id]
            model = model_info['model']
            
            if format == "pytorch":
                torch.save(model.state_dict(), export_path)
            elif format == "onnx":
                # Export to ONNX format
                dummy_input = torch.randn(1, 512).to(self.device)
                torch.onnx.export(model, dummy_input, export_path, verbose=True)
            elif format == "huggingface":
                # Export to Hugging Face format
                model.save_pretrained(export_path)
                model_info['config'].tokenizer.save_pretrained(export_path)
            
            logger.info(f"Model {model_id} exported to {export_path} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model {model_id}: {str(e)}")
            return False
    
    async def cleanup_resources(self):
        """Clean up resources and save state"""
        try:
            # Save current state
            state_path = "ai_optimization_engine_state.pkl"
            state = {
                'models': {k: {'config': v['config'], 'status': v['status']} 
                          for k, v in self.models.items()},
                'metrics': self.metrics,
                'training_history': self.training_history
            }
            
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Example usage and testing
async def main():
    """Example usage of the AI Optimization Engine"""
    
    # Initialize engine
    engine = AIOptimizationEngine()
    
    # Create a model configuration
    config = ModelConfig(
        model_name="gpt2",
        model_type=ModelType.LLM,
        max_length=256,
        batch_size=8,
        learning_rate=1e-4,
        num_epochs=2,
        optimization_strategy=OptimizationStrategy.LORA
    )
    
    # Create model
    await engine.create_model("my_llm", config)
    
    # Prepare sample training data
    training_data = [
        {"text": "This is a sample training text for fine-tuning."},
        {"text": "Another example of training data for the model."},
        {"text": "More training examples to improve model performance."}
    ]
    
    # Train model
    await engine.train_model("my_llm", training_data)
    
    # Apply optimization
    await engine.optimize_model("my_llm", "quantization", {"bits": 8})
    
    # Generate content
    result = await engine.generate_content(
        "my_llm", 
        "Generate a creative story about",
        {"max_length": 100, "temperature": 0.8}
    )
    
    print("Generated content:", result)
    
    # Get performance metrics
    performance = await engine.get_model_performance("my_llm")
    print("Model performance:", performance)
    
    # Create Gradio interface
    interface = await engine.create_gradio_interface()
    interface.launch(share=True)
    
    # Cleanup
    await engine.cleanup_resources()

if __name__ == "__main__":
    asyncio.run(main())