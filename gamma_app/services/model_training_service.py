"""
Gamma App - Model Training Service
Advanced model training and fine-tuning service for deep learning models
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import os
from datetime import datetime, timedelta
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup, AdamW
)
from diffusers import StableDiffusionPipeline, DDPMPipeline
from diffusers.optimization import get_scheduler
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from pathlib import Path
import wandb
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model types"""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    CUSTOM = "custom"
    FINE_TUNED = "fine_tuned"

class TrainingStatus(Enum):
    """Training status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class OptimizerType(Enum):
    """Optimizer types"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSprop = "rmsprop"

class SchedulerType(Enum):
    """Scheduler types"""
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: ModelType
    model_name: str
    task: str
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.LINEAR
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./models"
    logging_dir: str = "./logs"
    use_wandb: bool = False
    wandb_project: str = "gamma-app"
    use_accelerate: bool = False
    use_lora: bool = False
    lora_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingMetrics:
    """Training metrics"""
    epoch: int
    step: int
    train_loss: float
    eval_loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TrainingResult:
    """Training result"""
    success: bool
    model_path: str
    metrics: List[TrainingMetrics]
    best_metrics: Dict[str, float]
    training_time: float
    error: Optional[str] = None

class ModelTrainingService:
    """Advanced model training service"""
    
    def __init__(self):
        self.active_trainings = {}
        self.training_history = []
        self.accelerator = None
        self._initialize_accelerator()
    
    def _initialize_accelerator(self):
        """Initialize Accelerator for distributed training"""
        try:
            self.accelerator = Accelerator()
            logger.info("Initialized Accelerator for distributed training")
        except Exception as e:
            logger.warning(f"Could not initialize Accelerator: {e}")
    
    async def train_model(
        self,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        model: Optional[nn.Module] = None
    ) -> TrainingResult:
        """Train a model with given configuration"""
        try:
            training_id = str(uuid.uuid4())
            logger.info(f"Starting model training: {training_id}")
            
            # Initialize training
            start_time = time.time()
            metrics = []
            best_metrics = {}
            
            # Setup model
            if model is None:
                model = await self._load_model(config)
            
            # Setup optimizer and scheduler
            optimizer = self._create_optimizer(model, config)
            scheduler = self._create_scheduler(optimizer, config, len(train_dataloader))
            
            # Setup training components
            if config.use_accelerate and self.accelerator:
                model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
                    model, optimizer, train_dataloader, eval_dataloader
                )
            
            # Setup logging
            writer = SummaryWriter(config.logging_dir)
            if config.use_wandb:
                wandb.init(project=config.wandb_project, config=config.__dict__)
            
            # Training loop
            model.train()
            global_step = 0
            
            for epoch in range(config.num_epochs):
                epoch_metrics = await self._train_epoch(
                    model, optimizer, scheduler, train_dataloader,
                    config, epoch, global_step, writer
                )
                metrics.extend(epoch_metrics)
                global_step += len(train_dataloader)
                
                # Evaluation
                if eval_dataloader and (epoch + 1) % config.eval_strategy == 0:
                    eval_metrics = await self._evaluate_model(model, eval_dataloader, config)
                    metrics.extend(eval_metrics)
                    
                    # Update best metrics
                    if eval_metrics:
                        latest_eval = eval_metrics[-1]
                        if latest_eval.eval_loss < best_metrics.get('best_eval_loss', float('inf')):
                            best_metrics['best_eval_loss'] = latest_eval.eval_loss
                            best_metrics['best_accuracy'] = latest_eval.accuracy
                            best_metrics['best_f1_score'] = latest_eval.f1_score
                            
                            # Save best model
                            await self._save_model(model, config, "best")
                
                # Save checkpoint
                if (epoch + 1) % config.save_strategy == 0:
                    await self._save_model(model, config, f"epoch_{epoch+1}")
            
            # Final save
            final_model_path = await self._save_model(model, config, "final")
            
            # Cleanup
            writer.close()
            if config.use_wandb:
                wandb.finish()
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                success=True,
                model_path=final_model_path,
                metrics=metrics,
                best_metrics=best_metrics,
                training_time=training_time
            )
            
            self.training_history.append(result)
            logger.info(f"Training completed: {training_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return TrainingResult(
                success=False,
                model_path="",
                metrics=[],
                best_metrics={},
                training_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _load_model(self, config: TrainingConfig) -> nn.Module:
        """Load model based on configuration"""
        try:
            if config.model_type == ModelType.TRANSFORMER:
                if config.task == "text_classification":
                    model = AutoModel.from_pretrained(config.model_name)
                elif config.task == "text_generation":
                    model = AutoModelForCausalLM.from_pretrained(config.model_name)
                else:
                    model = AutoModel.from_pretrained(config.model_name)
            
            elif config.model_type == ModelType.DIFFUSION:
                if "stable-diffusion" in config.model_name.lower():
                    model = StableDiffusionPipeline.from_pretrained(config.model_name)
                else:
                    model = DDPMPipeline.from_pretrained(config.model_name)
            
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Apply LoRA if configured
            if config.use_lora:
                model = self._apply_lora(model, config)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _apply_lora(self, model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Apply LoRA to model"""
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_config.get('r', 16),
                lora_alpha=config.lora_config.get('lora_alpha', 32),
                lora_dropout=config.lora_config.get('lora_dropout', 0.1),
                target_modules=config.lora_config.get('target_modules', ["q_proj", "v_proj"])
            )
            
            model = get_peft_model(model, lora_config)
            logger.info("Applied LoRA to model")
            
            return model
            
        except Exception as e:
            logger.error(f"Error applying LoRA: {e}")
            raise
    
    def _create_optimizer(self, model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """Create optimizer"""
        try:
            if config.optimizer == OptimizerType.ADAM:
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
            elif config.optimizer == OptimizerType.ADAMW:
                optimizer = AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
            elif config.optimizer == OptimizerType.SGD:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    momentum=0.9
                )
            elif config.optimizer == OptimizerType.RMSprop:
                optimizer = optim.RMSprop(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer: {config.optimizer}")
            
            return optimizer
            
        except Exception as e:
            logger.error(f"Error creating optimizer: {e}")
            raise
    
    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        config: TrainingConfig,
        num_training_steps: int
    ) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        try:
            if config.scheduler == SchedulerType.LINEAR:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=config.warmup_steps,
                    num_training_steps=num_training_steps * config.num_epochs
                )
            elif config.scheduler == SchedulerType.COSINE:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config.num_epochs
                )
            elif config.scheduler == SchedulerType.STEP:
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.num_epochs // 3,
                    gamma=0.1
                )
            elif config.scheduler == SchedulerType.EXPONENTIAL:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=0.95
                )
            elif config.scheduler == SchedulerType.PLATEAU:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=2
                )
            else:
                raise ValueError(f"Unsupported scheduler: {config.scheduler}")
            
            return scheduler
            
        except Exception as e:
            logger.error(f"Error creating scheduler: {e}")
            raise
    
    async def _train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        dataloader: DataLoader,
        config: TrainingConfig,
        epoch: int,
        global_step: int,
        writer: SummaryWriter
    ) -> List[TrainingMetrics]:
        """Train for one epoch"""
        try:
            metrics = []
            total_loss = 0.0
            num_batches = len(dataloader)
            
            for batch_idx, batch in enumerate(dataloader):
                # Forward pass
                if config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                else:
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Backward pass
                if config.use_accelerate and self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                
                # Gradient clipping
                if config.max_grad_norm > 0:
                    if config.use_accelerate and self.accelerator:
                        self.accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Logging
                if batch_idx % config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    gradient_norm = self._calculate_gradient_norm(model)
                    
                    metric = TrainingMetrics(
                        epoch=epoch,
                        step=global_step + batch_idx,
                        train_loss=loss.item(),
                        learning_rate=current_lr,
                        gradient_norm=gradient_norm
                    )
                    metrics.append(metric)
                    
                    # TensorBoard logging
                    writer.add_scalar('Train/Loss', loss.item(), global_step + batch_idx)
                    writer.add_scalar('Train/LearningRate', current_lr, global_step + batch_idx)
                    writer.add_scalar('Train/GradientNorm', gradient_norm, global_step + batch_idx)
                    
                    # Wandb logging
                    if config.use_wandb:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/learning_rate': current_lr,
                            'train/gradient_norm': gradient_norm,
                            'epoch': epoch,
                            'step': global_step + batch_idx
                        })
                    
                    logger.info(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training epoch: {e}")
            raise
    
    async def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: TrainingConfig
    ) -> List[TrainingMetrics]:
        """Evaluate model"""
        try:
            model.eval()
            total_loss = 0.0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in dataloader:
                    if config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = model(**batch)
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                    else:
                        outputs = model(**batch)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                    
                    total_loss += loss.item()
                    
                    # Collect predictions and labels for metrics
                    if hasattr(outputs, 'logits') and 'labels' in batch:
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(batch['labels'].cpu().numpy())
            
            avg_loss = total_loss / len(dataloader)
            
            # Calculate metrics
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
            
            if all_predictions and all_labels:
                accuracy = accuracy_score(all_labels, all_predictions)
                precision, recall, f1_score, _ = precision_recall_fscore_support(
                    all_labels, all_predictions, average='weighted'
                )
            
            metric = TrainingMetrics(
                epoch=0,  # Will be set by caller
                step=0,   # Will be set by caller
                train_loss=0.0,
                eval_loss=avg_loss,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
            
            model.train()
            return [metric]
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return []
    
    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calculate gradient norm"""
        try:
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** (1. / 2)
        except Exception as e:
            logger.error(f"Error calculating gradient norm: {e}")
            return 0.0
    
    async def _save_model(self, model: nn.Module, config: TrainingConfig, suffix: str) -> str:
        """Save model"""
        try:
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / f"{config.model_name}_{suffix}"
            
            if config.use_lora:
                # Save LoRA weights
                model.save_pretrained(model_path)
            else:
                # Save full model
                model.save_pretrained(model_path)
            
            logger.info(f"Model saved to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def get_training_history(self) -> List[TrainingResult]:
        """Get training history"""
        return self.training_history.copy()
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        try:
            if not self.training_history:
                return {}
            
            stats = {
                'total_trainings': len(self.training_history),
                'successful_trainings': len([t for t in self.training_history if t.success]),
                'failed_trainings': len([t for t in self.training_history if not t.success]),
                'average_training_time': np.mean([t.training_time for t in self.training_history]),
                'total_training_time': sum([t.training_time for t in self.training_history])
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting training statistics: {e}")
            return {}
    
    def export_training_report(self, training_id: str, output_path: str) -> bool:
        """Export training report"""
        try:
            # Find training result
            training_result = None
            for result in self.training_history:
                if str(result.model_path) == training_id:
                    training_result = result
                    break
            
            if not training_result:
                return False
            
            # Create report
            report = {
                'training_id': training_id,
                'success': training_result.success,
                'model_path': training_result.model_path,
                'training_time': training_result.training_time,
                'best_metrics': training_result.best_metrics,
                'error': training_result.error,
                'metrics_history': [
                    {
                        'epoch': m.epoch,
                        'step': m.step,
                        'train_loss': m.train_loss,
                        'eval_loss': m.eval_loss,
                        'accuracy': m.accuracy,
                        'precision': m.precision,
                        'recall': m.recall,
                        'f1_score': m.f1_score,
                        'learning_rate': m.learning_rate,
                        'gradient_norm': m.gradient_norm,
                        'timestamp': m.timestamp.isoformat()
                    }
                    for m in training_result.metrics
                ]
            }
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Training report exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting training report: {e}")
            return False

# Global model training service instance
model_training_service = ModelTrainingService()

async def train_model(config: TrainingConfig, train_dataloader: DataLoader, eval_dataloader: DataLoader = None, model: nn.Module = None) -> TrainingResult:
    """Train model using global service"""
    return await model_training_service.train_model(config, train_dataloader, eval_dataloader, model)

def get_training_history() -> List[TrainingResult]:
    """Get training history using global service"""
    return model_training_service.get_training_history()

def get_training_statistics() -> Dict[str, Any]:
    """Get training statistics using global service"""
    return model_training_service.get_training_statistics()

def export_training_report(training_id: str, output_path: str) -> bool:
    """Export training report using global service"""
    return model_training_service.export_training_report(training_id, output_path)
























