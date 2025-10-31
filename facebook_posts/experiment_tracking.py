#!/usr/bin/env python3
"""
🔬 Experiment Tracking System for Gradient Clipping and NaN Handling
==================================================================

Comprehensive experiment tracking system integrating TensorBoard and Weights & Biases.
Provides unified logging, visualization, and experiment management capabilities for
numerical stability training experiments.
"""

import os
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import warnings

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

# Import wandb with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available. Install with: pip install wandb")

# Import Transformers-specific modules
try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        TrainingArguments, Trainer,
        PreTrainedModel, PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Language model tracking features will be disabled.")

# Import Diffusers-specific modules
try:
    from diffusers import (
        DiffusionPipeline, StableDiffusionPipeline, DDIMPipeline,
        DDPMPipeline, UNet2DConditionModel, AutoencoderKL,
        SchedulerMixin, DDIMScheduler, DDPMScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: Diffusers not available. Diffusion model tracking features will be disabled.")

# Import our centralized logging configuration
from logging_config import (
    get_logger, log_training_step, log_numerical_issue, 
    log_system_event, log_error_with_context, log_performance_metrics
)

warnings.filterwarnings('ignore')

logger = get_logger(__name__)

# =============================================================================
# EXPERIMENT TRACKING CONFIGURATION
# =============================================================================

@dataclass
class ProblemDefinition:
    """Comprehensive problem definition for the experiment"""
    
    # Core problem description
    problem_title: str = ""
    problem_description: str = ""
    problem_type: str = "classification"  # classification, regression, generation, etc.
    domain: str = "general"  # nlp, cv, audio, multimodal, etc.
    
    # Objectives and metrics
    primary_objective: str = ""
    success_metrics: List[str] = field(default_factory=list)
    baseline_performance: Optional[float] = None
    target_performance: Optional[float] = None
    
    # Constraints and requirements
    computational_constraints: str = ""
    time_constraints: str = ""
    accuracy_requirements: str = ""
    interpretability_requirements: str = ""
    
    # Business context
    business_value: str = ""
    stakeholders: List[str] = field(default_factory=list)
    deployment_context: str = ""

@dataclass
class DatasetAnalysis:
    """Comprehensive dataset analysis and characteristics"""
    
    # Dataset metadata
    dataset_name: str = ""
    dataset_source: str = ""
    dataset_version: str = ""
    dataset_size: Optional[int] = None
    
    # Data characteristics
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    feature_count: Optional[int] = None
    class_count: Optional[int] = None
    data_types: List[str] = field(default_factory=list)
    
    # Data quality
    missing_values_pct: Optional[float] = None
    duplicate_records_pct: Optional[float] = None
    outlier_pct: Optional[float] = None
    class_imbalance_ratio: Optional[float] = None
    
    # Data distribution
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    test_size: Optional[int] = None
    data_split_strategy: str = ""
    
    # Preprocessing requirements
    normalization_needed: bool = False
    encoding_needed: bool = False
    augmentation_strategy: str = ""
    preprocessing_steps: List[str] = field(default_factory=list)

@dataclass
class ExperimentConfig:
    """Enhanced configuration for experiment tracking with problem definition and dataset analysis"""
    
    # Experiment metadata
    experiment_name: str = "gradient_clipping_nan_handling"
    project_name: str = "blatam_academy_facebook_posts"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Problem definition and dataset analysis
    problem_definition: Optional[ProblemDefinition] = None
    dataset_analysis: Optional[DatasetAnalysis] = None
    
    # Tracking settings
    enable_tensorboard: bool = True
    enable_wandb: bool = True
    log_interval: int = 100  # Log every N steps
    save_interval: int = 1000  # Save model every N steps
    
    # Logging settings
    log_metrics: bool = True
    log_hyperparameters: bool = True
    log_model_architecture: bool = True
    log_gradients: bool = True
    log_images: bool = True
    log_text: bool = True
    
    # File paths
    tensorboard_dir: str = "runs/tensorboard"
    model_save_dir: str = "models"
    config_save_dir: str = "configs"
    
    # Wandb settings
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: str = "training"
    
    # Advanced settings
    sync_tensorboard: bool = True  # Sync TensorBoard logs to wandb
    resume_run: bool = False
    anonymous: bool = False
    
    # Numerical stability specific settings
    log_gradient_norms: bool = True
    log_nan_inf_counts: bool = True
    log_clipping_stats: bool = True
    log_performance_metrics: bool = True


@dataclass
class LanguageModelMetrics:
    """Metrics specific to language models and Transformers."""
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_score: Optional[float] = None
    token_accuracy: Optional[float] = None
    sequence_length: Optional[int] = None
    vocab_size: Optional[int] = None
    attention_weights_norm: Optional[float] = None
    layer_norm_stats: Optional[Dict[str, float]] = None
    gradient_flow: Optional[Dict[str, float]] = None

@dataclass
class DiffusionModelMetrics:
    """Metrics specific to diffusion models and image generation."""
    noise_level: Optional[float] = None
    denoising_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    image_quality_score: Optional[float] = None
    generation_time: Optional[float] = None
    memory_usage: Optional[float] = None
    scheduler_step: Optional[int] = None
    noise_prediction_loss: Optional[float] = None
    classifier_free_guidance: Optional[bool] = None
    prompt_embedding_norm: Optional[float] = None
    cross_attention_weights: Optional[torch.Tensor] = None
    latent_space_stats: Optional[Dict[str, float]] = None

@dataclass
class TrainingMetrics:
    """Enhanced training metrics with language model support."""
    loss: float = 0.0
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    nan_count: int = 0
    inf_count: int = 0
    clipping_applied: bool = False
    clipping_threshold: Optional[float] = None
    training_time: float = 0.0
    memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    # Language model specific metrics
    language_model_metrics: Optional[LanguageModelMetrics] = None
    
    # Diffusion model specific metrics
    diffusion_model_metrics: Optional[DiffusionModelMetrics] = None


@dataclass
class ModelCheckpoint:
    """Model checkpoint information"""
    epoch: int
    step: int
    loss: float
    metrics: Dict[str, Any]
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# EXPERIMENT TRACKER CLASS
# =============================================================================

class ExperimentTracker:
    """Enhanced experiment tracker with Transformers support."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment tracker."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize tracking systems
        self.tensorboard_writer = None
        self.wandb_run = None
        self.current_step = 0
        self.current_epoch = 0
        
        # Metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.checkpoints: List[ModelCheckpoint] = []
        
        # Transformers-specific tracking
        self.transformer_model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_config: Optional[AutoConfig] = None
        
        # Language model metrics history
        self.lm_metrics_history: List[LanguageModelMetrics] = []
        
        # Diffusers-specific tracking
        self.diffusion_pipeline: Optional[DiffusionPipeline] = None
        self.unet_model: Optional[UNet2DConditionModel] = None
        self.vae_model: Optional[AutoencoderKL] = None
        self.scheduler: Optional[SchedulerMixin] = None
        
        # Diffusion model metrics history
        self.diffusion_metrics_history: List[DiffusionModelMetrics] = []
        
        # Threading for async operations
        self.metrics_queue = queue.Queue()
        self.processing_thread = None
        self.stop_processing = False
        
        # Initialize tracking systems
        self._setup_tracking()
        self._start_processing_thread()
    
    def _setup_tracking(self):
        """Setup TensorBoard and Weights & Biases tracking."""
        try:
            # Setup TensorBoard
            if self.config.enable_tensorboard:
                self._setup_tensorboard()
            
            # Setup Weights & Biases
            if self.config.enable_wandb and WANDB_AVAILABLE:
                self._setup_wandb()
                
        except Exception as e:
            self.logger.error(f"Failed to setup tracking systems: {e}")
            traceback.print_exc()
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        try:
            # Create TensorBoard directory
            tb_dir = Path(self.config.tensorboard_dir)
            tb_dir.mkdir(parents=True, exist_ok=True)
            
            # Create unique run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = tb_dir / f"{self.config.experiment_name}_{timestamp}"
            run_dir.mkdir(exist_ok=True)
            
            self.tensorboard_writer = SummaryWriter(log_dir=str(run_dir))
            self.logger.info(f"TensorBoard logging initialized at: {run_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup TensorBoard: {e}")
            self.tensorboard_writer = None
    
    def _setup_wandb(self):
        """Setup Weights & Biases tracking."""
        try:
            if not WANDB_AVAILABLE:
                self.logger.warning("Weights & Biases not available")
                return
            
            # Initialize wandb run
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name or f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=self.config.tags,
                notes=self.config.notes,
                entity=self.config.wandb_entity,
                group=self.config.wandb_group,
                job_type=self.config.wandb_job_type,
                resume=self.config.resume_run,
                anonymous=self.config.anonymous,
                sync_tensorboard=self.config.sync_tensorboard
            )
            
            self.wandb_run = wandb.run
            self.logger.info("Weights & Biases tracking initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Weights & Biases: {e}")
            self.wandb_run = None
    
    def _start_processing_thread(self):
        """Start background thread for processing metrics."""
        self.processing_thread = threading.Thread(target=self._process_metrics_queue, daemon=True)
        self.processing_thread.start()
    
    def _process_metrics_queue(self):
        """Process metrics from the queue."""
        while not self.stop_processing:
            try:
                # Get metrics from queue with timeout
                metrics = self.metrics_queue.get(timeout=1.0)
                if metrics is None:  # Stop signal
                    break
                
                # Process metrics
                self._log_metrics_to_tracking_systems(metrics)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing metrics: {e}")
    
    def _log_metrics_to_tracking_systems(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard and Weights & Biases."""
        try:
            # Log to TensorBoard
            if self.tensorboard_writer:
                self._log_to_tensorboard(metrics)
            
            # Log to Weights & Biases
            if self.wandb_run:
                self._log_to_wandb(metrics)
                
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def _log_to_tensorboard(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard."""
        try:
            # Basic metrics
            if metrics.loss is not None:
                self.tensorboard_writer.add_scalar('Loss/Train', metrics.loss, self.current_step)
            
            if metrics.accuracy is not None:
                self.tensorboard_writer.add_scalar('Accuracy/Train', metrics.accuracy, self.current_step)
            
            if metrics.learning_rate is not None:
                self.tensorboard_writer.add_scalar('Learning_Rate', metrics.learning_rate, self.current_step)
            
            # Numerical stability metrics
            if metrics.gradient_norm is not None:
                self.tensorboard_writer.add_scalar('Gradients/Norm', metrics.gradient_norm, self.current_step)
            
            if metrics.nan_count > 0:
                self.tensorboard_writer.add_scalar('Numerical_Stability/NaN_Count', metrics.nan_count, self.current_step)
            
            if metrics.inf_count > 0:
                self.tensorboard_writer.add_scalar('Numerical_Stability/Inf_Count', metrics.inf_count, self.current_step)
            
            if metrics.clipping_applied:
                self.tensorboard_writer.add_scalar('Gradient_Clipping/Applied', 1.0, self.current_step)
                if metrics.clipping_threshold:
                    self.tensorboard_writer.add_scalar('Gradient_Clipping/Threshold', metrics.clipping_threshold, self.current_step)
            
            # Performance metrics
            if metrics.training_time > 0:
                self.tensorboard_writer.add_scalar('Performance/Training_Time', metrics.training_time, self.current_step)
            
            if metrics.memory_usage:
                self.tensorboard_writer.add_scalar('Performance/Memory_Usage_MB', metrics.memory_usage, self.current_step)
            
            if metrics.gpu_utilization:
                self.tensorboard_writer.add_scalar('Performance/GPU_Utilization', metrics.gpu_utilization, self.current_step)
            
            # Flush to disk
            self.tensorboard_writer.flush()
            
        except Exception as e:
            self.logger.error(f"Failed to log to TensorBoard: {e}")
    
    def _log_to_wandb(self, metrics: TrainingMetrics):
        """Log metrics to Weights & Biases."""
        try:
            if not self.wandb_run:
                return
            
            # Prepare wandb log data
            log_data = {
                'train/loss': metrics.loss,
                'train/step': self.current_step,
                'train/epoch': self.current_epoch,
            }
            
            # Add optional metrics
            if metrics.accuracy is not None:
                log_data['train/accuracy'] = metrics.accuracy
            
            if metrics.learning_rate is not None:
                log_data['train/learning_rate'] = metrics.learning_rate
            
            if metrics.gradient_norm is not None:
                log_data['gradients/norm'] = metrics.gradient_norm
            
            if metrics.nan_count > 0:
                log_data['numerical_stability/nan_count'] = metrics.nan_count
            
            if metrics.inf_count > 0:
                log_data['numerical_stability/inf_count'] = metrics.inf_count
            
            if metrics.clipping_applied:
                log_data['gradient_clipping/applied'] = 1.0
                if metrics.clipping_threshold:
                    log_data['gradient_clipping/threshold'] = metrics.clipping_threshold
            
            if metrics.training_time > 0:
                log_data['performance/training_time'] = metrics.training_time
            
            if metrics.memory_usage:
                log_data['performance/memory_usage_mb'] = metrics.memory_usage
            
            if metrics.gpu_utilization:
                log_data['performance/gpu_utilization'] = metrics.gpu_utilization
            
            # Log to wandb
            wandb.log(log_data, step=self.current_step)
            
        except Exception as e:
            self.logger.error(f"Failed to log to Weights & Biases: {e}")
    
    def log_training_step(self, 
                         loss: float, 
                         accuracy: Optional[float] = None,
                         learning_rate: Optional[float] = None,
                         gradient_norm: Optional[float] = None,
                         nan_count: int = 0,
                         inf_count: int = 0,
                         clipping_applied: bool = False,
                         clipping_threshold: Optional[float] = None,
                         training_time: float = 0.0,
                         memory_usage: Optional[float] = None,
                         gpu_utilization: Optional[float] = None):
        """Log a training step."""
        try:
            # Create metrics object
            metrics = TrainingMetrics(
                loss=loss,
                accuracy=accuracy,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                nan_count=nan_count,
                inf_count=inf_count,
                clipping_applied=clipping_applied,
                clipping_threshold=clipping_threshold,
                training_time=training_time,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Add to processing queue
            self.metrics_queue.put(metrics)
            
            # Update step counter
            self.current_step += 1
            
            # Log to centralized logging system
            log_training_step(
                step=self.current_step,
                epoch=self.current_epoch,
                loss=loss,
                accuracy=accuracy,
                learning_rate=learning_rate
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log training step: {e}")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch-level metrics."""
        try:
            self.current_epoch = epoch
            
            # Log epoch metrics to TensorBoard
            if self.tensorboard_writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f'Epoch/{key}', value, epoch)
            
            # Log epoch metrics to Weights & Biases
            if self.wandb_run:
                epoch_data = {f'epoch/{key}': value for key, value in metrics.items()}
                epoch_data['epoch'] = epoch
                wandb.log(epoch_data)
            
            # Log to centralized logging system
            log_performance_metrics(epoch=epoch, metrics=metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to log epoch: {e}")
    
    def log_model_architecture(self, model: nn.Module):
        """Log model architecture information."""
        try:
            if not self.config.log_model_architecture:
                return
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                # Add model graph
                dummy_input = torch.randn(1, 3, 224, 224)  # Adjust based on your model
                try:
                    self.tensorboard_writer.add_graph(model, dummy_input)
                except Exception as e:
                    self.logger.warning(f"Could not add model graph to TensorBoard: {e}")
                
                # Add model parameters
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.tensorboard_writer.add_histogram(f'Parameters/{name}', param.data, self.current_step)
            
            # Log to Weights & Biases
            if self.wandb_run:
                # Log model summary
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                wandb.log({
                    'model/total_parameters': total_params,
                    'model/trainable_parameters': trainable_params,
                    'model/step': self.current_step
                })
                
        except Exception as e:
            self.logger.error(f"Failed to log model architecture: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        try:
            if not self.config.log_hyperparameters:
                return
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                # Add hyperparameters as text
                hp_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
                self.tensorboard_writer.add_text('Hyperparameters', hp_text, 0)
            
            # Log to Weights & Biases
            if self.wandb_run:
                wandb.config.update(hyperparams)
                
        except Exception as e:
            self.logger.error(f"Failed to log hyperparameters: {e}")

    def log_problem_definition(self, problem_def: ProblemDefinition):
        """Log comprehensive problem definition to tracking systems."""
        try:
            problem_info = {
                "problem_title": problem_def.problem_title,
                "problem_description": problem_def.problem_description,
                "problem_type": problem_def.problem_type,
                "domain": problem_def.domain,
                "primary_objective": problem_def.primary_objective,
                "success_metrics": problem_def.success_metrics,
                "baseline_performance": problem_def.baseline_performance,
                "target_performance": problem_def.target_performance,
                "computational_constraints": problem_def.computational_constraints,
                "time_constraints": problem_def.time_constraints,
                "accuracy_requirements": problem_def.accuracy_requirements,
                "interpretability_requirements": problem_def.interpretability_requirements,
                "business_value": problem_def.business_value,
                "stakeholders": problem_def.stakeholders,
                "deployment_context": problem_def.deployment_context
            }
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                problem_text = f"""
**Problem Definition**

**Title:** {problem_def.problem_title}
**Description:** {problem_def.problem_description}
**Type:** {problem_def.problem_type}
**Domain:** {problem_def.domain}

**Objectives:**
- Primary: {problem_def.primary_objective}
- Success Metrics: {', '.join(problem_def.success_metrics)}
- Baseline Performance: {problem_def.baseline_performance}
- Target Performance: {problem_def.target_performance}

**Constraints:**
- Computational: {problem_def.computational_constraints}
- Time: {problem_def.time_constraints}
- Accuracy: {problem_def.accuracy_requirements}
- Interpretability: {problem_def.interpretability_requirements}

**Business Context:**
- Value: {problem_def.business_value}
- Stakeholders: {', '.join(problem_def.stakeholders)}
- Deployment: {problem_def.deployment_context}
                """
                self.tensorboard_writer.add_text("Problem_Definition", problem_text, 0)
                
            # Log to Weights & Biases
            if self.wandb_run:
                wandb.config.update({"problem_definition": problem_info})
                
            self.logger.info(f"Problem definition logged: {problem_def.problem_title}")
                
        except Exception as e:
            self.logger.error(f"Failed to log problem definition: {e}")

    def log_dataset_analysis(self, dataset_analysis: DatasetAnalysis):
        """Log comprehensive dataset analysis to tracking systems."""
        try:
            dataset_info = {
                "dataset_name": dataset_analysis.dataset_name,
                "dataset_source": dataset_analysis.dataset_source,
                "dataset_version": dataset_analysis.dataset_version,
                "dataset_size": dataset_analysis.dataset_size,
                "input_shape": dataset_analysis.input_shape,
                "output_shape": dataset_analysis.output_shape,
                "feature_count": dataset_analysis.feature_count,
                "class_count": dataset_analysis.class_count,
                "data_types": dataset_analysis.data_types,
                "missing_values_pct": dataset_analysis.missing_values_pct,
                "duplicate_records_pct": dataset_analysis.duplicate_records_pct,
                "outlier_pct": dataset_analysis.outlier_pct,
                "class_imbalance_ratio": dataset_analysis.class_imbalance_ratio,
                "train_size": dataset_analysis.train_size,
                "val_size": dataset_analysis.val_size,
                "test_size": dataset_analysis.test_size,
                "data_split_strategy": dataset_analysis.data_split_strategy,
                "normalization_needed": dataset_analysis.normalization_needed,
                "encoding_needed": dataset_analysis.encoding_needed,
                "augmentation_strategy": dataset_analysis.augmentation_strategy,
                "preprocessing_steps": dataset_analysis.preprocessing_steps
            }
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                dataset_text = f"""
**Dataset Analysis**

**Metadata:**
- Name: {dataset_analysis.dataset_name}
- Source: {dataset_analysis.dataset_source}
- Version: {dataset_analysis.dataset_version}
- Size: {dataset_analysis.dataset_size:,} samples

**Data Characteristics:**
- Input Shape: {dataset_analysis.input_shape}
- Output Shape: {dataset_analysis.output_shape}
- Features: {dataset_analysis.feature_count}
- Classes: {dataset_analysis.class_count}
- Data Types: {', '.join(dataset_analysis.data_types)}

**Data Quality:**
- Missing Values: {dataset_analysis.missing_values_pct}%
- Duplicates: {dataset_analysis.duplicate_records_pct}%
- Outliers: {dataset_analysis.outlier_pct}%
- Class Imbalance: {dataset_analysis.class_imbalance_ratio}

**Data Distribution:**
- Train: {dataset_analysis.train_size:,} samples
- Validation: {dataset_analysis.val_size:,} samples
- Test: {dataset_analysis.test_size:,} samples
- Split Strategy: {dataset_analysis.data_split_strategy}

**Preprocessing:**
- Normalization: {dataset_analysis.normalization_needed}
- Encoding: {dataset_analysis.encoding_needed}
- Augmentation: {dataset_analysis.augmentation_strategy}
- Steps: {', '.join(dataset_analysis.preprocessing_steps)}
                """
                self.tensorboard_writer.add_text("Dataset_Analysis", dataset_text, 0)
                
                # Log dataset statistics as scalars
                if dataset_analysis.dataset_size:
                    self.tensorboard_writer.add_scalar("Dataset/Total_Size", dataset_analysis.dataset_size, 0)
                if dataset_analysis.feature_count:
                    self.tensorboard_writer.add_scalar("Dataset/Feature_Count", dataset_analysis.feature_count, 0)
                if dataset_analysis.class_count:
                    self.tensorboard_writer.add_scalar("Dataset/Class_Count", dataset_analysis.class_count, 0)
                if dataset_analysis.missing_values_pct:
                    self.tensorboard_writer.add_scalar("Dataset/Missing_Values_Pct", dataset_analysis.missing_values_pct, 0)
                if dataset_analysis.class_imbalance_ratio:
                    self.tensorboard_writer.add_scalar("Dataset/Class_Imbalance_Ratio", dataset_analysis.class_imbalance_ratio, 0)
                
            # Log to Weights & Biases
            if self.wandb_run:
                wandb.config.update({"dataset_analysis": dataset_info})
                
                # Log dataset statistics
                if dataset_analysis.dataset_size:
                    wandb.log({"dataset_size": dataset_analysis.dataset_size}, step=0)
                if dataset_analysis.missing_values_pct:
                    wandb.log({"missing_values_pct": dataset_analysis.missing_values_pct}, step=0)
                    
            self.logger.info(f"Dataset analysis logged: {dataset_analysis.dataset_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to log dataset analysis: {e}")
    
    def log_gradients(self, model: nn.Module):
        """Log gradient information."""
        try:
            if not self.config.log_gradients:
                return
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.tensorboard_writer.add_histogram(f'Gradients/{name}', param.grad, self.current_step)
                        
        except Exception as e:
            self.logger.error(f"Failed to log gradients: {e}")
    
    def log_images(self, images: torch.Tensor, tag: str = "Images"):
        """Log images for visualization."""
        try:
            if not self.config.log_images:
                return
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_images(tag, images, self.current_step)
            
            # Log to Weights & Biases
            if self.wandb_run:
                # Convert to numpy and log
                if images.dim() == 4:  # Batch of images
                    for i, img in enumerate(images):
                        wandb.log({f"{tag}_{i}": wandb.Image(img.cpu().numpy())}, step=self.current_step)
                else:
                    wandb.log({tag: wandb.Image(images.cpu().numpy())}, step=self.current_step)
                    
        except Exception as e:
            self.logger.error(f"Failed to log images: {e}")
    
    def log_text(self, text: str, tag: str = "Text"):
        """Log text data."""
        try:
            if not self.config.log_text:
                return
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_text(tag, text, self.current_step)
            
            # Log to Weights & Biases
            if self.wandb_run:
                wandb.log({tag: wandb.Html(text)}, step=self.current_step)
                
        except Exception as e:
            self.logger.error(f"Failed to log text: {e}")
    
    def log_transformer_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer = None):
        """Log Transformers model architecture and configuration."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available. Skipping model logging.")
            return
        
        try:
            self.transformer_model = model
            self.tokenizer = tokenizer
            
            # Get model configuration
            if hasattr(model, 'config'):
                self.model_config = model.config
                
                # Log model architecture details
                model_info = {
                    'model_type': getattr(model.config, 'model_type', 'unknown'),
                    'architectures': getattr(model.config, 'architectures', []),
                    'num_layers': getattr(model.config, 'num_hidden_layers', 0),
                    'hidden_size': getattr(model.config, 'hidden_size', 0),
                    'num_attention_heads': getattr(model.config, 'num_attention_heads', 0),
                    'intermediate_size': getattr(model.config, 'intermediate_size', 0),
                    'vocab_size': getattr(model.config, 'vocab_size', 0),
                    'max_position_embeddings': getattr(model.config, 'max_position_embeddings', 0),
                    'type_vocab_size': getattr(model.config, 'type_vocab_size', 0),
                    'layer_norm_eps': getattr(model.config, 'layer_norm_eps', 1e-12),
                    'hidden_dropout_prob': getattr(model.config, 'hidden_dropout_prob', 0.1),
                    'attention_probs_dropout_prob': getattr(model.config, 'attention_probs_dropout_prob', 0.1),
                }
                
                # Log to tracking systems
                if self.tensorboard_writer:
                    for key, value in model_info.items():
                        if value is not None:
                            self.tensorboard_writer.add_text(f'Model_Config/{key}', str(value), 0)
                
                if self.wandb_run:
                    wandb.config.update(model_info)
                
                # Log tokenizer information if available
                if tokenizer:
                    tokenizer_info = {
                        'vocab_size': tokenizer.vocab_size,
                        'model_max_length': getattr(tokenizer, 'model_max_length', 'unknown'),
                        'pad_token': tokenizer.pad_token,
                        'unk_token': tokenizer.unk_token,
                        'cls_token': tokenizer.cls_token,
                        'sep_token': tokenizer.sep_token,
                        'mask_token': tokenizer.mask_token,
                    }
                    
                    if self.tensorboard_writer:
                        for key, value in tokenizer_info.items():
                            if value is not None:
                                self.tensorboard_writer.add_text(f'Tokenizer/{key}', str(value), 0)
                    
                    if self.wandb_run:
                        wandb.config.update(tokenizer_info)
                
                self.logger.info(f"Logged Transformers model: {model_info['model_type']}")
                
        except Exception as e:
            self.logger.error(f"Failed to log Transformers model: {e}")

    def log_language_model_metrics(self, 
                                 perplexity: Optional[float] = None,
                                 bleu_score: Optional[float] = None,
                                 rouge_score: Optional[float] = None,
                                 token_accuracy: Optional[float] = None,
                                 sequence_length: Optional[int] = None,
                                 attention_weights_norm: Optional[float] = None,
                                 layer_norm_stats: Optional[Dict[str, float]] = None,
                                 gradient_flow: Optional[Dict[str, float]] = None):
        """Log language model specific metrics."""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            # Create language model metrics
            lm_metrics = LanguageModelMetrics(
                perplexity=perplexity,
                bleu_score=bleu_score,
                rouge_score=rouge_score,
                token_accuracy=token_accuracy,
                sequence_length=sequence_length,
                attention_weights_norm=attention_weights_norm,
                layer_norm_stats=layer_norm_stats,
                gradient_flow=gradient_flow
            )
            
            # Store metrics
            self.lm_metrics_history.append(lm_metrics)
            
            # Log to tracking systems
            if self.tensorboard_writer:
                if perplexity is not None:
                    self.tensorboard_writer.add_scalar('Language_Model/Perplexity', perplexity, self.current_step)
                if bleu_score is not None:
                    self.tensorboard_writer.add_scalar('Language_Model/BLEU_Score', bleu_score, self.current_step)
                if rouge_score is not None:
                    self.tensorboard_writer.add_scalar('Language_Model/ROUGE_Score', rouge_score, self.current_step)
                if token_accuracy is not None:
                    self.tensorboard_writer.add_scalar('Language_Model/Token_Accuracy', token_accuracy, self.current_step)
                if attention_weights_norm is not None:
                    self.tensorboard_writer.add_scalar('Language_Model/Attention_Weights_Norm', attention_weights_norm, self.current_step)
                
                # Log layer norm statistics
                if layer_norm_stats:
                    for layer_name, norm_value in layer_norm_stats.items():
                        self.tensorboard_writer.add_scalar(f'Language_Model/Layer_Norm/{layer_name}', norm_value, self.current_step)
                
                # Log gradient flow
                if gradient_flow:
                    for layer_name, grad_value in gradient_flow.items():
                        self.tensorboard_writer.add_scalar(f'Language_Model/Gradient_Flow/{layer_name}', grad_value, self.current_step)
            
            if self.wandb_run:
                log_data = {}
                if perplexity is not None:
                    log_data['language_model/perplexity'] = perplexity
                if bleu_score is not None:
                    log_data['language_model/bleu_score'] = bleu_score
                if rouge_score is not None:
                    log_data['language_model/rouge_score'] = rouge_score
                if token_accuracy is not None:
                    log_data['language_model/token_accuracy'] = token_accuracy
                if attention_weights_norm is not None:
                    log_data['language_model/attention_weights_norm'] = attention_weights_norm
                
                if log_data:
                    wandb.log(log_data, step=self.current_step)
            
        except Exception as e:
            self.logger.error(f"Failed to log language model metrics: {e}")

    def log_attention_analysis(self, attention_weights: torch.Tensor, layer_idx: int = 0):
        """Log attention weight analysis for Transformers models."""
        if not TRANSFORMERS_AVAILABLE or attention_weights is None:
            return
        
        try:
            # Calculate attention statistics
            attention_norm = torch.norm(attention_weights, dim=-1).mean().item()
            attention_entropy = self._calculate_attention_entropy(attention_weights)
            attention_sparsity = self._calculate_attention_sparsity(attention_weights)
            
            # Log to tracking systems
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f'Attention_Layer_{layer_idx}/Norm', attention_norm, self.current_step)
                self.tensorboard_writer.add_scalar(f'Attention_Layer_{layer_idx}/Entropy', attention_entropy, self.current_step)
                self.tensorboard_writer.add_scalar(f'Attention_Layer_{layer_idx}/Sparsity', attention_sparsity, self.current_step)
                
                # Log attention heatmap periodically
                if self.current_step % 100 == 0:
                    # Sample attention weights for visualization
                    sample_attention = attention_weights[0, 0, :50, :50].detach().cpu().numpy()
                    self.tensorboard_writer.add_image(
                        f'Attention_Layer_{layer_idx}/Heatmap',
                        sample_attention[None, ...],
                        self.current_step,
                        dataformats='CHW'
                    )
            
            if self.wandb_run:
                wandb.log({
                    f'attention_layer_{layer_idx}/norm': attention_norm,
                    f'attention_layer_{layer_idx}/entropy': attention_entropy,
                    f'attention_layer_{layer_idx}/sparsity': attention_sparsity
                }, step=self.current_step)
                
        except Exception as e:
            self.logger.error(f"Failed to log attention analysis: {e}")

    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate entropy of attention weights."""
        try:
            # Normalize attention weights
            probs = torch.softmax(attention_weights, dim=-1)
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
            return entropy
        except Exception:
            return 0.0

    def _calculate_attention_sparsity(self, attention_weights: torch.Tensor) -> float:
        """Calculate sparsity of attention weights."""
        try:
            # Count near-zero attention weights
            near_zero = (attention_weights < 1e-6).float().mean().item()
            return near_zero
        except Exception:
            return 0.0

    def log_gradient_flow_analysis(self, model: PreTrainedModel):
        """Analyze and log gradient flow through Transformers model layers."""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            gradient_flow = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Calculate gradient norm for this parameter
                    grad_norm = param.grad.norm().item()
                    gradient_flow[name] = grad_norm
                    
                    # Log individual parameter gradients
                    if self.tensorboard_writer and self.current_step % 50 == 0:
                        self.tensorboard_writer.add_scalar(f'Gradients/{name}', grad_norm, self.current_step)
            
            # Log gradient flow summary
            if gradient_flow:
                total_grad_norm = sum(gradient_flow.values())
                max_grad_norm = max(gradient_flow.values())
                min_grad_norm = min(gradient_flow.values())
                
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar('Gradient_Flow/Total_Norm', total_grad_norm, self.current_step)
                    self.tensorboard_writer.add_scalar('Gradient_Flow/Max_Norm', max_grad_norm, self.current_step)
                    self.tensorboard_writer.add_scalar('Gradient_Flow/Min_Norm', min_grad_norm, self.current_step)
                
                if self.wandb_run:
                    wandb.log({
                        'gradient_flow/total_norm': total_grad_norm,
                        'gradient_flow/max_norm': max_grad_norm,
                        'gradient_flow/min_norm': min_grad_norm
                    }, step=self.current_step)
                
                # Store for later analysis
                self.log_language_model_metrics(gradient_flow=gradient_flow)
                
        except Exception as e:
            self.logger.error(f"Failed to analyze gradient flow: {e}")

    def create_language_model_visualization(self) -> Optional[plt.Figure]:
        """Create specialized visualizations for language models."""
        if not self.lm_metrics_history:
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Language Model Training Progress - {self.config.experiment_name}', fontsize=16)
            
            # Extract metrics
            steps = list(range(len(self.lm_metrics_history)))
            perplexities = [m.perplexity for m in self.lm_metrics_history if m.perplexity is not None]
            bleu_scores = [m.bleu_score for m in self.lm_metrics_history if m.bleu_score is not None]
            token_accuracies = [m.token_accuracy for m in self.lm_metrics_history if m.token_accuracy is not None]
            attention_norms = [m.attention_weights_norm for m in self.lm_metrics_history if m.attention_weights_norm is not None]
            
            # Plot 1: Perplexity
            if perplexities:
                axes[0, 0].plot(steps[:len(perplexities)], perplexities, 'b-', label='Perplexity')
                axes[0, 0].set_title('Language Model Perplexity')
                axes[0, 0].set_ylabel('Perplexity')
                axes[0, 0].legend()
            
            # Plot 2: BLEU Score
            if bleu_scores:
                axes[0, 1].plot(steps[:len(bleu_scores)], bleu_scores, 'g-', label='BLEU Score')
                axes[0, 1].set_title('BLEU Score')
                axes[0, 1].set_ylabel('BLEU Score')
                axes[0, 1].legend()
            
            # Plot 3: Token Accuracy
            if token_accuracies:
                axes[1, 0].plot(steps[:len(token_accuracies)], token_accuracies, 'r-', label='Token Accuracy')
                axes[1, 0].set_title('Token Accuracy')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].legend()
            
            # Plot 4: Attention Weights Norm
            if attention_norms:
                axes[1, 1].plot(steps[:len(attention_norms)], attention_norms, 'orange', label='Attention Norm')
                axes[1, 1].set_title('Attention Weights Norm')
                axes[1, 1].set_ylabel('Norm')
                axes[1, 1].legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Log to tracking systems
            if self.tensorboard_writer:
                self.tensorboard_writer.add_figure('Language_Model_Progress', fig, self.current_step)
            
            if self.wandb_run:
                wandb.log({'language_model_progress': wandb.Image(fig)}, step=self.current_step)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create language model visualization: {e}")
            return None

    def get_language_model_summary(self) -> Dict[str, Any]:
        """Get summary of language model training metrics."""
        if not self.lm_metrics_history:
            return {"error": "No language model metrics available"}
        
        try:
            # Calculate statistics
            perplexities = [m.perplexity for m in self.lm_metrics_history if m.perplexity is not None]
            bleu_scores = [m.bleu_score for m in self.lm_metrics_history if m.bleu_score is not None]
            token_accuracies = [m.token_accuracy for m in self.lm_metrics_history if m.token_accuracy is not None]
            
            summary = {
                "total_lm_metrics": len(self.lm_metrics_history),
                "perplexity_stats": {
                    "count": len(perplexities),
                    "min": min(perplexities) if perplexities else None,
                    "max": max(perplexities) if perplexities else None,
                    "final": perplexities[-1] if perplexities else None,
                    "mean": sum(perplexities) / len(perplexities) if perplexities else None
                },
                "bleu_score_stats": {
                    "count": len(bleu_scores),
                    "min": min(bleu_scores) if bleu_scores else None,
                    "max": max(bleu_scores) if bleu_scores else None,
                    "final": bleu_scores[-1] if bleu_scores else None,
                    "mean": sum(bleu_scores) / len(bleu_scores) if bleu_scores else None
                },
                "token_accuracy_stats": {
                    "count": len(token_accuracies),
                    "min": min(token_accuracies) if token_accuracies else None,
                    "max": max(token_accuracies) if token_accuracies else None,
                    "final": token_accuracies[-1] if token_accuracies else None,
                    "mean": sum(token_accuracies) / len(token_accuracies) if token_accuracies else None
                }
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to get language model summary: {e}"}

    # =============================================================================
    # DIFFUSION MODEL TRACKING METHODS
    # =============================================================================
    
    def log_diffusion_pipeline(self, pipeline: DiffusionPipeline, 
                              unet_model: Optional[UNet2DConditionModel] = None,
                              vae_model: Optional[AutoencoderKL] = None,
                              scheduler: Optional[SchedulerMixin] = None):
        """Log diffusion pipeline configuration and architecture."""
        if not DIFFUSERS_AVAILABLE:
            self.logger.warning("Diffusers not available. Skipping diffusion pipeline logging.")
            return
        
        try:
            self.diffusion_pipeline = pipeline
            self.unet_model = unet_model
            self.vae_model = vae_model
            self.scheduler = scheduler
            
            # Log pipeline configuration
            pipeline_config = {
                "pipeline_type": type(pipeline).__name__,
                "device": str(pipeline.device),
                "dtype": str(pipeline.dtype),
                "requires_safety_checker": hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None,
                "requires_watermarker": hasattr(pipeline, 'watermarker') and pipeline.watermarker is not None
            }
            
            # Log UNet configuration if available
            if unet_model:
                unet_config = {
                    "model_type": "UNet2DConditionModel",
                    "in_channels": unet_model.in_channels,
                    "out_channels": unet_model.out_channels,
                    "block_out_channels": unet_model.block_out_channels,
                    "down_block_types": unet_model.down_block_types,
                    "up_block_types": unet_model.up_block_types,
                    "cross_attention_dim": unet_model.cross_attention_dim,
                    "attention_head_dim": unet_model.attention_head_dim,
                    "num_attention_heads": unet_model.num_attention_heads,
                    "num_parameters": sum(p.numel() for p in unet_model.parameters())
                }
                pipeline_config["unet"] = unet_config
            
            # Log VAE configuration if available
            if vae_model:
                vae_config = {
                    "model_type": "AutoencoderKL",
                    "in_channels": vae_model.in_channels,
                    "out_channels": vae_model.out_channels,
                    "latent_channels": vae_model.latent_channels,
                    "sample_size": vae_model.sample_size,
                    "num_parameters": sum(p.numel() for p in vae_model.parameters())
                }
                pipeline_config["vae"] = vae_config
            
            # Log scheduler configuration if available
            if scheduler:
                scheduler_config = {
                    "scheduler_type": type(scheduler).__name__,
                    "num_train_timesteps": getattr(scheduler, 'num_train_timesteps', None),
                    "beta_start": getattr(scheduler, 'beta_start', None),
                    "beta_end": getattr(scheduler, 'beta_end', None),
                    "beta_schedule": getattr(scheduler, 'beta_schedule', None)
                }
                pipeline_config["scheduler"] = scheduler_config
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_text('Diffusion/Pipeline_Config', 
                                               json.dumps(pipeline_config, indent=2), 0)
                self.tensorboard_writer.add_scalar('Diffusion/Total_Parameters', 
                                                 sum(p.numel() for p in pipeline.parameters()), 0)
            
            # Log to Weights & Biases
            if self.wandb_run:
                wandb.config.update(pipeline_config)
                wandb.log({'diffusion/total_parameters': sum(p.numel() for p in pipeline.parameters())}, step=0)
            
            self.logger.info(f"Diffusion pipeline logged: {pipeline_config['pipeline_type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to log diffusion pipeline: {e}")
    
    def log_diffusion_metrics(self, 
                             noise_level: Optional[float] = None,
                             denoising_steps: Optional[int] = None,
                             guidance_scale: Optional[float] = None,
                             image_quality_score: Optional[float] = None,
                             generation_time: Optional[float] = None,
                             memory_usage: Optional[float] = None,
                             scheduler_step: Optional[int] = None,
                             noise_prediction_loss: Optional[float] = None,
                             classifier_free_guidance: Optional[bool] = None,
                             prompt_embedding_norm: Optional[float] = None,
                             cross_attention_weights: Optional[torch.Tensor] = None,
                             latent_space_stats: Optional[Dict[str, float]] = None):
        """Log diffusion model specific metrics."""
        if not DIFFUSERS_AVAILABLE:
            self.logger.warning("Diffusers not available. Skipping diffusion metrics logging.")
            return
        
        try:
            # Create diffusion metrics object
            diffusion_metrics = DiffusionModelMetrics(
                noise_level=noise_level,
                denoising_steps=denoising_steps,
                guidance_scale=guidance_scale,
                image_quality_score=image_quality_score,
                generation_time=generation_time,
                memory_usage=memory_usage,
                scheduler_step=scheduler_step,
                noise_prediction_loss=noise_prediction_loss,
                classifier_free_guidance=classifier_free_guidance,
                prompt_embedding_norm=prompt_embedding_norm,
                cross_attention_weights=cross_attention_weights,
                latent_space_stats=latent_space_stats
            )
            
            # Store metrics
            self.diffusion_metrics_history.append(diffusion_metrics)
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                if noise_level is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Noise_Level', noise_level, self.current_step)
                if denoising_steps is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Denoising_Steps', denoising_steps, self.current_step)
                if guidance_scale is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Guidance_Scale', guidance_scale, self.current_step)
                if image_quality_score is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Image_Quality_Score', image_quality_score, self.current_step)
                if generation_time is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Generation_Time', generation_time, self.current_step)
                if memory_usage is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Memory_Usage', memory_usage, self.current_step)
                if scheduler_step is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Scheduler_Step', scheduler_step, self.current_step)
                if noise_prediction_loss is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Noise_Prediction_Loss', noise_prediction_loss, self.current_step)
                if prompt_embedding_norm is not None:
                    self.tensorboard_writer.add_scalar('Diffusion/Prompt_Embedding_Norm', prompt_embedding_norm, self.current_step)
                
                # Log cross-attention weights as heatmap if available
                if cross_attention_weights is not None:
                    self._log_cross_attention_heatmap(cross_attention_weights, self.current_step)
                
                # Log latent space statistics if available
                if latent_space_stats:
                    for key, value in latent_space_stats.items():
                        self.tensorboard_writer.add_scalar(f'Diffusion/Latent_Space/{key}', value, self.current_step)
            
            # Log to Weights & Biases
            if self.wandb_run:
                wandb_metrics = {}
                if noise_level is not None:
                    wandb_metrics['diffusion/noise_level'] = noise_level
                if denoising_steps is not None:
                    wandb_metrics['diffusion/denoising_steps'] = denoising_steps
                if guidance_scale is not None:
                    wandb_metrics['diffusion/guidance_scale'] = guidance_scale
                if image_quality_score is not None:
                    wandb_metrics['diffusion/image_quality_score'] = image_quality_score
                if generation_time is not None:
                    wandb_metrics['diffusion/generation_time'] = generation_time
                if memory_usage is not None:
                    wandb_metrics['diffusion/memory_usage'] = memory_usage
                if scheduler_step is not None:
                    wandb_metrics['diffusion/scheduler_step'] = scheduler_step
                if noise_prediction_loss is not None:
                    wandb_metrics['diffusion/noise_prediction_loss'] = noise_prediction_loss
                if prompt_embedding_norm is not None:
                    wandb_metrics['diffusion/prompt_embedding_norm'] = prompt_embedding_norm
                if latent_space_stats:
                    for key, value in latent_space_stats.items():
                        wandb_metrics[f'diffusion/latent_space/{key}'] = value
                
                wandb.log(wandb_metrics, step=self.current_step)
            
            self.logger.debug(f"Diffusion metrics logged at step {self.current_step}")
            
        except Exception as e:
            self.logger.error(f"Failed to log diffusion metrics: {e}")
    
    def log_diffusion_generation_step(self, step: int, noise_prediction: torch.Tensor, 
                                    latent: torch.Tensor, guidance_scale: float = 1.0):
        """Log individual diffusion generation step details."""
        if not DIFFUSERS_AVAILABLE:
            return
        
        try:
            # Calculate noise prediction statistics
            noise_stats = {
                "mean": float(noise_prediction.mean()),
                "std": float(noise_prediction.std()),
                "min": float(noise_prediction.min()),
                "max": float(noise_prediction.max()),
                "norm": float(torch.norm(noise_prediction))
            }
            
            # Calculate latent space statistics
            latent_stats = {
                "mean": float(latent.mean()),
                "std": float(latent.std()),
                "min": float(latent.min()),
                "max": float(latent.max()),
                "norm": float(torch.norm(latent))
            }
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                for key, value in noise_stats.items():
                    self.tensorboard_writer.add_scalar(f'Diffusion/Step_{step}/Noise_{key}', value, self.current_step)
                for key, value in latent_stats.items():
                    self.tensorboard_writer.add_scalar(f'Diffusion/Step_{step}/Latent_{key}', value, self.current_step)
                self.tensorboard_writer.add_scalar(f'Diffusion/Step_{step}/Guidance_Scale', guidance_scale, self.current_step)
            
            # Log to Weights & Biases
            if self.wandb_run:
                wandb_metrics = {}
                for key, value in noise_stats.items():
                    wandb_metrics[f'diffusion/step_{step}/noise_{key}'] = value
                for key, value in latent_stats.items():
                    wandb_metrics[f'diffusion/step_{step}/latent_{key}'] = value
                wandb_metrics[f'diffusion/step_{step}/guidance_scale'] = guidance_scale
                
                wandb.log(wandb_metrics, step=self.current_step)
            
        except Exception as e:
            self.logger.error(f"Failed to log diffusion generation step: {e}")
    
    def _log_cross_attention_heatmap(self, attention_weights: torch.Tensor, step: int):
        """Log cross-attention weights as heatmap visualization."""
        try:
            # Convert to numpy and create heatmap
            attention_np = attention_weights.detach().cpu().numpy()
            
            # Create figure with subplots for different attention heads
            num_heads = min(attention_np.shape[0], 4)  # Show max 4 heads
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i in range(num_heads):
                if i < len(axes):
                    sns.heatmap(attention_np[i], ax=axes[i], cmap='viridis', 
                               cbar=True, square=True)
                    axes[i].set_title(f'Attention Head {i+1}')
                    axes[i].set_xlabel('Key Position')
                    axes[i].set_ylabel('Query Position')
            
            # Hide unused subplots
            for i in range(num_heads, len(axes)):
                axes[i].set_visible(False)
            
            # Log to TensorBoard
            if self.tensorboard_writer:
                self.tensorboard_writer.add_figure('Diffusion/Cross_Attention_Heatmap', fig, step)
            
            # Log to Weights & Biases
            if self.wandb_run:
                wandb.log({"diffusion/cross_attention_heatmap": wandb.Image(fig)}, step=step)
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Failed to log cross-attention heatmap: {e}")
    
    def create_diffusion_visualization(self) -> Optional[plt.Figure]:
        """Create specialized visualizations for diffusion models."""
        if not self.diffusion_metrics_history:
            return None
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # Plot 1: Noise level over time
            if any(m.noise_level is not None for m in self.diffusion_metrics_history):
                noise_levels = [m.noise_level for m in self.diffusion_metrics_history if m.noise_level is not None]
                steps = list(range(len(noise_levels)))
                axes[0].plot(steps, noise_levels, 'b-', linewidth=2)
                axes[0].set_title('Noise Level Over Time')
                axes[0].set_xlabel('Step')
                axes[0].set_ylabel('Noise Level')
                axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Generation time over time
            if any(m.generation_time is not None for m in self.diffusion_metrics_history):
                generation_times = [m.generation_time for m in self.diffusion_metrics_history if m.generation_time is not None]
                steps = list(range(len(generation_times)))
                axes[1].plot(steps, generation_times, 'g-', linewidth=2)
                axes[1].set_title('Generation Time Over Time')
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('Generation Time (s)')
                axes[1].set_yscale('log')
                axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Image quality score over time
            if any(m.image_quality_score is not None for m in self.diffusion_metrics_history):
                quality_scores = [m.image_quality_score for m in self.diffusion_metrics_history if m.image_quality_score is not None]
                steps = list(range(len(quality_scores)))
                axes[2].plot(steps, quality_scores, 'r-', linewidth=2)
                axes[2].set_title('Image Quality Score Over Time')
                axes[2].set_xlabel('Step')
                axes[2].set_ylabel('Quality Score')
                axes[2].grid(True, alpha=0.3)
            
            # Plot 4: Memory usage over time
            if any(m.memory_usage is not None for m in self.diffusion_metrics_history):
                memory_usage = [m.memory_usage for m in self.diffusion_metrics_history if m.memory_usage is not None]
                steps = list(range(len(memory_usage)))
                axes[3].plot(steps, memory_usage, 'm-', linewidth=2)
                axes[3].set_title('Memory Usage Over Time')
                axes[3].set_xlabel('Step')
                axes[3].set_ylabel('Memory Usage (MB)')
                axes[3].grid(True, alpha=0.3)
            
            # Plot 5: Guidance scale distribution
            if any(m.guidance_scale is not None for m in self.diffusion_metrics_history):
                guidance_scales = [m.guidance_scale for m in self.diffusion_metrics_history if m.guidance_scale is not None]
                axes[4].hist(guidance_scales, bins=20, alpha=0.7, color='orange', edgecolor='black')
                axes[4].set_title('Guidance Scale Distribution')
                axes[4].set_xlabel('Guidance Scale')
                axes[4].set_ylabel('Frequency')
                axes[4].grid(True, alpha=0.3)
            
            # Plot 6: Denoising steps distribution
            if any(m.denoising_steps is not None for m in self.diffusion_metrics_history):
                denoising_steps = [m.denoising_steps for m in self.diffusion_metrics_history if m.denoising_steps is not None]
                axes[5].hist(denoising_steps, bins=20, alpha=0.7, color='cyan', edgecolor='black')
                axes[5].set_title('Denoising Steps Distribution')
                axes[5].set_xlabel('Denoising Steps')
                axes[5].set_ylabel('Frequency')
                axes[5].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(6, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create diffusion visualization: {e}")
            return None
    
    def get_diffusion_summary(self) -> Dict[str, Any]:
        """Get summary of diffusion model metrics."""
        if not self.diffusion_metrics_history:
            return {"error": "No diffusion model metrics available"}
        
        try:
            # Calculate statistics for each metric type
            noise_levels = [m.noise_level for m in self.diffusion_metrics_history if m.noise_level is not None]
            generation_times = [m.generation_time for m in self.diffusion_metrics_history if m.generation_time is not None]
            quality_scores = [m.image_quality_score for m in self.diffusion_metrics_history if m.image_quality_score is not None]
            guidance_scales = [m.guidance_scale for m in self.diffusion_metrics_history if m.guidance_scale is not None]
            denoising_steps = [m.denoising_steps for m in self.diffusion_metrics_history if m.denoising_steps is not None]
            
            summary = {
                "total_diffusion_metrics": len(self.diffusion_metrics_history),
                "noise_level_stats": {
                    "count": len(noise_levels),
                    "min": min(noise_levels) if noise_levels else None,
                    "max": max(noise_levels) if noise_levels else None,
                    "final": noise_levels[-1] if noise_levels else None,
                    "mean": sum(noise_levels) / len(noise_levels) if noise_levels else None
                },
                "generation_time_stats": {
                    "count": len(generation_times),
                    "min": min(generation_times) if generation_times else None,
                    "max": max(generation_times) if generation_times else None,
                    "final": generation_times[-1] if generation_times else None,
                    "mean": sum(generation_times) / len(generation_times) if generation_times else None
                },
                "quality_score_stats": {
                    "count": len(quality_scores),
                    "min": min(quality_scores) if quality_scores else None,
                    "max": max(quality_scores) if quality_scores else None,
                    "final": quality_scores[-1] if quality_scores else None,
                    "mean": sum(quality_scores) / len(quality_scores) if quality_scores else None
                },
                "guidance_scale_stats": {
                    "count": len(guidance_scales),
                    "min": min(guidance_scales) if guidance_scales else None,
                    "max": max(guidance_scales) if guidance_scales else None,
                    "final": guidance_scales[-1] if guidance_scales else None,
                    "mean": sum(guidance_scales) / len(guidance_scales) if guidance_scales else None
                },
                "denoising_steps_stats": {
                    "count": len(denoising_steps),
                    "min": min(denoising_steps) if denoising_steps else None,
                    "max": max(denoising_steps) if denoising_steps else None,
                    "final": denoising_steps[-1] if denoising_steps else None,
                    "mean": sum(denoising_steps) / len(denoising_steps) if denoising_steps else None
                }
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to get diffusion summary: {e}"}

    # =============================================================================
    # ADVANCED CHECKPOINTING METHODS
    # =============================================================================

    def setup_advanced_checkpointing(self, checkpoint_config: Optional[Dict[str, Any]] = None):
        """
        Setup advanced checkpointing with CheckpointManager.
        
        Args:
            checkpoint_config: Configuration for checkpoint management
        """
        try:
            # Import CheckpointManager from modular structure
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent / "modular_structure"))
            
            from utils.checkpoint_manager import CheckpointManager, CheckpointConfig
            
            # Create checkpoint configuration
            if checkpoint_config is None:
                checkpoint_config = CheckpointConfig(
                    checkpoint_dir=self.config.model_save_dir,
                    save_interval=self.config.save_interval,
                    max_checkpoints=10,
                    save_best_only=False,
                    monitor_metric="val_loss",
                    monitor_mode="min",
                    save_optimizer=True,
                    save_scheduler=True,
                    save_metadata=True,
                    backup_checkpoints=True,
                    validate_checkpoints=True
                )
            else:
                checkpoint_config = CheckpointConfig(**checkpoint_config)
            
            # Initialize checkpoint manager
            self.checkpoint_manager = CheckpointManager(
                config=checkpoint_config,
                experiment_name=self.config.experiment_name
            )
            
            self.logger.info("Advanced checkpointing system initialized")
            
        except ImportError as e:
            self.logger.warning(f"Advanced checkpointing not available: {e}")
            self.logger.info("Falling back to basic checkpointing")
        except Exception as e:
            self.logger.error(f"Failed to setup advanced checkpointing: {e}")

    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints."""
        try:
            if hasattr(self, 'checkpoint_manager'):
                return self.checkpoint_manager.get_checkpoint_summary()
            else:
                return {
                    "message": "Advanced checkpointing not available",
                    "basic_checkpoints": len(self.checkpoints),
                    "checkpoints": [
                        {
                            "epoch": cp.epoch,
                            "step": cp.step,
                            "loss": cp.loss,
                            "metrics": cp.metrics
                        } for cp in self.checkpoints
                    ]
                }
        except Exception as e:
            return {"error": f"Failed to get checkpoint summary: {e}"}

    def list_checkpoints(self, sort_by: str = "timestamp", reverse: bool = True) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        try:
            if hasattr(self, 'checkpoint_manager'):
                checkpoints = self.checkpoint_manager.list_checkpoints(sort_by, reverse)
                return [asdict(cp) for cp in checkpoints]
            else:
                # Sort basic checkpoints
                sorted_checkpoints = sorted(
                    self.checkpoints,
                    key=lambda x: (x.epoch, x.step),
                    reverse=reverse
                )
                return [
                    {
                        "epoch": cp.epoch,
                        "step": cp.step,
                        "loss": cp.loss,
                        "metrics": cp.metrics,
                        "timestamp": "N/A"
                    } for cp in sorted_checkpoints
                ]
        except Exception as e:
            return [{"error": f"Failed to list checkpoints: {e}"}]

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the best checkpoint information."""
        try:
            if hasattr(self, 'checkpoint_manager'):
                best_cp = self.checkpoint_manager.get_best_checkpoint()
                return asdict(best_cp) if best_cp else None
            else:
                # Find best checkpoint based on loss
                if not self.checkpoints:
                    return None
                
                best_cp = min(self.checkpoints, key=lambda x: x.loss)
                return {
                    "epoch": best_cp.epoch,
                    "step": best_cp.step,
                    "loss": best_cp.loss,
                    "metrics": best_cp.metrics
                }
        except Exception as e:
            self.logger.error(f"Failed to get best checkpoint: {e}")
            return None

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint information."""
        try:
            if hasattr(self, 'checkpoint_manager'):
                latest_cp = self.checkpoint_manager.get_latest_checkpoint()
                return asdict(latest_cp) if latest_cp else None
            else:
                # Find latest checkpoint based on epoch and step
                if not self.checkpoints:
                    return None
                
                latest_cp = max(self.checkpoints, key=lambda x: (x.epoch, x.step))
                return {
                    "epoch": latest_cp.epoch,
                    "step": latest_cp.step,
                    "loss": latest_cp.loss,
                    "metrics": latest_cp.metrics
                }
        except Exception as e:
            self.logger.error(f"Failed to get latest checkpoint: {e}")
            return None

    def compare_checkpoints(self, checkpoint_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple checkpoints."""
        try:
            if hasattr(self, 'checkpoint_manager'):
                return self.checkpoint_manager.compare_checkpoints(checkpoint_ids)
            else:
                return {"error": "Advanced checkpoint comparison not available with basic checkpointing"}
        except Exception as e:
            return {"error": f"Failed to compare checkpoints: {e}"}

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        try:
            if hasattr(self, 'checkpoint_manager'):
                return self.checkpoint_manager.delete_checkpoint(checkpoint_id)
            else:
                return {"error": "Checkpoint deletion not available with basic checkpointing"}
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint: {e}")
            return False

    def export_checkpoint(self, checkpoint_id: str, export_path: str) -> bool:
        """Export a checkpoint to a different location."""
        try:
            if hasattr(self, 'checkpoint_manager'):
                return self.checkpoint_manager.export_checkpoint(checkpoint_id, export_path)
            else:
                return {"error": "Checkpoint export not available with basic checkpointing"}
        except Exception as e:
            self.logger.error(f"Failed to export checkpoint: {e}")
            return False

    def validate_checkpoint(self, checkpoint_id: str) -> bool:
        """Validate a checkpoint's integrity."""
        try:
            if hasattr(self, 'checkpoint_manager'):
                metadata = self.checkpoint_manager.get_checkpoint_info(checkpoint_id)
                if not metadata:
                    return False
                
                checkpoint_path = self.checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.pt"
                return self.checkpoint_manager._validate_checkpoint(checkpoint_path, metadata.checksum)
            else:
                return {"error": "Checkpoint validation not available with basic checkpointing"}
        except Exception as e:
            self.logger.error(f"Failed to validate checkpoint: {e}")
            return False

    def save_checkpoint(self, 
                       model: nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       step: int,
                       loss: float,
                       metrics: Dict[str, Any],
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       tags: Optional[List[str]] = None,
                       description: str = "",
                       force_save: bool = False) -> Optional[str]:
        """
        Save a model checkpoint with advanced checkpointing capabilities.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current step
            loss: Current loss
            metrics: Current metrics
            scheduler: Scheduler state (optional)
            tags: Tags for the checkpoint
            description: Description of the checkpoint
            force_save: Force save even if conditions not met
        
        Returns:
            Checkpoint ID if saved, None otherwise
        """
        try:
            # Use advanced checkpoint manager if available
            if hasattr(self, 'checkpoint_manager'):
                checkpoint_id = self.checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=step,
                    metrics=metrics,
                    config=self.config.__dict__,
                    tags=tags,
                    description=description,
                    force_save=force_save
                )
                
                if checkpoint_id:
                    # Log checkpoint to tracking systems
                    if self.tensorboard_writer:
                        self.tensorboard_writer.add_scalar('Checkpoints/Saved', 1.0, step)
                        self.tensorboard_writer.add_scalar('Checkpoints/Total', len(self.checkpoint_manager.checkpoints), step)
                    
                    if self.wandb_run:
                        wandb.log({
                            'checkpoints/saved': 1.0,
                            'checkpoints/total': len(self.checkpoint_manager.checkpoints)
                        }, step=step)
                    
                    self.logger.info(f"Advanced checkpoint saved: {checkpoint_id}")
                    return checkpoint_id
                
                return None
            
            # Fallback to basic checkpointing
            # Create checkpoint directory
            checkpoint_dir = Path(self.config.model_save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'metrics': metrics,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__
            }
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            
            # Create checkpoint object
            checkpoint = ModelCheckpoint(
                epoch=epoch,
                step=step,
                loss=loss,
                metrics=metrics,
                model_state=checkpoint_data['model_state_dict'],
                optimizer_state=checkpoint_data['optimizer_state_dict'],
                scheduler_state=checkpoint_data['scheduler_state_dict']
            )
            
            self.checkpoints.append(checkpoint)
            
            # Log checkpoint to tracking systems
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Checkpoints/Saved', 1.0, step)
            
            if self.wandb_run:
                wandb.log({'checkpoints/saved': 1.0}, step=step)
            
            self.logger.info(f"Basic checkpoint saved: {checkpoint_path}")
            return f"basic_checkpoint_epoch_{epoch}_step_{step}"
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[str] = None,
                       checkpoint_id: Optional[str] = None,
                       model: Optional[nn.Module] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Optional[Dict[str, Any]]:
        """
        Load a model checkpoint with advanced capabilities.
        
        Args:
            checkpoint_path: Direct path to checkpoint file
            checkpoint_id: ID of checkpoint to load (from advanced checkpoint manager)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
        
        Returns:
            Checkpoint data dictionary
        """
        try:
            # Use advanced checkpoint manager if available
            if hasattr(self, 'checkpoint_manager') and checkpoint_id:
                checkpoint_data = self.checkpoint_manager.load_checkpoint(
                    checkpoint_id=checkpoint_id,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                
                if checkpoint_data:
                    # Update current state
                    self.current_epoch = checkpoint_data.get('epoch', 0)
                    self.current_step = checkpoint_data.get('step', 0)
                    
                    self.logger.info(f"Advanced checkpoint loaded: {checkpoint_id}")
                    return checkpoint_data
                
                return None
            
            # Fallback to basic checkpoint loading
            if not checkpoint_path:
                self.logger.error("No checkpoint path or ID provided")
                return None
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                self.logger.error(f"Checkpoint not found: {checkpoint_path}")
                return None
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Load into model if provided
            if model and 'model_state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model_state_dict'])
                self.logger.info("Model state loaded from checkpoint")
            
            # Load into optimizer if provided
            if optimizer and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                self.logger.info("Optimizer state loaded from checkpoint")
            
            # Load into scheduler if provided
            if scheduler and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                self.logger.info("Scheduler state loaded from checkpoint")
            
            # Update current state
            self.current_epoch = checkpoint_data.get('epoch', 0)
            self.current_step = checkpoint_data.get('step', 0)
            
            self.logger.info(f"Basic checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def create_visualization(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Create training visualization plots."""
        try:
            if not self.metrics_history:
                self.logger.warning("No metrics to visualize")
                return {}
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - {self.config.experiment_name}', fontsize=16)
            
            # Extract data
            steps = list(range(len(self.metrics_history)))
            losses = [m.loss for m in self.metrics_history]
            accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
            gradient_norms = [m.gradient_norm for m in self.metrics_history if m.gradient_norm is not None]
            nan_counts = [m.nan_count for m in self.metrics_history]
            inf_counts = [m.inf_count for m in self.metrics_history]
            
            # Plot 1: Loss
            axes[0, 0].plot(steps, losses, 'b-', label='Training Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot 2: Accuracy (if available)
            if accuracies:
                axes[0, 1].plot(steps[:len(accuracies)], accuracies, 'g-', label='Accuracy')
                axes[0, 1].set_title('Training Accuracy')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Plot 3: Gradient Norms
            if gradient_norms:
                axes[1, 0].plot(steps[:len(gradient_norms)], gradient_norms, 'r-', label='Gradient Norm')
                axes[1, 0].set_title('Gradient Norms')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Norm')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Plot 4: Numerical Stability
            axes[1, 1].plot(steps, nan_counts, 'orange', label='NaN Count', alpha=0.7)
            axes[1, 1].plot(steps, inf_counts, 'red', label='Inf Count', alpha=0.7)
            axes[1, 1].set_title('Numerical Stability Issues')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Visualization saved to: {save_path}")
            
            # Log to tracking systems
            if self.tensorboard_writer:
                self.tensorboard_writer.add_figure('Training_Progress', fig, self.current_step)
            
            if self.wandb_run:
                wandb.log({'training_progress': wandb.Image(fig)}, step=self.current_step)
            
            # Return figure data
            visualization_data = {
                'figure': fig,
                'metrics_summary': {
                    'total_steps': len(self.metrics_history),
                    'final_loss': losses[-1] if losses else None,
                    'final_accuracy': accuracies[-1] if accuracies else None,
                    'total_nan_count': sum(nan_counts),
                    'total_inf_count': sum(inf_counts)
                }
            }
            
            return visualization_data
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
            return {}
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment."""
        try:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            # Calculate summary statistics
            losses = [m.loss for m in self.metrics_history]
            accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
            gradient_norms = [m.gradient_norm for m in self.metrics_history if m.gradient_norm is not None]
            nan_counts = [m.nan_count for m in self.metrics_history]
            inf_counts = [m.inf_count for m in self.metrics_history]
            
            summary = {
                'experiment_name': self.config.experiment_name,
                'project_name': self.config.project_name,
                'total_steps': len(self.metrics_history),
                'current_epoch': self.current_epoch,
                'current_step': self.current_step,
                'loss_stats': {
                    'min': min(losses) if losses else None,
                    'max': max(losses) if losses else None,
                    'mean': np.mean(losses) if losses else None,
                    'final': losses[-1] if losses else None
                },
                'accuracy_stats': {
                    'min': min(accuracies) if accuracies else None,
                    'max': max(accuracies) if accuracies else None,
                    'mean': np.mean(accuracies) if accuracies else None,
                    'final': accuracies[-1] if accuracies else None
                },
                'gradient_stats': {
                    'min': min(gradient_norms) if gradient_norms else None,
                    'max': max(gradient_norms) if gradient_norms else None,
                    'mean': np.mean(gradient_norms) if gradient_norms else None,
                    'final': gradient_norms[-1] if gradient_norms else None
                },
                'numerical_stability': {
                    'total_nan_count': sum(nan_counts),
                    'total_inf_count': sum(inf_counts),
                    'steps_with_nan': sum(1 for c in nan_counts if c > 0),
                    'steps_with_inf': sum(1 for c in inf_counts if c > 0)
                },
                'checkpoints': len(self.checkpoints),
                'tracking_systems': {
                    'tensorboard': self.tensorboard_writer is not None,
                    'wandb': self.wandb_run is not None
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment summary: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close the experiment tracker and cleanup resources."""
        try:
            # Stop processing thread
            self.stop_processing = True
            if self.processing_thread:
                self.metrics_queue.put(None)  # Send stop signal
                self.processing_thread.join(timeout=5.0)
            
            # Close TensorBoard writer
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            # Finish wandb run
            if self.wandb_run:
                wandb.finish()
            
            self.logger.info("Experiment tracker closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing experiment tracker: {e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_dataset_automatically(dataset, labels=None, dataset_name="unknown") -> DatasetAnalysis:
    """Automatically analyze a dataset and generate comprehensive analysis."""
    try:
        import pandas as pd
        
        analysis = DatasetAnalysis()
        analysis.dataset_name = dataset_name
        
        # Convert to pandas DataFrame if needed
        if isinstance(dataset, torch.utils.data.Dataset):
            # For PyTorch datasets, sample some data for analysis
            sample_size = min(1000, len(dataset))
            sample_data = []
            sample_labels = []
            
            for i in range(sample_size):
                data, label = dataset[i]
                if isinstance(data, torch.Tensor):
                    data = data.numpy()
                sample_data.append(data.flatten() if data.ndim > 1 else data)
                sample_labels.append(label)
            
            df = pd.DataFrame(sample_data)
            if labels is not None or sample_labels:
                df['target'] = sample_labels if sample_labels else labels[:sample_size]
                
        elif hasattr(dataset, 'shape'):  # NumPy array
            df = pd.DataFrame(dataset.reshape(dataset.shape[0], -1))
            if labels is not None:
                df['target'] = labels
                
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
            
        else:
            # Try to convert to DataFrame
            df = pd.DataFrame(dataset)
            if labels is not None:
                df['target'] = labels
        
        # Basic metadata
        analysis.dataset_size = len(df)
        analysis.feature_count = len(df.columns) - (1 if 'target' in df.columns else 0)
        analysis.data_types = list(df.dtypes.astype(str).unique())
        
        # Data quality analysis
        analysis.missing_values_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        analysis.duplicate_records_pct = (df.duplicated().sum() / len(df)) * 100
        
        # Outlier detection (using IQR method for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count += ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        analysis.outlier_pct = (outlier_count / (len(df) * len(numeric_cols))) * 100 if len(numeric_cols) > 0 else 0
        
        # Class analysis if target exists
        if 'target' in df.columns:
            unique_classes = df['target'].nunique()
            analysis.class_count = unique_classes
            
            # Class imbalance ratio (ratio of majority to minority class)
            class_counts = df['target'].value_counts()
            if len(class_counts) > 1:
                analysis.class_imbalance_ratio = class_counts.max() / class_counts.min()
        
        # Data shape analysis
        if hasattr(dataset, 'shape') and len(dataset.shape) > 1:
            analysis.input_shape = dataset.shape[1:]
        
        # Preprocessing recommendations
        analysis.normalization_needed = any(df[col].std() > 100 for col in numeric_cols) if len(numeric_cols) > 0 else False
        analysis.encoding_needed = len(df.select_dtypes(include=['object']).columns) > 0
        
        # Generate preprocessing steps
        preprocessing_steps = []
        if analysis.missing_values_pct > 5:
            preprocessing_steps.append("Handle missing values")
        if analysis.normalization_needed:
            preprocessing_steps.append("Normalize/standardize features")
        if analysis.encoding_needed:
            preprocessing_steps.append("Encode categorical variables")
        if analysis.outlier_pct > 10:
            preprocessing_steps.append("Handle outliers")
        if analysis.class_imbalance_ratio and analysis.class_imbalance_ratio > 3:
            preprocessing_steps.append("Address class imbalance")
            
        analysis.preprocessing_steps = preprocessing_steps
        
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze dataset automatically: {e}")
        # Return basic analysis with error info
        analysis = DatasetAnalysis()
        analysis.dataset_name = f"{dataset_name} (analysis failed)"
        return analysis

def create_problem_definition_template(problem_type: str = "classification", domain: str = "general") -> ProblemDefinition:
    """Create a template problem definition based on problem type and domain."""
    templates = {
        "classification": {
            "problem_title": "Classification Problem",
            "problem_description": "Classify input data into predefined categories",
            "primary_objective": "Maximize classification accuracy",
            "success_metrics": ["accuracy", "precision", "recall", "f1_score"],
            "baseline_performance": 0.5,
            "target_performance": 0.9
        },
        "regression": {
            "problem_title": "Regression Problem", 
            "problem_description": "Predict continuous numerical values",
            "primary_objective": "Minimize prediction error",
            "success_metrics": ["mse", "rmse", "mae", "r2_score"],
            "baseline_performance": None,
            "target_performance": None
        },
        "generation": {
            "problem_title": "Generation Problem",
            "problem_description": "Generate new data samples",
            "primary_objective": "Generate high-quality, diverse samples",
            "success_metrics": ["fid_score", "inception_score", "lpips", "human_evaluation"],
            "baseline_performance": None,
            "target_performance": None
        }
    }
    
    template = templates.get(problem_type, templates["classification"])
    
    problem_def = ProblemDefinition(
        problem_title=template["problem_title"],
        problem_description=template["problem_description"],
        problem_type=problem_type,
        domain=domain,
        primary_objective=template["primary_objective"],
        success_metrics=template["success_metrics"],
        baseline_performance=template["baseline_performance"],
        target_performance=template["target_performance"]
    )
    
    return problem_def

def create_experiment_tracker(config: ExperimentConfig) -> ExperimentTracker:
    """Create and configure an experiment tracker."""
    try:
        tracker = ExperimentTracker(config)
        
        # Log problem definition and dataset analysis if available
        if config.problem_definition:
            tracker.log_problem_definition(config.problem_definition)
            
        if config.dataset_analysis:
            tracker.log_dataset_analysis(config.dataset_analysis)
            
        return tracker
    except Exception as e:
        logger.error(f"Failed to create experiment tracker: {e}")
        raise


def get_default_experiment_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def create_experiment_config(**kwargs) -> ExperimentConfig:
    """Create experiment configuration with custom parameters."""
    default_config = get_default_experiment_config()
    
    # Update with provided parameters
    for key, value in kwargs.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)
    
    return default_config


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example of how to use the experiment tracker."""
    
    # Create configuration
    config = create_experiment_config(
        experiment_name="gradient_clipping_experiment",
        project_name="numerical_stability_research",
        enable_tensorboard=True,
        enable_wandb=True,
        log_interval=10
    )
    
    # Create tracker
    tracker = create_experiment_tracker(config)
    
    try:
        # Log hyperparameters
        hyperparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_grad_norm': 1.0,
            'clipping_type': 'norm'
        }
        tracker.log_hyperparameters(hyperparams)
        
        # Simulate training steps
        for step in range(100):
            # Simulate metrics
            loss = 1.0 / (step + 1) + np.random.normal(0, 0.1)
            accuracy = min(0.95, 0.5 + step * 0.005 + np.random.normal(0, 0.02))
            gradient_norm = np.random.exponential(0.5)
            
            # Log training step
            tracker.log_training_step(
                loss=loss,
                accuracy=accuracy,
                learning_rate=0.001,
                gradient_norm=gradient_norm,
                nan_count=np.random.poisson(0.1),
                inf_count=np.random.poisson(0.05),
                clipping_applied=gradient_norm > 1.0,
                clipping_threshold=1.0 if gradient_norm > 1.0 else None,
                training_time=np.random.exponential(0.1)
            )
            
            # Log every 10 steps
            if step % 10 == 0:
                tracker.log_gradients(None)  # Would pass actual model
        
        # Create visualization
        viz_data = tracker.create_visualization("training_progress.png")
        print("Visualization created:", viz_data.get('metrics_summary', {}))
        
        # Get experiment summary
        summary = tracker.get_experiment_summary()
        print("Experiment summary:", summary)
        
    finally:
        # Close tracker
        tracker.close()


if __name__ == "__main__":
    example_usage()
