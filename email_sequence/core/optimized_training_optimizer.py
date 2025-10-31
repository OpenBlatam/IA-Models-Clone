from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import json
import numpy as np
from pathlib import Path
from core.training_logger import (
from core.error_handling import ErrorHandler, ModelError, DataError
from core.early_stopping import EarlyStopping, EarlyStoppingConfig
from core.learning_rate_scheduling import LRScheduler, LRSchedulerConfig
from core.gradient_management import GradientManager, GradientConfig
from core.pytorch_debugging import PyTorchDebugger, create_pytorch_debugger, debug_training_session
from core.performance_optimizer import (
from core.multi_gpu_training import (
from core.gradient_accumulation import (
from core.mixed_precision_training import (
from core.code_profiler import (
    import torch.nn as nn
from typing import Any, List, Dict, Optional
import logging
"""
Optimized Training Optimizer

Enhanced training optimizer with comprehensive performance optimization
including memory optimization, computational efficiency, and training acceleration.
"""


    TrainingLogger, TrainingEventType, LogLevel, 
    create_training_logger, TrainingMetrics
)
    PerformanceOptimizer, PerformanceConfig, create_performance_optimizer,
    get_optimal_batch_size, benchmark_model_performance
)
    MultiGPUTrainer, MultiGPUConfig, create_multi_gpu_trainer,
    optimize_model_for_multi_gpu
)
    GradientAccumulator, GradientAccumulationConfig, create_gradient_accumulator,
    create_gradient_accumulation_trainer, calculate_optimal_accumulation_steps
)
    MixedPrecisionConfig, MixedPrecisionTrainer, create_mixed_precision_trainer,
    check_amp_compatibility
)
    CodeProfiler, ProfilerConfig, create_code_profiler
)


class OptimizedTrainingOptimizer:
    """Optimized training optimizer with comprehensive performance optimization and multi-GPU support"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        experiment_name: str = "optimized_email_sequence_training",
        log_dir: str = "logs",
        log_level: str = "INFO",
        device: str = "auto",
        debug_mode: bool = False,
        enable_pytorch_debugging: bool = False,
        performance_config: Optional[PerformanceConfig] = None,
        multi_gpu_config: Optional[MultiGPUConfig] = None,
        gradient_accumulation_config: Optional[GradientAccumulationConfig] = None,
        mixed_precision_config: Optional[MixedPrecisionConfig] = None,
        **kwargs
    ):
        """Initialize the optimized training optimizer"""
        
        # Setup device
        match device:
    case "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize training logger
        self.logger = create_training_logger(
            experiment_name=experiment_name,
            log_dir=log_dir,
            log_level=log_level,
            enable_visualization=True,
            enable_metrics_logging=True
        )
        
        # Initialize error handler
        self.error_handler = ErrorHandler(debug_mode=True)
        
        # Initialize performance optimizer
        if performance_config is None:
            performance_config = PerformanceConfig()
        self.performance_optimizer = create_performance_optimizer(
            logger=self.logger,
            device=device,
            **{k: v for k, v in kwargs.items() if k in PerformanceConfig.__annotations__}
        )
        
        # Initialize multi-GPU trainer
        if multi_gpu_config is None:
            multi_gpu_config = MultiGPUConfig()
        self.multi_gpu_trainer = create_multi_gpu_trainer(
            logger=self.logger,
            device=device,
            **{k: v for k, v in kwargs.items() if k in MultiGPUConfig.__annotations__}
        )
        
        # Initialize gradient accumulation
        if gradient_accumulation_config is None:
            gradient_accumulation_config = GradientAccumulationConfig()
        self.gradient_accumulation_config = gradient_accumulation_config
        self.gradient_accumulator = None  # Will be initialized after optimizer setup
        
        # Initialize PyTorch debugger
        self.debug_mode = debug_mode
        self.enable_pytorch_debugging = enable_pytorch_debugging
        if self.enable_pytorch_debugging:
            self.debugger = create_pytorch_debugger(
                logger=self.logger,
                debug_mode=debug_mode,
                enable_anomaly_detection=True,
                enable_profiling=kwargs.get("enable_profiling", False),
                enable_memory_tracking=True,
                enable_gradient_checking=True,
                log_dir=f"{log_dir}/debug"
            )
        else:
            self.debugger = None
        
        # Optimize model and data loaders with multi-GPU support
        self.model = self._optimize_model_with_multi_gpu(model)
        self.train_loader = self._optimize_dataloader_with_multi_gpu(train_loader)
        self.val_loader = self._optimize_dataloader_with_multi_gpu(val_loader) if val_loader else None
        
        # Training configuration
        self.config = {
            "learning_rate": kwargs.get("learning_rate", 0.001),
            "batch_size": kwargs.get("batch_size", 32),
            "max_epochs": kwargs.get("max_epochs", 100),
            "weight_decay": kwargs.get("weight_decay", 1e-5),
            "gradient_clip": kwargs.get("gradient_clip", 1.0),
            "early_stopping_patience": kwargs.get("early_stopping_patience", 10),
            "checkpoint_dir": kwargs.get("checkpoint_dir", "checkpoints"),
            "save_interval": kwargs.get("save_interval", 5),
            "debug_mode": debug_mode,
            "enable_pytorch_debugging": enable_pytorch_debugging,
            "performance_config": performance_config.__dict__,
            "multi_gpu_config": multi_gpu_config.__dict__,
            **kwargs
        }
        
        # Initialize optimizers and schedulers
        self._setup_optimizers()
        self._setup_schedulers()
        self._setup_early_stopping()
        self._setup_gradient_management()
        
        # Initialize gradient accumulation after optimizer setup
        self._setup_gradient_accumulation()
        
        # Initialize mixed precision training
        self._setup_mixed_precision(mixed_precision_config, **kwargs)
        
        # Initialize code profiler
        self._setup_code_profiler(**kwargs)
        
        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
        
        # Performance tracking
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.learning_rates = []
        self.gradient_norms = []
        self.performance_metrics = []
        
        # Resource monitoring
        self.memory_usage = []
        self.gpu_usage = []
        
        # Start performance monitoring
        self.performance_optimizer.start_monitoring()
        
        self.logger.log_info(f"Optimized training optimizer initialized on device: {self.device}")
        self.logger.log_info(f"Training configuration: {json.dumps(self.config, indent=2)}")
        
        if self.debugger:
            self.logger.log_info("PyTorch debugging tools enabled")
        
        # Log multi-GPU information
        training_info = self.multi_gpu_trainer.get_training_info()
        self.logger.log_info(f"Multi-GPU training info: {json.dumps(training_info, indent=2)}")
        
        # Log optimization recommendations
        recommendations = self.performance_optimizer.get_optimization_recommendations()
        if recommendations:
            self.logger.log_info(f"Optimization recommendations: {recommendations}")
    
    def _optimize_model_with_multi_gpu(self, model: nn.Module) -> nn.Module:
        """Optimize model with both performance and multi-GPU optimizations"""
        
        try:
            # First apply performance optimizations
            model = self.performance_optimizer.optimize_model(model)
            
            # Then apply multi-GPU optimizations
            model = self.multi_gpu_trainer.initialize_training(model)
            
            self.logger.log_info("Model optimized with performance and multi-GPU optimizations")
            return model
            
        except Exception as e:
            self.logger.log_error(e, "Model optimization", "optimize_model_with_multi_gpu")
            return model
    
    def _optimize_dataloader_with_multi_gpu(self, dataloader: DataLoader) -> DataLoader:
        """Optimize DataLoader with multi-GPU support"""
        
        try:
            # First apply performance optimizations
            dataloader = self.performance_optimizer.optimize_dataloader(dataloader)
            
            # Then apply multi-GPU optimizations if needed
            if hasattr(dataloader, 'dataset'):
                dataloader = self.multi_gpu_trainer.setup_dataloader(
                    dataloader.dataset,
                    dataloader.batch_size,
                    num_workers=getattr(dataloader, 'num_workers', 4),
                    pin_memory=getattr(dataloader, 'pin_memory', True),
                    persistent_workers=getattr(dataloader, 'persistent_workers', True)
                )
            
            return dataloader
            
        except Exception as e:
            self.logger.log_error(e, "DataLoader optimization", "optimize_dataloader_with_multi_gpu")
            return dataloader
    
    def _setup_optimizers(self) -> Any:
        """Setup and optimize optimizers"""
        
        try:
            # Main optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
            
            # Optimize optimizer
            self.optimizer = self.performance_optimizer.optimize_optimizer(self.optimizer)
            
            # Additional optimizers for different parameter groups
            self.optimizers = {
                "main": self.optimizer,
                "embedding": optim.AdamW(
                    [p for name, p in self.model.named_parameters() if "embedding" in name],
                    lr=self.config["learning_rate"] * 0.1
                ) if any("embedding" in name for name, _ in self.model.named_parameters()) else None,
                "classifier": optim.AdamW(
                    [p for name, p in self.model.named_parameters() if "classifier" in name],
                    lr=self.config["learning_rate"] * 2.0
                ) if any("classifier" in name for name, _ in self.model.named_parameters()) else None
            }
            
            # Optimize additional optimizers
            for name, opt in self.optimizers.items():
                if opt is not None:
                    self.optimizers[name] = self.performance_optimizer.optimize_optimizer(opt)
            
            self.logger.log_info("Optimizers initialized and optimized successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Optimizer setup", "setup_optimizers")
            raise ModelError(f"Failed to setup optimizers: {str(e)}")
    
    def _setup_schedulers(self) -> Any:
        """Setup learning rate schedulers"""
        
        try:
            # Main scheduler
            self.scheduler = LRScheduler(
                self.optimizer,
                LRSchedulerConfig(
                    scheduler_type="cosine",
                    warmup_steps=1000,
                    max_steps=self.config["max_epochs"] * len(self.train_loader)
                )
            )
            
            # Additional schedulers for different parameter groups
            self.schedulers = {
                "main": self.scheduler
            }
            
            if self.optimizers["embedding"]:
                self.schedulers["embedding"] = LRScheduler(
                    self.optimizers["embedding"],
                    LRSchedulerConfig(scheduler_type="linear", warmup_steps=500)
                )
            
            if self.optimizers["classifier"]:
                self.schedulers["classifier"] = LRScheduler(
                    self.optimizers["classifier"],
                    LRSchedulerConfig(scheduler_type="step", step_size=5, gamma=0.9)
                )
            
            self.logger.log_info("Learning rate schedulers initialized successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Scheduler setup", "setup_schedulers")
            raise ModelError(f"Failed to setup schedulers: {str(e)}")
    
    def _setup_early_stopping(self) -> Any:
        """Setup early stopping"""
        
        try:
            self.early_stopping = EarlyStopping(
                EarlyStoppingConfig(
                    patience=self.config["early_stopping_patience"],
                    min_delta=1e-4,
                    restore_best_weights=True,
                    monitor="validation_loss"
                )
            )
            
            self.logger.log_info("Early stopping initialized successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Early stopping setup", "setup_early_stopping")
            raise ModelError(f"Failed to setup early stopping: {str(e)}")
    
    def _setup_gradient_management(self) -> Any:
        """Setup gradient management"""
        
        try:
            self.gradient_manager = GradientManager(
                GradientConfig(
                    clip_type="norm",
                    clip_value=self.config["gradient_clip"],
                    detect_nan_inf=True,
                    fix_nan_inf=True
                )
            )
            
            self.logger.log_info("Gradient management initialized successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Gradient management setup", "setup_gradient_management")
            raise ModelError(f"Failed to setup gradient management: {str(e)}")
    
    def _setup_gradient_accumulation(self) -> Any:
        """Setup gradient accumulation"""
        
        try:
            # Initialize gradient accumulator
            self.gradient_accumulator = create_gradient_accumulator(
                model=self.model,
                optimizer=self.optimizer,
                accumulation_steps=self.gradient_accumulation_config.accumulation_steps,
                effective_batch_size=self.gradient_accumulation_config.effective_batch_size,
                logger=self.logger,
                **{k: v for k, v in self.gradient_accumulation_config.__dict__.items() 
                   if k not in ['accumulation_steps', 'effective_batch_size']}
            )
            
            # Update configuration
            self.config["gradient_accumulation_config"] = self.gradient_accumulation_config.__dict__
            
            self.logger.log_info("Gradient accumulation initialized successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Gradient accumulation setup", "setup_gradient_accumulation")
            raise ModelError(f"Failed to setup gradient accumulation: {str(e)}")
    
    def _setup_mixed_precision(self, mixed_precision_config: Optional[MixedPrecisionConfig], **kwargs):
        """Setup mixed precision training"""
        
        try:
            # Initialize mixed precision configuration
            if mixed_precision_config is None:
                mixed_precision_config = MixedPrecisionConfig(
                    enable_amp=kwargs.get("enable_amp", True),
                    dtype=kwargs.get("amp_dtype", torch.float16),
                    enable_grad_scaler=kwargs.get("enable_grad_scaler", True),
                    init_scale=kwargs.get("amp_init_scale", 2**16),
                    growth_factor=kwargs.get("amp_growth_factor", 2.0),
                    backoff_factor=kwargs.get("amp_backoff_factor", 0.5),
                    growth_interval=kwargs.get("amp_growth_interval", 2000),
                    enable_monitoring=kwargs.get("amp_monitoring", True),
                    log_amp_stats=kwargs.get("amp_log_stats", True),
                    track_memory_usage=kwargs.get("amp_track_memory", True),
                    validate_amp=kwargs.get("amp_validate", True),
                    check_compatibility=kwargs.get("amp_check_compatibility", True)
                )
            
            self.mixed_precision_config = mixed_precision_config
            
            # Check model compatibility with mixed precision
            if mixed_precision_config.check_compatibility:
                compatibility = check_amp_compatibility(self.model)
                if not compatibility["compatible"]:
                    self.logger.log_warning(f"Mixed precision compatibility issues: {compatibility}")
                else:
                    self.logger.log_info("Model is compatible with mixed precision training")
            
            # Initialize mixed precision trainer
            self.mp_trainer = create_mixed_precision_trainer(
                model=self.model,
                optimizer=self.optimizer,
                enable_amp=mixed_precision_config.enable_amp,
                dtype=mixed_precision_config.dtype,
                enable_grad_scaler=mixed_precision_config.enable_grad_scaler,
                logger=self.logger,
                init_scale=mixed_precision_config.init_scale,
                growth_factor=mixed_precision_config.growth_factor,
                backoff_factor=mixed_precision_config.backoff_factor,
                growth_interval=mixed_precision_config.growth_interval,
                enable_monitoring=mixed_precision_config.enable_monitoring,
                log_amp_stats=mixed_precision_config.log_amp_stats,
                track_memory_usage=mixed_precision_config.track_memory_usage
            )
            
            # Add mixed precision config to training config
            self.config["mixed_precision_config"] = mixed_precision_config.__dict__
            
            self.logger.log_info("Mixed precision training initialized successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Mixed precision setup", "setup_mixed_precision")
            # Don't raise error, fall back to standard precision
            self.logger.log_warning("Failed to setup mixed precision, falling back to standard precision")
            self.mp_trainer = None
    
    def _setup_code_profiler(self, **kwargs) -> Any:
        """Setup code profiler for performance analysis"""
        
        try:
            # Create profiler configuration
            profiler_config = ProfilerConfig(
                enable_profiling=kwargs.get("enable_profiling", True),
                profile_level=kwargs.get("profile_level", "detailed"),
                save_profiles=kwargs.get("save_profiles", True),
                profile_dir=f"{self.config['checkpoint_dir']}/profiles",
                enable_performance_monitoring=kwargs.get("enable_performance_monitoring", True),
                monitor_interval=kwargs.get("monitor_interval", 1.0),
                track_memory=kwargs.get("track_memory", True),
                track_cpu=kwargs.get("track_cpu", True),
                track_gpu=kwargs.get("track_gpu", True),
                profile_data_loading=kwargs.get("profile_data_loading", True),
                profile_preprocessing=kwargs.get("profile_preprocessing", True),
                profile_forward_pass=kwargs.get("profile_forward_pass", True),
                profile_backward_pass=kwargs.get("profile_backward_pass", True),
                profile_optimizer_step=kwargs.get("profile_optimizer_step", True),
                enable_memory_profiling=kwargs.get("enable_memory_profiling", True),
                enable_gpu_profiling=kwargs.get("enable_gpu_profiling", True),
                generate_reports=kwargs.get("generate_reports", True),
                create_visualizations=kwargs.get("create_visualizations", True)
            )
            
            # Initialize code profiler
            self.code_profiler = create_code_profiler(
                enable_profiling=profiler_config.enable_profiling,
                profile_level=profiler_config.profile_level,
                save_profiles=profiler_config.save_profiles,
                logger=self.logger,
                **profiler_config.__dict__
            )
            
            # Start performance monitoring
            if profiler_config.enable_performance_monitoring:
                self.code_profiler.start_performance_monitoring()
            
            # Add profiler config to training config
            self.config["profiler_config"] = profiler_config.__dict__
            
            self.logger.log_info("Code profiler initialized successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Code profiler setup", "setup_code_profiler")
            self.logger.log_warning("Failed to setup code profiler, continuing without profiling")
            self.code_profiler = None
    
    def _get_resource_usage(self) -> Tuple[float, float]:
        """Get current resource usage"""
        
        try:
            # Memory usage
            memory_usage = psutil.virtual_memory().percent / 100.0
            
            # GPU usage
            gpu_usage = 0.0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            return memory_usage, gpu_usage
            
        except Exception as e:
            self.logger.log_warning(f"Failed to get resource usage: {e}")
            return 0.0, 0.0
    
    def _calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate training metrics"""
        
        try:
            # Loss calculation
            if hasattr(self.model, 'loss_fn'):
                loss = self.model.loss_fn(outputs, targets)
            else:
                loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Accuracy calculation
            if outputs.dim() > 1:
                accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
            else:
                accuracy = ((outputs > 0.5) == targets).float().mean().item()
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            return {
                "loss": loss.item(),
                "accuracy": accuracy,
                "learning_rate": current_lr
            }
            
        except Exception as e:
            self.logger.log_error(e, "Metrics calculation", "calculate_metrics")
            return {"loss": float('inf'), "accuracy": 0.0, "learning_rate": 0.0}
    
    def _train_batch_with_accumulation(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Train a single batch with gradient accumulation and mixed precision"""
        
        try:
            inputs, targets = batch_data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            batch_start_time = time.time()
            
            # Use mixed precision training if available
            if self.mp_trainer is not None:
                # Use mixed precision trainer for forward pass and loss calculation
                step_metrics = self.mp_trainer.train_step(
                    (inputs, targets),
                    lambda outputs, targets: self._calculate_metrics(outputs, targets)["loss"],
                    self.device
                )
                
                loss = step_metrics["loss"]
                accuracy = step_metrics["accuracy"]
                batch_time = step_metrics["step_time"]
                
                # Get mixed precision statistics
                amp_stats = self.mp_trainer.get_amp_stats()
                
            else:
                # Standard precision training
                outputs = self.model(inputs)
                loss = self._calculate_metrics(outputs, targets)["loss"]
                
                # Calculate accuracy
                if outputs.dim() > 1:
                    accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                else:
                    accuracy = ((outputs > 0.5) == targets).float().mean().item()
                
                batch_time = time.time() - batch_start_time
                amp_stats = {}
            
            # Accumulate gradients (mixed precision handles this internally)
            if self.mp_trainer is None:
                should_update = self.gradient_accumulator.accumulate_gradients(loss)
            else:
                # Mixed precision trainer handles gradient accumulation internally
                should_update = True  # Mixed precision updates every step
            
            # Get gradient norm
            gradient_norm = self._calculate_gradient_norm()
            
            # Update schedulers only when parameters are updated
            if should_update and self.mp_trainer is None:
                for scheduler in self.schedulers.values():
                    scheduler.step()
            
            # Calculate final metrics
            batch_size = inputs.size(0)
            
            metrics = {
                "loss": loss.item() if hasattr(loss, 'item') else loss,
                "accuracy": accuracy,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "gradient_norm": gradient_norm,
                "batch_time": batch_time,
                "batch_size": batch_size,
                "should_update": should_update,
                "amp_enabled": self.mp_trainer is not None
            }
            
            # Add mixed precision statistics if available
            if amp_stats:
                metrics.update({
                    "amp_scale": amp_stats.get("current_scale"),
                    "amp_usage_ratio": amp_stats.get("amp_usage_ratio"),
                    "overflow_ratio": amp_stats.get("overflow_ratio"),
                    "memory_savings": amp_stats.get("avg_memory_savings", 0.0)
                })
            
            # Add accumulation info if not using mixed precision
            if self.mp_trainer is None:
                metrics.update({
                    "accumulation_step": self.gradient_accumulator.accumulation_step,
                    "accumulation_progress": self.gradient_accumulator.accumulation_step / self.gradient_accumulation_config.accumulation_steps,
                })
            
            # Record GPU metrics for multi-GPU training
            self.multi_gpu_trainer.record_gpu_metrics()
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(e, f"Batch {self.current_batch}", "train_batch_with_accumulation")
            return {"loss": float('inf'), "accuracy": 0.0, "learning_rate": 0.0, "gradient_norm": 0.0}
    
    def _calculate_gradient_norm(self) -> float:
        """Calculate gradient norm across all parameters"""
        
        try:
            total_norm = 0.0
            param_count = 0
            
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                return total_norm
            else:
                return 0.0
                
        except Exception as e:
            self.logger.log_error(e, "Gradient norm calculation", "calculate_gradient_norm")
            return 0.0
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate the model for one epoch with performance optimization and multi-GPU support"""
        
        if self.val_loader is None:
            return {}
        
        try:
            self.model.eval()
            total_loss = 0.0
            total_accuracy = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_data in self.val_loader:
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Use debugging for validation if enabled
                    if self.debugger:
                        outputs = self.debugger.debug_forward_pass(self.model, inputs, layer_hooks=False)
                    else:
                        outputs = self.model(inputs)
                    
                    metrics = self._calculate_metrics(outputs, targets)
                    
                    total_loss += metrics["loss"]
                    total_accuracy += metrics["accuracy"]
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            
            return {
                "validation_loss": avg_loss,
                "validation_accuracy": avg_accuracy
            }
            
        except Exception as e:
            self.logger.log_error(e, "Validation", "validate_epoch")
            return {"validation_loss": float('inf'), "validation_accuracy": 0.0}
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with multi-GPU support"""
        
        try:
            checkpoint_dir = Path(self.config["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Get model state dict (handle DataParallel/DistributedDataParallel)
            if hasattr(self.model, 'module'):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "config": self.config,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "performance_summary": self.performance_optimizer.get_performance_summary(),
                "multi_gpu_summary": self.multi_gpu_trainer.get_performance_summary()
            }
            
            # Save regular checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best checkpoint
            if is_best:
                best_checkpoint_path = checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, best_checkpoint_path)
            
            self.logger.log_checkpoint(str(checkpoint_path), metrics)
            
        except Exception as e:
            self.logger.log_error(e, "Checkpoint saving", "save_checkpoint")
    
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint with multi-GPU support"""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state dict (handle DataParallel/DistributedDataParallel)
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            self.current_epoch = checkpoint["epoch"]
            self.best_metric = checkpoint["best_metric"]
            self.best_epoch = checkpoint["best_epoch"]
            
            self.logger.log_info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint["epoch"]
            
        except Exception as e:
            self.logger.log_error(e, "Checkpoint loading", "load_checkpoint")
            raise ModelError(f"Failed to load checkpoint: {str(e)}")
    
    async def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with performance optimization and multi-GPU support"""
        
        try:
            with self.logger.epoch_context(epoch, len(self.train_loader)):
                epoch_start_time = time.time()
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                num_batches = 0
                
                # Use PyTorch debugging context if enabled
                debug_context = self.debugger.anomaly_detection() if self.debugger else nullcontext()
                
                with debug_context:
                    for batch_idx, batch_data in enumerate(self.train_loader):
                        with self.logger.batch_context(batch_idx, len(self.train_loader)):
                            # Train batch
                            batch_metrics = self._train_batch(batch_data)
                            
                            # Update epoch metrics
                            epoch_loss += batch_metrics["loss"]
                            epoch_accuracy += batch_metrics["accuracy"]
                            num_batches += 1
                            
                            # Update step counter
                            self.current_step += 1
                            
                            # Log batch metrics
                            memory_usage, gpu_usage = self._get_resource_usage()
                            
                            batch_metrics.update({
                                "memory_usage": memory_usage,
                                "gpu_usage": gpu_usage
                            })
                            
                            self.logger.end_batch(batch_metrics)
                            
                            # Log resource usage
                            self.logger.log_resource_usage(memory_usage, gpu_usage)
                            
                            # Log gradient updates
                            if "gradient_norm" in batch_metrics:
                                self.logger.log_gradient_update(
                                    batch_metrics["gradient_norm"],
                                    self.config["gradient_clip"]
                                )
                
                # Calculate epoch metrics
                avg_epoch_loss = epoch_loss / num_batches
                avg_epoch_accuracy = epoch_accuracy / num_batches
                epoch_duration = time.time() - epoch_start_time
                
                epoch_metrics = {
                    "loss": avg_epoch_loss,
                    "accuracy": avg_epoch_accuracy,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch_duration": epoch_duration
                }
                
                # Validation
                if self.val_loader:
                    validation_metrics = self._validate_epoch()
                    epoch_metrics.update(validation_metrics)
                    self.logger.log_validation(validation_metrics)
                
                # Update best metric
                current_metric = epoch_metrics.get("validation_loss", epoch_metrics["loss"])
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    is_best = True
                else:
                    is_best = False
                
                # Save checkpoint
                if epoch % self.config["save_interval"] == 0 or is_best:
                    self._save_checkpoint(epoch, epoch_metrics, is_best)
                
                # Early stopping check
                if self.val_loader:
                    should_stop = self.early_stopping(epoch_metrics["validation_loss"])
                    if should_stop:
                        self.logger.log_early_stopping(
                            "Validation loss did not improve",
                            self.best_epoch,
                            self.best_metric
                        )
                        return epoch_metrics
                
                self.logger.end_epoch(epoch_metrics)
                return epoch_metrics
                
        except Exception as e:
            self.logger.log_error(e, f"Epoch {epoch}", "train_epoch")
            raise
    
    async def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop with performance optimization and multi-GPU support"""
        
        try:
            # Resume from checkpoint if provided
            if resume_from:
                start_epoch = self._load_checkpoint(resume_from)
            else:
                start_epoch = 0
            
            # Use debugging training session if enabled
            if self.debugger:
                debug_context = debug_training_session(
                    model=self.model,
                    logger=self.logger,
                    debug_mode=self.debug_mode,
                    enable_anomaly_detection=True,
                    enable_profiling=self.config.get("enable_profiling", False),
                    enable_memory_tracking=True,
                    enable_gradient_checking=True
                )
            else:
                debug_context = nullcontext()
            
            with debug_context as debugger:
                # Start training session
                with self.logger.training_context(self.config["max_epochs"], self.config):
                    for epoch in range(start_epoch, self.config["max_epochs"]):
                        self.current_epoch = epoch
                        
                        # Train epoch
                        epoch_metrics = await self.train_epoch(epoch)
                        
                        # Check for early stopping
                        if self.early_stopping.should_stop:
                            self.logger.log_info("Early stopping triggered")
                            break
                        
                        # Log learning rate changes
                        if epoch > 0:
                            old_lr = self.learning_rates[-1] if self.learning_rates else 0.0
                            new_lr = epoch_metrics["learning_rate"]
                            if abs(new_lr - old_lr) > 1e-6:
                                self.logger.log_learning_rate_change(old_lr, new_lr, "Scheduler update")
                        
                        # Store metrics
                        self.training_losses.append(epoch_metrics["loss"])
                        self.training_accuracies.append(epoch_metrics["accuracy"])
                        self.learning_rates.append(epoch_metrics["learning_rate"])
                        
                        if "validation_loss" in epoch_metrics:
                            self.validation_losses.append(epoch_metrics["validation_loss"])
                            self.validation_accuracies.append(epoch_metrics["validation_accuracy"])
                    
                    # Final validation
                    if self.val_loader:
                        final_validation = self._validate_epoch()
                        self.logger.log_validation(final_validation)
                    
                    # Training summary
                    training_summary = {
                        "total_epochs": self.current_epoch + 1,
                        "best_epoch": self.best_epoch,
                        "best_metric": self.best_metric,
                        "final_training_loss": self.training_losses[-1] if self.training_losses else float('inf'),
                        "final_validation_loss": self.validation_losses[-1] if self.validation_losses else float('inf'),
                        "early_stopping_triggered": self.early_stopping.should_stop
                    }
                    
                    # Add performance summary
                    performance_summary = self.performance_optimizer.get_performance_summary()
                    training_summary["performance_summary"] = performance_summary
                    
                    # Add multi-GPU summary
                    multi_gpu_summary = self.multi_gpu_trainer.get_performance_summary()
                    training_summary["multi_gpu_summary"] = multi_gpu_summary
                    
                    # Add debug summary if available
                    if debugger:
                        training_summary["debug_summary"] = debugger.get_debug_summary()
                    
                    self.logger.end_training(training_summary)
                    
                    return training_summary
                
        except Exception as e:
            self.logger.log_error(e, "Training", "train")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary with performance and multi-GPU metrics"""
        
        summary = {
            **self.logger.get_training_summary(),
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "training_accuracies": self.training_accuracies,
            "validation_accuracies": self.validation_accuracies,
            "learning_rates": self.learning_rates,
            "gradient_norms": self.gradient_norms,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric
        }
        
        # Add performance summary
        performance_summary = self.performance_optimizer.get_performance_summary()
        summary["performance_summary"] = performance_summary
        
        # Add multi-GPU summary
        multi_gpu_summary = self.multi_gpu_trainer.get_performance_summary()
        summary["multi_gpu_summary"] = multi_gpu_summary
        
        # Add debug summary if available
        if self.debugger:
            summary["debug_summary"] = self.debugger.get_debug_summary()
        
        return summary
    
    def create_training_visualizations(self, save_path: str = None):
        """Create comprehensive training visualizations including multi-GPU metrics"""
        
        try:
            self.logger.create_visualizations(save_path)
            
            # Create performance-specific visualizations
            performance_summary = self.performance_optimizer.get_performance_summary()
            if performance_summary:
                self.logger.log_info(f"Performance summary: {json.dumps(performance_summary, indent=2)}")
            
            # Create multi-GPU specific visualizations
            multi_gpu_summary = self.multi_gpu_trainer.get_performance_summary()
            if multi_gpu_summary:
                self.logger.log_info(f"Multi-GPU summary: {json.dumps(multi_gpu_summary, indent=2)}")
                
        except Exception as e:
            self.logger.log_error(e, "Visualization creation", "create_training_visualizations")
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance with multi-GPU support"""
        
        try:
            benchmark_results = benchmark_model_performance(
                self.model, self.train_loader, num_iterations
            )
            
            self.logger.log_info(f"Performance benchmark: {json.dumps(benchmark_results, indent=2)}")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.log_error(e, "Performance benchmarking", "benchmark_performance")
            return {}
    
    def get_optimal_batch_size(self, target_memory_usage: float = 0.8) -> int:
        """Get optimal batch size for current model and memory constraints"""
        
        try:
            # Get input size from first batch
            sample_batch = next(iter(self.train_loader))
            input_size = sample_batch[0].shape[1:]  # Remove batch dimension
            
            optimal_batch_size = get_optimal_batch_size(
                self.model, input_size, target_memory_usage
            )
            
            self.logger.log_info(f"Optimal batch size: {optimal_batch_size}")
            
            return optimal_batch_size
            
        except Exception as e:
            self.logger.log_error(e, "Optimal batch size calculation", "get_optimal_batch_size")
            return 32  # Default fallback
    
    def cleanup(self) -> Any:
        """Cleanup all resources"""
        
        try:
            # Stop performance monitoring
            self.performance_optimizer.stop_monitoring()
            
            # Cleanup multi-GPU training
            self.multi_gpu_trainer.cleanup()
            
            # Cleanup gradient accumulation
            if self.gradient_accumulator:
                self.gradient_accumulator.cleanup()
            
            # Cleanup mixed precision training
            if self.mp_trainer:
                self.mp_trainer.cleanup()
            
            # Cleanup code profiler
            if self.code_profiler:
                self.code_profiler.cleanup()
            
            # Cleanup PyTorch debugger
            if self.debugger:
                self.debugger.cleanup()
            
            # Save final training summary
            summary = self.get_training_summary()
            summary_path = Path(self.logger.log_dir) / "final_training_summary.json"
            with open(summary_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(summary, f, indent=2)
            
            # Create final visualizations
            self.create_training_visualizations()
            
            self.logger.log_info("Training optimizer cleanup completed successfully")
            
        except Exception as e:
            self.logger.log_error(e, "Cleanup", "cleanup")
            self.logger.log_warning("Some cleanup operations may have failed")

    def _train_batch(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Train a single batch with performance optimization, multi-GPU support, and gradient accumulation"""
        
        # Use gradient accumulation if enabled
        if self.gradient_accumulator and self.gradient_accumulation_config.accumulation_steps > 1:
            return self._train_batch_with_accumulation(batch_data)
        
        try:
            inputs, targets = batch_data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            batch_start_time = time.time()
            
            # Use PyTorch debugging tools if enabled
            if self.debugger:
                debug_info = self.debugger.debug_training_step(
                    model=self.model,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets),
                    optimizer=self.optimizer,
                    gradient_threshold=self.config["gradient_clip"]
                )
                
                # Extract metrics from debug info
                loss = debug_info.get("loss", float('inf'))
                outputs = debug_info.get("forward_outputs")
                
                # Calculate accuracy
                if outputs is not None and outputs.dim() > 1:
                    accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                else:
                    accuracy = 0.0
                
                gradient_norm = debug_info.get("gradient_norm", 0.0)
                
            else:
                # Use performance-optimized training step
                loss, metrics = self.performance_optimizer.optimize_training_step(
                    model=self.model,
                    inputs=inputs,
                    targets=targets,
                    loss_fn=lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets),
                    optimizer=self.optimizer
                )
                
                # Extract metrics
                accuracy = self._calculate_metrics(loss, targets)["accuracy"]
                gradient_norm = self.gradient_manager.clip_gradients(self.model)
                
                # Update schedulers
                for scheduler in self.schedulers.values():
                    scheduler.step()
            
            # Calculate final metrics
            batch_time = time.time() - batch_start_time
            batch_size = inputs.size(0)
            
            metrics = {
                "loss": loss.item() if hasattr(loss, 'item') else loss,
                "accuracy": accuracy,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "gradient_norm": gradient_norm,
                "batch_time": batch_time,
                "batch_size": batch_size
            }
            
            # Record GPU metrics for multi-GPU training
            self.multi_gpu_trainer.record_gpu_metrics()
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(e, f"Batch {self.current_batch}", "train_batch")
            return {"loss": float('inf'), "accuracy": 0.0, "learning_rate": 0.0, "gradient_norm": 0.0}


# Utility context manager for null context
class nullcontext:
    def __enter__(self) -> Any:
        return None
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        pass


# Utility functions
def create_optimized_training_optimizer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    experiment_name: str = "optimized_email_sequence_training",
    debug_mode: bool = False,
    enable_pytorch_debugging: bool = False,
    performance_config: Optional[PerformanceConfig] = None,
    multi_gpu_config: Optional[MultiGPUConfig] = None,
    **kwargs
) -> OptimizedTrainingOptimizer:
    """Create an optimized training optimizer with default settings"""
    
    return OptimizedTrainingOptimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
        debug_mode=debug_mode,
        enable_pytorch_debugging=enable_pytorch_debugging,
        performance_config=performance_config,
        multi_gpu_config=multi_gpu_config,
        **kwargs
    )


async def train_model_with_optimization(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    experiment_name: str = "optimized_email_sequence_training",
    debug_mode: bool = False,
    enable_pytorch_debugging: bool = False,
    performance_config: Optional[PerformanceConfig] = None,
    multi_gpu_config: Optional[MultiGPUConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to train a model with comprehensive optimization and multi-GPU support"""
    
    optimizer = create_optimized_training_optimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
        debug_mode=debug_mode,
        enable_pytorch_debugging=enable_pytorch_debugging,
        performance_config=performance_config,
        multi_gpu_config=multi_gpu_config,
        **kwargs
    )
    
    try:
        # Benchmark performance before training
        benchmark_results = optimizer.benchmark_performance()
        print(f"Pre-training benchmark: {benchmark_results}")
        
        # Get optimal batch size
        optimal_batch_size = optimizer.get_optimal_batch_size()
        print(f"Optimal batch size: {optimal_batch_size}")
        
        # Get multi-GPU information
        training_info = optimizer.multi_gpu_trainer.get_training_info()
        print(f"Multi-GPU training info: {json.dumps(training_info, indent=2)}")
        
        # Train model
        results = await optimizer.train()
        
        # Benchmark performance after training
        post_benchmark_results = optimizer.benchmark_performance()
        print(f"Post-training benchmark: {post_benchmark_results}")
        
        return results
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    # Example usage
    
    # Simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.linear = nn.Linear(10, 2)
            self.loss_fn = nn.CrossEntropyLoss()
        
        def forward(self, x) -> Any:
            return self.linear(x)
    
    # Create dummy data
    train_data = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(100)]
    val_data = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(20)]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Train model with optimization and multi-GPU support
    model = SimpleModel()
    
    async def main():
        
    """main function."""
# Create multi-GPU configuration
        multi_gpu_config = MultiGPUConfig(
            training_mode="auto",
            enable_data_parallel=True,
            enable_distributed=False,
            enable_gpu_monitoring=True
        )
        
        results = await train_model_with_optimization(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_name="test_optimized_multi_gpu_training",
            debug_mode=True,
            enable_pytorch_debugging=True,
            multi_gpu_config=multi_gpu_config,
            max_epochs=5,
            learning_rate=0.001,
            enable_mixed_precision=True,
            enable_compile=True,
            num_workers=4
        )
        print(f"Training completed: {results}")
    
    asyncio.run(main()) 