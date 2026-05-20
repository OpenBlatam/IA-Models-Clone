from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import psutil
import GPUtil
from pathlib import Path
import json
import traceback
from core.training_logger import (
from core.error_handling import ErrorHandler, ModelError, DataError
from core.early_stopping import EarlyStopping, EarlyStoppingConfig
from core.learning_rate_scheduling import LRScheduler, LRSchedulerConfig
from core.gradient_management import GradientManager, GradientConfig
from core.pytorch_debugging import PyTorchDebugger, create_pytorch_debugger, debug_training_session
    import torch.nn as nn
from typing import Any, List, Dict, Optional
import logging
"""
Enhanced Training Optimizer with Comprehensive Logging

Advanced training optimizer that integrates with the training logger
to provide detailed tracking of training progress, errors, and performance metrics.
"""


    TrainingLogger, TrainingEventType, LogLevel, 
    create_training_logger, TrainingMetrics
)


class EnhancedTrainingOptimizer:
    """Enhanced training optimizer with comprehensive logging and error handling"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        experiment_name: str = "email_sequence_training",
        log_dir: str = "logs",
        log_level: str = "INFO",
        device: str = "auto",
        debug_mode: bool = False,
        enable_pytorch_debugging: bool = False,
        **kwargs
    ):
        """Initialize the enhanced training optimizer"""
        
        # Setup device
        match device:
    case "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
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
            **kwargs
        }
        
        # Initialize optimizers and schedulers
        self._setup_optimizers()
        self._setup_schedulers()
        self._setup_early_stopping()
        self._setup_gradient_management()
        
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
        
        # Resource monitoring
        self.memory_usage = []
        self.gpu_usage = []
        
        self.logger.log_info(f"Enhanced training optimizer initialized on device: {self.device}")
        self.logger.log_info(f"Training configuration: {json.dumps(self.config, indent=2)}")
        
        if self.debugger:
            self.logger.log_info("PyTorch debugging tools enabled")
    
    def _setup_optimizers(self) -> Any:
        """Setup optimizers"""
        
        try:
            # Main optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
            
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
            
            self.logger.log_info("Optimizers initialized successfully")
            
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
    
    def _train_batch(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Train a single batch with debugging support"""
        
        try:
            inputs, targets = batch_data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
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
                # Standard training without debugging
                # Forward pass
                self.model.train()
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self._calculate_metrics(outputs, targets)["loss"]
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_tensor = torch.tensor(loss, requires_grad=True)
                loss_tensor.backward()
                
                # Gradient management
                gradient_norm = self.gradient_manager.clip_gradients(self.model)
                
                # Optimizer step
                self.optimizer.step()
                
                # Update schedulers
                for scheduler in self.schedulers.values():
                    scheduler.step()
                
                # Calculate accuracy
                if outputs.dim() > 1:
                    accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                else:
                    accuracy = ((outputs > 0.5) == targets).float().mean().item()
            
            # Calculate final metrics
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "gradient_norm": gradient_norm
            }
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(e, f"Batch {self.current_batch}", "train_batch")
            return {"loss": float('inf'), "accuracy": 0.0, "learning_rate": 0.0, "gradient_norm": 0.0}
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate the model for one epoch with debugging support"""
        
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
        """Save model checkpoint"""
        
        try:
            checkpoint_dir = Path(self.config["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "config": self.config,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch
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
        """Load model checkpoint"""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
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
        """Train for one epoch with debugging support"""
        
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
                            batch_start_time = time.time()
                            
                            # Train batch
                            batch_metrics = self._train_batch(batch_data)
                            
                            # Update epoch metrics
                            epoch_loss += batch_metrics["loss"]
                            epoch_accuracy += batch_metrics["accuracy"]
                            num_batches += 1
                            
                            # Update step counter
                            self.current_step += 1
                            
                            # Log batch metrics
                            batch_duration = time.time() - batch_start_time
                            memory_usage, gpu_usage = self._get_resource_usage()
                            
                            batch_metrics.update({
                                "training_time": batch_duration,
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
        """Main training loop with debugging support"""
        
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
                    
                    # Add debug summary if available
                    if debugger:
                        training_summary["debug_summary"] = debugger.get_debug_summary()
                    
                    self.logger.end_training(training_summary)
                    
                    return training_summary
                
        except Exception as e:
            self.logger.log_error(e, "Training", "train")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        
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
        
        # Add debug summary if available
        if self.debugger:
            summary["debug_summary"] = self.debugger.get_debug_summary()
        
        return summary
    
    def create_training_visualizations(self, save_path: str = None):
        """Create comprehensive training visualizations"""
        
        try:
            self.logger.create_visualizations(save_path)
        except Exception as e:
            self.logger.log_error(e, "Visualization creation", "create_training_visualizations")
    
    def cleanup(self) -> Any:
        """Cleanup resources"""
        
        try:
            # Save debug report if debugger is available
            if self.debugger:
                self.debugger.save_debug_report()
            
            self.logger.cleanup()
            self.logger.log_info("Training optimizer cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")


# Utility context manager for null context
class nullcontext:
    def __enter__(self) -> Any:
        return None
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        pass


# Utility functions
def create_enhanced_training_optimizer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    experiment_name: str = "email_sequence_training",
    debug_mode: bool = False,
    enable_pytorch_debugging: bool = False,
    **kwargs
) -> EnhancedTrainingOptimizer:
    """Create an enhanced training optimizer with default settings"""
    
    return EnhancedTrainingOptimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
        debug_mode=debug_mode,
        enable_pytorch_debugging=enable_pytorch_debugging,
        **kwargs
    )


async def train_model_with_logging(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    experiment_name: str = "email_sequence_training",
    debug_mode: bool = False,
    enable_pytorch_debugging: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to train a model with comprehensive logging and debugging"""
    
    optimizer = create_enhanced_training_optimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
        debug_mode=debug_mode,
        enable_pytorch_debugging=enable_pytorch_debugging,
        **kwargs
    )
    
    try:
        results = await optimizer.train()
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
    
    # Train model with debugging
    model = SimpleModel()
    
    async def main():
        
    """main function."""
results = await train_model_with_logging(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_name="test_training_with_debugging",
            debug_mode=True,
            enable_pytorch_debugging=True,
            max_epochs=5,
            learning_rate=0.001
        )
        print(f"Training completed: {results}")
    
    asyncio.run(main()) 