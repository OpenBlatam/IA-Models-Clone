#!/usr/bin/env python3
"""
GPU Optimization and Mixed Precision Training for Blaze AI
Implements proper GPU utilization with descriptive naming and PEP 8 compliance
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging with descriptive names
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GPUTrainingConfiguration:
    """Configuration for GPU-optimized training with descriptive names"""
    batch_size_per_gpu: int = 32
    learning_rate_initial: float = 1e-4
    maximum_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clipping_threshold: float = 1.0
    weight_decay_factor: float = 1e-5
    mixed_precision_enabled: bool = True
    gradient_accumulation_steps: int = 1
    data_loader_workers: int = 4
    pin_memory_enabled: bool = True
    automatic_mixed_precision: bool = True


class GPUOptimizedTrainer:
    """GPU-optimized trainer with mixed precision and descriptive naming"""
    
    def __init__(self, training_config: GPUTrainingConfiguration):
        self.training_config = training_config
        self.logger = logger
        self.device = self._setup_optimal_gpu_device()
        self.gradient_scaler = None
        self._initialize_mixed_precision_training()
        
    def _setup_optimal_gpu_device(self) -> torch.device:
        """Setup optimal GPU device with automatic selection"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU")
            return torch.device("cpu")
        
        # Enable optimal CUDA settings for performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Select GPU with most available memory
        optimal_gpu_id = self._select_gpu_with_most_memory()
        selected_device = torch.device(f"cuda:{optimal_gpu_id}")
        
        # Set device and memory fraction
        torch.cuda.set_device(optimal_gpu_id)
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        gpu_name = torch.cuda.get_device_name(optimal_gpu_id)
        self.logger.info(f"Selected GPU: {gpu_name} (ID: {optimal_gpu_id})")
        
        return selected_device
    
    def _select_gpu_with_most_memory(self) -> int:
        """Select GPU with most available memory"""
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return 0
        
        # Get memory info for all GPUs
        gpu_memory_info = []
        for gpu_id in range(gpu_count):
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            free_memory = total_memory - allocated_memory
            gpu_memory_info.append((gpu_id, free_memory))
        
        # Sort by free memory and return best GPU
        gpu_memory_info.sort(key=lambda x: x[1], reverse=True)
        optimal_gpu_id = gpu_memory_info[0][0]
        
        return optimal_gpu_id
    
    def _initialize_mixed_precision_training(self):
        """Initialize mixed precision training components"""
        if (self.training_config.mixed_precision_enabled and 
            torch.cuda.is_available()):
            try:
                self.gradient_scaler = GradScaler()
                self.logger.info("Mixed precision training initialized")
            except ImportError:
                self.logger.warning("Mixed precision not available")
                self.training_config.mixed_precision_enabled = False
    
    def create_optimized_data_loader(self, dataset, 
                                   custom_batch_size: Optional[int] = None) -> DataLoader:
        """Create optimized DataLoader with GPU-friendly settings"""
        effective_batch_size = custom_batch_size or self.training_config.batch_size_per_gpu
        
        # Calculate optimal number of workers
        cpu_count = os.cpu_count() or 4
        optimal_worker_count = min(self.training_config.data_loader_workers, cpu_count)
        
        # Create optimized DataLoader
        optimized_data_loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=optimal_worker_count,
            pin_memory=self.training_config.pin_memory_enabled and torch.cuda.is_available(),
            persistent_workers=optimal_worker_count > 0,
            drop_last=True
        )
        
        self.logger.info(f"DataLoader optimized: {optimal_worker_count} workers, "
                        f"batch_size={effective_batch_size}")
        
        return optimized_data_loader
    
    def create_gpu_optimized_optimizer(self, neural_network_model: nn.Module) -> optim.Optimizer:
        """Create GPU-optimized optimizer with parameter grouping"""
        # Group parameters for different learning rates
        parameter_groups = self._create_parameter_groups(neural_network_model)
        
        # Create AdamW optimizer with optimal settings
        gpu_optimizer = optim.AdamW(
            parameter_groups,
            lr=self.training_config.learning_rate_initial,
            weight_decay=self.training_config.weight_decay_factor,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.logger.info(f"GPU optimizer created: AdamW with "
                        f"lr={self.training_config.learning_rate_initial}")
        
        return gpu_optimizer
    
    def _create_parameter_groups(self, neural_network_model: nn.Module) -> List[Dict]:
        """Create parameter groups for different learning rates"""
        # Parameters that should not have weight decay
        no_weight_decay_parameters = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        parameter_groups = [
            {
                'params': [param for name, param in neural_network_model.named_parameters()
                          if not any(no_decay in name for no_decay in no_weight_decay_parameters)],
                'weight_decay': self.training_config.weight_decay_factor
            },
            {
                'params': [param for name, param in neural_network_model.named_parameters()
                          if any(no_decay in name for no_decay in no_weight_decay_parameters)],
                'weight_decay': 0.0
            }
        ]
        
        return parameter_groups
    
    def execute_training_step(self, neural_network_model: nn.Module, 
                            training_batch: Tuple, 
                            gpu_optimizer: optim.Optimizer, 
                            loss_criterion: nn.Module) -> Dict[str, float]:
        """Execute optimized training step with mixed precision"""
        neural_network_model.train()
        
        # Move batch to GPU device
        gpu_training_batch = tuple(batch_item.to(self.device) for batch_item in training_batch)
        
        # Forward pass with mixed precision
        if (self.training_config.mixed_precision_enabled and 
            self.gradient_scaler is not None):
            with autocast():
                model_outputs = neural_network_model(*gpu_training_batch[:-1])
                training_loss = loss_criterion(model_outputs, gpu_training_batch[-1])
        else:
            model_outputs = neural_network_model(*gpu_training_batch[:-1])
            training_loss = loss_criterion(model_outputs, gpu_training_batch[-1])
        
        # Backward pass with gradient scaling
        if (self.training_config.mixed_precision_enabled and 
            self.gradient_scaler is not None):
            self.gradient_scaler.scale(training_loss).backward()
        else:
            training_loss.backward()
        
        # Apply gradient clipping
        if self.training_config.gradient_clipping_threshold > 0:
            if (self.training_config.mixed_precision_enabled and 
                self.gradient_scaler is not None):
                self.gradient_scaler.unscale_(gpu_optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                neural_network_model.parameters(), 
                self.training_config.gradient_clipping_threshold
            )
        
        # Optimizer step with gradient scaling
        if (self.training_config.mixed_precision_enabled and 
            self.gradient_scaler is not None):
            self.gradient_scaler.step(gpu_optimizer)
            self.gradient_scaler.update()
        else:
            gpu_optimizer.step()
        
        gpu_optimizer.zero_grad()
        
        return {'training_loss': training_loss.item()}
    
    def execute_validation_step(self, neural_network_model: nn.Module, 
                              validation_batch: Tuple, 
                              loss_criterion: nn.Module) -> Dict[str, float]:
        """Execute optimized validation step"""
        neural_network_model.eval()
        
        with torch.no_grad():
            gpu_validation_batch = tuple(batch_item.to(self.device) 
                                       for batch_item in validation_batch)
            
            if (self.training_config.mixed_precision_enabled and 
                self.gradient_scaler is not None):
                with autocast():
                    model_outputs = neural_network_model(*gpu_validation_batch[:-1])
                    validation_loss = loss_criterion(model_outputs, gpu_validation_batch[-1])
            else:
                model_outputs = neural_network_model(*gpu_validation_batch[:-1])
                validation_loss = loss_criterion(model_outputs, gpu_validation_batch[-1])
        
        return {'validation_loss': validation_loss.item()}
    
    def train_complete_model(self, neural_network_model: nn.Module,
                           training_data_loader: DataLoader,
                           validation_data_loader: DataLoader,
                           loss_criterion: nn.Module,
                           model_save_path: str) -> Dict[str, Any]:
        """Complete training loop with GPU optimization"""
        # Move model to GPU
        neural_network_model = neural_network_model.to(self.device)
        
        # Create optimizer and scheduler
        gpu_optimizer = self.create_gpu_optimized_optimizer(neural_network_model)
        learning_rate_scheduler = self._create_learning_rate_scheduler(gpu_optimizer)
        
        # Training state variables
        best_validation_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        self.logger.info("Starting GPU-optimized training...")
        
        for current_epoch in range(self.training_config.maximum_epochs):
            # Training phase
            training_metrics = self._train_single_epoch(
                neural_network_model, training_data_loader, gpu_optimizer, loss_criterion
            )
            
            # Validation phase
            validation_metrics = self._validate_single_epoch(
                neural_network_model, validation_data_loader, loss_criterion
            )
            
            # Update learning rate
            learning_rate_scheduler.step(validation_metrics['validation_loss'])
            
            # Combine metrics
            epoch_metrics = {**training_metrics, **validation_metrics}
            training_history.append(epoch_metrics)
            
            # Log progress
            current_learning_rate = gpu_optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {current_epoch + 1}/{self.training_config.maximum_epochs} - "
                f"Train Loss: {epoch_metrics['training_loss']:.4f}, "
                f"Val Loss: {epoch_metrics['validation_loss']:.4f}, "
                f"LR: {current_learning_rate:.2e}"
            )
            
            # Early stopping logic
            if self.training_config.early_stopping_patience > 0:
                if epoch_metrics['validation_loss'] < best_validation_loss:
                    best_validation_loss = epoch_metrics['validation_loss']
                    patience_counter = 0
                    
                    # Save best model
                    self._save_model_checkpoint(
                        neural_network_model, gpu_optimizer, current_epoch, 
                        epoch_metrics, model_save_path
                    )
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.training_config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {current_epoch + 1}")
                    break
        
        return {
            'training_history': training_history,
            'best_validation_loss': best_validation_loss,
            'final_epoch': current_epoch + 1
        }
    
    def _create_learning_rate_scheduler(self, gpu_optimizer: optim.Optimizer):
        """Create learning rate scheduler"""
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            gpu_optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
        
        return scheduler
    
    def _train_single_epoch(self, neural_network_model: nn.Module,
                           training_data_loader: DataLoader,
                           gpu_optimizer: optim.Optimizer,
                           loss_criterion: nn.Module) -> Dict[str, float]:
        """Train for single epoch with gradient accumulation"""
        neural_network_model.train()
        total_training_loss = 0.0
        batch_count = len(training_data_loader)
        
        for batch_index, training_batch in enumerate(training_data_loader):
            # Gradient accumulation
            if (batch_index % self.training_config.gradient_accumulation_steps == 0):
                step_loss = self.execute_training_step(
                    neural_network_model, training_batch, gpu_optimizer, loss_criterion
                )
                total_training_loss += step_loss['training_loss']
                
                # Log progress
                if batch_index % 10 == 0:
                    self.logger.info(f"Batch {batch_index}/{batch_count}, "
                                   f"Loss: {step_loss['training_loss']:.4f}")
        
        average_training_loss = total_training_loss / batch_count
        return {'training_loss': average_training_loss}
    
    def _validate_single_epoch(self, neural_network_model: nn.Module,
                              validation_data_loader: DataLoader,
                              loss_criterion: nn.Module) -> Dict[str, float]:
        """Validate for single epoch"""
        neural_network_model.eval()
        total_validation_loss = 0.0
        batch_count = len(validation_data_loader)
        
        with torch.no_grad():
            for validation_batch in validation_data_loader:
                step_loss = self.execute_validation_step(
                    neural_network_model, validation_batch, loss_criterion
                )
                total_validation_loss += step_loss['validation_loss']
        
        average_validation_loss = total_validation_loss / batch_count
        return {'validation_loss': average_validation_loss}
    
    def _save_model_checkpoint(self, neural_network_model: nn.Module,
                              gpu_optimizer: optim.Optimizer,
                              current_epoch: int,
                              epoch_metrics: Dict,
                              model_save_path: str):
        """Save model checkpoint with comprehensive information"""
        checkpoint_data = {
            'epoch': current_epoch,
            'model_state_dict': neural_network_model.state_dict(),
            'optimizer_state_dict': gpu_optimizer.state_dict(),
            'metrics': epoch_metrics,
            'training_config': self.training_config.__dict__,
            'gpu_device_info': str(self.device)
        }
        
        # Create timestamped filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"{model_save_path}_epoch_{current_epoch}_{timestamp}.pt"
        
        torch.save(checkpoint_data, checkpoint_filename)
        self.logger.info(f"Model checkpoint saved: {checkpoint_filename}")
    
    def get_gpu_memory_utilization(self) -> Dict[str, Any]:
        """Get comprehensive GPU memory utilization information"""
        memory_info = {}
        
        if torch.cuda.is_available():
            gpu_memory_details = []
            
            for gpu_id in range(torch.cuda.device_count()):
                allocated_memory = torch.cuda.memory_allocated(gpu_id)
                reserved_memory = torch.cuda.memory_reserved(gpu_id)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                
                gpu_memory_details.append({
                    'gpu_id': gpu_id,
                    'allocated_memory_bytes': allocated_memory,
                    'reserved_memory_bytes': reserved_memory,
                    'total_memory_bytes': total_memory,
                    'free_memory_bytes': total_memory - reserved_memory,
                    'utilization_percentage': (reserved_memory / total_memory) * 100
                })
            
            memory_info['gpu_memory'] = gpu_memory_details
        
        return memory_info
    
    def optimize_batch_size_for_memory(self, neural_network_model: nn.Module,
                                     sample_input_tensor: torch.Tensor,
                                     maximum_memory_usage: float = 0.8) -> int:
        """Dynamically optimize batch size based on GPU memory"""
        if not torch.cuda.is_available():
            return self.training_config.batch_size_per_gpu
        
        neural_network_model.eval()
        gpu_device = next(neural_network_model.parameters()).device
        
        # Start with configured batch size
        current_batch_size = self.training_config.batch_size_per_gpu
        
        while current_batch_size > 1:
            try:
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Test current batch size
                test_input_batch = sample_input_tensor.repeat(current_batch_size, 1, 1, 1)
                test_input_batch = test_input_batch.to(gpu_device)
                
                with torch.no_grad():
                    _ = neural_network_model(test_input_batch)
                
                # Check memory usage
                allocated_memory = torch.cuda.memory_allocated(gpu_device)
                total_memory = torch.cuda.get_device_properties(gpu_device).total_memory
                memory_usage_ratio = allocated_memory / total_memory
                
                if memory_usage_ratio <= maximum_memory_usage:
                    self.logger.info(f"Optimal batch size found: {current_batch_size}")
                    return current_batch_size
                
                current_batch_size //= 2
                
            except RuntimeError as runtime_error:
                if "out of memory" in str(runtime_error):
                    current_batch_size //= 2
                    torch.cuda.empty_cache()
                else:
                    raise runtime_error
        
        self.logger.warning(f"Could not find optimal batch size, using: {current_batch_size}")
        return max(1, current_batch_size)


def create_gpu_optimized_configuration() -> GPUTrainingConfiguration:
    """Create GPU-optimized configuration based on system capabilities"""
    configuration = GPUTrainingConfiguration()
    
    # Auto-adjust based on GPU capabilities
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory_gb >= 24:  # High-end GPU (RTX 4090, A100)
            configuration.batch_size_per_gpu = 64
            configuration.mixed_precision_enabled = True
            configuration.automatic_mixed_precision = True
        elif gpu_memory_gb >= 12:  # Mid-range GPU (RTX 3080, RTX 4080)
            configuration.batch_size_per_gpu = 32
            configuration.mixed_precision_enabled = True
            configuration.automatic_mixed_precision = True
        else:  # Low-end GPU (RTX 3060, GTX 1660)
            configuration.batch_size_per_gpu = 16
            configuration.mixed_precision_enabled = False
            configuration.automatic_mixed_precision = False
    
    return configuration


def main():
    """Main execution function"""
    print("Starting Blaze AI GPU Optimization and Mixed Precision Training...")
    
    # Create configuration
    gpu_config = create_gpu_optimized_configuration()
    print(f"GPU Configuration: {gpu_config}")
    
    # Create trainer
    gpu_trainer = GPUOptimizedTrainer(gpu_config)
    
    # Get GPU memory information
    memory_utilization = gpu_trainer.get_gpu_memory_utilization()
    print(f"GPU Memory Utilization: {memory_utilization}")
    
    print("GPU optimization setup completed successfully!")


if __name__ == "__main__":
    main()
