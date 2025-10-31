#!/usr/bin/env python3
"""
Real-time Training Visualization Demo with Gradio
Live demonstration of training dynamics with numerical stability monitoring.
Enhanced with robust error handling and input validation.
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
import queue
import json
import logging
import traceback
import warnings
from pathlib import Path

# Import our gradient clipping and NaN handling system
from gradient_clipping_nan_handling import (
    GradientClippingConfig,
    NaNHandlingConfig,
    NumericalStabilityManager,
    ClippingType,
    NaNHandlingType
)

# Import centralized logging configuration
from logging_config import (
    setup_logging, get_logger, log_training_step, log_numerical_issue,
    log_system_event, log_error_with_context, log_performance_metrics
)

# Setup comprehensive logging
loggers = setup_logging(
    log_dir="logs",
    log_level="INFO",
    enable_file_logging=True,
    enable_console_logging=True
)

# Get specific loggers
logger = loggers['main']
training_logger = loggers['training']
error_logger = loggers['errors']
stability_logger = loggers['stability']
system_logger = loggers['system']

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class RealTimeTrainingDemo:
    """Real-time training demonstration with live visualization and error handling."""

    def __init__(self):
        self.training_thread = None
        self.stop_training = False
        self.training_queue = queue.Queue()
        self.current_model = None
        self.current_optimizer = None
        self.stability_manager = None
        self.training_data = {
            'steps': [],
            'losses': [],
            'stability_scores': [],
            'clipping_ratios': [],
            'nan_counts': [],
            'inf_counts': [],
            'overflow_counts': [],
            'gradient_norms': [],
            'learning_rates': []
        }
        self.current_step = 0
        self.is_training = False
        self.error_count = 0
        self.max_errors = 10
        self.training_lock = threading.Lock()
        
    def _validate_model_parameters(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Validate model architecture parameters."""
        if input_dim < 1 or input_dim > 1000:
            raise ValidationError("Input dimension must be between 1 and 1000")
        if hidden_dim < 1 or hidden_dim > 2000:
            raise ValidationError("Hidden dimension must be between 1 and 2000")
        if output_dim < 1 or output_dim > 1000:
            raise ValidationError("Output dimension must be between 1 and 1000")
        if hidden_dim < input_dim // 4:
            raise ValidationError("Hidden dimension should be at least 1/4 of input dimension")
        if hidden_dim < output_dim:
            raise ValidationError("Hidden dimension should be at least as large as output dimension")
    
    def _validate_training_parameters(self, batch_size: int, num_epochs: int, 
                                    issue_frequency: float, learning_rate: float) -> None:
        """Validate training parameters."""
        if batch_size < 1 or batch_size > 512:
            raise ValidationError("Batch size must be between 1 and 512")
        if num_epochs < 1 or num_epochs > 100:
            raise ValidationError("Number of epochs must be between 1 and 100")
        if not (0 <= issue_frequency <= 1):
            raise ValidationError("Issue frequency must be between 0 and 1")
        if learning_rate <= 0 or learning_rate > 1:
            raise ValidationError("Learning rate must be between 0.0001 and 1")
    
    def _validate_stability_parameters(self, max_norm: float, adaptive_threshold: float) -> None:
        """Validate stability configuration parameters."""
        if max_norm <= 0 or max_norm > 100:
            raise ValidationError("Max norm must be between 0.001 and 100")
        if adaptive_threshold <= 0 or adaptive_threshold > 10:
            raise ValidationError("Adaptive threshold must be between 0.001 and 10")
    
    def _handle_error(self, error: Exception, operation: str) -> str:
        """Centralized error handling with logging and user-friendly messages."""
        self.error_count += 1
        error_msg = f"❌ Error in {operation}: {str(error)}"
        
        # Log detailed error information
        logger.error(f"Error in {operation}: {str(error)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Provide specific guidance based on error type
        if isinstance(error, ValidationError):
            error_msg += "\n\n💡 **Validation Error**: Please check your input parameters and try again."
        elif isinstance(error, RuntimeError):
            error_msg += "\n\n💡 **Runtime Error**: This might be due to insufficient memory or GPU issues."
        elif isinstance(error, ValueError):
            error_msg += "\n\n💡 **Value Error**: Please verify your input values are within valid ranges."
        elif isinstance(error, OSError):
            error_msg += "\n\n💡 **System Error**: Check if you have sufficient disk space and permissions."
        elif isinstance(error, torch.cuda.OutOfMemoryError):
            error_msg += "\n\n💡 **Memory Error**: Try reducing batch size or model dimensions."
        else:
            error_msg += "\n\n💡 **Unexpected Error**: Please try again or contact support if the issue persists."
        
        # Add error count information
        if self.error_count >= self.max_errors:
            error_msg += f"\n\n⚠️ **Warning**: You've encountered {self.error_count} errors. Consider restarting the demo."
        
        return error_msg
    
    def _safe_model_creation(self, model_func, *args, **kwargs):
        """Safely create models with error handling."""
        try:
            return model_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Model creation failed: {str(e)}")
            raise
    
    def create_training_model(self, model_type: str, input_dim: int, hidden_dim: int,
                            output_dim: int, learning_rate: float) -> str:
        """Create a model for real-time training demonstration with validation."""
        try:
            # Validate parameters
            self._validate_model_parameters(input_dim, hidden_dim, output_dim)
            self._validate_training_parameters(32, 5, 0.2, learning_rate)
            
            # Check if training is in progress
            if self.is_training:
                return "❌ Cannot create model while training is in progress. Please stop training first."
            
            # Check if model already exists
            if self.current_model is not None:
                return "ℹ️ Model already exists. Use 'Reset Model' to create a new one."
            
            # Create model based on type
            if model_type == "Feedforward":
                self.current_model = self._safe_model_creation(
                    lambda: nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim // 2, output_dim)
                    )
                )
            elif model_type == "Deep":
                layers = []
                current_dim = input_dim
                for i in range(4):
                    layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ])
                    current_dim = hidden_dim
                layers.append(nn.Linear(current_dim, output_dim))
                self.current_model = self._safe_model_creation(lambda: nn.Sequential(*layers))
            elif model_type == "Wide":
                self.current_model = self._safe_model_creation(
                    lambda: nn.Sequential(
                        nn.Linear(input_dim, hidden_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, output_dim)
                    )
                )
            else:
                raise ValidationError(f"Unknown model type: {model_type}")
            
            # Initialize weights safely
            try:
                for module in self.current_model.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.constant_(module.bias, 0.1)
            except Exception as e:
                logger.warning(f"Weight initialization warning: {str(e)}")
                # Continue with default weights
            
            # Create optimizer safely
            try:
                self.current_optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
            except Exception as e:
                logger.error(f"Optimizer creation failed: {str(e)}")
                raise RuntimeError("Failed to create optimizer")
            
            # Reset training data
            self.reset_training_data()
            
            param_count = sum(p.numel() for p in self.current_model.parameters())
            return f"✅ {model_type} model created successfully!\nParameters: {param_count:,}\nLearning rate: {learning_rate}"
            
        except Exception as e:
            return self._handle_error(e, "model creation")
    
    def configure_stability_manager(self, clipping_type: str, max_norm: float,
                                  nan_handling: str, adaptive_threshold: float) -> str:
        """Configure the numerical stability manager with validation."""
        try:
            if self.current_model is None:
                raise ValidationError("Please create a model first")
            
            # Validate parameters
            self._validate_stability_parameters(max_norm, adaptive_threshold)
            
            # Validate enum values
            try:
                clipping_enum = getattr(ClippingType, clipping_type.upper())
            except AttributeError:
                raise ValidationError(f"Invalid clipping type: {clipping_type}")
            
            try:
                nan_enum = getattr(NaNHandlingType, nan_handling.upper())
            except AttributeError:
                raise ValidationError(f"Invalid NaN handling type: {nan_handling}")
            
            # Create stability configuration
            clipping_config = GradientClippingConfig(
                clipping_type=clipping_enum,
                max_norm=max_norm,
                adaptive_threshold=adaptive_threshold,
                monitor_clipping=True,
                log_clipping_stats=True,
                save_clipping_history=True
            )
            
            nan_config = NaNHandlingConfig(
                handling_type=nan_enum,
                detect_nan=True,
                detect_inf=True,
                detect_overflow=True,
                monitor_nan=True,
                log_nan_stats=True,
                save_nan_history=True
            )
            
            # Create stability manager safely
            try:
                self.stability_manager = NumericalStabilityManager(clipping_config, nan_config)
            except Exception as e:
                logger.error(f"Stability manager creation failed: {str(e)}")
                raise RuntimeError("Failed to create stability manager")
            
            return f"✅ Stability manager configured!\nClipping: {clipping_type}\nMax norm: {max_norm}\nNaN handling: {nan_handling}"
            
        except Exception as e:
            return self._handle_error(e, "stability configuration")
    
    def reset_model(self) -> str:
        """Reset the current model and start fresh."""
        try:
            # Stop training if running
            if self.is_training:
                self.stop_training_loop()
                time.sleep(1)  # Wait for training to stop
            
            with self.training_lock:
                self.current_model = None
                self.current_optimizer = None
                self.stability_manager = None
                self.reset_training_data()
                self.error_count = 0  # Reset error count
            
            return "✅ Model reset successfully! Ready to create a new model."
        except Exception as e:
            return self._handle_error(e, "model reset")
    
    def reset_training_data(self):
        """Reset training data safely."""
        try:
            with self.training_lock:
                self.training_data = {
                    'steps': [],
                    'losses': [],
                    'stability_scores': [],
                    'clipping_ratios': [],
                    'nan_counts': [],
                    'inf_counts': [],
                    'overflow_counts': [],
                    'gradient_norms': [],
                    'learning_rates': []
                }
                self.current_step = 0
        except Exception as e:
            logger.error(f"Failed to reset training data: {str(e)}")
    
    def start_training(self, batch_size: int, num_epochs: int, introduce_issues: bool,
                       issue_frequency: float, progress=gr.Progress()) -> str:
        """Start real-time training in a separate thread with validation."""
        try:
            # Validate parameters
            self._validate_training_parameters(batch_size, num_epochs, issue_frequency, 0.01)
            
            if self.current_model is None or self.stability_manager is None:
                raise ValidationError("Please create a model and configure stability manager first")
            
            if self.is_training:
                return "❌ Training is already in progress."
            
            # Check system resources
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"GPU memory cleanup warning: {str(e)}")
            
            # Check memory constraints
            if batch_size > 256:
                logger.warning(f"Large batch size ({batch_size}) may cause memory issues")
            
            with self.training_lock:
                self.is_training = True
                self.stop_training = False
            
            # Start training thread
            self.training_thread = threading.Thread(
                target=self._training_loop,
                args=(batch_size, num_epochs, introduce_issues, issue_frequency, progress)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            
            return "✅ Training started! Check the visualization tab for live updates."
            
        except Exception as e:
            return self._handle_error(e, "starting training")
    
    def stop_training_loop(self) -> str:
        """Stop the training loop safely."""
        try:
            if not self.is_training:
                return "ℹ️ No training in progress."
            
            with self.training_lock:
                self.stop_training = True
                self.is_training = False
            
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=5)
                if self.training_thread.is_alive():
                    logger.warning("Training thread did not stop gracefully")
            
            return "✅ Training stopped successfully."
            
        except Exception as e:
            return self._handle_error(e, "stopping training")
    
    def _training_loop(self, batch_size: int, num_epochs: int, introduce_issues: bool,
                       issue_frequency: float, progress):
        """Main training loop running in separate thread with error handling."""
        try:
            model = self.current_model
            optimizer = self.current_optimizer
            stability_manager = self.stability_manager
            
            if model is None or optimizer is None or stability_manager is None:
                logger.error("Training loop: Required components are missing")
                return
            
            # Generate synthetic dataset safely
            try:
                if model[0].in_features == 1:  # Regression
                    x = torch.linspace(-5, 5, 1000).unsqueeze(1)
                    y = 0.5 * x**2 + 0.3 * x + 0.1 + 0.1 * torch.randn_like(x)
                else:  # Classification or other
                    x = torch.randn(1000, model[0].in_features)
                    y = torch.randint(0, model[-1].out_features, (1000,))
                    if model[-1].out_features == 1:
                        y = y.float()
                
                dataset = torch.utils.data.TensorDataset(x, y)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            except Exception as e:
                logger.error(f"Failed to create dataset: {str(e)}")
                return
            
            total_steps = len(dataloader) * num_epochs
            current_step = 0
            
            for epoch in range(num_epochs):
                if self.stop_training:
                    break
                
                for batch_x, batch_y in dataloader:
                    if self.stop_training:
                        break
                    
                    try:
                        # Forward pass with comprehensive error handling
                        try:
                            optimizer.zero_grad()
                            output = self._safe_forward_pass(model, batch_x)
                        except Exception as e:
                            logger.error(f"Forward pass failed for step {current_step + 1}: {str(e)}")
                            continue
                        
                        # Calculate loss with error handling
                        try:
                            task_type = "regression" if model[-1].out_features == 1 else "classification"
                            loss = self._safe_loss_calculation(output, batch_y, task_type)
                        except Exception as e:
                            logger.error(f"Loss calculation failed for step {current_step + 1}: {str(e)}")
                            continue
                        
                        # Backward pass with error handling
                        try:
                            self._safe_backward_pass(loss)
                        except Exception as e:
                            logger.error(f"Backward pass failed for step {current_step + 1}: {str(e)}")
                            continue
                        
                        # Introduce numerical issues if requested
                        if introduce_issues and np.random.random() < issue_frequency:
                            self._introduce_numerical_issues(model)
                        
                        # Apply stability measures safely
                        try:
                            stability_result = stability_manager.step(model, loss, optimizer)
                        except Exception as e:
                            logger.warning(f"Stability step warning: {str(e)}")
                            # Create a basic stability result
                            stability_result = {
                                'stability_score': 0.5,
                                'clipping_stats': {'clipping_ratio': 0.0},
                                'nan_stats': {
                                    'nan_detected': False,
                                    'inf_detected': False,
                                    'overflow_detected': False
                                }
                            }
                        
                        # Optimizer step safely using new safe method
                        try:
                            self._safe_optimizer_step(optimizer)
                        except Exception as e:
                            logger.warning(f"Optimizer step warning: {str(e)}")
                            continue
                        
                        # Update training data safely
                        with self.training_lock:
                            self.current_step += 1
                            self.training_data['steps'].append(self.current_step)
                            self.training_data['losses'].append(loss.item())
                            self.training_data['stability_scores'].append(stability_result['stability_score'])
                            self.training_data['clipping_ratios'].append(stability_result['clipping_stats'].get('clipping_ratio', 0.0))
                            self.training_data['nan_counts'].append(1 if stability_result['nan_stats']['nan_detected'] else 0)
                            self.training_data['inf_counts'].append(1 if stability_result['nan_stats']['inf_detected'] else 0)
                            self.training_data['overflow_counts'].append(1 if stability_result['nan_stats']['overflow_detected'] else 0)
                            
                            # Calculate gradient norm safely
                            try:
                                total_norm = 0.0
                                for p in model.parameters():
                                    if p.grad is not None:
                                        param_norm = p.grad.data.norm(2)
                                        total_norm += param_norm.item() ** 2
                                total_norm = total_norm ** 0.5
                                self.training_data['gradient_norms'].append(total_norm)
                            except Exception as e:
                                logger.warning(f"Gradient norm calculation warning: {str(e)}")
                                self.training_data['gradient_norms'].append(0.0)
                            
                            # Current learning rate
                            try:
                                current_lr = optimizer.param_groups[0]['lr']
                                self.training_data['learning_rates'].append(current_lr)
                            except Exception as e:
                                logger.warning(f"Learning rate access warning: {str(e)}")
                                self.training_data['learning_rates'].append(0.01)
                        
                        # Update progress
                        current_step += 1
                        progress(current_step / total_steps, desc=f"Training step {current_step}/{total_steps}")
                        
                        # Small delay for visualization
                        time.sleep(0.01)
                        
                        # Limit data points for performance
                        if len(self.training_data['steps']) > 1000:
                            with self.training_lock:
                                for key in self.training_data:
                                    self.training_data[key] = self.training_data[key][-500:]
                    
                    except Exception as step_error:
                        logger.warning(f"Training step {current_step + 1} failed: {str(step_error)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Training loop error: {str(e)}")
        finally:
            with self.training_lock:
                self.is_training = False
    
    def _introduce_numerical_issues(self, model: nn.Module):
        """Introduce numerical issues in gradients safely."""
        try:
            for param in model.parameters():
                if param.grad is not None:
                    if np.random.random() < 0.5:
                        # Introduce NaN
                        param.grad.data[0, 0] = float('nan')
                    else:
                        # Introduce overflow
                        param.grad.data *= 1e6
        except Exception as e:
            logger.warning(f"Failed to introduce numerical issues: {str(e)}")
    
    def get_training_status(self) -> str:
        """Get current training status with error information."""
        try:
            if not self.is_training:
                return "Training not in progress."
            
            if not self.training_data['steps']:
                return "Training started, waiting for first step..."
            
            with self.training_lock:
                current_loss = self.training_data['losses'][-1]
                current_stability = self.training_data['stability_scores'][-1]
                current_step = self.training_data['steps'][-1]
                error_count = self.error_count
            
            status = f"🔄 Training in progress...\n"
            status += f"Current step: {current_step}\n"
            status += f"Current loss: {current_loss:.6f}\n"
            status += f"Current stability: {current_stability:.4f}\n"
            status += f"Total steps completed: {len(self.training_data['steps'])}\n"
            status += f"Error count: {error_count}"
            
            if error_count > 0:
                status += f"\n⚠️ **Warning**: {error_count} errors encountered"
            
            return status
            
        except Exception as e:
            return f"❌ Error getting training status: {str(e)}"
    
    def generate_live_plots(self) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        """Generate live training plots with error handling."""
        try:
            with self.training_lock:
                if not self.training_data['steps']:
                    # Return empty plots with helpful messages
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.text(0.5, 0.5, 'No training data available\nStart training first!', 
                             ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                    ax1.set_title('Training Loss & Stability')
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.text(0.5, 0.5, 'No training data available\nStart training first!', 
                             ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                    ax2.set_title('Gradient Analysis')
                    
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    ax3.text(0.5, 0.5, 'No training data available\nStart training first!', 
                             ha='center', va='center', transform=ax3.transAxes, fontsize=14)
                    ax3.set_title('Numerical Issues')
                    
                    return fig1, fig2, fig3
                
                # Validate data before plotting
                if len(self.training_data['steps']) != len(self.training_data['losses']):
                    raise ValueError("Training data arrays have mismatched lengths")
                
                # Plot 1: Training Loss & Stability
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Loss over time with error handling
                try:
                    ax1.plot(self.training_data['steps'], self.training_data['losses'], 'b-', linewidth=2, alpha=0.8)
                    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Training Step')
                    ax1.set_ylabel('Loss')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_yscale('log')
                except Exception as e:
                    logger.error(f"Failed to plot loss: {str(e)}")
                    ax1.text(0.5, 0.5, 'Failed to plot loss data', ha='center', va='center', transform=ax1.transAxes)
                
                # Stability score over time
                try:
                    ax2.plot(self.training_data['steps'], self.training_data['stability_scores'], 'g-', linewidth=2, alpha=0.8)
                    ax2.set_title('Numerical Stability Score Over Time', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Training Step')
                    ax2.set_ylabel('Stability Score')
                    ax2.set_ylim(0, 1)
                    ax2.grid(True, alpha=0.3)
                except Exception as e:
                    logger.error(f"Failed to plot stability: {str(e)}")
                    ax2.text(0.5, 0.5, 'Failed to plot stability data', ha='center', va='center', transform=ax2.transAxes)
                
                fig1.tight_layout()
                
                # Plot 2: Gradient Analysis
                fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Gradient norms over time
                try:
                    ax1.plot(self.training_data['steps'], self.training_data['gradient_norms'], 'r-', linewidth=2, alpha=0.8)
                    ax1.set_title('Gradient Norms Over Time', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Training Step')
                    ax1.set_ylabel('Gradient Norm')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_yscale('log')
                except Exception as e:
                    logger.error(f"Failed to plot gradient norms: {str(e)}")
                    ax1.text(0.5, 0.5, 'Failed to plot gradient norm data', ha='center', va='center', transform=ax1.transAxes)
                
                # Clipping ratios over time
                try:
                    ax2.plot(self.training_data['steps'], self.training_data['clipping_ratios'], 'orange', linewidth=2, alpha=0.8)
                    ax2.set_title('Gradient Clipping Ratios Over Time', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Training Step')
                    ax2.set_ylabel('Clipping Ratio')
                    ax2.grid(True, alpha=0.3)
                except Exception as e:
                    logger.error(f"Failed to plot clipping ratios: {str(e)}")
                    ax2.text(0.5, 0.5, 'Failed to plot clipping ratio data', ha='center', va='center', transform=ax2.transAxes)
                
                fig2.tight_layout()
                
                # Plot 3: Numerical Issues
                fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Numerical issues over time
                try:
                    ax1.plot(self.training_data['steps'], self.training_data['nan_counts'], 'r-', label='NaN', alpha=0.8, linewidth=2)
                    ax1.plot(self.training_data['steps'], self.training_data['inf_counts'], 'orange', label='Inf', alpha=0.8, linewidth=2)
                    ax1.plot(self.training_data['steps'], self.training_data['overflow_counts'], 'yellow', label='Overflow', alpha=0.8, linewidth=2)
                    ax1.set_title('Numerical Issues Over Time', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Training Step')
                    ax1.set_ylabel('Issue Count')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                except Exception as e:
                    logger.error(f"Failed to plot numerical issues: {str(e)}")
                    ax1.text(0.5, 0.5, 'Failed to plot numerical issues data', ha='center', va='center', transform=ax1.transAxes)
                
                # Cumulative issues
                try:
                    cumulative_nan = np.cumsum(self.training_data['nan_counts'])
                    cumulative_inf = np.cumsum(self.training_data['inf_counts'])
                    cumulative_overflow = np.cumsum(self.training_data['overflow_counts'])
                    
                    ax2.plot(self.training_data['steps'], cumulative_nan, 'r-', label='Cumulative NaN', alpha=0.8, linewidth=2)
                    ax2.plot(self.training_data['steps'], cumulative_inf, 'orange', label='Cumulative Inf', alpha=0.8, linewidth=2)
                    ax2.plot(self.training_data['steps'], cumulative_overflow, 'yellow', label='Cumulative Overflow', alpha=0.8, linewidth=2)
                    ax2.set_title('Cumulative Numerical Issues', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Training Step')
                    ax2.set_ylabel('Cumulative Count')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                except Exception as e:
                    logger.error(f"Failed to plot cumulative issues: {str(e)}")
                    ax2.text(0.5, 0.5, 'Failed to plot cumulative issues data', ha='center', va='center', transform=ax2.transAxes)
                
                fig3.tight_layout()
                
                return fig1, fig2, fig3
                
        except Exception as e:
            logger.error(f"Failed to generate plots: {str(e)}")
            # Return error plots
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.text(0.5, 0.5, f'Failed to generate plots:\n{str(e)}', 
                     ha='center', va='center', transform=ax1.transAxes, fontsize=12, color='red')
            ax1.set_title('Plot Generation Error')
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.text(0.5, 0.5, 'Please check your training data\nand try again', 
                     ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('No Data Available')
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.text(0.5, 0.5, 'Please check your training data\nand try again', 
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('No Data Available')
            
            return fig1, fig2, fig3
    
    def get_training_summary(self) -> str:
        """Get a summary of the training session with error handling."""
        try:
            with self.training_lock:
                if not self.training_data['steps']:
                    return "No training data available. Please start training first."
                
                # Validate data integrity
                if len(self.training_data['steps']) != len(self.training_data['losses']):
                    return "❌ Training data corrupted. Please restart training."
                
                total_steps = len(self.training_data['steps'])
                
                # Calculate statistics safely
                try:
                    avg_loss = np.mean(self.training_data['losses'])
                    avg_stability = np.mean(self.training_data['stability_scores'])
                    avg_clipping = np.mean(self.training_data['clipping_ratios'])
                    total_nan = sum(self.training_data['nan_counts'])
                    total_inf = sum(self.training_data['inf_counts'])
                    total_overflow = sum(self.training_data['overflow_counts'])
                except Exception as e:
                    logger.error(f"Failed to calculate statistics: {str(e)}")
                    return f"❌ Failed to calculate training statistics: {str(e)}"
                
                summary = f"📊 Real-time Training Summary\n"
                summary += f"{'='*40}\n"
                summary += f"Total Steps: {total_steps}\n"
                summary += f"Average Loss: {avg_loss:.6f}\n"
                summary += f"Average Stability Score: {avg_stability:.4f}\n"
                summary += f"Average Clipping Ratio: {avg_clipping:.4f}\n"
                summary += f"Total NaN Issues: {total_nan}\n"
                summary += f"Total Inf Issues: {total_inf}\n"
                summary += f"Total Overflow Issues: {total_overflow}\n"
                summary += f"Numerical Issues Rate: {(total_nan + total_inf + total_overflow) / total_steps * 100:.2f}%\n"
                summary += f"Training Status: {'🔄 Active' if self.is_training else '⏹️ Stopped'}\n"
                summary += f"Error Count: {self.error_count}"
                
                if self.error_count > 0:
                    summary += f"\n⚠️ **Warning**: {self.error_count} errors encountered during training"
                
                return summary
                
        except Exception as e:
            return f"❌ Error generating training summary: {str(e)}"
    
    def clear_error_count(self) -> str:
        """Clear the error count."""
        try:
            with self.training_lock:
                old_count = self.error_count
                self.error_count = 0
            return f"✅ Error count cleared (was {old_count})"
        except Exception as e:
            return f"❌ Error clearing error count: {str(e)}"

    def _safe_data_generation(self, batch_size: int, input_features: int, output_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Safely generate synthetic training data with error handling."""
        try:
            # Validate input parameters
            if batch_size <= 0 or input_features <= 0 or output_features <= 0:
                raise ValueError("Batch size, input features, and output features must be positive")
            
            if batch_size > 10000:
                logger.warning(f"Large batch size ({batch_size}) may cause memory issues")
            
            # Generate input data safely
            try:
                x = torch.randn(batch_size, input_features)
                if not torch.isfinite(x).all():
                    raise RuntimeError("Generated input data contains NaN or Inf values")
            except RuntimeError as e:
                logger.error(f"Failed to generate input data: {str(e)}")
                # Fallback to smaller batch size
                if batch_size > 100:
                    logger.info("Retrying with smaller batch size")
                    return self._safe_data_generation(100, input_features, output_features)
                else:
                    raise RuntimeError("Data generation failed even with small batch size")
            
            # Generate target data safely
            try:
                if output_features == 1:
                    # Regression task
                    target = torch.randn(batch_size, output_features)
                else:
                    # Classification task
                    target = torch.randint(0, output_features, (batch_size,))
                    if output_features > 1:
                        target = torch.nn.functional.one_hot(target, num_classes=output_features).float()
                
                if not torch.isfinite(target).all():
                    raise RuntimeError("Generated target data contains NaN or Inf values")
            except RuntimeError as e:
                logger.error(f"Failed to generate target data: {str(e)}")
                raise RuntimeError("Target data generation failed")
            
            return x, target
            
        except Exception as e:
            logger.error(f"Data generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate training data: {str(e)}")
    
    def _safe_forward_pass(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Safely perform forward pass with comprehensive error handling."""
        try:
            # Validate inputs
            if model is None:
                raise ValueError("Model is None")
            if x is None or not isinstance(x, torch.Tensor):
                raise ValueError("Input x must be a valid tensor")
            
            # Check tensor properties
            if not torch.isfinite(x).all():
                raise RuntimeError("Input tensor contains NaN or Inf values")
            
            if x.requires_grad:
                logger.warning("Input tensor requires grad, this may cause memory issues")
            
            # Perform forward pass
            try:
                with torch.no_grad() if not x.requires_grad else torch.enable_grad():
                    output = model(x)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("CUDA out of memory during forward pass")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError("GPU memory exhausted during forward pass")
                elif "input size" in str(e).lower():
                    logger.error("Input size mismatch during forward pass")
                    raise ValueError("Input tensor dimensions don't match model expectations")
                else:
                    logger.error(f"Forward pass failed: {str(e)}")
                    raise RuntimeError(f"Forward pass failed: {str(e)}")
            
            # Validate output
            if output is None:
                raise RuntimeError("Model returned None output")
            
            if not torch.isfinite(output).all():
                logger.warning("Model output contains NaN or Inf values")
                # Try to clean the output
                output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise RuntimeError(f"Forward pass failed: {str(e)}")
    
    def _safe_loss_calculation(self, output: torch.Tensor, target: torch.Tensor, task_type: str = "regression") -> torch.Tensor:
        """Safely calculate loss with error handling."""
        try:
            # Validate inputs
            if output is None or target is None:
                raise ValueError("Output and target tensors cannot be None")
            
            if not isinstance(output, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise ValueError("Output and target must be tensors")
            
            # Check tensor shapes compatibility
            if output.shape[0] != target.shape[0]:
                raise ValueError(f"Batch size mismatch: output {output.shape[0]} vs target {target.shape[0]}")
            
            # Check for numerical issues in inputs
            if not torch.isfinite(output).all():
                logger.warning("Output tensor contains NaN or Inf values")
                output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
            
            if not torch.isfinite(target).all():
                logger.warning("Target tensor contains NaN or Inf values")
                target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
            
            # Calculate loss based on task type
            try:
                if task_type == "regression":
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(output.squeeze(), target)
                elif task_type == "classification":
                    if target.dim() == 1:
                        loss_fn = nn.CrossEntropyLoss()
                        loss = loss_fn(output, target)
                    else:
                        loss_fn = nn.BCEWithLogitsLoss()
                        loss = loss_fn(output, target)
                else:
                    # Default to MSE loss
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(output.squeeze(), target)
                
            except RuntimeError as e:
                logger.error(f"Loss calculation failed: {str(e)}")
                if "size mismatch" in str(e).lower():
                    raise ValueError("Output and target tensor shapes are incompatible for loss calculation")
                else:
                    raise RuntimeError(f"Loss calculation failed: {str(e)}")
            
            # Validate loss value
            if not torch.isfinite(loss):
                logger.error(f"Invalid loss value: {loss.item()}")
                raise RuntimeError(f"Loss calculation produced invalid value: {loss.item()}")
            
            if loss.item() < 0:
                logger.warning(f"Negative loss value: {loss.item()}")
            
            return loss
            
        except Exception as e:
            logger.error(f"Loss calculation failed: {str(e)}")
            raise RuntimeError(f"Loss calculation failed: {str(e)}")
    
    def _safe_backward_pass(self, loss: torch.Tensor) -> None:
        """Safely perform backward pass with error handling."""
        try:
            # Validate loss
            if loss is None:
                raise ValueError("Loss tensor is None")
            
            if not isinstance(loss, torch.Tensor):
                raise ValueError("Loss must be a tensor")
            
            if not loss.requires_grad:
                logger.warning("Loss tensor doesn't require gradients")
                return
            
            # Check for numerical issues
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss tensor contains invalid values: {loss.item()}")
            
            # Perform backward pass
            try:
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("CUDA out of memory during backward pass")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError("GPU memory exhausted during backward pass")
                elif "gradient computation" in str(e).lower():
                    logger.error("Gradient computation failed")
                    raise RuntimeError("Gradient computation failed during backward pass")
                else:
                    logger.error(f"Backward pass failed: {str(e)}")
                    raise RuntimeError(f"Backward pass failed: {str(e)}")
            
            # Validate gradients
            try:
                for name, param in self.current_model.named_parameters():
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            logger.warning(f"Parameter {name} has invalid gradients")
                            # Zero invalid gradients
                            param.grad.data = torch.where(
                                torch.isfinite(param.grad.data),
                                param.grad.data,
                                torch.zeros_like(param.grad.data)
                            )
            except Exception as e:
                logger.warning(f"Gradient validation warning: {str(e)}")
            
        except Exception as e:
            logger.error(f"Backward pass failed: {str(e)}")
            raise RuntimeError(f"Backward pass failed: {str(e)}")
    
    def _safe_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Safely perform optimizer step with error handling."""
        try:
            # Validate optimizer
            if optimizer is None:
                raise ValueError("Optimizer is None")
            
            # Check if gradients exist
            has_gradients = False
            for param in self.current_model.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_gradients = True
                    break
            
            if not has_gradients:
                logger.warning("No valid gradients found for optimizer step")
                return
            
            # Perform optimizer step
            try:
                optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("CUDA out of memory during optimizer step")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError("GPU memory exhausted during optimizer step")
                elif "invalid gradient" in str(e).lower():
                    logger.error("Invalid gradients detected during optimizer step")
                    # Try to clean gradients and retry
                    for param in self.current_model.parameters():
                        if param.grad is not None:
                            param.grad.data = torch.where(
                                torch.isfinite(param.grad.data),
                                param.grad.data,
                                torch.zeros_like(param.grad.data)
                            )
                    # Retry once
                    try:
                        optimizer.step()
                    except Exception as retry_error:
                        raise RuntimeError(f"Optimizer step failed even after gradient cleaning: {str(retry_error)}")
                else:
                    logger.error(f"Optimizer step failed: {str(e)}")
                    raise RuntimeError(f"Optimizer step failed: {str(e)}")
            
            # Zero gradients safely
            try:
                optimizer.zero_grad()
            except Exception as e:
                logger.warning(f"Failed to zero gradients: {str(e)}")
            
        except Exception as e:
            logger.error(f"Optimizer step failed: {str(e)}")
            raise RuntimeError(f"Optimizer step failed: {str(e)}")


def create_realtime_training_interface():
    """Create the real-time training demo interface with error handling."""
    demo = RealTimeTrainingDemo()
    
    with gr.Blocks(title="Real-time Training Visualization Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚀 Real-time Training Visualization Demo
        
        Live demonstration of training dynamics with numerical stability monitoring.
        Watch as your model trains in real-time with comprehensive visualization of
        loss, stability, gradients, and numerical issues.
        
        **Features:**
        - 🔄 Real-time training with live updates
        - 📊 Dynamic visualization of training metrics
        - 🚨 Numerical stability monitoring
        - 🎯 Controlled numerical issue injection
        - 📈 Comprehensive training analysis
        - 🛡️ Robust error handling and recovery
        """)
        
        with gr.Tabs():
            # Tab 1: Model Setup
            with gr.Tab("🏗️ Model Setup"):
                gr.Markdown("### Configure your training model")
                
                with gr.Row():
                    with gr.Column():
                        model_type = gr.Dropdown(
                            choices=["Feedforward", "Deep", "Wide"],
                            value="Feedforward",
                            label="Model Architecture"
                        )
                        input_dim = gr.Slider(minimum=1, maximum=20, value=2, step=1, label="Input Dimension")
                        hidden_dim = gr.Slider(minimum=16, maximum=256, value=64, step=16, label="Hidden Dimension")
                        output_dim = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Output Dimension")
                        learning_rate = gr.Slider(minimum=0.0001, maximum=0.1, value=0.01, step=0.0001, label="Learning Rate")
                        
                        create_model_btn = gr.Button("🚀 Create Training Model", variant="primary")
                        model_status = gr.Textbox(label="Model Status", lines=3)
                        
                        # Reset button for error recovery
                        reset_model_btn = gr.Button("🔄 Reset Model", variant="secondary", size="sm")
                    
                    with gr.Column():
                        gr.Markdown("### Model Information")
                        model_info = gr.Markdown("No model created yet.")
                
                create_model_btn.click(
                    fn=demo.create_training_model,
                    inputs=[model_type, input_dim, hidden_dim, output_dim, learning_rate],
                    outputs=model_status
                )
                
                reset_model_btn.click(
                    fn=demo.reset_model,
                    outputs=model_status
                )
            
            # Tab 2: Stability Configuration
            with gr.Tab("⚙️ Stability Configuration"):
                gr.Markdown("### Configure numerical stability parameters")
                
                with gr.Row():
                    with gr.Column():
                        clipping_type = gr.Dropdown(
                            choices=["NORM", "VALUE", "GLOBAL_NORM", "ADAPTIVE", "LAYER_WISE", "PERCENTILE", "EXPONENTIAL"],
                            value="NORM",
                            label="Gradient Clipping Type"
                        )
                        max_norm = gr.Slider(minimum=0.1, maximum=10.0, value=1.0, step=0.1, label="Max Norm")
                        adaptive_threshold = gr.Slider(minimum=0.01, maximum=2.0, value=0.1, step=0.01, label="Adaptive Threshold")
                        
                    with gr.Column():
                        nan_handling = gr.Dropdown(
                            choices=["DETECT", "REPLACE", "SKIP", "GRADIENT_ZEROING", "ADAPTIVE", "GRADIENT_SCALING"],
                            value="ADAPTIVE",
                            label="NaN/Inf Handling"
                        )
                        
                        config_btn = gr.Button("🔧 Configure Stability", variant="primary")
                        config_status = gr.Textbox(label="Configuration Status", lines=3)
                
                config_btn.click(
                    fn=demo.configure_stability_manager,
                    inputs=[clipping_type, max_norm, nan_handling, adaptive_threshold],
                    outputs=config_status
                )
            
            # Tab 3: Training Control
            with gr.Tab("🎮 Training Control"):
                gr.Markdown("### Control real-time training")
                
                with gr.Row():
                    with gr.Column():
                        batch_size = gr.Slider(minimum=8, maximum=128, value=32, step=8, label="Batch Size")
                        num_epochs = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Epochs")
                        introduce_issues = gr.Checkbox(label="Introduce Numerical Issues", value=True)
                        issue_frequency = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Issue Frequency")
                        
                        with gr.Row():
                            start_btn = gr.Button("▶️ Start Training", variant="primary")
                            stop_btn = gr.Button("⏹️ Stop Training", variant="primary")
                        
                        training_status = gr.Textbox(label="Training Status", lines=6)
                    
                    with gr.Column():
                        gr.Markdown("### Training Information")
                        status_btn = gr.Button("📊 Get Current Status", variant="secondary")
                        current_status = gr.Textbox(label="Current Status", lines=8)
                
                start_btn.click(
                    fn=demo.start_training,
                    inputs=[batch_size, num_epochs, introduce_issues, issue_frequency],
                    outputs=training_status
                )
                
                stop_btn.click(
                    fn=demo.stop_training_loop,
                    outputs=training_status
                )
                
                status_btn.click(
                    fn=demo.get_training_status,
                    outputs=current_status
                )
            
            # Tab 4: Live Visualization
            with gr.Tab("📊 Live Visualization"):
                gr.Markdown("### Real-time training visualization")
                
                with gr.Row():
                    plot_btn = gr.Button("📈 Generate Live Plots", variant="primary")
                    summary_btn = gr.Button("📋 Training Summary", variant="primary")
                
                with gr.Row():
                    plot1 = gr.Plot(label="Training Loss & Stability")
                    plot2 = gr.Plot(label="Gradient Analysis")
                
                with gr.Row():
                    plot3 = gr.Plot(label="Numerical Issues")
                    summary_text = gr.Textbox(label="Training Summary", lines=10)
                
                plot_btn.click(
                    fn=demo.generate_live_plots,
                    outputs=[plot1, plot2, plot3]
                )
                
                summary_btn.click(
                    fn=demo.get_training_summary,
                    outputs=summary_text
                )
            
            # Tab 5: Error Handling & Recovery
            with gr.Tab("🛡️ Error Handling"):
                gr.Markdown("### Monitor and recover from errors")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### **Error Recovery**")
                        clear_errors_btn = gr.Button("🧹 Clear Error Count", variant="secondary", size="sm")
                        
                        error_info = gr.Markdown("""
                        **Error Handling Features:**
                        - ✅ Input validation for all parameters
                        - ✅ Graceful error recovery during training
                        - ✅ Detailed error logging and monitoring
                        - ✅ User-friendly error messages
                        - ✅ Automatic fallback configurations
                        - ✅ Memory management and cleanup
                        - ✅ Thread-safe operations
                        """)
                    
                    with gr.Column():
                        gr.Markdown("### **System Status**")
                        error_status_btn = gr.Button("📊 Check Error Status", variant="primary")
                        error_status = gr.Textbox(label="Error Status", lines=8)
                
                # Connect error handling buttons
                clear_errors_btn.click(
                    fn=demo.clear_error_count,
                    outputs=error_status
                )
                
                error_status_btn.click(
                    fn=demo.get_training_status,
                    outputs=error_status
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Real-time Training Visualization Demo** | Built with Gradio & Matplotlib
        
        Watch your model train in real-time with comprehensive numerical stability monitoring.
        Robust error handling ensures a smooth experience even when issues occur.
        """)
    
    return interface


def main():
    """Main function to launch the real-time training demo interface with error handling."""
    try:
        interface = create_realtime_training_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7862,  # Different port from other interfaces
            share=True,
            show_error=True,
            show_tips=True
        )
    except Exception as e:
        print(f"Failed to launch real-time training demo: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
