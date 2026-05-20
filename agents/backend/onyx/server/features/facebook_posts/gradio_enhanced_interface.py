#!/usr/bin/env python3
"""
Enhanced User-Friendly Interface for Gradient Clipping & NaN Handling
Improved UX design with guided workflows, visual feedback, and robust error handling.
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import traceback
import logging
from pathlib import Path
import warnings

# Import our gradient clipping and NaN handling system
from gradient_clipping_nan_handling import (
    GradientClippingConfig,
    NaNHandlingConfig,
    NumericalStabilityManager,
    ClippingType,
    NaNHandlingType,
    create_training_wrapper
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


class EnhancedTrainingSimulator:
    """Enhanced training simulator with user-friendly interface and robust error handling."""
    
    def __init__(self):
        self.current_model = None
        self.current_optimizer = None
        self.stability_manager = None
        self.training_history = {
            'steps': [],
            'losses': [],
            'stability_scores': [],
            'clipping_ratios': [],
            'nan_counts': [],
            'inf_counts': [],
            'overflow_counts': []
        }
        self.current_step = 0
        self.workflow_stage = "setup"  # setup, configured, training, completed
        self.error_count = 0
        self.max_errors = 5
        
    def _validate_input_dimensions(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Validate input dimensions for model creation."""
        if input_size < 1 or input_size > 1000:
            raise ValidationError("Input size must be between 1 and 1000")
        if hidden_size < 1 or hidden_size > 2000:
            raise ValidationError("Hidden size must be between 1 and 2000")
        if output_size < 1 or output_size > 1000:
            raise ValidationError("Output size must be between 1 and 1000")
        if hidden_size < input_size // 2:
            raise ValidationError("Hidden size should be at least half of input size for proper learning")
        if hidden_size < output_size:
            raise ValidationError("Hidden size should be at least as large as output size")
    
    def _validate_stability_parameters(self, max_norm: float, adaptive_threshold: float) -> None:
        """Validate stability configuration parameters."""
        if max_norm <= 0 or max_norm > 100:
            raise ValidationError("Max norm must be between 0.001 and 100")
        if adaptive_threshold <= 0 or adaptive_threshold > 10:
            raise ValidationError("Adaptive threshold must be between 0.001 and 10")
    
    def _validate_training_parameters(self, num_steps: int, batch_size: int, 
                                    nan_prob: float, inf_prob: float, overflow_prob: float) -> None:
        """Validate training parameters."""
        if num_steps < 1 or num_steps > 1000:
            raise ValidationError("Number of training steps must be between 1 and 1000")
        if batch_size < 1 or batch_size > 512:
            raise ValidationError("Batch size must be between 1 and 512")
        if not (0 <= nan_prob <= 1):
            raise ValidationError("NaN probability must be between 0 and 1")
        if not (0 <= inf_prob <= 1):
            raise ValidationError("Inf probability must be between 0 and 1")
        if not (0 <= overflow_prob <= 1):
            raise ValidationError("Overflow probability must be between 0 and 1")
        if nan_prob + inf_prob + overflow_prob > 1.0:
            raise ValidationError("Sum of all probabilities should not exceed 1.0")
    
    def _handle_error(self, error: Exception, operation: str) -> str:
        """Centralized error handling with comprehensive logging and user-friendly messages."""
        start_time = time.time()
        self.error_count += 1
        error_msg = f"❌ Error in {operation}: {str(error)}"
        
        # Log detailed error information with context
        log_error_with_context(
            error_logger,
            error=error,
            operation=operation,
            context={
                "error_count": self.error_count,
                "max_errors": self.max_errors,
                "workflow_stage": self.workflow_stage,
                "current_step": self.current_step
            },
            recovery_attempted=False
        )
        
        # Log system event for error
        log_system_event(
            system_logger,
            event_type="training_error",
            description=f"Error occurred in {operation}",
            details={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_count": self.error_count,
                "workflow_stage": self.workflow_stage
            },
            level="error"
        )
        
        # Provide specific guidance based on error type
        if isinstance(error, ValidationError):
            error_msg += "\n\n💡 **Validation Error**: Please check your input parameters and try again."
            logger.warning(f"Validation error in {operation}: {str(error)}")
        elif isinstance(error, RuntimeError):
            error_msg += "\n\n💡 **Runtime Error**: This might be due to insufficient memory or GPU issues."
            logger.error(f"Runtime error in {operation}: {str(error)}")
        elif isinstance(error, ValueError):
            error_msg += "\n\n💡 **Value Error**: Please verify your input values are within valid ranges."
            logger.warning(f"Value error in {operation}: {str(error)}")
        elif isinstance(error, OSError):
            error_msg += "\n\n💡 **System Error**: Check if you have sufficient disk space and permissions."
            logger.error(f"System error in {operation}: {str(error)}")
        else:
            error_msg += "\n\n💡 **Unexpected Error**: Please try again or contact support if the issue persists."
            logger.error(f"Unexpected error in {operation}: {str(error)}")
        
        # Add error count information
        if self.error_count >= self.max_errors:
            error_msg += f"\n\n⚠️ **Warning**: You've encountered {self.error_count} errors. Consider restarting the interface."
            logger.critical(f"Maximum error count reached: {self.error_count}")
            
            # Log critical system event
            log_system_event(
                system_logger,
                event_type="max_errors_reached",
                description="Maximum error count reached, interface may be unstable",
                details={"error_count": self.error_count, "max_errors": self.max_errors},
                level="critical"
            )
        
        # Log performance metrics for error handling
        duration = time.time() - start_time
        log_performance_metrics(
            error_logger,
            metrics={
                "error_count": self.error_count,
                "error_type": type(error).__name__,
                "operation": operation,
                "workflow_stage": self.workflow_stage
            },
            operation=f"error_handling_{operation}",
            duration=duration
        )
        
        return error_msg
    
    def _safe_model_creation(self, model_func, *args, **kwargs):
        """Safely create models with error handling."""
        try:
            return model_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Model creation failed: {str(e)}")
            raise
    
    def create_simple_model(self) -> str:
        """Create a simple model for beginners with error handling."""
        try:
            # Validate that we can create a model
            if self.current_model is not None:
                return "ℹ️ Model already exists. Use 'Reset Model' to create a new one."
            
            # Create model with error handling
            self.current_model = self._safe_model_creation(
                lambda: nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                )
            )
            
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
                self.current_optimizer = torch.optim.Adam(self.current_model.parameters(), lr=0.01)
            except Exception as e:
                logger.error(f"Optimizer creation failed: {str(e)}")
                raise RuntimeError("Failed to create optimizer")
            
            self.reset_training_history()
            self.workflow_stage = "setup"
            
            param_count = sum(p.numel() for p in self.current_model.parameters())
            return f"✅ Simple model created successfully!\nParameters: {param_count:,}\nReady for stability configuration."
            
        except Exception as e:
            return self._handle_error(e, "simple model creation")
    
    def create_advanced_model(self, model_type: str, input_size: int, hidden_size: int, output_size: int) -> str:
        """Create an advanced model with custom parameters and validation."""
        try:
            # Input validation
            self._validate_input_dimensions(input_size, hidden_size, output_size)
            
            if self.current_model is not None:
                return "ℹ️ Model already exists. Use 'Reset Model' to create a new one."
            
            # Create model based on type
            if model_type == "Sequential":
                self.current_model = self._safe_model_creation(
                    lambda: nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size // 2, output_size)
                    )
                )
            elif model_type == "Deep":
                layers = []
                current_size = input_size
                for i in range(4):
                    layers.extend([
                        nn.Linear(current_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ])
                    current_size = hidden_size
                layers.append(nn.Linear(current_size, output_size))
                self.current_model = self._safe_model_creation(lambda: nn.Sequential(*layers))
            elif model_type == "Wide":
                self.current_model = self._safe_model_creation(
                    lambda: nn.Sequential(
                        nn.Linear(input_size, hidden_size * 2),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, output_size)
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
            
            # Create optimizer safely
            try:
                self.current_optimizer = torch.optim.Adam(self.current_model.parameters(), lr=0.01)
            except Exception as e:
                logger.error(f"Optimizer creation failed: {str(e)}")
                raise RuntimeError("Failed to create optimizer")
            
            self.reset_training_history()
            self.workflow_stage = "setup"
            
            param_count = sum(p.numel() for p in self.current_model.parameters())
            return f"✅ {model_type} model created successfully!\nParameters: {param_count:,}\nReady for stability configuration."
            
        except Exception as e:
            return self._handle_error(e, "advanced model creation")
    
    def configure_basic_stability(self) -> str:
        """Configure basic stability settings for beginners with error handling."""
        try:
            if self.current_model is None:
                raise ValidationError("Please create a model first")
            
            # Create basic stability configuration
            clipping_config = GradientClippingConfig(
                clipping_type=ClippingType.NORM,
                max_norm=1.0,
                monitor_clipping=True,
                log_clipping_stats=True,
                save_clipping_history=True
            )
            
            nan_config = NaNHandlingConfig(
                handling_type=NaNHandlingType.ADAPTIVE,
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
            
            self.workflow_stage = "configured"
            
            return "✅ Basic stability configured!\nClipping: NORM (max norm: 1.0)\nNaN handling: ADAPTIVE\nReady for training!"
            
        except Exception as e:
            return self._handle_error(e, "basic stability configuration")
    
    def configure_advanced_stability(self, clipping_type: str, max_norm: float, 
                                   nan_handling: str, adaptive_threshold: float) -> str:
        """Configure advanced stability settings with validation."""
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
            
            # Create advanced stability configuration
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
            
            self.workflow_stage = "configured"
            
            return f"✅ Advanced stability configured!\nClipping: {clipping_type}\nMax norm: {max_norm}\nNaN handling: {nan_handling}\nReady for training!"
            
        except Exception as e:
            return self._handle_error(e, "advanced stability configuration")
    
    def reset_model(self) -> str:
        """Reset the current model and start fresh."""
        try:
            self.current_model = None
            self.current_optimizer = None
            self.stability_manager = None
            self.reset_training_history()
            self.workflow_stage = "setup"
            self.error_count = 0  # Reset error count
            
            return "✅ Model reset successfully! Ready to create a new model."
        except Exception as e:
            return self._handle_error(e, "model reset")
    
    def reset_training_history(self):
        """Reset training history safely."""
        try:
            self.training_history = {
                'steps': [],
                'losses': [],
                'stability_scores': [],
                'clipping_ratios': [],
                'nan_counts': [],
                'inf_counts': [],
                'overflow_counts': []
            }
            self.current_step = 0
        except Exception as e:
            logger.error(f"Failed to reset training history: {str(e)}")
    
    def run_guided_training(self, num_steps: int, progress=gr.Progress()) -> str:
        """Run guided training with automatic configuration and comprehensive logging."""
        start_time = time.time()
        
        try:
            # Log training start
            log_system_event(
                system_logger,
                event_type="guided_training_started",
                description=f"Guided training started with {num_steps} steps",
                details={
                    "num_steps": num_steps,
                    "workflow_stage": self.workflow_stage,
                    "model_exists": self.current_model is not None,
                    "stability_manager_exists": self.stability_manager is not None
                }
            )
            
            # Validate input
            self._validate_training_parameters(num_steps, 32, 0.1, 0.05, 0.1)
            
            if self.current_model is None:
                raise ValidationError("Please create a model first")
            
            if self.stability_manager is None:
                # Auto-configure basic stability
                self.configure_basic_stability()
                logger.info("Auto-configured basic stability for guided training")
            
            # Check available memory (basic check)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cache cleared")
            except Exception as e:
                logger.warning(f"GPU memory cleanup warning: {str(e)}")
            
            results = []
            self.workflow_stage = "training"
            step_success_count = 0
            step_error_count = 0
            
            # Log training configuration
            training_logger.info(f"Starting guided training: {num_steps} steps, model: {type(self.current_model).__name__}")
            
            for step in range(num_steps):
                step_start_time = time.time()
                
                try:
                    progress(step / num_steps, desc=f"Training step {step + 1}/{num_steps}")
                    
                    # Generate synthetic data safely using new safe methods
                    try:
                        x, target = self._safe_data_generation(32, self.current_model[0].in_features, self.current_model[-1].out_features)
                    except Exception as e:
                        logger.error(f"Data generation failed for step {step + 1}: {str(e)}")
                        results.append(f"Step {step + 1}: ❌ Data generation failed - {str(e)}")
                        step_error_count += 1
                        continue
                    
                    # Forward pass with comprehensive error handling
                    try:
                        self.current_optimizer.zero_grad()
                        output = self._safe_forward_pass(self.current_model, x)
                    except Exception as e:
                        logger.error(f"Forward pass failed for step {step + 1}: {str(e)}")
                        results.append(f"Step {step + 1}: ❌ Forward pass failed - {str(e)}")
                        step_error_count += 1
                        continue
                    
                    # Loss calculation with error handling
                    try:
                        loss = self._safe_loss_calculation(output, target, "regression")
                    except Exception as e:
                        logger.error(f"Loss calculation failed for step {step + 1}: {str(e)}")
                        results.append(f"Step {step + 1}: ❌ Loss calculation failed - {str(e)}")
                        step_error_count += 1
                        continue
                    
                    # Backward pass with error handling
                    try:
                        self._safe_backward_pass(loss)
                    except Exception as e:
                        logger.error(f"Backward pass failed for step {step + 1}: {str(e)}")
                        results.append(f"Step {step + 1}: ❌ Backward pass failed - {str(e)}")
                        step_error_count += 1
                        continue
                    
                    # Introduce some numerical issues occasionally
                    if np.random.random() < 0.1:
                        self._introduce_numerical_issues(0.1, 0.05, 0.1)
                        logger.info(f"Introduced numerical issues in step {step + 1}")
                    
                    # Apply stability measures safely
                    try:
                        stability_result = self.stability_manager.step(self.current_model, loss, self.current_optimizer)
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
                        self._safe_optimizer_step(self.current_optimizer)
                    except Exception as e:
                        logger.warning(f"Optimizer step warning: {str(e)}")
                        # Skip this step if optimizer fails
                        step_error_count += 1
                        continue
                    
                    # Update history safely
                    self.current_step += 1
                    self.training_history['steps'].append(self.current_step)
                    self.training_history['losses'].append(loss.item())
                    self.training_history['stability_scores'].append(stability_result['stability_score'])
                    self.training_history['clipping_ratios'].append(stability_result['clipping_stats'].get('clipping_ratio', 0.0))
                    self.training_history['nan_counts'].append(1 if stability_result['nan_stats']['nan_detected'] else 0)
                    self.training_history['inf_counts'].append(1 if stability_result['nan_stats']['inf_detected'] else 0)
                    self.training_history['overflow_counts'].append(1 if stability_result['nan_stats']['overflow_detected'] else 0)
                    
                    # Log training step with comprehensive metrics
                    step_duration = time.time() - step_start_time
                    log_training_step(
                        training_logger,
                        step=step + 1,
                        epoch=1,  # Single epoch for guided training
                        loss=loss.item(),
                        stability_score=stability_result['stability_score'],
                        gradient_norm=stability_result['clipping_stats'].get('gradient_norm', 0.0),
                        clipping_ratio=stability_result['clipping_stats'].get('clipping_ratio', 0.0),
                        step_duration=step_duration,
                        numerical_issues=stability_result['nan_stats']['nan_detected'] or 
                                       stability_result['nan_stats']['inf_detected'] or 
                                       stability_result['nan_stats']['overflow_detected']
                    )
                    
                    results.append(f"Step {step + 1}: Loss={loss.item():.6f}, Stability={stability_result['stability_score']:.4f}")
                    step_success_count += 1
                    
                except Exception as step_error:
                    step_error_count += 1
                    logger.error(f"Unexpected error in training step {step + 1}: {str(step_error)}")
                    results.append(f"Step {step + 1}: ❌ Unexpected error - {str(step_error)}")
                    continue
                    
                    time.sleep(0.01)
                    
                except Exception as step_error:
                    logger.warning(f"Training step {step + 1} failed: {str(step_error)}")
                    results.append(f"Step {step + 1}: ❌ Failed - {str(step_error)}")
                    continue
            
            # Log training completion
            total_duration = time.time() - start_time
            success_rate = step_success_count / num_steps if num_steps > 0 else 0
            
            log_system_event(
                system_logger,
                event_type="guided_training_completed",
                description=f"Guided training completed successfully",
                details={
                    "total_steps": num_steps,
                    "successful_steps": step_success_count,
                    "failed_steps": step_error_count,
                    "success_rate": f"{success_rate:.2%}",
                    "total_duration_seconds": total_duration,
                    "final_loss": self.training_history['losses'][-1] if self.training_history['losses'] else None,
                    "final_stability_score": self.training_history['stability_scores'][-1] if self.training_history['stability_scores'] else None
                }
            )
            
            # Log performance metrics for entire training session
            log_performance_metrics(
                training_logger,
                metrics={
                    "total_steps": num_steps,
                    "successful_steps": step_success_count,
                    "failed_steps": step_error_count,
                    "success_rate": success_rate,
                    "final_loss": self.training_history['losses'][-1] if self.training_history['losses'] else None,
                    "final_stability_score": self.training_history['stability_scores'][-1] if self.training_history['stability_scores'] else None,
                    "total_nan_issues": sum(self.training_history['nan_counts']),
                    "total_inf_issues": sum(self.training_history['inf_counts']),
                    "total_overflow_issues": sum(self.training_history['overflow_counts'])
                },
                operation="guided_training_session",
                duration=total_duration
            )
            
            training_logger.info(f"Guided training completed: {step_success_count}/{num_steps} steps successful, "
                               f"success rate: {success_rate:.2%}, duration: {total_duration:.2f}s")
            
            self.workflow_stage = "completed"
            return f"✅ Guided training completed!\n\n" + "\n".join(results[-5:])  # Show last 5 results
            
        except Exception as e:
            self.workflow_stage = "configured"  # Reset to previous stage
            return self._handle_error(e, "guided training")
    
    def run_custom_training(self, batch_size: int, introduce_nan_prob: float,
                           introduce_inf_prob: float, introduce_overflow_prob: float) -> str:
        """Run custom training with user-defined parameters and validation."""
        try:
            # Validate all parameters
            self._validate_training_parameters(1, batch_size, introduce_nan_prob, introduce_inf_prob, introduce_overflow_prob)
            
            if self.current_model is None or self.stability_manager is None:
                raise ValidationError("Please create a model and configure stability manager first")
            
            # Check memory constraints
            if batch_size > 256:
                logger.warning(f"Large batch size ({batch_size}) may cause memory issues")
            
            # Generate synthetic data safely using new safe methods
            try:
                x, target = self._safe_data_generation(batch_size, self.current_model[0].in_features, self.current_model[-1].out_features)
            except Exception as e:
                logger.error(f"Data generation failed: {str(e)}")
                raise RuntimeError(f"Failed to generate training data: {str(e)}")
            
            # Forward pass with comprehensive error handling
            try:
                self.current_optimizer.zero_grad()
                output = self._safe_forward_pass(self.current_model, x)
            except Exception as e:
                logger.error(f"Forward pass failed: {str(e)}")
                raise RuntimeError(f"Forward pass failed: {str(e)}")
            
            # Loss calculation with error handling
            try:
                loss = self._safe_loss_calculation(output, target, "regression")
            except Exception as e:
                logger.error(f"Loss calculation failed: {str(e)}")
                raise RuntimeError(f"Loss calculation failed: {str(e)}")
            
            # Backward pass with error handling
            try:
                self._safe_backward_pass(loss)
            except Exception as e:
                logger.error(f"Backward pass failed: {str(e)}")
                raise RuntimeError(f"Backward pass failed: {str(e)}")
            
            # Introduce numerical issues based on probabilities
            self._introduce_numerical_issues(introduce_nan_prob, introduce_inf_prob, introduce_overflow_prob)
            
            # Apply stability measures safely
            try:
                stability_result = self.stability_manager.step(self.current_model, loss, self.current_optimizer)
            except Exception as e:
                logger.error(f"Stability step failed: {str(e)}")
                raise RuntimeError("Stability measures failed to apply")
            
            # Optimizer step safely using new safe method
            try:
                self._safe_optimizer_step(self.current_optimizer)
            except Exception as e:
                logger.error(f"Optimizer step failed: {str(e)}")
                raise RuntimeError("Training step failed")
            
            # Update training history safely
            self.current_step += 1
            self.training_history['steps'].append(self.current_step)
            self.training_history['losses'].append(loss.item())
            self.training_history['stability_scores'].append(stability_result['stability_score'])
            self.training_history['clipping_ratios'].append(stability_result['clipping_stats'].get('clipping_ratio', 0.0))
            self.training_history['nan_counts'].append(1 if stability_result['nan_stats']['nan_detected'] else 0)
            self.training_history['inf_counts'].append(1 if stability_result['nan_stats']['inf_detected'] else 0)
            self.training_history['overflow_counts'].append(1 if stability_result['nan_stats']['overflow_detected'] else 0)
            
            self.workflow_stage = "completed"
            
            # Prepare result message
            result_msg = f"✅ Training step {self.current_step} completed!\n"
            result_msg += f"Loss: {loss.item():.6f}\n"
            result_msg += f"Stability Score: {stability_result['stability_score']:.4f}\n"
            result_msg += f"Clipping Ratio: {stability_result['clipping_stats'].get('clipping_ratio', 0.0):.4f}\n"
            result_msg += f"NaN Detected: {stability_result['nan_stats']['nan_detected']}\n"
            result_msg += f"Inf Detected: {stability_result['nan_stats']['inf_detected']}\n"
            result_msg += f"Overflow Detected: {stability_result['nan_stats']['overflow_detected']}\n"
            result_msg += f"Handling Action: {stability_result['nan_stats'].get('handling_action', 'N/A')}"
            
            return result_msg
            
        except Exception as e:
            return self._handle_error(e, "custom training")
    
    def _introduce_numerical_issues(self, nan_prob: float, inf_prob: float, overflow_prob: float):
        """Introduce numerical issues in gradients based on probabilities safely."""
        try:
            for param in self.current_model.parameters():
                if param.grad is not None:
                    # Introduce NaN
                    if np.random.random() < nan_prob:
                        param.grad.data[0, 0] = float('nan')
                    
                    # Introduce Inf
                    if np.random.random() < inf_prob:
                        param.grad.data[0, 0] = float('inf')
                    
                    # Introduce overflow
                    if np.random.random() < overflow_prob:
                        param.grad.data *= 1e6
        except Exception as e:
            logger.warning(f"Failed to introduce numerical issues: {str(e)}")
    
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
                elif task_type == "classification":
                    if target.dim() == 1:
                        loss_fn = nn.CrossEntropyLoss()
                    else:
                        loss_fn = nn.BCEWithLogitsLoss()
                else:
                    # Default to MSE loss
                    loss_fn = nn.MSELoss()
                
                loss = loss_fn(output, target)
                
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
    
    def get_workflow_status(self) -> str:
        """Get current workflow status with error information."""
        try:
            status_map = {
                "setup": "🔄 Setup Phase - Model created, ready for configuration",
                "configured": "⚙️ Configured - Stability manager ready, can start training",
                "training": "🏃‍♂️ Training - Currently running training steps",
                "completed": "✅ Completed - Training finished, ready for analysis"
            }
            
            current_status = status_map.get(self.workflow_stage, "❓ Unknown status")
            
            status_text = f"📊 Workflow Status\n"
            status_text += f"{'='*30}\n"
            status_text += f"Current Stage: {current_status}\n"
            status_text += f"Model Created: {'✅ Yes' if self.current_model else '❌ No'}\n"
            status_text += f"Stability Configured: {'✅ Yes' if self.stability_manager else '❌ No'}\n"
            status_text += f"Training Steps: {len(self.training_history['steps'])}\n"
            status_text += f"Error Count: {self.error_count}\n"
            
            if self.error_count > 0:
                status_text += f"⚠️ **Warning**: {self.error_count} errors encountered\n"
            
            if self.workflow_stage == "completed":
                status_text += f"\n🎉 Ready for analysis and visualization!"
            
            return status_text
            
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"
    
    def generate_training_plots(self) -> Tuple[plt.Figure, plt.Figure]:
        """Generate training visualization plots with error handling."""
        try:
            if not self.training_history['steps']:
                # Return empty plots with helpful messages
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.text(0.5, 0.5, 'No training data available\nRun training first!', 
                         ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Training Progress')
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.text(0.5, 0.5, 'No training data available\nRun training first!', 
                         ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Numerical Stability')
                
                return fig1, fig2
            
            # Validate data before plotting
            if len(self.training_history['steps']) != len(self.training_history['losses']):
                raise ValueError("Training data arrays have mismatched lengths")
            
            # Plot 1: Training Progress
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Loss over time with error handling
            try:
                ax1.plot(self.training_history['steps'], self.training_history['losses'], 'b-', linewidth=2, alpha=0.8)
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
                ax2.plot(self.training_history['steps'], self.training_history['stability_scores'], 'g-', linewidth=2, alpha=0.8)
                ax2.set_title('Numerical Stability Score Over Time', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Training Step')
                ax2.set_ylabel('Stability Score')
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
            except Exception as e:
                logger.error(f"Failed to plot stability: {str(e)}")
                ax2.text(0.5, 0.5, 'Failed to plot stability data', ha='center', va='center', transform=ax2.transAxes)
            
            fig1.tight_layout()
            
            # Plot 2: Numerical Stability
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Clipping ratio over time
            try:
                ax1.plot(self.training_history['steps'], self.training_history['clipping_ratios'], 'r-', linewidth=2, alpha=0.8)
                ax1.set_title('Gradient Clipping Ratio Over Time', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Training Step')
                ax1.set_ylabel('Clipping Ratio')
                ax1.grid(True, alpha=0.3)
            except Exception as e:
                logger.error(f"Failed to plot clipping: {str(e)}")
                ax1.text(0.5, 0.5, 'Failed to plot clipping data', ha='center', va='center', transform=ax1.transAxes)
            
            # Numerical issues over time
            try:
                ax2.plot(self.training_history['steps'], self.training_history['nan_counts'], 'r-', label='NaN', alpha=0.8, linewidth=2)
                ax2.plot(self.training_history['steps'], self.training_history['inf_counts'], 'orange', label='Inf', alpha=0.8, linewidth=2)
                ax2.plot(self.training_history['steps'], self.training_history['overflow_counts'], 'yellow', label='Overflow', alpha=0.8, linewidth=2)
                ax2.set_title('Numerical Issues Over Time', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Training Step')
                ax2.set_ylabel('Issue Count')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            except Exception as e:
                logger.error(f"Failed to plot numerical issues: {str(e)}")
                ax2.text(0.5, 0.5, 'Failed to plot numerical issues data', ha='center', va='center', transform=ax2.transAxes)
            
            fig2.tight_layout()
            
            return fig1, fig2
            
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
            
            return fig1, fig2
    
    def get_training_summary(self) -> str:
        """Get a summary of the training session with error handling."""
        try:
            if not self.training_history['steps']:
                return "No training data available. Please run training first."
            
            # Validate data integrity
            if len(self.training_history['steps']) != len(self.training_history['losses']):
                return "❌ Training data corrupted. Please restart training."
            
            total_steps = len(self.training_history['steps'])
            
            # Calculate statistics safely
            try:
                avg_loss = np.mean(self.training_history['losses'])
                avg_stability = np.mean(self.training_history['stability_scores'])
                avg_clipping = np.mean(self.training_history['clipping_ratios'])
                total_nan = sum(self.training_history['nan_counts'])
                total_inf = sum(self.training_history['inf_counts'])
                total_overflow = sum(self.training_history['overflow_counts'])
            except Exception as e:
                logger.error(f"Failed to calculate statistics: {str(e)}")
                return f"❌ Failed to calculate training statistics: {str(e)}"
            
            summary = f"📊 Training Session Summary\n"
            summary += f"{'='*40}\n"
            summary += f"Total Steps: {total_steps}\n"
            summary += f"Average Loss: {avg_loss:.6f}\n"
            summary += f"Average Stability Score: {avg_stability:.4f}\n"
            summary += f"Average Clipping Ratio: {avg_clipping:.4f}\n"
            summary += f"Total NaN Issues: {total_nan}\n"
            summary += f"Total Inf Issues: {total_inf}\n"
            summary += f"Total Overflow Issues: {total_overflow}\n"
            summary += f"Numerical Issues Rate: {(total_nan + total_inf + total_overflow) / total_steps * 100:.2f}%"
            
            return summary
            
        except Exception as e:
            return f"❌ Error generating training summary: {str(e)}"


def create_enhanced_interface():
    """Create the enhanced user-friendly interface with error handling."""
    simulator = EnhancedTrainingSimulator()
    
    with gr.Blocks(title="Enhanced Gradient Clipping & NaN Handling Interface", theme=gr.themes.Soft()) as interface:
        # Header with clear description
        gr.Markdown("""
        # 🚀 **Enhanced Gradient Clipping & NaN/Inf Handling Interface**
        
        Welcome to the user-friendly interface for exploring numerical stability in deep learning!
        This interface guides you through the complete workflow from model creation to training analysis.
        
        **🛡️ Robust Error Handling**: Built-in validation and graceful error recovery
        **📊 Real-time Monitoring**: Track progress and identify issues early
        **🔄 Auto-recovery**: Automatic fallbacks when operations fail
        """)
        
        with gr.Tabs():
            # Tab 1: Quick Start (Beginner Friendly)
            with gr.Tab("🚀 Quick Start"):
                gr.Markdown("### Get started quickly with guided workflows")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### **Step 1: Create Model**")
                        simple_model_btn = gr.Button("🎯 Create Simple Model", variant="primary", size="lg")
                        model_status = gr.Textbox(label="Model Status", lines=3)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### **Step 2: Configure Stability**")
                        basic_stability_btn = gr.Button("⚙️ Configure Basic Stability", variant="primary", size="lg")
                        stability_status = gr.Textbox(label="Stability Status", lines=3)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### **Step 3: Run Training**")
                        guided_training_btn = gr.Button("🏃‍♂️ Start Guided Training", variant="primary", size="lg")
                        training_status = gr.Textbox(label="Training Status", lines=3)
                
                # Quick start controls
                num_steps = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Number of Training Steps")
                
                # Reset button for error recovery
                reset_btn = gr.Button("🔄 Reset Model", variant="secondary", size="sm")
                
                # Connect quick start buttons
                simple_model_btn.click(
                    fn=simulator.create_simple_model,
                    outputs=model_status
                )
                
                basic_stability_btn.click(
                    fn=simulator.configure_basic_stability,
                    outputs=stability_status
                )
                
                guided_training_btn.click(
                    fn=simulator.run_guided_training,
                    inputs=[num_steps],
                    outputs=training_status
                )
                
                reset_btn.click(
                    fn=simulator.reset_model,
                    outputs=[model_status, stability_status, training_status]
                )
            
            # Tab 2: Advanced Setup
            with gr.Tab("⚙️ Advanced Setup"):
                gr.Markdown("### Create custom models and configurations")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### **Custom Model Creation**")
                        model_type = gr.Dropdown(
                            choices=["Sequential", "Deep", "Wide"],
                            value="Sequential",
                            label="Model Architecture"
                        )
                        input_size = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Input Size")
                        hidden_size = gr.Slider(minimum=10, maximum=500, value=50, step=10, label="Hidden Size")
                        output_size = gr.Slider(minimum=1, maximum=50, value=1, step=1, label="Output Size")
                        
                        create_advanced_btn = gr.Button("🏗️ Create Advanced Model", variant="primary")
                        advanced_model_status = gr.Textbox(label="Advanced Model Status", lines=3)
                    
                    with gr.Column():
                        gr.Markdown("### **Advanced Stability Configuration**")
                        clipping_type = gr.Dropdown(
                            choices=["NORM", "VALUE", "GLOBAL_NORM", "ADAPTIVE", "LAYER_WISE", "PERCENTILE", "EXPONENTIAL"],
                            value="NORM",
                            label="Gradient Clipping Type"
                        )
                        max_norm = gr.Slider(minimum=0.1, maximum=10.0, value=1.0, step=0.1, label="Max Norm")
                        adaptive_threshold = gr.Slider(minimum=0.01, maximum=2.0, value=0.1, step=0.01, label="Adaptive Threshold")
                        
                        nan_handling = gr.Dropdown(
                            choices=["DETECT", "REPLACE", "SKIP", "GRADIENT_ZEROING", "ADAPTIVE", "GRADIENT_SCALING"],
                            value="ADAPTIVE",
                            label="NaN/Inf Handling"
                        )
                        
                        config_advanced_btn = gr.Button("🔧 Configure Advanced Stability", variant="primary")
                        advanced_stability_status = gr.Textbox(label="Advanced Stability Status", lines=3)
                
                # Connect advanced setup buttons
                create_advanced_btn.click(
                    fn=simulator.create_advanced_model,
                    inputs=[model_type, input_size, hidden_size, output_size],
                    outputs=advanced_model_status
                )
                
                config_advanced_btn.click(
                    fn=simulator.configure_advanced_stability,
                    inputs=[clipping_type, max_norm, nan_handling, adaptive_threshold],
                    outputs=advanced_stability_status
                )
            
            # Tab 3: Training & Analysis
            with gr.Tab("🏃‍♂️ Training & Analysis"):
                gr.Markdown("### Run training and analyze results")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### **Custom Training**")
                        batch_size = gr.Slider(minimum=8, maximum=128, value=32, step=8, label="Batch Size")
                        introduce_nan_prob = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="NaN Probability")
                        introduce_inf_prob = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.01, label="Inf Probability")
                        introduce_overflow_prob = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="Overflow Probability")
                        
                        custom_training_btn = gr.Button("🎯 Run Custom Training", variant="primary")
                        custom_training_status = gr.Textbox(label="Custom Training Status", lines=6)
                    
                    with gr.Column():
                        gr.Markdown("### **Analysis Tools**")
                        plot_btn = gr.Button("📈 Generate Plots", variant="primary")
                        summary_btn = gr.Button("📋 Training Summary", variant="primary")
                        status_btn = gr.Button("📊 Workflow Status", variant="secondary")
                
                with gr.Row():
                    plot1 = gr.Plot(label="Training Progress")
                    plot2 = gr.Plot(label="Numerical Stability")
                
                with gr.Row():
                    summary_text = gr.Textbox(label="Training Summary", lines=10)
                    workflow_status = gr.Textbox(label="Workflow Status", lines=8)
                
                # Connect training & analysis buttons
                custom_training_btn.click(
                    fn=simulator.run_custom_training,
                    inputs=[batch_size, introduce_nan_prob, introduce_inf_prob, introduce_overflow_prob],
                    outputs=custom_training_status
                )
                
                plot_btn.click(
                    fn=simulator.generate_training_plots,
                    outputs=[plot1, plot2]
                )
                
                summary_btn.click(
                    fn=simulator.get_training_summary,
                    outputs=summary_text
                )
                
                status_btn.click(
                    fn=simulator.get_workflow_status,
                    outputs=workflow_status
                )
            
            # Tab 4: Error Handling
            with gr.Tab("🛡️ Error Handling"):
                gr.Markdown("### Monitor and recover from errors")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### **Error Recovery**")
                        reset_all_btn = gr.Button("🔄 Reset Everything", variant="secondary", size="lg")
                        clear_errors_btn = gr.Button("🧹 Clear Error Count", variant="secondary", size="sm")
                        
                        error_info = gr.Markdown("""
                        **Error Handling Features:**
                        - ✅ Input validation for all parameters
                        - ✅ Graceful error recovery
                        - ✅ Detailed error logging
                        - ✅ User-friendly error messages
                        - ✅ Automatic fallback configurations
                        - ✅ Memory management and cleanup
                        """)
                    
                    with gr.Column():
                        gr.Markdown("### **System Status**")
                        system_status_btn = gr.Button("📊 Check System Status", variant="primary")
                        system_status = gr.Textbox(label="System Status", lines=10)
                
                # Connect error handling buttons
                reset_all_btn.click(
                    fn=simulator.reset_model,
                    outputs=[model_status, stability_status, training_status, advanced_model_status, advanced_stability_status]
                )
                
                clear_errors_btn.click(
                    fn=lambda: setattr(simulator, 'error_count', 0),
                    outputs=workflow_status
                )
                
                system_status_btn.click(
                    fn=simulator.get_workflow_status,
                    outputs=system_status
                )
        
        # Footer with workflow guidance and error handling info
        gr.Markdown("---")
        gr.Markdown("""
        ### 💡 **Workflow Guidance & Error Handling**
        
        **🚀 Quick Start**: Perfect for beginners - follow the guided workflow to learn the basics
        **⚙️ Advanced Setup**: For experienced users who want custom configurations
        **🏃‍♂️ Training & Analysis**: Run training and analyze results with comprehensive tools
        **🛡️ Error Handling**: Monitor errors and recover gracefully with built-in tools
        
        **Pro Tip**: Start with Quick Start to understand the concepts, then explore Advanced Setup for customization!
        **Error Recovery**: Use the Reset buttons to recover from errors and start fresh.
        """)
        
        gr.Markdown("""
        ---
        **Enhanced Interface** | Built with Gradio | Blatam Academy Project
        
        Designed for intuitive learning and experimentation with numerical stability concepts.
        Robust error handling ensures a smooth user experience even when issues occur.
        """)
    
    return interface


def main():
    """Main function to launch the enhanced interface."""
    try:
        interface = create_enhanced_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7864,  # Different port for the enhanced interface
            share=True,
            show_error=True,
            show_tips=True
        )
    except Exception as e:
        print(f"Failed to launch interface: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
