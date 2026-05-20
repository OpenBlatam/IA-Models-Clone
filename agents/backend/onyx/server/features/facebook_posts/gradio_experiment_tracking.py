#!/usr/bin/env python3
"""
🔬 Gradio Interface for Experiment Tracking System
==================================================

Interactive Gradio interface for the experiment tracking system.
Provides configuration, monitoring, and visualization capabilities for
TensorBoard and Weights & Biases integration.
"""

import os
import sys
import time
import json
import random
import subprocess
import threading
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our experiment tracking system
from experiment_tracking import (
    ExperimentTracker, ExperimentConfig, create_experiment_config,
    create_experiment_tracker, TrainingMetrics, LanguageModelMetrics,
    DiffusionModelMetrics, ProblemDefinition, DatasetAnalysis,
    analyze_dataset_automatically, create_problem_definition_template
)

# Import our centralized logging configuration
from logging_config import (
    get_logger, log_training_step, log_numerical_issue, 
    log_system_event, log_error_with_context, log_performance_metrics
)

warnings.filterwarnings('ignore')

logger = get_logger(__name__)

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Global experiment tracker
current_tracker: Optional[ExperimentTracker] = None
training_thread: Optional[threading.Thread] = None
stop_training = False

# Default configurations
DEFAULT_CONFIGS = {
    "Basic Tracking": {
        "experiment_name": "basic_experiment",
        "project_name": "blatam_academy_facebook_posts",
        "enable_tensorboard": True,
        "enable_wandb": False,
        "log_interval": 10
    },
    "Full Tracking": {
        "experiment_name": "full_experiment",
        "project_name": "blatam_academy_facebook_posts",
        "enable_tensorboard": True,
        "enable_wandb": True,
        "log_interval": 5,
        "log_gradients": True,
        "log_images": True,
        "log_text": True
    },
    "Numerical Stability Focus": {
        "experiment_name": "numerical_stability_experiment",
        "project_name": "blatam_academy_facebook_posts",
        "enable_tensorboard": True,
        "enable_wandb": True,
        "log_interval": 1,
        "log_gradient_norms": True,
        "log_nan_inf_counts": True,
        "log_clipping_stats": True
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port: int = 6006) -> int:
    """Find an available port starting from start_port."""
    port = start_port
    while port < start_port + 100:
        if check_port_available(port):
            return port
        port += 1
    return start_port

def launch_tensorboard(log_dir: str, port: int = 6006) -> str:
    """Launch TensorBoard in a subprocess."""
    try:
        if not os.path.exists(log_dir):
            return f"Error: Log directory {log_dir} does not exist"
        
        # Find available port
        available_port = find_available_port(port)
        
        # Launch TensorBoard
        cmd = f"tensorboard --logdir={log_dir} --port={available_port} --host=0.0.0.0"
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment to see if it starts successfully
        time.sleep(2)
        if process.poll() is None:
            return f"TensorBoard launched successfully on port {available_port}. Access at: http://localhost:{available_port}"
        else:
            stdout, stderr = process.communicate()
            return f"Failed to launch TensorBoard: {stderr}"
            
    except Exception as e:
        return f"Error launching TensorBoard: {e}"

def create_dummy_model() -> nn.Module:
    """Create a dummy model for demonstration purposes."""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

def simulate_training_data(num_steps: int = 100) -> List[Dict[str, float]]:
    """Simulate training data for demonstration."""
    data = []
    for step in tqdm(range(num_steps), desc="Generating Training Data", unit="step"):
        # Simulate realistic training metrics
        base_loss = 2.0 * np.exp(-step / 50)  # Exponential decay
        noise = np.random.normal(0, 0.1)
        loss = max(0.01, base_loss + noise)
        
        accuracy = min(0.95, 0.3 + 0.6 * (1 - np.exp(-step / 30)))
        accuracy += np.random.normal(0, 0.02)
        accuracy = max(0.0, min(1.0, accuracy))
        
        gradient_norm = np.random.exponential(0.8) + 0.1
        nan_count = np.random.poisson(0.05)
        inf_count = np.random.poisson(0.02)
        
        data.append({
            'loss': loss,
            'accuracy': accuracy,
            'gradient_norm': gradient_norm,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'clipping_applied': gradient_norm > 1.5,
            'clipping_threshold': 1.5 if gradient_norm > 1.5 else None,
            'training_time': np.random.exponential(0.05),
            'memory_usage': np.random.uniform(100, 500),
            'gpu_utilization': np.random.uniform(20, 80)
        })
    
    return data

def create_dummy_transformer_model():
    """Create a dummy Transformers model for demonstration purposes."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        # Create a simple config for demonstration
        config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256
        )
        
        # Create model and tokenizer
        model = AutoModel.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to create dummy transformer model: {e}")
        return None, None

def simulate_language_model_training(num_steps: int = 100) -> List[Dict[str, float]]:
    """Simulate language model training data for demonstration."""
    data = []
    for step in tqdm(range(num_steps), desc="Generating Language Model Data", unit="step"):
        # Simulate realistic language model metrics
        base_perplexity = 50.0 * np.exp(-step / 30)  # Exponential decay
        noise = np.random.normal(0, 2.0)
        perplexity = max(1.0, base_perplexity + noise)
        
        # BLEU score improvement
        bleu_score = min(0.8, 0.1 + 0.6 * (1 - np.exp(-step / 25)))
        bleu_score += np.random.normal(0, 0.02)
        bleu_score = max(0.0, min(1.0, bleu_score))
        
        # Token accuracy
        token_accuracy = min(0.95, 0.3 + 0.6 * (1 - np.exp(-step / 20)))
        token_accuracy += np.random.normal(0, 0.01)
        token_accuracy = max(0.0, min(1.0, token_accuracy))
        
        # Attention weights norm
        attention_norm = np.random.exponential(0.5) + 0.1
        
        # Sequence length variation
        sequence_length = np.random.randint(64, 512)
        
        data.append({
            'perplexity': perplexity,
            'bleu_score': bleu_score,
            'token_accuracy': token_accuracy,
            'attention_weights_norm': attention_norm,
            'sequence_length': sequence_length,
            'loss': np.random.exponential(0.8) + 0.1,
            'gradient_norm': np.random.exponential(0.6) + 0.1,
            'nan_count': np.random.poisson(0.02),
            'inf_count': np.random.poisson(0.01)
        })
    
    return data

def start_language_model_training_simulation(num_steps: int, 
                                           learning_rate: float,
                                           batch_size: int,
                                           max_grad_norm: float,
                                           model_type: str) -> str:
    """Start a language model training simulation."""
    global current_tracker, training_thread, stop_training
    
    if not current_tracker:
        return "❌ No experiment tracker available. Please create one first."
    
    if training_thread and training_thread.is_alive():
        return "❌ Training already in progress. Please stop it first."
    
    stop_training = False
    
    def lm_training_simulation():
        try:
            # Create dummy transformer model
            model, tokenizer = create_dummy_transformer_model()
            if model and tokenizer:
                # Log transformer model
                current_tracker.log_transformer_model(model, tokenizer)
            
            # Simulate training data
            training_data = simulate_language_model_training(num_steps)
            
            for step, metrics in enumerate(tqdm(training_data, desc="Language Model Training", unit="step")):
                if stop_training:
                    break
                
                # Log training step
                current_tracker.log_training_step(
                    loss=metrics['loss'],
                    learning_rate=learning_rate,
                    gradient_norm=metrics['gradient_norm'],
                    nan_count=metrics['nan_count'],
                    inf_count=metrics['inf_count'],
                    training_time=np.random.exponential(0.05),
                    memory_usage=np.random.uniform(200, 800),
                    gpu_utilization=np.random.uniform(30, 90)
                )
                
                # Log language model specific metrics
                current_tracker.log_language_model_metrics(
                    perplexity=metrics['perplexity'],
                    bleu_score=metrics['bleu_score'],
                    token_accuracy=metrics['token_accuracy'],
                    attention_weights_norm=metrics['attention_weights_norm'],
                    sequence_length=metrics['sequence_length']
                )
                
                # Log attention analysis periodically
                if step % 20 == 0 and model:
                    # Simulate attention weights
                    batch_size_sim = 2
                    seq_len = metrics['sequence_length']
                    attention_weights = torch.randn(batch_size_sim, 4, seq_len, seq_len)  # 4 attention heads
                    current_tracker.log_attention_analysis(attention_weights, layer_idx=0)
                
                # Log gradient flow analysis periodically
                if step % 30 == 0 and model:
                    current_tracker.log_gradient_flow_analysis(model)
                
                # Small delay to simulate real training
                time.sleep(0.1)
            
            # Log final epoch
            if not stop_training:
                final_metrics = {
                    'final_perplexity': training_data[-1]['perplexity'],
                    'final_bleu_score': training_data[-1]['bleu_score'],
                    'final_token_accuracy': training_data[-1]['token_accuracy'],
                    'total_steps': len(training_data),
                    'model_type': model_type
                }
                current_tracker.log_epoch(1, final_metrics)
                
        except Exception as e:
            logger.error(f"Language model training simulation error: {e}")
    
    # Start training thread
    training_thread = threading.Thread(target=lm_training_simulation, daemon=True)
    training_thread.start()
    
    return f"🚀 Language Model Training Simulation Started!\n\n" \
           f"Model Type: {model_type}\n" \
           f"Steps: {num_steps}\n" \
           f"Learning Rate: {learning_rate}\n" \
           f"Batch Size: {batch_size}\n" \
           f"Max Gradient Norm: {max_grad_norm}\n\n" \
           f"Check the experiment tracker for real-time updates!"

def create_language_model_visualization() -> Tuple[plt.Figure, str]:
    """Create language model specific visualization."""
    global current_tracker
    
    if not current_tracker:
        return None, "❌ No experiment tracker available."
    
    try:
        # Create language model visualization
        figure = current_tracker.create_language_model_visualization()
        
        if not figure:
            return None, "❌ No language model visualization data available."
        
        # Get language model summary
        lm_summary = current_tracker.get_language_model_summary()
        
        if "error" in lm_summary:
            summary_text = f"❌ Error getting language model summary: {lm_summary['error']}"
        else:
            summary_text = f"📊 Language Model Training Summary\n\n"
            summary_text += f"Total Metrics: {lm_summary['total_lm_metrics']}\n\n"
            
            if lm_summary['perplexity_stats']['final']:
                summary_text += f"Final Perplexity: {lm_summary['perplexity_stats']['final']:.4f}\n"
                summary_text += f"Perplexity Range: {lm_summary['perplexity_stats']['min']:.4f} - {lm_summary['perplexity_stats']['max']:.4f}\n"
            
            if lm_summary['bleu_score_stats']['final']:
                summary_text += f"Final BLEU Score: {lm_summary['bleu_score_stats']['final']:.4f}\n"
                summary_text += f"BLEU Score Range: {lm_summary['bleu_score_stats']['min']:.4f} - {lm_summary['bleu_score_stats']['max']:.4f}\n"
            
            if lm_summary['token_accuracy_stats']['final']:
                summary_text += f"Final Token Accuracy: {lm_summary['token_accuracy_stats']['final']:.4f}\n"
                summary_text += f"Token Accuracy Range: {lm_summary['token_accuracy_stats']['min']:.4f} - {lm_summary['token_accuracy_stats']['max']:.4f}\n"
            
            summary_text += f"\nVisualization has been logged to TensorBoard and Weights & Biases!"
        
        return figure, summary_text
        
    except Exception as e:
        logger.error(f"Failed to create language model visualization: {e}")
        return None, f"❌ Error creating language model visualization: {e}"

# =============================================================================
# DIFFUSION MODEL FUNCTIONS
# =============================================================================

def create_dummy_diffusion_pipeline():
    """Create a dummy diffusion pipeline for demonstration purposes."""
    try:
        # This is a mock function - in a real scenario, you'd load actual models
        # For demonstration, we'll create mock data structures
        
        class MockDiffusionPipeline:
            def __init__(self):
                self.device = "cpu"
                self.dtype = torch.float32
                self.safety_checker = None
                self.watermarker = None
            
            def parameters(self):
                # Return mock parameters
                return [torch.nn.Parameter(torch.randn(100, 100))]
        
        class MockUNet:
            def __init__(self):
                self.in_channels = 4
                self.out_channels = 4
                self.block_out_channels = (128, 256, 512, 512)
                self.down_block_types = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
                self.up_block_types = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
                self.cross_attention_dim = 768
                self.attention_head_dim = 8
                self.num_attention_heads = 8
            
            def parameters(self):
                return [torch.nn.Parameter(torch.randn(100, 100))]
        
        class MockVAE:
            def __init__(self):
                self.in_channels = 3
                self.out_channels = 3
                self.latent_channels = 4
                self.sample_size = 512
            
            def parameters(self):
                return [torch.nn.Parameter(torch.randn(100, 100))]
        
        class MockScheduler:
            def __init__(self):
                self.num_train_timesteps = 1000
                self.beta_start = 0.00085
                self.beta_end = 0.012
                self.beta_schedule = "scaled_linear"
        
        return MockDiffusionPipeline(), MockUNet(), MockVAE(), MockScheduler()
        
    except Exception as e:
        logger.error(f"Failed to create dummy diffusion pipeline: {e}")
        return None, None, None, None

def simulate_diffusion_generation(num_generations: int = 50) -> List[Dict[str, float]]:
    """Simulate diffusion model generation data for demonstration."""
    try:
        generation_data = []
        
        for i in tqdm(range(num_generations), desc="Generating Diffusion Data", unit="generation"):
            # Simulate realistic diffusion metrics
            noise_level = 1.0 - (i / num_generations) * 0.8  # Decreasing noise
            denoising_steps = np.random.randint(20, 50)
            guidance_scale = np.random.uniform(7.0, 15.0)
            image_quality_score = np.random.uniform(0.6, 0.95)
            generation_time = np.random.uniform(2.0, 8.0)
            memory_usage = np.random.uniform(2000, 8000)  # MB
            scheduler_step = i
            noise_prediction_loss = np.random.uniform(0.1, 0.5)
            classifier_free_guidance = np.random.choice([True, False])
            prompt_embedding_norm = np.random.uniform(0.8, 1.2)
            
            # Simulate cross-attention weights (4 attention heads, 64x64 attention matrix)
            cross_attention_weights = torch.randn(4, 64, 64)
            
            # Simulate latent space statistics
            latent_space_stats = {
                "mean": np.random.uniform(-0.1, 0.1),
                "std": np.random.uniform(0.8, 1.2),
                "min": np.random.uniform(-2.0, -1.0),
                "max": np.random.uniform(1.0, 2.0)
            }
            
            generation_data.append({
                'noise_level': noise_level,
                'denoising_steps': denoising_steps,
                'guidance_scale': guidance_scale,
                'image_quality_score': image_quality_score,
                'generation_time': generation_time,
                'memory_usage': memory_usage,
                'scheduler_step': scheduler_step,
                'noise_prediction_loss': noise_prediction_loss,
                'classifier_free_guidance': classifier_free_guidance,
                'prompt_embedding_norm': prompt_embedding_norm,
                'cross_attention_weights': cross_attention_weights,
                'latent_space_stats': latent_space_stats
            })
        
        return generation_data
        
    except Exception as e:
        logger.error(f"Failed to simulate diffusion generation: {e}")
        return []

def start_diffusion_generation_simulation(num_generations: int, 
                                        guidance_scale: float,
                                        denoising_steps: int,
                                        model_type: str) -> str:
    """Start a diffusion model generation simulation."""
    global current_tracker, training_thread, stop_training
    
    if not current_tracker:
        return "❌ No experiment tracker available. Please create one first."
    
    if training_thread and training_thread.is_alive():
        return "❌ Generation already in progress. Please stop it first."
    
    stop_training = False
    
    def diffusion_generation_simulation():
        try:
            # Create dummy diffusion pipeline
            pipeline, unet, vae, scheduler = create_dummy_diffusion_pipeline()
            
            if pipeline:
                # Log diffusion pipeline
                current_tracker.log_diffusion_pipeline(pipeline, unet, vae, scheduler)
            
            # Simulate generation data
            generation_data = simulate_diffusion_generation(num_generations)
            
            for step, metrics in enumerate(tqdm(generation_data, desc="Diffusion Generation", unit="step")):
                if stop_training:
                    break
                
                # Log diffusion metrics
                current_tracker.log_diffusion_metrics(
                    noise_level=metrics['noise_level'],
                    denoising_steps=metrics['denoising_steps'],
                    guidance_scale=metrics['guidance_scale'],
                    image_quality_score=metrics['image_quality_score'],
                    generation_time=metrics['generation_time'],
                    memory_usage=metrics['memory_usage'],
                    scheduler_step=metrics['scheduler_step'],
                    noise_prediction_loss=metrics['noise_prediction_loss'],
                    classifier_free_guidance=metrics['classifier_free_guidance'],
                    prompt_embedding_norm=metrics['prompt_embedding_norm'],
                    cross_attention_weights=metrics['cross_attention_weights'],
                    latent_space_stats=metrics['latent_space_stats']
                )
                
                # Log individual generation step details
                if step % 10 == 0:
                    # Simulate noise prediction and latent tensors
                    noise_prediction = torch.randn(1, 4, 64, 64)
                    latent = torch.randn(1, 4, 64, 64)
                    current_tracker.log_diffusion_generation_step(
                        step, noise_prediction, latent, metrics['guidance_scale']
                    )
                
                # Small delay to simulate real generation
                time.sleep(0.1)
            
            # Log final generation summary
            if not stop_training:
                final_metrics = {
                    'total_generations': len(generation_data),
                    'final_quality_score': generation_data[-1]['image_quality_score'],
                    'avg_generation_time': np.mean([m['generation_time'] for m in generation_data]),
                    'model_type': model_type
                }
                current_tracker.log_epoch(1, final_metrics)
                
        except Exception as e:
            logger.error(f"Diffusion generation simulation error: {e}")
    
    # Start generation thread
    training_thread = threading.Thread(target=diffusion_generation_simulation, daemon=True)
    training_thread.start()
    
    return f"🎨 Diffusion Generation Simulation Started!\n\n" \
           f"Model Type: {model_type}\n" \
           f"Generations: {num_generations}\n" \
           f"Guidance Scale: {guidance_scale}\n" \
           f"Denoising Steps: {denoising_steps}\n\n" \
           f"Check the experiment tracker for real-time updates!"

def create_diffusion_visualization() -> Tuple[plt.Figure, str]:
    """Create diffusion model specific visualization."""
    global current_tracker
    
    if not current_tracker:
        return None, "❌ No experiment tracker available."
    
    try:
        # Create diffusion visualization
        figure = current_tracker.create_diffusion_visualization()
        
        if not figure:
            return None, "❌ No diffusion visualization data available."
        
        # Get diffusion summary
        diffusion_summary = current_tracker.get_diffusion_summary()
        
        if "error" in diffusion_summary:
            summary_text = f"❌ Error getting diffusion summary: {diffusion_summary['error']}"
        else:
            summary_text = f"🎨 Diffusion Model Generation Summary\n\n"
            summary_text += f"Total Generations: {diffusion_summary['total_diffusion_metrics']}\n\n"
            
            if diffusion_summary['quality_score_stats']['final']:
                summary_text += f"Final Quality Score: {diffusion_summary['quality_score_stats']['final']:.4f}\n"
                summary_text += f"Quality Score Range: {diffusion_summary['quality_score_stats']['min']:.4f} - {diffusion_summary['quality_score_stats']['max']:.4f}\n"
            
            if diffusion_summary['generation_time_stats']['final']:
                summary_text += f"Final Generation Time: {diffusion_summary['generation_time_stats']['final']:.2f}s\n"
                summary_text += f"Generation Time Range: {diffusion_summary['generation_time_stats']['min']:.2f}s - {diffusion_summary['generation_time_stats']['max']:.2f}s\n"
            
            if diffusion_summary['guidance_scale_stats']['final']:
                summary_text += f"Final Guidance Scale: {diffusion_summary['guidance_scale_stats']['final']:.2f}\n"
                summary_text += f"Guidance Scale Range: {diffusion_summary['guidance_scale_stats']['min']:.2f} - {diffusion_summary['guidance_scale_stats']['max']:.2f}\n"
            
            if diffusion_summary['denoising_steps_stats']['final']:
                summary_text += f"Final Denoising Steps: {diffusion_summary['denoising_steps_stats']['final']}\n"
                summary_text += f"Denoising Steps Range: {diffusion_summary['denoising_steps_stats']['min']} - {diffusion_summary['denoising_steps_stats']['max']}\n"
            
            summary_text += f"\nVisualization has been logged to TensorBoard and Weights & Biases!"
        
        return figure, summary_text
        
    except Exception as e:
        logger.error(f"Failed to create diffusion visualization: {e}")
        return None, f"❌ Error creating diffusion visualization: {e}"

# =============================================================================
# GRADIO INTERFACE FUNCTIONS
# =============================================================================

def create_problem_definition_interface(problem_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Create a problem definition from user inputs."""
    try:
        problem_def = ProblemDefinition(
            problem_title=problem_data.get("problem_title", ""),
            problem_description=problem_data.get("problem_description", ""),
            problem_type=problem_data.get("problem_type", "classification"),
            domain=problem_data.get("domain", "general"),
            primary_objective=problem_data.get("primary_objective", ""),
            success_metrics=problem_data.get("success_metrics", []),
            baseline_performance=problem_data.get("baseline_performance"),
            target_performance=problem_data.get("target_performance"),
            computational_constraints=problem_data.get("computational_constraints", ""),
            time_constraints=problem_data.get("time_constraints", ""),
            accuracy_requirements=problem_data.get("accuracy_requirements", ""),
            interpretability_requirements=problem_data.get("interpretability_requirements", ""),
            business_value=problem_data.get("business_value", ""),
            stakeholders=problem_data.get("stakeholders", []),
            deployment_context=problem_data.get("deployment_context", "")
        )
        
        # Convert to dict for JSON serialization
        problem_dict = {
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
        
        status = f"✅ Problem Definition Created!\n\n" \
                f"Title: {problem_def.problem_title}\n" \
                f"Type: {problem_def.problem_type}\n" \
                f"Domain: {problem_def.domain}\n" \
                f"Objective: {problem_def.primary_objective}\n" \
                f"Metrics: {', '.join(problem_def.success_metrics)}"
        
        return status, problem_dict
        
    except Exception as e:
        logger.error(f"Failed to create problem definition: {e}")
        return f"❌ Error creating problem definition: {e}", {}

def create_dataset_analysis_interface(dataset_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Create a dataset analysis from user inputs."""
    try:
        dataset_analysis = DatasetAnalysis(
            dataset_name=dataset_data.get("dataset_name", ""),
            dataset_source=dataset_data.get("dataset_source", ""),
            dataset_version=dataset_data.get("dataset_version", ""),
            dataset_size=dataset_data.get("dataset_size"),
            input_shape=dataset_data.get("input_shape"),
            output_shape=dataset_data.get("output_shape"),
            feature_count=dataset_data.get("feature_count"),
            class_count=dataset_data.get("class_count"),
            data_types=dataset_data.get("data_types", []),
            missing_values_pct=dataset_data.get("missing_values_pct"),
            duplicate_records_pct=dataset_data.get("duplicate_records_pct"),
            outlier_pct=dataset_data.get("outlier_pct"),
            class_imbalance_ratio=dataset_data.get("class_imbalance_ratio"),
            train_size=dataset_data.get("train_size"),
            val_size=dataset_data.get("val_size"),
            test_size=dataset_data.get("test_size"),
            data_split_strategy=dataset_data.get("data_split_strategy", ""),
            normalization_needed=dataset_data.get("normalization_needed", False),
            encoding_needed=dataset_data.get("encoding_needed", False),
            augmentation_strategy=dataset_data.get("augmentation_strategy", ""),
            preprocessing_steps=dataset_data.get("preprocessing_steps", [])
        )
        
        # Convert to dict for JSON serialization
        analysis_dict = {
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
        
        status = f"✅ Dataset Analysis Created!\n\n" \
                f"Dataset: {dataset_analysis.dataset_name}\n" \
                f"Size: {dataset_analysis.dataset_size:,} samples\n" \
                f"Features: {dataset_analysis.feature_count}\n" \
                f"Classes: {dataset_analysis.class_count}\n" \
                f"Missing Values: {dataset_analysis.missing_values_pct}%\n" \
                f"Preprocessing Steps: {len(dataset_analysis.preprocessing_steps)}"
        
        return status, analysis_dict
        
    except Exception as e:
        logger.error(f"Failed to create dataset analysis: {e}")
        return f"❌ Error creating dataset analysis: {e}", {}

def create_experiment_tracker_interface(config_data: Dict[str, Any]) -> str:
    """Create and configure an experiment tracker."""
    global current_tracker
    
    try:
        # Close existing tracker
        if current_tracker:
            current_tracker.close()
        
        # Create new configuration
        config = create_experiment_config(**config_data)
        
        # Add problem definition and dataset analysis if provided
        if "problem_definition" in config_data:
            problem_def = ProblemDefinition(**config_data["problem_definition"])
            config.problem_definition = problem_def
            
        if "dataset_analysis" in config_data:
            dataset_analysis = DatasetAnalysis(**config_data["dataset_analysis"])
            config.dataset_analysis = dataset_analysis
        
        # Create tracker
        current_tracker = create_experiment_tracker(config)
        
        # Log hyperparameters
        current_tracker.log_hyperparameters(config_data)
        
        # Log model architecture if we have a model
        dummy_model = create_dummy_model()
        current_tracker.log_model_architecture(dummy_model)
        
        status = f"✅ Experiment tracker created successfully!\n\n" \
               f"Experiment: {config.experiment_name}\n" \
               f"Project: {config.project_name}\n" \
               f"TensorBoard: {'✅' if config.enable_tensorboard else '❌'}\n" \
               f"Weights & Biases: {'✅' if config.enable_wandb else '❌'}\n" \
               f"Log Interval: {config.log_interval} steps"
        
        if config.problem_definition:
            status += f"\n\n🎯 Problem Definition: {config.problem_definition.problem_title}"
        
        if config.dataset_analysis:
            status += f"\n📊 Dataset Analysis: {config.dataset_analysis.dataset_name}"
            
        return status
               
    except Exception as e:
        logger.error(f"Failed to create experiment tracker: {e}")
        return f"❌ Error creating experiment tracker: {e}"

def start_training_simulation(num_steps: int, 
                            learning_rate: float,
                            batch_size: int,
                            max_grad_norm: float) -> str:
    """Start a training simulation with the experiment tracker."""
    global current_tracker, training_thread, stop_training
    
    if not current_tracker:
        return "❌ No experiment tracker available. Please create one first."
    
    if training_thread and training_thread.is_alive():
        return "❌ Training already in progress. Please stop it first."
    
    stop_training = False
    
    def training_simulation():
        try:
            # Simulate training data
            training_data = simulate_training_data(num_steps)
            
            for step, metrics in enumerate(tqdm(training_data, desc="Training Simulation")):
                if stop_training:
                    break
                
                # Log training step
                current_tracker.log_training_step(
                    loss=metrics['loss'],
                    accuracy=metrics['accuracy'],
                    learning_rate=learning_rate,
                    gradient_norm=metrics['gradient_norm'],
                    nan_count=metrics['nan_count'],
                    inf_count=metrics['inf_count'],
                    clipping_applied=metrics['clipping_applied'],
                    clipping_threshold=metrics['clipping_threshold'],
                    training_time=metrics['training_time'],
                    memory_usage=metrics['memory_usage'],
                    gpu_utilization=metrics['gpu_utilization']
                )
                
                # Log gradients periodically
                if step % 10 == 0:
                    dummy_model = create_dummy_model()
                    current_tracker.log_gradients(dummy_model)
                
                # Small delay to simulate real training
                time.sleep(0.1)
            
            # Log final epoch
            if not stop_training:
                final_metrics = {
                    'final_loss': training_data[-1]['loss'],
                    'final_accuracy': training_data[-1]['accuracy'],
                    'total_steps': len(training_data),
                    'avg_training_time': np.mean([m['training_time'] for m in training_data])
                }
                current_tracker.log_epoch(1, final_metrics)
                
        except Exception as e:
            logger.error(f"Training simulation error: {e}")
    
    # Start training thread
    training_thread = threading.Thread(target=training_simulation, daemon=True)
    training_thread.start()
    
    return f"🚀 Training simulation started!\n\n" \
           f"Steps: {num_steps}\n" \
           f"Learning Rate: {learning_rate}\n" \
           f"Batch Size: {batch_size}\n" \
           f"Max Gradient Norm: {max_grad_norm}\n\n" \
           f"Check the experiment tracker for real-time updates!"

def stop_training_simulation() -> str:
    """Stop the training simulation."""
    global stop_training, training_thread
    
    stop_training = True
    
    if training_thread and training_thread.is_alive():
        training_thread.join(timeout=5.0)
        training_thread = None
    
    return "⏹️ Training simulation stopped."

def get_experiment_status() -> str:
    """Get current experiment status."""
    global current_tracker
    
    if not current_tracker:
        return "❌ No experiment tracker available."
    
    try:
        summary = current_tracker.get_experiment_summary()
        
        if "error" in summary:
            return f"❌ Error getting status: {summary['error']}"
        
        status = f"📊 Experiment Status\n\n"
        status += f"Experiment: {summary['experiment_name']}\n"
        status += f"Project: {summary['project_name']}\n"
        status += f"Total Steps: {summary['total_steps']}\n"
        status += f"Current Epoch: {summary['current_epoch']}\n"
        status += f"Current Step: {summary['current_step']}\n\n"
        
        if summary['loss_stats']['final']:
            status += f"Final Loss: {summary['loss_stats']['final']:.4f}\n"
            status += f"Loss Range: {summary['loss_stats']['min']:.4f} - {summary['loss_stats']['max']:.4f}\n"
        
        if summary['accuracy_stats']['final']:
            status += f"Final Accuracy: {summary['accuracy_stats']['final']:.4f}\n"
            status += f"Accuracy Range: {summary['accuracy_stats']['min']:.4f} - {summary['accuracy_stats']['max']:.4f}\n"
        
        status += f"\nNumerical Stability:\n"
        status += f"Total NaN Count: {summary['numerical_stability']['total_nan_count']}\n"
        status += f"Total Inf Count: {summary['numerical_stability']['total_inf_count']}\n"
        status += f"Steps with NaN: {summary['numerical_stability']['steps_with_nan']}\n"
        status += f"Steps with Inf: {summary['numerical_stability']['steps_with_inf']}\n\n"
        
        status += f"Checkpoints: {summary['checkpoints']}\n"
        status += f"Tracking Systems:\n"
        status += f"  TensorBoard: {'✅' if summary['tracking_systems']['tensorboard'] else '❌'}\n"
        status += f"  Weights & Biases: {'✅' if summary['tracking_systems']['wandb'] else '❌'}"
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get experiment status: {e}")
        return f"❌ Error getting status: {e}"

def create_training_visualization() -> Tuple[plt.Figure, str]:
    """Create training visualization plots."""
    global current_tracker
    
    if not current_tracker:
        return None, "❌ No experiment tracker available."
    
    try:
        # Create visualization
        viz_data = current_tracker.create_visualization()
        
        if not viz_data:
            return None, "❌ No visualization data available."
        
        figure = viz_data['figure']
        metrics_summary = viz_data['metrics_summary']
        
        # Create summary text
        summary_text = f"📈 Training Visualization Created\n\n"
        summary_text += f"Total Steps: {metrics_summary['total_steps']}\n"
        if metrics_summary['final_loss']:
            summary_text += f"Final Loss: {metrics_summary['final_loss']:.4f}\n"
        if metrics_summary['final_accuracy']:
            summary_text += f"Final Accuracy: {metrics_summary['final_accuracy']:.4f}\n"
        summary_text += f"Total NaN Count: {metrics_summary['total_nan_count']}\n"
        summary_text += f"Total Inf Count: {metrics_summary['total_inf_count']}\n\n"
        summary_text += f"Visualization has been logged to TensorBoard and Weights & Biases!"
        
        return figure, summary_text
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        return None, f"❌ Error creating visualization: {e}"

def save_experiment_config(config_data: Dict[str, Any], filename: str) -> str:
    """Save experiment configuration to file."""
    try:
        if not filename.endswith('.json'):
            filename += '.json'
        
        config_path = Path("configs") / filename
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return f"✅ Configuration saved to: {config_path}"
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return f"❌ Error saving configuration: {e}"

def load_experiment_config(filename: str) -> Tuple[Dict[str, Any], str]:
    """Load experiment configuration from file."""
    try:
        if not filename.endswith('.json'):
            filename += '.json'
        
        config_path = Path("configs") / filename
        
        if not config_path.exists():
            return {}, f"❌ Configuration file not found: {config_path}"
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return config_data, f"✅ Configuration loaded from: {config_path}"
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}, f"❌ Error loading configuration: {e}"

def launch_tensorboard_interface() -> str:
    """Launch TensorBoard for the current experiment."""
    global current_tracker
    
    if not current_tracker:
        return "❌ No experiment tracker available."
    
    try:
        # Get TensorBoard directory from config
        tb_dir = current_tracker.config.tensorboard_dir
        
        # Launch TensorBoard
        result = launch_tensorboard(tb_dir)
        return result
        
    except Exception as e:
        logger.error(f"Failed to launch TensorBoard: {e}")
        return f"❌ Error launching TensorBoard: {e}"

def get_default_config(config_name: str) -> Tuple[Dict[str, Any], str]:
    """Get a default configuration by name."""
    if config_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[config_name], f"✅ Loaded default configuration: {config_name}"
    else:
        return {}, f"❌ Unknown default configuration: {config_name}"

# =============================================================================
# GRADIO INTERFACE SETUP
# =============================================================================

def create_gradio_interface():
    """Create the Gradio interface for experiment tracking."""
    
    with gr.Blocks(title="🔬 Experiment Tracking System", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🔬 Experiment Tracking System
        
        Comprehensive experiment tracking with TensorBoard and Weights & Biases integration.
        Monitor training progress, visualize metrics, and manage experiments efficiently.
        **Now with enhanced Transformers and Language Model support!**
        """)
        
        with gr.Tabs():
            
            # Problem Definition Tab
            with gr.Tab("🎯 Problem Definition"):
                gr.Markdown("### Define Your Machine Learning Problem")
                gr.Markdown("Start every project with a clear problem definition to ensure focused development and measurable success.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Core problem description
                        problem_title = gr.Textbox(
                            label="Problem Title",
                            placeholder="e.g., Image Classification for Medical Diagnosis"
                        )
                        problem_description = gr.Textbox(
                            label="Problem Description",
                            lines=3,
                            placeholder="Detailed description of the problem you're solving..."
                        )
                        problem_type = gr.Dropdown(
                            label="Problem Type",
                            choices=["classification", "regression", "generation", "segmentation", "detection", "clustering"],
                            value="classification"
                        )
                        domain = gr.Dropdown(
                            label="Domain",
                            choices=["computer_vision", "nlp", "audio", "tabular", "time_series", "multimodal", "general"],
                            value="general"
                        )
                        
                    with gr.Column(scale=1):
                        # Objectives and metrics
                        primary_objective = gr.Textbox(
                            label="Primary Objective",
                            placeholder="e.g., Maximize classification accuracy while minimizing false negatives"
                        )
                        success_metrics = gr.Textbox(
                            label="Success Metrics (comma-separated)",
                            placeholder="e.g., accuracy, precision, recall, f1_score"
                        )
                        baseline_performance = gr.Number(
                            label="Baseline Performance",
                            placeholder="Current/expected baseline performance"
                        )
                        target_performance = gr.Number(
                            label="Target Performance",
                            placeholder="Target performance to achieve"
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Constraints
                        computational_constraints = gr.Textbox(
                            label="Computational Constraints",
                            placeholder="e.g., Must run on single GPU, inference < 100ms"
                        )
                        time_constraints = gr.Textbox(
                            label="Time Constraints",
                            placeholder="e.g., Development deadline, training time limits"
                        )
                        accuracy_requirements = gr.Textbox(
                            label="Accuracy Requirements",
                            placeholder="e.g., Minimum 95% accuracy required for production"
                        )
                        
                    with gr.Column(scale=1):
                        # Business context
                        interpretability_requirements = gr.Textbox(
                            label="Interpretability Requirements",
                            placeholder="e.g., Model decisions must be explainable to medical professionals"
                        )
                        business_value = gr.Textbox(
                            label="Business Value",
                            placeholder="e.g., Reduce diagnostic time by 50%, save $1M annually"
                        )
                        stakeholders = gr.Textbox(
                            label="Stakeholders (comma-separated)",
                            placeholder="e.g., medical_team, product_manager, compliance_officer"
                        )
                        deployment_context = gr.Textbox(
                            label="Deployment Context",
                            placeholder="e.g., Hospital environment, real-time inference, batch processing"
                        )
                
                with gr.Row():
                    create_problem_btn = gr.Button("🎯 Create Problem Definition", variant="primary")
                    load_problem_template_btn = gr.Button("📋 Load Template")
                    
                problem_output = gr.Textbox(
                    label="Problem Definition Status",
                    lines=5, interactive=False
                )
                
                # Store problem definition data
                problem_definition_data = gr.JSON(visible=False)
            
            # Dataset Analysis Tab
            with gr.Tab("📊 Dataset Analysis"):
                gr.Markdown("### Analyze Your Dataset")
                gr.Markdown("Comprehensive dataset analysis ensures you understand your data characteristics and preprocessing needs.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Dataset metadata
                        dataset_name = gr.Textbox(
                            label="Dataset Name",
                            placeholder="e.g., CIFAR-10, Custom Medical Images"
                        )
                        dataset_source = gr.Textbox(
                            label="Dataset Source",
                            placeholder="e.g., Kaggle, Internal Collection, Public Repository"
                        )
                        dataset_version = gr.Textbox(
                            label="Dataset Version",
                            placeholder="e.g., v1.0, 2024-01-15"
                        )
                        dataset_size = gr.Number(
                            label="Dataset Size (total samples)",
                            placeholder="Total number of samples"
                        )
                        
                    with gr.Column(scale=1):
                        # Data characteristics
                        feature_count = gr.Number(
                            label="Feature Count",
                            placeholder="Number of features/dimensions"
                        )
                        class_count = gr.Number(
                            label="Class Count",
                            placeholder="Number of classes (for classification)"
                        )
                        data_types = gr.Textbox(
                            label="Data Types (comma-separated)",
                            placeholder="e.g., float32, int64, string, categorical"
                        )
                        input_shape = gr.Textbox(
                            label="Input Shape",
                            placeholder="e.g., (224, 224, 3) for images, (512,) for text"
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Data quality
                        missing_values_pct = gr.Number(
                            label="Missing Values %",
                            placeholder="Percentage of missing values"
                        )
                        duplicate_records_pct = gr.Number(
                            label="Duplicate Records %",
                            placeholder="Percentage of duplicate records"
                        )
                        outlier_pct = gr.Number(
                            label="Outliers %",
                            placeholder="Percentage of outlier values"
                        )
                        class_imbalance_ratio = gr.Number(
                            label="Class Imbalance Ratio",
                            placeholder="Ratio of majority to minority class"
                        )
                        
                    with gr.Column(scale=1):
                        # Data distribution
                        train_size = gr.Number(
                            label="Training Set Size",
                            placeholder="Number of training samples"
                        )
                        val_size = gr.Number(
                            label="Validation Set Size",
                            placeholder="Number of validation samples"
                        )
                        test_size = gr.Number(
                            label="Test Set Size",
                            placeholder="Number of test samples"
                        )
                        data_split_strategy = gr.Textbox(
                            label="Data Split Strategy",
                            placeholder="e.g., Random 70/15/15, Stratified, Time-based"
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Preprocessing
                        normalization_needed = gr.Checkbox(
                            label="Normalization Needed",
                            value=False
                        )
                        encoding_needed = gr.Checkbox(
                            label="Encoding Needed",
                            value=False
                        )
                        
                    with gr.Column(scale=1):
                        augmentation_strategy = gr.Textbox(
                            label="Augmentation Strategy",
                            placeholder="e.g., Random crops, rotations, noise injection"
                        )
                        preprocessing_steps = gr.Textbox(
                            label="Preprocessing Steps (comma-separated)",
                            placeholder="e.g., resize, normalize, tokenize, encode_categorical"
                        )
                
                with gr.Row():
                    create_dataset_analysis_btn = gr.Button("📊 Create Dataset Analysis", variant="primary")
                    auto_analyze_btn = gr.Button("🔍 Auto-Analyze Sample")
                    
                dataset_analysis_output = gr.Textbox(
                    label="Dataset Analysis Status",
                    lines=5, interactive=False
                )
                
                # Store dataset analysis data
                dataset_analysis_data = gr.JSON(visible=False)
            
            # Configuration Tab
            with gr.Tab("⚙️ Configuration"):
                gr.Markdown("### Experiment Configuration")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Basic settings
                        experiment_name = gr.Textbox(
                            label="Experiment Name",
                            value="gradient_clipping_experiment",
                            placeholder="Enter experiment name"
                        )
                        
                        project_name = gr.Textbox(
                            label="Project Name",
                            value="blatam_academy_facebook_posts",
                            placeholder="Enter project name"
                        )
                        
                        run_name = gr.Textbox(
                            label="Run Name (Optional)",
                            placeholder="Enter run name or leave empty for auto-generation"
                        )
                        
                        tags = gr.Textbox(
                            label="Tags (comma-separated)",
                            placeholder="deep-learning, numerical-stability, research"
                        )
                        
                        notes = gr.Textbox(
                            label="Notes",
                            placeholder="Enter experiment notes",
                            lines=3
                        )
                    
                    with gr.Column(scale=1):
                        # Tracking settings
                        enable_tensorboard = gr.Checkbox(
                            label="Enable TensorBoard",
                            value=True
                        )
                        
                        enable_wandb = gr.Checkbox(
                            label="Enable Weights & Biases",
                            value=True
                        )
                        
                        log_interval = gr.Slider(
                            label="Log Interval (steps)",
                            minimum=1,
                            maximum=100,
                            value=10,
                            step=1
                        )
                        
                        save_interval = gr.Slider(
                            label="Save Interval (steps)",
                            minimum=100,
                            maximum=10000,
                            value=1000,
                            step=100
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Advanced settings
                        log_gradients = gr.Checkbox(
                            label="Log Gradients",
                            value=True
                        )
                        
                        log_images = gr.Checkbox(
                            label="Log Images",
                            value=False
                        )
                        
                        log_text = gr.Checkbox(
                            label="Log Text",
                            value=False
                        )
                        
                        log_gradient_norms = gr.Checkbox(
                            label="Log Gradient Norms",
                            value=True
                        )
                        
                        log_nan_inf_counts = gr.Checkbox(
                            label="Log NaN/Inf Counts",
                            value=True
                        )
                    
                    with gr.Column(scale=1):
                        # File paths
                        tensorboard_dir = gr.Textbox(
                            label="TensorBoard Directory",
                            value="runs/tensorboard"
                        )
                        
                        model_save_dir = gr.Textbox(
                            label="Model Save Directory",
                            value="models"
                        )
                        
                        config_save_dir = gr.Textbox(
                            label="Config Save Directory",
                            value="configs"
                        )
                
                with gr.Row():
                    create_btn = gr.Button("🚀 Create Experiment Tracker", variant="primary")
                    status_output = gr.Textbox(
                        label="Status",
                        lines=5,
                        interactive=False
                    )
                
                with gr.Row():
                    # Default configurations
                    default_config_dropdown = gr.Dropdown(
                        label="Load Default Configuration",
                        choices=list(DEFAULT_CONFIGS.keys()),
                        value=None
                    )
                    
                    load_default_btn = gr.Button("📋 Load Default")
                
                with gr.Row():
                    # Save/Load configuration
                    config_filename = gr.Textbox(
                        label="Configuration Filename",
                        placeholder="my_experiment_config",
                        value=""
                    )
                    
                    save_config_btn = gr.Button("💾 Save Configuration")
                    load_config_btn = gr.Button("📂 Load Configuration")
                
                # Problem Definition event handlers
                create_problem_btn.click(
                    fn=lambda *args: create_problem_definition_interface({
                        "problem_title": args[0],
                        "problem_description": args[1],
                        "problem_type": args[2],
                        "domain": args[3],
                        "primary_objective": args[4],
                        "success_metrics": args[5].split(",") if args[5] else [],
                        "baseline_performance": args[6],
                        "target_performance": args[7],
                        "computational_constraints": args[8],
                        "time_constraints": args[9],
                        "accuracy_requirements": args[10],
                        "interpretability_requirements": args[11],
                        "business_value": args[12],
                        "stakeholders": args[13].split(",") if args[13] else [],
                        "deployment_context": args[14]
                    }),
                    inputs=[
                        problem_title, problem_description, problem_type, domain,
                        primary_objective, success_metrics, baseline_performance, target_performance,
                        computational_constraints, time_constraints, accuracy_requirements,
                        interpretability_requirements, business_value, stakeholders, deployment_context
                    ],
                    outputs=[problem_output, problem_definition_data]
                )
                
                load_problem_template_btn.click(
                    fn=lambda pt: create_problem_definition_template(pt, "general"),
                    inputs=[problem_type],
                    outputs=[problem_definition_data]
                )
                
                # Dataset Analysis event handlers
                create_dataset_analysis_btn.click(
                    fn=lambda *args: create_dataset_analysis_interface({
                        "dataset_name": args[0],
                        "dataset_source": args[1],
                        "dataset_version": args[2],
                        "dataset_size": args[3],
                        "feature_count": args[4],
                        "class_count": args[5],
                        "data_types": args[6].split(",") if args[6] else [],
                        "input_shape": args[7],
                        "missing_values_pct": args[8],
                        "duplicate_records_pct": args[9],
                        "outlier_pct": args[10],
                        "class_imbalance_ratio": args[11],
                        "train_size": args[12],
                        "val_size": args[13],
                        "test_size": args[14],
                        "data_split_strategy": args[15],
                        "normalization_needed": args[16],
                        "encoding_needed": args[17],
                        "augmentation_strategy": args[18],
                        "preprocessing_steps": args[19].split(",") if args[19] else []
                    }),
                    inputs=[
                        dataset_name, dataset_source, dataset_version, dataset_size,
                        feature_count, class_count, data_types, input_shape,
                        missing_values_pct, duplicate_records_pct, outlier_pct, class_imbalance_ratio,
                        train_size, val_size, test_size, data_split_strategy,
                        normalization_needed, encoding_needed, augmentation_strategy, preprocessing_steps
                    ],
                    outputs=[dataset_analysis_output, dataset_analysis_data]
                )
                
                auto_analyze_btn.click(
                    fn=lambda: (
                        "🔍 Auto-analysis feature will analyze your dataset automatically when you provide data.\n"
                        "This feature can detect data types, missing values, outliers, and recommend preprocessing steps.",
                        {}
                    ),
                    outputs=[dataset_analysis_output, dataset_analysis_data]
                )

                # Event handlers for configuration
                create_btn.click(
                    fn=lambda *args: create_experiment_tracker_interface({
                        "experiment_name": args[0],
                        "project_name": args[1],
                        "run_name": args[2],
                        "tags": args[3].split(",") if args[3] else [],
                        "notes": args[4],
                        "enable_tensorboard": args[5],
                        "enable_wandb": args[6],
                        "log_interval": args[7],
                        "save_interval": args[8],
                        "log_gradients": args[9],
                        "log_images": args[10],
                        "log_text": args[11],
                        "log_gradient_norms": args[12],
                        "log_nan_inf_counts": args[13],
                        "tensorboard_dir": args[14],
                        "model_save_dir": args[15],
                        "config_save_dir": args[16],
                        "problem_definition": problem_definition_data.value if hasattr(problem_definition_data, 'value') and problem_definition_data.value else None,
                        "dataset_analysis": dataset_analysis_data.value if hasattr(dataset_analysis_data, 'value') and dataset_analysis_data.value else None
                    }),
                    inputs=[
                        experiment_name, project_name, run_name, tags, notes,
                        enable_tensorboard, enable_wandb, log_interval, save_interval,
                        log_gradients, log_images, log_text, log_gradient_norms, log_nan_inf_counts,
                        tensorboard_dir, model_save_dir, config_save_dir
                    ],
                    outputs=status_output
                )
                
                load_default_btn.click(
                    fn=get_default_config,
                    inputs=[default_config_dropdown],
                    outputs=[gr.JSON(), gr.Textbox()]
                )
                
                save_config_btn.click(
                    fn=save_experiment_config,
                    inputs=[
                        gr.JSON(value={
                            "experiment_name": experiment_name.value,
                            "project_name": project_name.value,
                            "run_name": run_name.value,
                            "tags": tags.value.split(",") if tags.value else [],
                            "notes": notes.value,
                            "enable_tensorboard": enable_tensorboard.value,
                            "enable_wandb": enable_wandb.value,
                            "log_interval": log_interval.value,
                            "save_interval": save_interval.value,
                            "log_gradients": log_gradients.value,
                            "log_images": log_images.value,
                            "log_text": log_text.value,
                            "log_gradient_norms": log_gradient_norms.value,
                            "log_nan_inf_counts": log_nan_inf_counts.value,
                            "tensorboard_dir": tensorboard_dir.value,
                            "model_save_dir": model_save_dir.value,
                            "config_save_dir": config_save_dir.value
                        }),
                        config_filename
                    ],
                    outputs=status_output
                )
                
                load_config_btn.click(
                    fn=load_experiment_config,
                    inputs=[config_filename],
                    outputs=[gr.JSON(), status_output]
                )
            
            # Training Tab
            with gr.Tab("🚀 Training"):
                gr.Markdown("### Training Simulation")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        num_steps = gr.Slider(
                            label="Number of Training Steps",
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10
                        )
                        
                        learning_rate = gr.Slider(
                            label="Learning Rate",
                            minimum=0.0001,
                            maximum=0.1,
                            value=0.001,
                            step=0.0001
                        )
                    
                    with gr.Column(scale=1):
                        batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=8,
                            maximum=128,
                            value=32,
                            step=8
                        )
                        
                        max_grad_norm = gr.Slider(
                            label="Max Gradient Norm",
                            minimum=0.1,
                            maximum=10.0,
                            value=1.0,
                            step=0.1
                        )
                
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Training", variant="stop")
                    status_btn = gr.Button("📊 Get Status")
                
                training_output = gr.Textbox(
                    label="Training Status",
                    lines=8,
                    interactive=False
                )
                
                # Event handlers for training
                start_btn.click(
                    fn=start_training_simulation,
                    inputs=[num_steps, learning_rate, batch_size, max_grad_norm],
                    outputs=training_output
                )
                
                stop_btn.click(
                    fn=stop_training_simulation,
                    outputs=training_output
                )
                
                status_btn.click(
                    fn=get_experiment_status,
                    outputs=training_output
                )
            
            # Checkpointing Tab
            with gr.Tab("💾 Checkpointing"):
                gr.Markdown("### Advanced Checkpoint Management")
                gr.Markdown("Manage model checkpoints with advanced features including validation, comparison, and recovery.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Checkpoint configuration
                        gr.Markdown("#### Checkpoint Configuration")
                        save_interval = gr.Slider(
                            label="Save Interval (epochs)",
                            minimum=1,
                            maximum=50,
                            value=5,
                            step=1
                        )
                        max_checkpoints = gr.Slider(
                            label="Max Checkpoints to Keep",
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1
                        )
                        save_best_only = gr.Checkbox(
                            label="Save Best Only",
                            value=False
                        )
                        monitor_metric = gr.Textbox(
                            label="Monitor Metric",
                            value="val_loss",
                            placeholder="Metric to monitor for best checkpoint"
                        )
                        monitor_mode = gr.Dropdown(
                            label="Monitor Mode",
                            choices=["min", "max"],
                            value="min"
                        )
                        
                    with gr.Column(scale=1):
                        # Checkpoint actions
                        gr.Markdown("#### Checkpoint Actions")
                        setup_checkpointing_btn = gr.Button("🔧 Setup Advanced Checkpointing", variant="primary")
                        save_checkpoint_btn = gr.Button("💾 Save Checkpoint")
                        load_checkpoint_btn = gr.Button("📂 Load Checkpoint")
                        list_checkpoints_btn = gr.Button("📋 List Checkpoints")
                        
                with gr.Row():
                    with gr.Column(scale=1):
                        # Checkpoint management
                        gr.Markdown("#### Checkpoint Management")
                        checkpoint_id_input = gr.Textbox(
                            label="Checkpoint ID",
                            placeholder="Enter checkpoint ID to load/delete/export"
                        )
                        export_path = gr.Textbox(
                            label="Export Path",
                            placeholder="Path to export checkpoint to"
                        )
                        delete_checkpoint_btn = gr.Button("🗑️ Delete Checkpoint", variant="stop")
                        export_checkpoint_btn = gr.Button("📤 Export Checkpoint")
                        validate_checkpoint_btn = gr.Button("✅ Validate Checkpoint")
                        
                    with gr.Column(scale=1):
                        # Checkpoint comparison
                        gr.Markdown("#### Checkpoint Comparison")
                        compare_checkpoints_input = gr.Textbox(
                            label="Checkpoint IDs to Compare (comma-separated)",
                            placeholder="checkpoint_id1,checkpoint_id2,checkpoint_id3"
                        )
                        compare_checkpoints_btn = gr.Button("🔍 Compare Checkpoints")
                        
                with gr.Row():
                    # Checkpoint information
                    checkpoint_summary_btn = gr.Button("📊 Get Checkpoint Summary")
                    get_best_checkpoint_btn = gr.Button("🏆 Get Best Checkpoint")
                    get_latest_checkpoint_btn = gr.Button("🕒 Get Latest Checkpoint")
                
                # Checkpoint outputs
                checkpoint_output = gr.Textbox(
                    label="Checkpoint Status",
                    lines=8,
                    interactive=False
                )
                checkpoint_list_output = gr.JSON(
                    label="Checkpoint List",
                    visible=True
                )
                checkpoint_comparison_output = gr.JSON(
                    label="Checkpoint Comparison",
                    visible=True
                )
                
                # Event handlers for checkpointing
                setup_checkpointing_btn.click(
                    fn=setup_advanced_checkpointing,
                    inputs=[save_interval, max_checkpoints, save_best_only, monitor_metric, monitor_mode],
                    outputs=checkpoint_output
                )
                
                save_checkpoint_btn.click(
                    fn=save_checkpoint_interface,
                    outputs=checkpoint_output
                )
                
                load_checkpoint_btn.click(
                    fn=load_checkpoint_interface,
                    inputs=[checkpoint_id_input],
                    outputs=checkpoint_output
                )
                
                list_checkpoints_btn.click(
                    fn=list_checkpoints_interface,
                    outputs=[checkpoint_list_output, checkpoint_output]
                )
                
                delete_checkpoint_btn.click(
                    fn=delete_checkpoint_interface,
                    inputs=[checkpoint_id_input],
                    outputs=checkpoint_output
                )
                
                export_checkpoint_btn.click(
                    fn=export_checkpoint_interface,
                    inputs=[checkpoint_id_input, export_path],
                    outputs=checkpoint_output
                )
                
                validate_checkpoint_btn.click(
                    fn=validate_checkpoint_interface,
                    inputs=[checkpoint_id_input],
                    outputs=checkpoint_output
                )
                
                compare_checkpoints_btn.click(
                    fn=compare_checkpoints_interface,
                    inputs=[compare_checkpoints_input],
                    outputs=[checkpoint_comparison_output, checkpoint_output]
                )
                
                checkpoint_summary_btn.click(
                    fn=get_checkpoint_summary_interface,
                    outputs=checkpoint_output
                )
                
                get_best_checkpoint_btn.click(
                    fn=get_best_checkpoint_interface,
                    outputs=checkpoint_output
                )
                
                get_latest_checkpoint_btn.click(
                    fn=get_latest_checkpoint_interface,
                    outputs=checkpoint_output
                )
            
            # Monitoring Tab
            with gr.Tab("📊 Monitoring"):
                gr.Markdown("### Experiment Monitoring and Visualization")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        viz_btn = gr.Button("📈 Create Visualization", variant="primary")
                        summary_btn = gr.Button("📋 Get Summary")
                        launch_tb_btn = gr.Button("🔍 Launch TensorBoard")
                    
                    with gr.Column(scale=1):
                        # Visualization output
                        viz_output = gr.Plot(label="Training Progress")
                        summary_output = gr.Textbox(
                            label="Experiment Summary",
                            lines=10,
                            interactive=False
                        )
                
                # Event handlers for monitoring
                viz_btn.click(
                    fn=create_training_visualization,
                    outputs=[viz_output, summary_output]
                )
                
                summary_btn.click(
                    fn=get_experiment_status,
                    outputs=summary_output
                )
                
                launch_tb_btn.click(
                    fn=launch_tensorboard_interface,
                    outputs=summary_output
                )
            
            # New Transformers Tab
            with gr.Tab("🤖 Transformers & Language Models"):
                gr.Markdown("### Language Model Training and Analysis")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Model configuration
                        model_type = gr.Dropdown(
                            label="Model Type",
                            choices=[
                                "BERT", "GPT-2", "T5", "RoBERTa", "DistilBERT",
                                "Custom Transformer", "Language Model"
                            ],
                            value="BERT"
                        )
                        
                        num_steps_lm = gr.Slider(
                            label="Number of Training Steps",
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10
                        )
                        
                        learning_rate_lm = gr.Slider(
                            label="Learning Rate",
                            minimum=0.00001,
                            maximum=0.01,
                            value=0.0001,
                            step=0.00001
                        )
                    
                    with gr.Column(scale=1):
                        # Training parameters
                        batch_size_lm = gr.Slider(
                            label="Batch Size",
                            minimum=4,
                            maximum=64,
                            value=16,
                            step=4
                        )
                        
                        max_grad_norm_lm = gr.Slider(
                            label="Max Gradient Norm",
                            minimum=0.1,
                            maximum=5.0,
                            value=1.0,
                            step=0.1
                        )
                        
                        sequence_length = gr.Slider(
                            label="Max Sequence Length",
                            minimum=64,
                            maximum=1024,
                            value=512,
                            step=64
                        )
                
                with gr.Row():
                    start_lm_btn = gr.Button("🚀 Start Language Model Training", variant="primary")
                    stop_lm_btn = gr.Button("⏹️ Stop Training", variant="stop")
                    lm_status_btn = gr.Button("📊 Get LM Status")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Language model specific controls
                        lm_viz_btn = gr.Button("📈 Create LM Visualization", variant="primary")
                        attention_analysis_btn = gr.Button("🔍 Analyze Attention")
                        gradient_flow_btn = gr.Button("🌊 Analyze Gradient Flow")
                    
                    with gr.Column(scale=1):
                        # Output displays
                        lm_training_output = gr.Textbox(
                            label="Language Model Training Status",
                            lines=6,
                            interactive=False
                        )
                
                with gr.Row():
                    # Language model visualization
                    lm_viz_output = gr.Plot(label="Language Model Training Progress")
                    lm_summary_output = gr.Textbox(
                        label="Language Model Summary",
                        lines=8,
                        interactive=False
                    )
                
                # Event handlers for language model tab
                start_lm_btn.click(
                    fn=start_language_model_training_simulation,
                    inputs=[num_steps_lm, learning_rate_lm, batch_size_lm, max_grad_norm_lm, model_type],
                    outputs=lm_training_output
                )
                
                stop_lm_btn.click(
                    fn=stop_training_simulation,
                    outputs=lm_training_output
                )
                
                lm_status_btn.click(
                    fn=get_experiment_status,
                    outputs=lm_training_output
                )
                
                lm_viz_btn.click(
                    fn=create_language_model_visualization,
                    outputs=[lm_viz_output, lm_summary_output]
                )
                
                attention_analysis_btn.click(
                    fn=lambda: "🔍 Attention analysis will be performed during training simulation. Check TensorBoard for detailed attention heatmaps and statistics.",
                    outputs=lm_summary_output
                )
                
                gradient_flow_btn.click(
                    fn=lambda: "🌊 Gradient flow analysis will be performed during training simulation. Check TensorBoard for detailed gradient flow statistics.",
                    outputs=lm_summary_output
                )
            
            # New Diffusion Models Tab
            with gr.Tab("🎨 Diffusion Models & Image Generation"):
                gr.Markdown("### Diffusion Model Generation and Analysis")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Model configuration
                        diffusion_model_type = gr.Dropdown(
                            label="Model Type",
                            choices=[
                                "Stable Diffusion", "DDIM", "DDPM", "Custom Diffusion",
                                "Latent Diffusion", "Image Generation"
                            ],
                            value="Stable Diffusion"
                        )
                        
                        num_generations = gr.Slider(
                            label="Number of Generations",
                            minimum=10,
                            maximum=200,
                            value=50,
                            step=10
                        )
                        
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5
                        )
                    
                    with gr.Column(scale=1):
                        # Generation parameters
                        denoising_steps = gr.Slider(
                            label="Denoising Steps",
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=5
                        )
                        
                        image_size = gr.Dropdown(
                            label="Image Size",
                            choices=["256x256", "512x512", "768x768", "1024x1024"],
                            value="512x512"
                        )
                        
                        batch_size_diffusion = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1
                        )
                
                with gr.Row():
                    start_diffusion_btn = gr.Button("🎨 Start Diffusion Generation", variant="primary")
                    stop_diffusion_btn = gr.Button("⏹️ Stop Generation", variant="stop")
                    diffusion_status_btn = gr.Button("📊 Get Diffusion Status")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Diffusion model specific controls
                        diffusion_viz_btn = gr.Button("📈 Create Diffusion Visualization", variant="primary")
                        attention_heatmap_btn = gr.Button("🔥 Show Attention Heatmaps")
                        latent_analysis_btn = gr.Button("🔍 Analyze Latent Space")
                    
                    with gr.Column(scale=1):
                        # Output displays
                        diffusion_output = gr.Textbox(
                            label="Diffusion Generation Status",
                            lines=6,
                            interactive=False
                        )
                
                with gr.Row():
                    # Diffusion model visualization
                    diffusion_viz_output = gr.Plot(label="Diffusion Generation Progress")
                    diffusion_summary_output = gr.Textbox(
                        label="Diffusion Model Summary",
                        lines=8,
                        interactive=False
                    )
                
                # Event handlers for diffusion model tab
                start_diffusion_btn.click(
                    fn=start_diffusion_generation_simulation,
                    inputs=[num_generations, guidance_scale, denoising_steps, diffusion_model_type],
                    outputs=diffusion_output
                )
                
                stop_diffusion_btn.click(
                    fn=stop_training_simulation,
                    outputs=diffusion_output
                )
                
                diffusion_status_btn.click(
                    fn=get_experiment_status,
                    outputs=diffusion_output
                )
                
                diffusion_viz_btn.click(
                    fn=create_diffusion_visualization,
                    outputs=[diffusion_viz_output, diffusion_summary_output]
                )
                
                attention_heatmap_btn.click(
                    fn=lambda: "🔥 Attention heatmaps will be generated during generation simulation. Check TensorBoard for detailed cross-attention visualizations.",
                    outputs=diffusion_summary_output
                )
                
                latent_analysis_btn.click(
                    fn=lambda: "🔍 Latent space analysis will be performed during generation simulation. Check TensorBoard for detailed latent space statistics.",
                    outputs=diffusion_summary_output
                )
        
        # Footer
        gr.Markdown("""
        ---
        **🔬 Experiment Tracking System** | Built with Gradio, TensorBoard, Weights & Biases, Transformers, and Diffusers
        
        Monitor your deep learning experiments with comprehensive logging, visualization, and tracking capabilities.
        **Enhanced support for language models, attention analysis, gradient flow monitoring, diffusion model generation, and advanced checkpointing.**
        """)
    
    return interface

# =============================================================================
# CHECKPOINTING INTERFACE FUNCTIONS
# =============================================================================

def setup_advanced_checkpointing(save_interval: int, max_checkpoints: int, save_best_only: bool, 
                                monitor_metric: str, monitor_mode: str) -> str:
    """Setup advanced checkpointing system."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        checkpoint_config = {
            'save_interval': save_interval,
            'max_checkpoints': max_checkpoints,
            'save_best_only': save_best_only,
            'monitor_metric': monitor_metric,
            'monitor_mode': monitor_mode,
            'save_optimizer': True,
            'save_scheduler': True,
            'save_metadata': True,
            'backup_checkpoints': True,
            'validate_checkpoints': True
        }
        
        experiment_tracker.setup_advanced_checkpointing(checkpoint_config)
        return f"✅ Advanced checkpointing system initialized with:\n" \
               f"• Save interval: {save_interval} epochs\n" \
               f"• Max checkpoints: {max_checkpoints}\n" \
               f"• Save best only: {save_best_only}\n" \
               f"• Monitor metric: {monitor_metric} ({monitor_mode})"
    
    except Exception as e:
        return f"❌ Failed to setup advanced checkpointing: {e}"

def save_checkpoint_interface() -> str:
    """Interface for saving a checkpoint."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        # Create dummy model and optimizer for demonstration
        import torch
        import torch.nn as nn
        
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate checkpoint data
        epoch = experiment_tracker.current_epoch + 1
        step = experiment_tracker.current_step + 100
        loss = 0.5 + np.random.normal(0, 0.1)
        metrics = {
            'val_loss': loss,
            'accuracy': 0.85 + np.random.normal(0, 0.05),
            'learning_rate': 0.001
        }
        
        checkpoint_id = experiment_tracker.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            loss=loss,
            metrics=metrics,
            tags=['demo', 'simulation'],
            description=f"Demo checkpoint from Gradio interface"
        )
        
        if checkpoint_id:
            return f"✅ Checkpoint saved successfully!\n" \
                   f"• Checkpoint ID: {checkpoint_id}\n" \
                   f"• Epoch: {epoch}, Step: {step}\n" \
                   f"• Loss: {loss:.4f}\n" \
                   f"• Metrics: {metrics}"
        else:
            return "⚠️ Checkpoint not saved (conditions not met or error occurred)"
    
    except Exception as e:
        return f"❌ Failed to save checkpoint: {e}"

def load_checkpoint_interface(checkpoint_id: str) -> str:
    """Interface for loading a checkpoint."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        if not checkpoint_id:
            return "❌ Please provide a checkpoint ID"
        
        # Create dummy model and optimizer for demonstration
        import torch
        import torch.nn as nn
        
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_data = experiment_tracker.load_checkpoint(
            checkpoint_id=checkpoint_id,
            model=model,
            optimizer=optimizer
        )
        
        if checkpoint_data:
            return f"✅ Checkpoint loaded successfully!\n" \
                   f"• Checkpoint ID: {checkpoint_id}\n" \
                   f"• Epoch: {checkpoint_data.get('epoch', 'N/A')}\n" \
                   f"• Step: {checkpoint_data.get('step', 'N/A')}\n" \
                   f"• Loss: {checkpoint_data.get('loss', 'N/A')}\n" \
                   f"• Model and optimizer states loaded"
        else:
            return f"❌ Failed to load checkpoint: {checkpoint_id}"
    
    except Exception as e:
        return f"❌ Failed to load checkpoint: {e}"

def list_checkpoints_interface():
    """Interface for listing checkpoints."""
    try:
        if not experiment_tracker:
            return [], "❌ No experiment tracker available. Please create one first."
        
        checkpoints = experiment_tracker.list_checkpoints()
        
        if checkpoints and not isinstance(checkpoints[0], dict):
            return [], f"❌ Error listing checkpoints: {checkpoints}"
        
        return checkpoints, f"✅ Found {len(checkpoints)} checkpoints"
    
    except Exception as e:
        return [], f"❌ Failed to list checkpoints: {e}"

def delete_checkpoint_interface(checkpoint_id: str) -> str:
    """Interface for deleting a checkpoint."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        if not checkpoint_id:
            return "❌ Please provide a checkpoint ID"
        
        success = experiment_tracker.delete_checkpoint(checkpoint_id)
        
        if success:
            return f"✅ Checkpoint deleted successfully: {checkpoint_id}"
        else:
            return f"❌ Failed to delete checkpoint: {checkpoint_id}"
    
    except Exception as e:
        return f"❌ Failed to delete checkpoint: {e}"

def export_checkpoint_interface(checkpoint_id: str, export_path: str) -> str:
    """Interface for exporting a checkpoint."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        if not checkpoint_id:
            return "❌ Please provide a checkpoint ID"
        
        if not export_path:
            export_path = f"exported_checkpoints/{checkpoint_id}.pt"
        
        success = experiment_tracker.export_checkpoint(checkpoint_id, export_path)
        
        if success:
            return f"✅ Checkpoint exported successfully!\n" \
                   f"• Checkpoint ID: {checkpoint_id}\n" \
                   f"• Export path: {export_path}"
        else:
            return f"❌ Failed to export checkpoint: {checkpoint_id}"
    
    except Exception as e:
        return f"❌ Failed to export checkpoint: {e}"

def validate_checkpoint_interface(checkpoint_id: str) -> str:
    """Interface for validating a checkpoint."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        if not checkpoint_id:
            return "❌ Please provide a checkpoint ID"
        
        is_valid = experiment_tracker.validate_checkpoint(checkpoint_id)
        
        if is_valid:
            return f"✅ Checkpoint validation successful: {checkpoint_id}"
        else:
            return f"❌ Checkpoint validation failed: {checkpoint_id}"
    
    except Exception as e:
        return f"❌ Failed to validate checkpoint: {e}"

def compare_checkpoints_interface(checkpoint_ids: str):
    """Interface for comparing checkpoints."""
    try:
        if not experiment_tracker:
            return {}, "❌ No experiment tracker available. Please create one first."
        
        if not checkpoint_ids:
            return {}, "❌ Please provide checkpoint IDs to compare"
        
        # Parse checkpoint IDs
        checkpoint_id_list = [cp_id.strip() for cp_id in checkpoint_ids.split(',')]
        
        comparison = experiment_tracker.compare_checkpoints(checkpoint_id_list)
        
        if comparison and 'error' not in comparison:
            return comparison, f"✅ Successfully compared {len(checkpoint_id_list)} checkpoints"
        else:
            return {}, f"❌ Failed to compare checkpoints: {comparison.get('error', 'Unknown error')}"
    
    except Exception as e:
        return {}, f"❌ Failed to compare checkpoints: {e}"

def get_checkpoint_summary_interface() -> str:
    """Interface for getting checkpoint summary."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        summary = experiment_tracker.get_checkpoint_summary()
        
        if summary and 'error' not in summary:
            return f"📊 Checkpoint Summary:\n" \
                   f"• Total checkpoints: {summary.get('total_checkpoints', 0)}\n" \
                   f"• Total size: {summary.get('total_size_mb', 0):.2f} MB\n" \
                   f"• Best checkpoint: {summary.get('best_checkpoint', 'None')}\n" \
                   f"• Latest checkpoint: {summary.get('latest_checkpoint', 'None')}"
        else:
            return f"❌ Failed to get checkpoint summary: {summary.get('error', 'Unknown error')}"
    
    except Exception as e:
        return f"❌ Failed to get checkpoint summary: {e}"

def get_best_checkpoint_interface() -> str:
    """Interface for getting best checkpoint."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        best_checkpoint = experiment_tracker.get_best_checkpoint()
        
        if best_checkpoint:
            return f"🏆 Best Checkpoint:\n" \
                   f"• Epoch: {best_checkpoint.get('epoch', 'N/A')}\n" \
                   f"• Step: {best_checkpoint.get('step', 'N/A')}\n" \
                   f"• Loss: {best_checkpoint.get('loss', 'N/A')}\n" \
                   f"• Metrics: {best_checkpoint.get('metrics', {})}"
        else:
            return "❌ No best checkpoint found"
    
    except Exception as e:
        return f"❌ Failed to get best checkpoint: {e}"

def get_latest_checkpoint_interface() -> str:
    """Interface for getting latest checkpoint."""
    try:
        if not experiment_tracker:
            return "❌ No experiment tracker available. Please create one first."
        
        latest_checkpoint = experiment_tracker.get_latest_checkpoint()
        
        if latest_checkpoint:
            return f"🕒 Latest Checkpoint:\n" \
                   f"• Epoch: {latest_checkpoint.get('epoch', 'N/A')}\n" \
                   f"• Step: {latest_checkpoint.get('step', 'N/A')}\n" \
                   f"• Loss: {latest_checkpoint.get('loss', 'N/A')}\n" \
                   f"• Metrics: {latest_checkpoint.get('metrics', {})}"
        else:
            return "❌ No latest checkpoint found"
    
    except Exception as e:
        return f"❌ Failed to get latest checkpoint: {e}"

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to launch the Gradio interface."""
    try:
        # Create the interface
        interface = create_gradio_interface()
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
