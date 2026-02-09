"""
Config Generator - Generador de configuraciones
=================================================

Genera archivos de configuración para proyectos de Deep Learning.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Generador de configuraciones"""
    
    def __init__(self):
        """Inicializa el generador de configuraciones"""
        pass
    
    def generate_training_config(
        self,
        config_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera archivo de configuración YAML para entrenamiento"""
        
        config_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_content = f'''# Training Configuration
# {'=' * 60}
# Configuración para entrenamiento de modelos de Deep Learning

# Model Configuration
model:
  name: "{keywords.get('model_architecture', 'transformer')}"
  model_name: "bert-base-uncased"  # Para transformers
  num_labels: null  # Ajustar según necesidad
  task_type: "{keywords.get('ai_type', 'classification')}"

# Training Configuration
training:
  batch_size: 32
  learning_rate: 2.0e-5
  num_epochs: 3
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  warmup_steps: 500
  
  # Mixed Precision
  use_mixed_precision: true
  
  # Early Stopping
  early_stopping:
    enabled: true
    patience: 7
    min_delta: 0.0
    mode: "min"  # min or max

# Data Configuration
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  num_workers: 4
  pin_memory: true

# Optimizer Configuration
optimizer:
  type: "adamw"  # adamw, adam, sgd
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Scheduler Configuration
scheduler:
  type: "cosine"  # cosine, step, exponential, plateau
  T_max: 10  # Para cosine
  step_size: 10  # Para step
  gamma: 0.1  # Para step/exponential

# Logging and Checkpointing
logging:
  log_dir: "./logs"
  log_steps: 100
  save_steps: 1000
  eval_steps: 500
  
checkpointing:
  enabled: true
  checkpoint_dir: "./checkpoints"
  save_best: true
  metric_for_best: "loss"

# Device Configuration
device:
  type: "cuda"  # cuda or cpu
  device_id: 0  # Para multi-GPU

# Experiment Tracking
tracking:
  enabled: true
  type: "tensorboard"  # tensorboard or wandb
  project_name: "{project_info['name']}"
'''
        
        (config_dir / "training_config.yaml").write_text(yaml_content, encoding="utf-8")
    
    def generate_model_config(
        self,
        config_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera archivo de configuración para el modelo"""
        
        config_dir.mkdir(parents=True, exist_ok=True)
        
        if keywords.get("is_diffusion"):
            model_config = f'''# Diffusion Model Configuration
model_id: "runwayml/stable-diffusion-v1-5"
use_xl: false
num_inference_steps: 50
guidance_scale: 7.5
image_height: 512
image_width: 512
'''
        elif keywords.get("is_transformer") or keywords.get("is_llm"):
            model_config = f'''# Transformer/LLM Configuration
model_name: "bert-base-uncased"
task_type: "{keywords.get('ai_type', 'classification')}"
num_labels: null
max_length: 512
'''
        else:
            model_config = f'''# Custom Model Configuration
input_size: 768
hidden_sizes: [128, 64, 32]
num_classes: 10
dropout_rate: 0.2
use_batch_norm: true
'''
        
        (config_dir / "model_config.yaml").write_text(model_config, encoding="utf-8")

