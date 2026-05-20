"""
Templates - Code Templates
==========================

Code templates for common deep learning workflows.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_training_template(
    model_type: str = 'transformer',
    use_pipeline: bool = True
) -> str:
    """
    Get training script template.
    
    Args:
        model_type: Type of model
        use_pipeline: Use high-level pipeline
        
    Returns:
        Training script template
    """
    if use_pipeline:
        template = '''"""
Training Script - Generated Template
====================================
"""

from pathlib import Path
from core.deep_learning.pipelines import TrainingPipeline
from core.deep_learning.presets import get_model_preset, get_training_preset
from core.deep_learning.utils import set_seed, get_device

# Set seed for reproducibility
set_seed(42)
device = get_device()

# Load your dataset here
# train_dataset = ...
# val_dataset = ...
# test_dataset = ...

# Get presets
model_config = get_model_preset('transformer_medium')
training_config = get_training_preset('standard')

# Create and run pipeline
pipeline = TrainingPipeline()
pipeline.setup(
    model_config=model_config,
    training_config=training_config,
    experiment_name="my_experiment"
)

results = pipeline.train(train_dataset, val_dataset, test_dataset)
print(f"Training completed! Test metrics: {results['test_metrics']}")
'''
    else:
        template = '''"""
Training Script - Generated Template
====================================
"""

from pathlib import Path
import torch
from core.deep_learning.models import create_model
from core.deep_learning.data import create_dataloader
from core.deep_learning.training import (
    Trainer, TrainingConfig, EarlyStopping,
    create_optimizer, create_scheduler
)
from core.deep_learning.evaluation import evaluate_model
from core.deep_learning.utils import set_seed, get_device, ExperimentTracker

# Set seed for reproducibility
set_seed(42)
device = get_device()

# Create model
model = create_model('transformer', {
    'vocab_size': 10000,
    'd_model': 512,
    'num_heads': 8,
    'num_layers': 6
})
model = model.to(device)

# Create data loaders
# train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
# val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

# Create optimizer and scheduler
optimizer = create_optimizer(model, 'adamw', learning_rate=1e-4)
scheduler = create_scheduler(optimizer, 'cosine', num_epochs=10)

# Training config
config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    use_mixed_precision=True,
    early_stopping=EarlyStopping(patience=5)
)

# Create trainer
trainer = Trainer(model, config, optimizer, scheduler)

# Train
history = trainer.train(train_loader, val_loader)

# Evaluate
# metrics = evaluate_model(model, test_loader, device)
'''
    
    return template


def get_inference_template() -> str:
    """Get inference script template."""
    template = '''"""
Inference Script - Generated Template
======================================
"""

from pathlib import Path
from core.deep_learning.pipelines import InferencePipeline
from core.deep_learning.models import TransformerModel
from core.deep_learning.utils import get_device

device = get_device()

# Load model from checkpoint
inference = InferencePipeline()
inference.load_from_checkpoint(
    checkpoint_path=Path("checkpoints/best_model.pt"),
    model_class=TransformerModel
)

# Run inference
inputs = {
    'input_ids': torch.randint(0, 10000, (1, 512))
}

predictions = inference.predict(inputs, return_probabilities=True)
print(f"Predictions: {predictions}")
'''
    return template


def get_config_template() -> str:
    """Get configuration file template (YAML)."""
    template = '''# Model Configuration
model:
  type: transformer
  vocab_size: 10000
  d_model: 512
  num_heads: 8
  num_layers: 6
  d_ff: 2048
  max_seq_len: 512
  dropout: 0.1

# Training Configuration
training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  optimizer: adamw
  scheduler: cosine
  use_mixed_precision: true
  gradient_accumulation_steps: 1
  early_stopping_patience: 5
  use_tensorboard: true
  use_wandb: false

# Data Configuration
data:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  shuffle: true

# Experiment Configuration
experiment:
  name: my_experiment
  seed: 42
  save_dir: checkpoints
  log_dir: logs
'''
    return template


def generate_project_structure(base_path: Path) -> None:
    """
    Generate standard project structure.
    
    Args:
        base_path: Base path for project
    """
    base_path = Path(base_path)
    
    directories = [
        'models',
        'data',
        'training',
        'evaluation',
        'inference',
        'configs',
        'checkpoints',
        'logs',
        'notebooks',
        'scripts'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    for directory in ['models', 'data', 'training', 'evaluation', 'inference']:
        (base_path / directory / '__init__.py').touch()
    
    # Create template files
    (base_path / 'configs' / 'config.yaml').write_text(get_config_template())
    (base_path / 'scripts' / 'train.py').write_text(get_training_template())
    (base_path / 'scripts' / 'inference.py').write_text(get_inference_template())
    
    # Create README
    readme = f"""# Deep Learning Project

Generated project structure for deep learning workflows.

## Structure

- `models/`: Custom model definitions
- `data/`: Data loading and preprocessing
- `training/`: Training scripts
- `evaluation/`: Evaluation scripts
- `inference/`: Inference scripts
- `configs/`: Configuration files
- `checkpoints/`: Model checkpoints
- `logs/`: Training logs
- `notebooks/`: Jupyter notebooks
- `scripts/`: Utility scripts

## Quick Start

1. Configure your model and training in `configs/config.yaml`
2. Run training: `python scripts/train.py`
3. Run inference: `python scripts/inference.py`

## Documentation

See the main documentation for more details.
"""
    (base_path / 'README.md').write_text(readme)
    
    logger.info(f"Project structure generated at {base_path}")



