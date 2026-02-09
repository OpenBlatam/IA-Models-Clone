"""
Cargador de configuración YAML
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True


@dataclass
class ModelConfig:
    """Configuración de modelo"""
    base_model: str = "gpt2"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


@dataclass
class ExperimentConfig:
    """Configuración de experimento"""
    tracker: str = "wandb"
    project_name: str = "social-media-identity-clone"
    experiment_name: Optional[str] = None


class ConfigLoader:
    """Cargador de configuración desde YAML"""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Carga configuración desde archivo YAML"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def load_training_config(config_path: str) -> TrainingConfig:
        """Carga configuración de entrenamiento"""
        config = ConfigLoader.load(config_path)
        training = config.get("training", {})
        
        return TrainingConfig(
            num_epochs=training.get("num_epochs", 10),
            batch_size=training.get("batch_size", 8),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
            learning_rate=training.get("learning_rate", 5e-5),
            weight_decay=training.get("weight_decay", 0.01),
            warmup_steps=training.get("warmup_steps", 100),
            max_grad_norm=training.get("max_grad_norm", 1.0),
            optimizer=training.get("optimizer", "adamw"),
            scheduler=training.get("scheduler", "cosine"),
            mixed_precision=training.get("mixed_precision", True)
        )
    
    @staticmethod
    def load_model_config(config_path: str) -> ModelConfig:
        """Carga configuración de modelo"""
        config = ConfigLoader.load(config_path)
        model = config.get("model", {})
        lora = model.get("lora", {})
        
        return ModelConfig(
            base_model=model.get("base_model", "gpt2"),
            lora_r=lora.get("r", 8),
            lora_alpha=lora.get("lora_alpha", 16),
            lora_dropout=lora.get("lora_dropout", 0.1)
        )




