"""
Configuración de ML
===================

Configuración para modelos de machine learning.
"""

import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings


class MLConfig(BaseSettings):
    """Configuración de ML."""
    
    # Modelos
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    generation_model: str = "microsoft/DialoGPT-medium"
    image_model: str = "runwayml/stable-diffusion-v1-5"
    use_sd_xl: bool = False
    
    # Dispositivos
    device: Optional[str] = None  # Auto-detect
    use_cuda: bool = True
    torch_dtype: str = "float16"  # float16 o float32
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Entrenamiento
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    
    # Generación de imágenes
    image_width: int = 512
    image_height: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # Optimizaciones avanzadas
    use_onnx: bool = False
    onnx_model_path: Optional[str] = None
    batch_size_embeddings: int = 64  # Aumentado para mejor rendimiento
    enable_model_prefetch: bool = True
    max_prefetched_models: int = 3
    use_fast_attention: bool = True
    enable_kv_cache: bool = True
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "manuales-hogar-ai"
    wandb_api_key: Optional[str] = os.getenv("WANDB_API_KEY")
    
    # Directorios
    models_dir: str = "./models"
    checkpoints_dir: str = "./checkpoints"
    outputs_dir: str = "./outputs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_ml_config() -> MLConfig:
    """Obtener configuración de ML."""
    return MLConfig()

