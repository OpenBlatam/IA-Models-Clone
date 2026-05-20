"""
Helpers adicionales para el generador de Deep Learning
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_framework_from_code(code: str) -> Optional[str]:
    """
    Detecta el framework usado en un código.
    
    Args:
        code: Código a analizar
        
    Returns:
        Nombre del framework detectado o None
    """
    code_lower = code.lower()
    
    # PyTorch
    if any(keyword in code_lower for keyword in ["torch", "pytorch", "nn.module", "torch.nn"]):
        return "pytorch"
    
    # TensorFlow
    if any(keyword in code_lower for keyword in ["tensorflow", "tf.", "keras", "tf.keras"]):
        return "tensorflow"
    
    # JAX
    if any(keyword in code_lower for keyword in ["jax", "jnp", "jax.numpy"]):
        return "jax"
    
    # ONNX
    if "onnx" in code_lower:
        return "onnx"
    
    return None


def detect_model_type_from_code(code: str) -> Optional[str]:
    """
    Detecta el tipo de modelo en un código.
    
    Args:
        code: Código a analizar
        
    Returns:
        Tipo de modelo detectado o None
    """
    code_lower = code.lower()
    
    # Transformer
    if any(keyword in code_lower for keyword in ["transformer", "attention", "multiheadattention"]):
        return "transformer"
    
    # CNN
    if any(keyword in code_lower for keyword in ["conv2d", "convolution", "cnn"]):
        return "cnn"
    
    # RNN/LSTM/GRU
    if "lstm" in code_lower:
        return "lstm"
    if "gru" in code_lower:
        return "gru"
    if "rnn" in code_lower:
        return "rnn"
    
    # GAN
    if "generator" in code_lower and "discriminator" in code_lower:
        return "gan"
    
    # VAE
    if "variational" in code_lower and "encoder" in code_lower:
        return "vae"
    
    # Diffusion
    if "diffusion" in code_lower or "ddpm" in code_lower:
        return "diffusion"
    
    # LLM
    if any(keyword in code_lower for keyword in ["llm", "language model", "gpt", "bert"]):
        return "llm"
    
    # Vision Transformer
    if "vision transformer" in code_lower or "vit" in code_lower:
        return "vision_transformer"
    
    return None


def analyze_code_file(file_path: Path) -> Dict[str, Any]:
    """
    Analiza un archivo de código y detecta framework y tipo de modelo.
    
    Args:
        file_path: Ruta al archivo
        
    Returns:
        Diccionario con información detectada
    """
    try:
        code = file_path.read_text(encoding="utf-8")
        
        framework = detect_framework_from_code(code)
        model_type = detect_model_type_from_code(code)
        
        return {
            "file": str(file_path),
            "framework": framework,
            "model_type": model_type,
            "detected": framework is not None or model_type is not None
        }
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return {
            "file": str(file_path),
            "error": str(e),
            "detected": False
        }


def suggest_generator_config(
    framework: Optional[str] = None,
    model_type: Optional[str] = None,
    project_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Sugiere una configuración de generador basada en análisis.
    
    Args:
        framework: Framework conocido (opcional)
        model_type: Tipo de modelo conocido (opcional)
        project_path: Ruta del proyecto para analizar (opcional)
        
    Returns:
        Configuración sugerida
    """
    config = {}
    
    # Analizar proyecto si se proporciona
    if project_path and project_path.exists():
        python_files = list(project_path.rglob("*.py"))
        
        frameworks_found = []
        model_types_found = []
        
        for py_file in python_files[:10]:  # Limitar a 10 archivos
            analysis = analyze_code_file(py_file)
            if analysis.get("framework"):
                frameworks_found.append(analysis["framework"])
            if analysis.get("model_type"):
                model_types_found.append(analysis["model_type"])
        
        # Usar el más común
        if frameworks_found and not framework:
            from collections import Counter
            framework = Counter(frameworks_found).most_common(1)[0][0]
        
        if model_types_found and not model_type:
            from collections import Counter
            model_type = Counter(model_types_found).most_common(1)[0][0]
    
    if framework:
        config["framework"] = framework
    else:
        config["framework"] = "pytorch"  # Default
    
    if model_type:
        config["model_type"] = model_type
    
    return config


def get_framework_info(framework: str) -> Dict[str, Any]:
    """
    Retorna información sobre un framework específico.
    
    Args:
        framework: Nombre del framework
        
    Returns:
        Información del framework
    """
    frameworks_info = {
        "pytorch": {
            "name": "PyTorch",
            "description": "Deep learning framework by Facebook",
            "common_imports": ["torch", "torch.nn", "torch.optim"],
            "version": "2.0+",
            "use_cases": ["Research", "Production", "Prototyping"]
        },
        "tensorflow": {
            "name": "TensorFlow",
            "description": "Deep learning framework by Google",
            "common_imports": ["tensorflow", "tensorflow.keras"],
            "version": "2.x",
            "use_cases": ["Production", "Deployment", "Mobile"]
        },
        "jax": {
            "name": "JAX",
            "description": "High-performance ML library by Google",
            "common_imports": ["jax", "jax.numpy", "flax"],
            "version": "0.4+",
            "use_cases": ["Research", "High Performance", "Scientific Computing"]
        },
        "onnx": {
            "name": "ONNX",
            "description": "Open Neural Network Exchange format",
            "common_imports": ["onnx", "onnxruntime"],
            "version": "1.14+",
            "use_cases": ["Model Exchange", "Deployment", "Interoperability"]
        }
    }
    
    return frameworks_info.get(framework.lower(), {
        "name": framework,
        "description": "Unknown framework",
        "common_imports": [],
        "version": "Unknown",
        "use_cases": []
    })


def get_model_type_info(model_type: str) -> Dict[str, Any]:
    """
    Retorna información sobre un tipo de modelo específico.
    
    Args:
        model_type: Tipo de modelo
        
    Returns:
        Información del tipo de modelo
    """
    model_types_info = {
        "transformer": {
            "name": "Transformer",
            "description": "Attention-based architecture",
            "use_cases": ["NLP", "Vision", "Multimodal"],
            "common_layers": ["MultiHeadAttention", "FeedForward", "LayerNorm"]
        },
        "cnn": {
            "name": "Convolutional Neural Network",
            "description": "For image processing",
            "use_cases": ["Computer Vision", "Image Classification"],
            "common_layers": ["Conv2d", "MaxPool2d", "BatchNorm2d"]
        },
        "llm": {
            "name": "Large Language Model",
            "description": "Large-scale language models",
            "use_cases": ["NLP", "Text Generation", "Chatbots"],
            "common_layers": ["Transformer", "Embedding", "Attention"]
        },
        "diffusion": {
            "name": "Diffusion Model",
            "description": "For generative tasks",
            "use_cases": ["Image Generation", "Text-to-Image"],
            "common_layers": ["UNet", "Attention", "Time Embedding"]
        }
    }
    
    return model_types_info.get(model_type.lower(), {
        "name": model_type,
        "description": "Unknown model type",
        "use_cases": [],
        "common_layers": []
    })


def generate_config_template(
    framework: str = "pytorch",
    model_type: str = "transformer"
) -> Dict[str, Any]:
    """
    Genera una plantilla de configuración.
    
    Args:
        framework: Framework a usar
        model_type: Tipo de modelo
        
    Returns:
        Plantilla de configuración
    """
    from core.deep_learning_generator import validate_generator_config
    
    config = {
        "framework": framework,
        "model_type": model_type,
        "config": {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "dropout": 0.1
        }
    }
    
    # Validar
    is_valid, error = validate_generator_config(config)
    if not is_valid:
        logger.warning(f"Generated config is invalid: {error}")
    
    return config

