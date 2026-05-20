"""
Model Generator - Generador de arquitecturas de modelos
========================================================

Genera modelos especializados: Transformer, Diffusion, Custom
Siguiendo mejores prácticas de PyTorch y arquitecturas modulares.
Incluye soporte para LoRA, P-tuning, y otras técnicas de fine-tuning eficiente.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelGenerator:
    """Generador de arquitecturas de modelos de Deep Learning"""
    
    def __init__(self):
        """Inicializa el generador de modelos"""
        pass
    
    def generate(
        self,
        models_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera la arquitectura del modelo según el tipo detectado.
        
        Args:
            models_dir: Directorio donde generar los modelos
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar __init__.py para el módulo de modelos
        self._generate_models_init(models_dir, keywords)
        
        # Generar base model para herencia
        self._generate_base_model(models_dir, keywords, project_info)
        
        # Generar modelos según el tipo
        if keywords.get("is_diffusion"):
            self._generate_diffusion_model(models_dir, keywords, project_info)
        
        if keywords.get("is_llm") or keywords.get("is_transformer"):
            self._generate_transformer_model(models_dir, keywords, project_info)
            # Generar utilidades de fine-tuning eficiente
            self._generate_lora_utils(models_dir, keywords, project_info)
        
        if keywords.get("is_deep_learning") and not keywords.get("is_transformer") and not keywords.get("is_diffusion"):
            self._generate_custom_model(models_dir, keywords, project_info)
    
    def _generate_models_init(
        self,
        models_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de modelos"""
        
        imports = ["from .base_model import BaseModel"]
        
        if keywords.get("is_transformer") or keywords.get("is_llm"):
            imports.append("from .transformer_model import TransformerModel, FineTuner")
            imports.append("from .lora_utils import apply_lora, LoraConfig")
        
        if keywords.get("is_diffusion"):
            imports.append("from .diffusion_model import DiffusionModel")
        
        if keywords.get("is_deep_learning"):
            imports.append("from .custom_model import CustomModel")
        
        init_content = f'''"""
Models Module - Arquitecturas de modelos de Deep Learning
{'=' * 60}

Este módulo contiene todas las arquitecturas de modelos.
Sigue mejores prácticas de PyTorch y arquitecturas modulares.
"""

{chr(10).join(imports)}

__all__ = [
    "BaseModel",
'''
        
        if keywords.get("is_transformer") or keywords.get("is_llm"):
            init_content += '    "TransformerModel",\n    "FineTuner",\n    "apply_lora",\n    "LoraConfig",\n'
        
        if keywords.get("is_diffusion"):
            init_content += '    "DiffusionModel",\n'
        
        if keywords.get("is_deep_learning"):
            init_content += '    "CustomModel",\n'
        
        init_content += "]\n"
        
        (models_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_base_model(
        self,
        models_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera clase base para todos los modelos con mejoras"""
        
        base_model_content = '''"""
Base Model - Clase base para todos los modelos
===============================================

Proporciona funcionalidad común para todos los modelos de Deep Learning.
Incluye manejo de errores, logging, y validación.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Clase base abstracta para todos los modelos.
    
    Proporciona funcionalidad común:
    - Gestión de dispositivo (CPU/GPU)
    - Guardado y carga de modelos
    - Modos de entrenamiento/evaluación
    - Manejo de errores robusto
    - Validación de inputs
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Inicializa el modelo base.
        
        Args:
            device: Dispositivo a usar (cuda, cpu, o None para auto-detectar)
        
        Raises:
            RuntimeError: Si hay problemas con el dispositivo
        """
        super().__init__()
        
        try:
            self.device = device or self._detect_device()
            logger.info(f"Modelo inicializado en dispositivo: {self.device}")
            
            # Verificar disponibilidad de CUDA si se solicita
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA solicitado pero no disponible, usando CPU")
                self.device = "cpu"
        
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            self.device = "cpu"
            logger.info("Usando CPU como fallback")
    
    def _detect_device(self) -> str:
        """Detecta el mejor dispositivo disponible"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Mueve un tensor al dispositivo del modelo.
        
        Args:
            tensor: Tensor a mover
            
        Returns:
            Tensor en el dispositivo correcto
        """
        try:
            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error moviendo tensor a {self.device}: {e}")
            raise
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        **kwargs
    ) -> None:
        """
        Guarda un checkpoint del modelo.
        
        Args:
            path: Ruta donde guardar
            **kwargs: Información adicional a guardar
        
        Raises:
            IOError: Si hay problemas guardando el archivo
        """
        try:
            path = Path(path)
            checkpoint = {
                "model_state_dict": self.state_dict(),
                "device": self.device,
                "model_class": self.__class__.__name__,
                **kwargs
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint guardado en {path}")
        
        except Exception as e:
            logger.error(f"Error guardando checkpoint: {e}")
            raise IOError(f"No se pudo guardar checkpoint en {path}: {e}")
    
    def load_checkpoint(
        self,
        path: Union[str, Path],
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Carga un checkpoint del modelo.
        
        Args:
            path: Ruta del checkpoint
            strict: Si cargar estrictamente (todas las claves deben coincidir)
        
        Returns:
            Diccionario con información del checkpoint
        
        Raises:
            FileNotFoundError: Si el archivo no existe
            RuntimeError: Si hay problemas cargando el checkpoint
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint no encontrado: {path}")
            
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            logger.info(f"Checkpoint cargado desde {path}")
            return checkpoint
        
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error cargando checkpoint: {e}")
            raise RuntimeError(f"No se pudo cargar checkpoint desde {path}: {e}")
    
    def validate_input(self, *args, **kwargs) -> bool:
        """
        Valida inputs del modelo (puede ser sobrescrito por subclases).
        
        Args:
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            True si los inputs son válidos
        """
        return True
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass - debe ser implementado por subclases.
        
        Args:
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            Salida del modelo
        """
        raise NotImplementedError("Subclases deben implementar forward()")
    
    def predict(
        self,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Realiza predicción (modo evaluación).
        
        Args:
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            Diccionario con predicciones
        
        Raises:
            ValueError: Si los inputs no son válidos
        """
        if not self.validate_input(*args, **kwargs):
            raise ValueError("Inputs no válidos")
        
        self.eval()
        with torch.no_grad():
            try:
                return self._predict_impl(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error en predicción: {e}")
                raise
    
    @abstractmethod
    def _predict_impl(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementación de predicción - debe ser implementado por subclases.
        
        Args:
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            Diccionario con predicciones
        """
        raise NotImplementedError("Subclases deben implementar _predict_impl()")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_class": self.__class__.__name__,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "training_mode": self.training,
        }
'''
        
        (models_dir / "base_model.py").write_text(base_model_content, encoding="utf-8")
    
    def _generate_lora_utils(
        self,
        models_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades para LoRA (Low-Rank Adaptation)"""
        
        lora_content = '''"""
LoRA Utilities - Utilidades para fine-tuning eficiente con LoRA
================================================================

Implementa LoRA (Low-Rank Adaptation) para fine-tuning eficiente de modelos grandes.
Reduce significativamente el número de parámetros a entrenar.
"""

from dataclasses import dataclass
from typing import Optional, List, Union
import torch
import torch.nn as nn
import logging

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT no disponible. Instala con: pip install peft")

logger = logging.getLogger(__name__)


@dataclass
class LoraConfig:
    """
    Configuración para LoRA.
    
    Attributes:
        r: Rango de la adaptación (rank)
        lora_alpha: Factor de escalado
        target_modules: Módulos objetivo para aplicar LoRA
        lora_dropout: Dropout para LoRA
        bias: Tipo de bias (none, all, lora_only)
        task_type: Tipo de tarea
    """
    r: int = 8
    lora_alpha: int = 16
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.1
    bias: str = "none"  # none, all, lora_only
    task_type: str = "FEATURE_EXTRACTION"  # FEATURE_EXTRACTION, SEQ_CLS, etc.


def apply_lora(
    model: nn.Module,
    config: Optional[LoraConfig] = None,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Aplica LoRA a un modelo.
    
    Args:
        model: Modelo a adaptar
        config: Configuración de LoRA (opcional)
        target_modules: Módulos objetivo (si no se especifica en config)
    
    Returns:
        Modelo con LoRA aplicado
    
    Raises:
        ImportError: Si PEFT no está disponible
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT no está disponible. Instala con: pip install peft"
        )
    
    if config is None:
        config = LoraConfig()
    
    # Determinar módulos objetivo
    if config.target_modules is None:
        if target_modules is None:
            # Módulos por defecto para transformers comunes
            config.target_modules = ["query", "key", "value", "dense"]
        else:
            config.target_modules = target_modules
    
    # Crear configuración de PEFT
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=getattr(TaskType, config.task_type, TaskType.FEATURE_EXTRACTION),
    )
    
    # Aplicar LoRA
    try:
        model = get_peft_model(model, peft_config)
        logger.info(f"LoRA aplicado: r={config.r}, alpha={config.lora_alpha}")
        logger.info(f"Parámetros entrenables: {model.num_parameters()}")
        return model
    
    except Exception as e:
        logger.error(f"Error aplicando LoRA: {e}")
        raise


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Imprime información sobre parámetros entrenables.
    
    Args:
        model: Modelo a analizar
    """
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(
        f"Parámetros entrenables: {trainable_params:,} || "
        f"Total parámetros: {all_param:,} || "
        f"Porcentaje entrenable: {100 * trainable_params / all_param:.2f}%"
    )
'''
        
        (models_dir / "lora_utils.py").write_text(lora_content, encoding="utf-8")
    
    def _generate_transformer_model(
        self,
        models_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera modelo Transformer/LLM mejorado con LoRA support"""
        
        # Leer el contenido actual y mejorarlo
        model_content = '''"""
Transformer Model - Modelo basado en arquitectura Transformer
==============================================================

Implementa un modelo Transformer usando la librería Transformers de HuggingFace.
Sigue mejores prácticas para fine-tuning y inferencia.
Incluye soporte para LoRA y otras técnicas de fine-tuning eficiente.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from typing import Optional, Dict, Any, List, Union
import logging
import torch.nn.functional as F

from .base_model import BaseModel
from .lora_utils import apply_lora, LoraConfig, print_trainable_parameters

logger = logging.getLogger(__name__)


class TransformerModel(BaseModel):
    """
    Modelo Transformer personalizado.
    
    Utiliza modelos pre-entrenados de HuggingFace y permite fine-tuning.
    Soporta LoRA para fine-tuning eficiente.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: Optional[int] = None,
        task_type: str = "classification",  # classification, generation, qa
        device: Optional[str] = None,
        use_lora: bool = False,
        lora_config: Optional[LoraConfig] = None,
    ):
        """
        Inicializa el modelo Transformer.
        
        Args:
            model_name: Nombre del modelo pre-entrenado de HuggingFace
            num_labels: Número de clases para clasificación (opcional)
            task_type: Tipo de tarea (classification, generation, qa)
            device: Dispositivo a usar (cuda, cpu, o None para auto-detectar)
            use_lora: Si usar LoRA para fine-tuning eficiente
            lora_config: Configuración de LoRA (opcional)
        
        Raises:
            ValueError: Si la configuración es inválida
            RuntimeError: Si hay problemas cargando el modelo
        """
        super().__init__(device=device)
        
        self.model_name = model_name
        self.task_type = task_type
        self.use_lora = use_lora
        
        try:
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Cargar modelo según el tipo de tarea
            if task_type == "classification" and num_labels:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                )
            elif task_type == "generation":
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            else:
                self.model = AutoModel.from_pretrained(model_name)
            
            # Aplicar LoRA si se solicita
            if use_lora:
                try:
                    self.model = apply_lora(self.model, lora_config)
                    print_trainable_parameters(self.model)
                except ImportError:
                    logger.warning("PEFT no disponible, continuando sin LoRA")
                    self.use_lora = False
            
        self.model.to(self.device)
        self.model.train()
        
        # Optimizaciones de velocidad agresivas
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Más rápido
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Habilitar flash attention si está disponible
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Flash attention habilitado")
            except:
                pass
            # Optimizar allocator
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except:
                pass
        
        logger.info(f"Modelo {model_name} cargado en {self.device}")
            if use_lora:
                logger.info("LoRA habilitado para fine-tuning eficiente")
        
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            raise RuntimeError(f"No se pudo cargar modelo: {e}")
    
    def validate_input(self, text: Union[str, List[str]], max_length: int = 512) -> bool:
        """
        Valida inputs del modelo.
        
        Args:
            text: Texto o lista de textos
            max_length: Longitud máxima
        
        Returns:
            True si los inputs son válidos
        """
        if isinstance(text, str):
            return len(text) > 0 and len(text) <= max_length * 10  # Aproximado
        elif isinstance(text, list):
            return all(len(t) > 0 for t in text) and len(text) > 0
        return False
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass del modelo.
        
        Args:
            input_ids: IDs de tokens de entrada
            attention_mask: Máscara de atención
            labels: Etiquetas para entrenamiento (opcional)
            **kwargs: Argumentos adicionales
        
        Returns:
            Salida del modelo
        """
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            return outputs
        
        except Exception as e:
            logger.error(f"Error en forward pass: {e}")
            raise
    
    def encode(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Codifica textos en embeddings.
        
        Args:
            texts: Lista de textos a codificar
            max_length: Longitud máxima de secuencia
        
        Returns:
            Tensor con embeddings
        
        Raises:
            ValueError: Si los textos no son válidos
        """
        if not self.validate_input(texts, max_length):
            raise ValueError("Textos no válidos para encoding")
        
        self.model.eval()
        with torch.no_grad():
            try:
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                
                # Extraer embeddings (usar [CLS] token o pooling)
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                else:
                    embeddings = outputs.pooler_output
                
                return embeddings
            
            except Exception as e:
                logger.error(f"Error en encoding: {e}")
                raise
    
    def _predict_impl(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Realiza predicción en un texto.
        
        Args:
            text: Texto de entrada
            max_length: Longitud máxima
        
        Returns:
            Diccionario con predicción y metadata
        """
        if not self.validate_input(text, max_length):
            raise ValueError(f"Texto no válido: longitud {len(text)}")
        
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        
        if self.task_type == "classification":
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probs[0].cpu().numpy().tolist(),
            }
        elif self.task_type == "generation":
            generated_ids = self.model.generate(
                **encoded,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            return {
                "generated_text": generated_text,
            }
        
        return {"output": outputs}
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo y tokenizer.
        
        Args:
            path: Ruta donde guardar
        """
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Modelo guardado en {path}")
        
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            raise


class FineTuner:
    """
    Clase para fine-tuning de modelos Transformer.
    
    Implementa mejores prácticas para entrenamiento:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Experiment tracking
    """
    
    def __init__(
        self,
        model: TransformerModel,
        training_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Inicializa el fine-tuner.
        
        Args:
            model: Modelo Transformer a entrenar
            training_args: Argumentos de entrenamiento
        """
        self.model = model
        self.training_args = training_args or {}
        
        # Configuración por defecto
        self.default_args = {
            "output_dir": "./results",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "logging_steps": 100,
            "eval_steps": 500,
            "save_steps": 1000,
            "fp16": torch.cuda.is_available(),
            "dataloader_num_workers": 4,
            "load_best_model_at_end": True,
            "metric_for_best_model": "accuracy",
            "greater_is_better": True,
            "report_to": ["tensorboard"],  # Experiment tracking
        }
        self.default_args.update(self.training_args)
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        compute_metrics=None,
    ):
        """
        Entrena el modelo.
        
        Args:
            train_dataset: Dataset de entrenamiento
            eval_dataset: Dataset de evaluación (opcional)
            compute_metrics: Función para calcular métricas (opcional)
        
        Returns:
            Trainer entrenado
        """
        training_args = TrainingArguments(**self.default_args)
        
        data_collator = DataCollatorWithPadding(
            tokenizer=self.model.tokenizer,
            padding=True
        )
        
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        logger.info("Iniciando entrenamiento...")
        try:
            trainer.train()
            logger.info("Entrenamiento completado")
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {e}")
            raise
        
        return trainer
'''
        
        (models_dir / "transformer_model.py").write_text(model_content, encoding="utf-8")
    
    def _generate_diffusion_model(
        self,
        models_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera modelo de difusión mejorado"""
        
        # Mantener el contenido existente pero mejorado con mejor manejo de errores
        model_content = '''"""
Diffusion Model - Modelo de generación de imágenes usando Diffusers
=====================================================================

Implementa un pipeline de difusión usando la librería Diffusers de HuggingFace.
Soporta Stable Diffusion, Stable Diffusion XL, y otros modelos.
Incluye optimizaciones de memoria y manejo robusto de errores.
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from diffusers.utils import make_image_grid
from PIL import Image
from typing import Optional, Dict, Any, List, Union
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class DiffusionModel(BaseModel):
    """
    Modelo de difusión para generación de imágenes.
    
    Utiliza modelos pre-entrenados de HuggingFace Diffusers.
    Incluye optimizaciones de memoria y manejo robusto de errores.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        use_xl: bool = False,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_optimizations: bool = True,
    ):
        """
        Inicializa el modelo de difusión.
        
        Args:
            model_id: ID del modelo en HuggingFace
            use_xl: Si usar Stable Diffusion XL
            device: Dispositivo a usar (cuda, cpu, o None para auto-detectar)
            dtype: Tipo de datos (float16 para GPU, float32 para CPU)
            enable_optimizations: Si habilitar optimizaciones de memoria
        
        Raises:
            RuntimeError: Si hay problemas cargando el modelo
        """
        super().__init__(device=device)
        
        self.model_id = model_id
        self.use_xl = use_xl
        
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            # Cargar pipeline según el tipo
            if use_xl:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    variant="fp16" if dtype == torch.float16 else None,
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    variant="fp16" if dtype == torch.float16 else None,
                )
            
            # Optimizar scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
        # Mover a dispositivo y optimizar
        if self.device == "cuda" and enable_optimizations:
            self.pipeline = self.pipeline.to(self.device)
            
            # Optimizaciones de memoria ultra agresivas (máxima velocidad)
            self.pipeline.enable_attention_slicing(1)  # Slice size 1 para máximo paralelismo
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()  # Tiling para imágenes grandes
            
            # Optimizaciones de velocidad ultra agresivas
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Optimización xformers habilitada")
                except Exception:
                    logger.warning("xformers no disponible, continuando sin optimización")
            
            # Habilitar CPU offload para modelos grandes
            try:
                self.pipeline.enable_model_cpu_offload()
                logger.info("CPU offload habilitado")
            except:
                pass  # No todos los pipelines lo soportan
            
            # Optimizaciones adicionales ultra agresivas
            try:
                # Habilitar sequential CPU offload para máxima velocidad
                if hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                    self.pipeline.enable_sequential_cpu_offload()
                
                # Deshabilitar safety checker si existe (más rápido)
                if hasattr(self.pipeline, "safety_checker"):
                    self.pipeline.safety_checker = None
                    logger.info("Safety checker deshabilitado para velocidad")
            except:
                pass
            
            logger.info(f"Modelo de difusión {model_id} cargado en {self.device}")
        
        except Exception as e:
            logger.error(f"Error cargando modelo de difusión: {e}")
            raise RuntimeError(f"No se pudo cargar modelo de difusión: {e}")
    
    def validate_input(self, prompt: str, **kwargs) -> bool:
        """
        Valida inputs para generación.
        
        Args:
            prompt: Prompt de texto
            **kwargs: Argumentos adicionales
        
        Returns:
            True si los inputs son válidos
        """
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return False
        if len(prompt) > 1000:  # Límite razonable
            logger.warning(f"Prompt muy largo: {len(prompt)} caracteres")
        return True
    
    def forward(self, *args, **kwargs):
        """Forward pass - no aplica directamente para diffusion models"""
        raise NotImplementedError("Use generate() method for diffusion models")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Genera imágenes a partir de un prompt de texto.
        
        Args:
            prompt: Texto descriptivo de la imagen
            negative_prompt: Prompt negativo (qué evitar)
            num_inference_steps: Número de pasos de inferencia
            guidance_scale: Escala de guía (mayor = más fiel al prompt)
            height: Altura de la imagen
            width: Ancho de la imagen
            num_images_per_prompt: Número de imágenes a generar
            seed: Semilla para reproducibilidad
        
        Returns:
            Lista de imágenes PIL generadas
        
        Raises:
            ValueError: Si el prompt no es válido
            RuntimeError: Si hay errores durante la generación
        """
        if not self.validate_input(prompt):
            raise ValueError("Prompt no válido")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Optimizar steps para velocidad EXTREMA
        if num_inference_steps > 10:
            try:
                from diffusers import DPMSolverSinglestepScheduler
                if not isinstance(self.pipeline.scheduler, DPMSolverSinglestepScheduler):
                    self.pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
                        self.pipeline.scheduler.config
                    )
                # Reducir a mínimo para máxima velocidad
                num_inference_steps = min(num_inference_steps, 10)
                logger.info(f"Optimizado a {num_inference_steps} steps para velocidad EXTREMA")
            except:
                pass
        
        try:
            with torch.autocast(
                self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                enabled=self.device == "cuda",
                cache_enabled=True,  # Caché de autocast para velocidad
            ):
                images = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    output_type="pil",  # Directamente PIL
                ).images
            
            logger.info(f"Generadas {len(images)} imágenes")
            return images
        
        except Exception as e:
            logger.error(f"Error generando imágenes: {e}")
            raise RuntimeError(f"Error durante generación: {e}")
    
    def _predict_impl(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Implementación de predicción para diffusion models"""
        images = self.generate(prompt, **kwargs)
        return {"images": images, "count": len(images)}
'''
        
        (models_dir / "diffusion_model.py").write_text(model_content, encoding="utf-8")
    
    def _generate_custom_model(
        self,
        models_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera modelo personalizado de PyTorch mejorado"""
        
        model_content = '''"""
Custom Neural Network Model - Modelo de red neuronal personalizado
====================================================================

Implementa una arquitectura de red neuronal usando PyTorch.
Sigue mejores prácticas: inicialización de pesos, normalización, etc.
Incluye validación de inputs y manejo robusto de errores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import logging
import math

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class CustomModel(BaseModel):
    """
    Modelo de red neuronal personalizado.
    
    Implementa una arquitectura modular y extensible.
    Incluye inicialización adecuada de pesos y normalización.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = None,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        device: Optional[str] = None,
    ):
        """
        Inicializa el modelo.
        
        Args:
            input_size: Tamaño de entrada
            hidden_sizes: Lista de tamaños de capas ocultas
            num_classes: Número de clases de salida
            dropout_rate: Tasa de dropout
            use_batch_norm: Si usar normalización por lotes
            device: Dispositivo a usar
        
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        super().__init__(device=device)
        
        if input_size <= 0:
            raise ValueError("input_size debe ser mayor que 0")
        if num_classes <= 0:
            raise ValueError("num_classes debe ser mayor que 0")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate debe estar entre 0 y 1")
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or [128, 64, 32]
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Construir capas
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            if hidden_size <= 0:
                raise ValueError(f"hidden_size en posición {i} debe ser mayor que 0")
            
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.layers = nn.Sequential(*layers)
        self.layers.to(self.device)
        
        # Inicializar pesos
        self._initialize_weights()
        
        logger.info(
            f"Modelo inicializado: input={input_size}, "
            f"hidden={self.hidden_sizes}, output={num_classes}"
        )
    
    def _initialize_weights(self) -> None:
        """Inicializa los pesos usando mejores prácticas"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization para ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def validate_input(self, x: torch.Tensor) -> bool:
        """
        Valida inputs del modelo.
        
        Args:
            x: Tensor de entrada
        
        Returns:
            True si el input es válido
        """
        if not isinstance(x, torch.Tensor):
            return False
        if x.dim() < 2:
            return False
        if x.size(-1) != self.input_size:
            return False
        # Verificar NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Input contiene NaN o Inf")
            return False
        return True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada
        
        Returns:
            Tensor de salida
        
        Raises:
            ValueError: Si el input no es válido
        """
        if not self.validate_input(x):
            raise ValueError(f"Input inválido: shape={x.shape}, expected={self.input_size}")
        
        x = x.to(self.device)
        
        try:
            output = self.layers(x)
            # Verificar NaN/Inf en salida
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.error("Output contiene NaN o Inf")
                raise RuntimeError("Output contiene valores inválidos")
            return output
        
        except Exception as e:
            logger.error(f"Error en forward pass: {e}")
            raise
    
    def _predict_impl(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Realiza predicción (modo evaluación).
        
        Args:
            x: Tensor de entrada
        
        Returns:
            Diccionario con predicciones
        """
        x = x.to(self.device)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        return {
            "predictions": predictions.cpu().numpy().tolist(),
            "probabilities": probs.cpu().numpy().tolist(),
            "logits": logits.cpu().numpy().tolist(),
        }
'''
        
        (models_dir / "custom_model.py").write_text(model_content, encoding="utf-8")
