"""
Postprocessing Generator - Generador de utilidades de postprocesamiento
=========================================================================

Genera utilidades para postprocesamiento de outputs:
- Output formatting
- Prediction decoding
- Result aggregation
- Confidence calibration
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PostprocessingGenerator:
    """Generador de utilidades de postprocesamiento"""
    
    def __init__(self):
        """Inicializa el generador de postprocesamiento"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de postprocesamiento.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        postprocessing_dir = utils_dir / "postprocessing"
        postprocessing_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_output_formatter(postprocessing_dir, keywords, project_info)
        self._generate_prediction_decoder(postprocessing_dir, keywords, project_info)
        self._generate_postprocessing_init(postprocessing_dir, keywords)
    
    def _generate_postprocessing_init(
        self,
        postprocessing_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de postprocesamiento"""
        
        init_content = '''"""
Postprocessing Utilities Module
=================================

Utilidades para postprocesamiento de outputs de modelos.
"""

from .output_formatter import (
    OutputFormatter,
    format_predictions,
    format_classification_output,
    format_regression_output,
)
from .prediction_decoder import (
    PredictionDecoder,
    decode_predictions,
    decode_classification,
    decode_sequence,
)

__all__ = [
    "OutputFormatter",
    "format_predictions",
    "format_classification_output",
    "format_regression_output",
    "PredictionDecoder",
    "decode_predictions",
    "decode_classification",
    "decode_sequence",
]
'''
        
        (postprocessing_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_output_formatter(
        self,
        postprocessing_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera formateador de outputs"""
        
        formatter_content = '''"""
Output Formatter - Formateador de outputs
==========================================

Utilidades para formatear outputs de modelos.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Formateador de outputs de modelos.
    
    Formatea outputs para diferentes tipos de tareas.
    """
    
    def __init__(self):
        """Inicializa el formateador"""
        pass
    
    def format_predictions(
        self,
        predictions: Any,
        task_type: str = "classification",
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Formatea predicciones según el tipo de tarea.
        
        Args:
            predictions: Predicciones del modelo
            task_type: Tipo de tarea (classification, regression, generation)
            class_names: Nombres de clases (opcional)
        
        Returns:
            Diccionario con predicciones formateadas
        """
        if task_type == "classification":
            return self.format_classification_output(predictions, class_names)
        elif task_type == "regression":
            return self.format_regression_output(predictions)
        elif task_type == "generation":
            return self.format_generation_output(predictions)
        else:
            raise ValueError(f"Tipo de tarea no soportado: {task_type}")
    
    def format_classification_output(
        self,
        predictions: Any,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Formatea outputs de clasificación.
        
        Args:
            predictions: Predicciones (logits o probabilidades)
            class_names: Nombres de clases (opcional)
        
        Returns:
            Diccionario con predicciones formateadas
        """
        # Convertir a numpy si es tensor
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # Aplicar softmax si son logits
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            # Verificar si son logits (valores grandes)
            if predictions.max() > 1.0 or predictions.min() < 0.0:
                try:
                    from scipy.special import softmax
                    probs = softmax(predictions, axis=-1)
                except ImportError:
                    # Softmax manual
                    exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
                    probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
            else:
                probs = predictions
        else:
            probs = predictions
        
        # Obtener clase predicha
        if probs.ndim > 1:
            predicted_class = np.argmax(probs, axis=-1)
            confidence = np.max(probs, axis=-1)
        else:
            predicted_class = np.argmax(probs)
            confidence = np.max(probs)
        
        # Formatear resultado
        result = {
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
            "probabilities": probs.tolist() if probs.ndim <= 2 else probs.flatten().tolist(),
        }
        
        if class_names:
            result["predicted_class_name"] = class_names[int(predicted_class)]
            result["class_probabilities"] = {
                class_names[i]: float(probs[i]) if probs.ndim == 1 else float(probs[0][i])
                for i in range(len(class_names))
            }
        
        return result
    
    def format_regression_output(
        self,
        predictions: Any,
    ) -> Dict[str, Any]:
        """
        Formatea outputs de regresión.
        
        Args:
            predictions: Predicciones del modelo
        
        Returns:
            Diccionario con predicciones formateadas
        """
        # Convertir a numpy si es tensor
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # Aplanar si es necesario
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        result = {
            "predicted_value": float(predictions[0]) if len(predictions) == 1 else predictions.tolist(),
        }
        
        if len(predictions) > 1:
            result["statistics"] = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
            }
        
        return result
    
    def format_generation_output(
        self,
        predictions: Any,
    ) -> Dict[str, Any]:
        """
        Formatea outputs de generación.
        
        Args:
            predictions: Predicciones generadas
        
        Returns:
            Diccionario con predicciones formateadas
        """
        result = {
            "generated_text": str(predictions) if not isinstance(predictions, (list, tuple)) else predictions,
        }
        
        if isinstance(predictions, (list, tuple)):
            result["num_tokens"] = len(predictions)
            result["generated_sequences"] = predictions
        
        return result


def format_predictions(
    predictions: Any,
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para formatear predicciones.
    
    Args:
        predictions: Predicciones a formatear
        **kwargs: Argumentos adicionales
    
    Returns:
        Predicciones formateadas
    """
    formatter = OutputFormatter()
    return formatter.format_predictions(predictions, **kwargs)


def format_classification_output(
    predictions: Any,
    **kwargs,
) -> Dict[str, Any]:
    """
    Función helper para formatear outputs de clasificación.
    
    Args:
        predictions: Predicciones a formatear
        **kwargs: Argumentos adicionales
    
    Returns:
        Outputs formateados
    """
    formatter = OutputFormatter()
    return formatter.format_classification_output(predictions, **kwargs)


def format_regression_output(
    predictions: Any,
) -> Dict[str, Any]:
    """
    Función helper para formatear outputs de regresión.
    
    Args:
        predictions: Predicciones a formatear
    
    Returns:
        Outputs formateados
    """
    formatter = OutputFormatter()
    return formatter.format_regression_output(predictions)
'''
        
        (postprocessing_dir / "output_formatter.py").write_text(formatter_content, encoding="utf-8")
    
    def _generate_prediction_decoder(
        self,
        postprocessing_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera decodificador de predicciones"""
        
        decoder_content = '''"""
Prediction Decoder - Decodificador de predicciones
====================================================

Utilidades para decodificar predicciones de modelos.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PredictionDecoder:
    """
    Decodificador de predicciones.
    
    Decodifica predicciones para diferentes tipos de tareas.
    """
    
    def __init__(self):
        """Inicializa el decodificador"""
        pass
    
    def decode_predictions(
        self,
        predictions: Any,
        task_type: str = "classification",
        tokenizer: Optional[Any] = None,
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Decodifica predicciones según el tipo de tarea.
        
        Args:
            predictions: Predicciones del modelo
            task_type: Tipo de tarea (classification, generation, sequence)
            tokenizer: Tokenizer (opcional, para generación)
        
        Returns:
            Predicciones decodificadas
        """
        if task_type == "classification":
            return self.decode_classification(predictions)
        elif task_type == "generation":
            return self.decode_sequence(predictions, tokenizer)
        elif task_type == "sequence":
            return self.decode_sequence(predictions, tokenizer)
        else:
            raise ValueError(f"Tipo de tarea no soportado: {task_type}")
    
    def decode_classification(
        self,
        predictions: Any,
        class_names: Optional[List[str]] = None,
    ) -> Union[int, str, Dict[str, Any]]:
        """
        Decodifica predicciones de clasificación.
        
        Args:
            predictions: Predicciones (logits, probabilidades o índices)
            class_names: Nombres de clases (opcional)
        
        Returns:
            Clase predicha (índice, nombre o diccionario)
        """
        # Convertir a numpy si es tensor
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # Si es un escalar o índice directo
        if predictions.ndim == 0 or (predictions.ndim == 1 and len(predictions) == 1):
            class_idx = int(predictions.item() if predictions.ndim == 0 else predictions[0])
        else:
            # Obtener clase con mayor probabilidad
            if predictions.ndim > 1:
                class_idx = int(np.argmax(predictions, axis=-1)[0])
            else:
                class_idx = int(np.argmax(predictions))
        
        if class_names:
            if class_idx < len(class_names):
                return class_names[class_idx]
            else:
                logger.warning(f"Índice de clase {class_idx} fuera de rango")
                return class_idx
        
        return class_idx
    
    def decode_sequence(
        self,
        predictions: Any,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """
        Decodifica secuencia de tokens a texto.
        
        Args:
            predictions: Predicciones (tokens o índices)
            tokenizer: Tokenizer para decodificar (opcional)
        
        Returns:
            Texto decodificado
        """
        # Convertir a numpy si es tensor
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        # Si es lista de tokens
        if isinstance(predictions, (list, tuple)):
            token_ids = predictions
        elif predictions.ndim > 0:
            # Aplanar si es necesario
            token_ids = predictions.flatten().tolist()
        else:
            token_ids = [int(predictions.item())]
        
        # Decodificar con tokenizer si está disponible
        if tokenizer:
            try:
                if hasattr(tokenizer, "decode"):
                    return tokenizer.decode(token_ids)
                elif hasattr(tokenizer, "convert_ids_to_tokens"):
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    return " ".join(tokens)
            except Exception as e:
                logger.warning(f"Error decodificando con tokenizer: {e}")
        
        # Decodificación básica (asumiendo que son caracteres ASCII)
        try:
            return "".join([chr(int(t)) if isinstance(t, (int, np.integer)) and 32 <= int(t) <= 126 else str(t) for t in token_ids])
        except:
            return " ".join([str(t) for t in token_ids])


def decode_predictions(
    predictions: Any,
    **kwargs,
) -> Union[str, List[str], Dict[str, Any]]:
    """
    Función helper para decodificar predicciones.
    
    Args:
        predictions: Predicciones a decodificar
        **kwargs: Argumentos adicionales
    
    Returns:
        Predicciones decodificadas
    """
    decoder = PredictionDecoder()
    return decoder.decode_predictions(predictions, **kwargs)


def decode_classification(
    predictions: Any,
    **kwargs,
) -> Union[int, str, Dict[str, Any]]:
    """
    Función helper para decodificar clasificación.
    
    Args:
        predictions: Predicciones a decodificar
        **kwargs: Argumentos adicionales
    
    Returns:
        Clase predicha
    """
    decoder = PredictionDecoder()
    return decoder.decode_classification(predictions, **kwargs)


def decode_sequence(
    predictions: Any,
    **kwargs,
) -> str:
    """
    Función helper para decodificar secuencia.
    
    Args:
        predictions: Predicciones a decodificar
        **kwargs: Argumentos adicionales
    
    Returns:
        Secuencia decodificada
    """
    decoder = PredictionDecoder()
    return decoder.decode_sequence(predictions, **kwargs)
'''
        
        (postprocessing_dir / "prediction_decoder.py").write_text(decoder_content, encoding="utf-8")

