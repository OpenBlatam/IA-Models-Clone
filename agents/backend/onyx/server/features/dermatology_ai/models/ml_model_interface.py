"""
Interfaz para modelos de Machine Learning
"""

from typing import Dict, List, Optional, Any
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import cv2


class MLModelInterface(ABC):
    """Interfaz base para modelos ML"""
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict:
        """
        Realiza predicción sobre una imagen
        
        Args:
            image: Imagen como numpy array
            
        Returns:
            Diccionario con predicciones
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str):
        """
        Carga un modelo desde archivo
        
        Args:
            model_path: Path al modelo
        """
        pass


class SkinAnalysisMLModel(MLModelInterface):
    """Modelo ML para análisis de piel (placeholder para integración futura)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el modelo
        
        Args:
            model_path: Path al modelo (opcional)
        """
        self.model = None
        self.model_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Carga modelo ML
        
        Args:
            model_path: Path al modelo
        """
        # Placeholder - en producción cargaría modelo real (TensorFlow, PyTorch, etc.)
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Aquí se cargaría el modelo real
        # Ejemplo:
        # import tensorflow as tf
        # self.model = tf.keras.models.load_model(model_path)
        
        self.model_loaded = True
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Realiza predicción ML
        
        Args:
            image: Imagen como numpy array
            
        Returns:
            Diccionario con predicciones ML
        """
        if not self.model_loaded:
            # Si no hay modelo, retornar predicciones basadas en análisis tradicional
            return {
                "ml_enabled": False,
                "message": "Modelo ML no cargado, usando análisis tradicional"
            }
        
        # Preprocesar imagen
        processed = self._preprocess_image(image)
        
        # Realizar predicción (placeholder)
        # predictions = self.model.predict(processed)
        
        # Retornar estructura de predicciones
        return {
            "ml_enabled": True,
            "predictions": {
                "skin_type": "normal",  # predictions[0]
                "conditions": [],  # predictions[1]
                "quality_scores": {}  # predictions[2]
            },
            "confidence": 0.85
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa imagen para modelo ML
        
        Args:
            image: Imagen original
            
        Returns:
            Imagen preprocesada
        """
        # Redimensionar a tamaño del modelo (ej: 224x224)
        target_size = (224, 224)
        
        if len(image.shape) == 3:
            processed = cv2.resize(image, target_size)
        else:
            processed = cv2.resize(image, target_size)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        # Normalizar a [0, 1]
        processed = processed.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch
        processed = np.expand_dims(processed, axis=0)
        
        return processed


class ModelManager:
    """Gestor de modelos ML"""
    
    def __init__(self):
        """Inicializa el gestor de modelos"""
        self.models: Dict[str, MLModelInterface] = {}
        self.models_enabled = False
    
    def register_model(self, name: str, model: MLModelInterface):
        """
        Registra un modelo
        
        Args:
            name: Nombre del modelo
            model: Instancia del modelo
        """
        self.models[name] = model
        self.models_enabled = True
    
    def get_model(self, name: str) -> Optional[MLModelInterface]:
        """Obtiene un modelo por nombre"""
        return self.models.get(name)
    
    def predict_with_model(self, model_name: str, image: np.ndarray) -> Dict:
        """
        Realiza predicción con un modelo específico
        
        Args:
            model_name: Nombre del modelo
            image: Imagen
            
        Returns:
            Predicciones
        """
        model = self.get_model(model_name)
        
        if not model:
            raise ValueError(f"Modelo no encontrado: {model_name}")
        
        return model.predict(image)
    
    def is_ml_enabled(self) -> bool:
        """Verifica si ML está habilitado"""
        return self.models_enabled and len(self.models) > 0

