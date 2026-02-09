"""
AI Models Integration - TensorFlow and PyTorch (optimizado)
============================================================

Integración con TensorFlow y PyTorch para modelos de IA en robot humanoide.
Incluye validaciones, manejo de errores robusto, y optimizaciones.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available.")

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available.")

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Excepción personalizada para errores de modelos."""
    pass


class TensorFlowModel:
    """
    Wrapper para modelos TensorFlow (optimizado).
    
    Incluye validaciones, manejo de errores, y optimizaciones.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_shape: Tuple[int, ...] = (32,),
        output_size: int = 32
    ):
        """
        Inicializar modelo TensorFlow (optimizado).
        
        Args:
            model_path: Ruta al modelo guardado
            input_shape: Forma de entrada del modelo
            output_size: Tamaño de salida del modelo
            
        Raises:
            ImportError: Si TensorFlow no está disponible
            ModelError: Si hay error al cargar el modelo
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available.")
        
        # Guard clauses
        if input_shape and len(input_shape) == 0:
            raise ValueError("input_shape cannot be empty")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        
        try:
            if model_path:
                self.load_model(model_path)
            else:
                self.model = self._create_default_model()
            
            logger.info(f"TensorFlow model initialized - Input: {input_shape}, Output: {output_size}")
        except Exception as e:
            logger.error(f"Error initializing TensorFlow model: {e}", exc_info=True)
            raise ModelError(f"Failed to initialize TensorFlow model: {str(e)}")
    
    def _create_default_model(self) -> tf.keras.Model:
        """
        Crear modelo por defecto para control humanoide (optimizado).
        
        Returns:
            Modelo TensorFlow compilado
        """
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=self.input_shape),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.output_size, activation='linear')
            ])
            
            # Compilar modelo
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            logger.debug(f"Created default TensorFlow model with shape {self.input_shape}")
            return model
        except Exception as e:
            logger.error(f"Error creating default model: {e}", exc_info=True)
            raise ModelError(f"Failed to create default model: {str(e)}")
    
    def load_model(self, model_path: str) -> None:
        """
        Cargar modelo desde archivo (optimizado).
        
        Args:
            model_path: Ruta al modelo guardado
            
        Raises:
            ModelError: Si hay error al cargar el modelo
        """
        if not model_path or not model_path.strip():
            raise ValueError("model_path cannot be empty")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = tf.keras.models.load_model(str(model_path))
            logger.info(f"Loaded TensorFlow model from {model_path}")
            
            # Validar que el modelo tiene la forma esperada
            if hasattr(self.model, 'input_shape'):
                logger.debug(f"Model input shape: {self.model.input_shape}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Realizar predicción (optimizado).
        
        Args:
            input_data: Datos de entrada
            
        Returns:
            Predicción del modelo
            
        Raises:
            ModelError: Si hay error en la predicción
        """
        if self.model is None:
            raise ModelError("Model not initialized")
        
        if input_data is None or input_data.size == 0:
            raise ValueError("input_data cannot be empty")
        
        try:
            # Validar forma de entrada
            expected_shape = self.model.input_shape[1:]  # Excluir batch dimension
            if input_data.shape[1:] != expected_shape:
                raise ValueError(
                    f"Input shape mismatch: expected {expected_shape}, "
                    f"got {input_data.shape[1:]}"
                )
            
            prediction = self.model.predict(input_data, verbose=0)
            return prediction
        except Exception as e:
            logger.error(f"Error in prediction: {e}", exc_info=True)
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> tf.keras.callbacks.History:
        """
        Entrenar modelo (optimizado).
        
        Args:
            x_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            epochs: Número de épocas
            batch_size: Tamaño de batch
            validation_split: Proporción de datos para validación
            
        Returns:
            Historial de entrenamiento
            
        Raises:
            ModelError: Si hay error en el entrenamiento
        """
        if self.model is None:
            raise ModelError("Model not initialized")
        
        # Guard clauses
        if x_train is None or x_train.size == 0:
            raise ValueError("x_train cannot be empty")
        if y_train is None or y_train.size == 0:
            raise ValueError("y_train cannot be empty")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        
        try:
            if not self.model._is_compiled:
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            history = self.model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0
            )
            
            logger.info(f"Model trained for {epochs} epochs")
            return history
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            raise ModelError(f"Training failed: {str(e)}")


class PyTorchModel(nn.Module):
    """
    Modelo PyTorch para control humanoide (optimizado).
    
    Incluye validaciones, dropout para regularización, y mejor arquitectura.
    """
    
    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 128,
        output_size: int = 32,
        dropout_rate: float = 0.2
    ):
        """
        Inicializar modelo PyTorch (optimizado).
        
        Args:
            input_size: Tamaño de entrada (número de articulaciones)
            hidden_size: Tamaño de capa oculta
            output_size: Tamaño de salida
            dropout_rate: Tasa de dropout para regularización
            
        Raises:
            ImportError: Si PyTorch no está disponible
            ValueError: Si los parámetros son inválidos
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
        
        # Guard clauses
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        super(PyTorchModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
        logger.info(
            f"PyTorch model initialized - Input: {input_size}, "
            f"Hidden: {hidden_size}, Output: {output_size}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (optimizado).
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Realizar predicción (optimizado).
        
        Args:
            input_data: Datos de entrada como numpy array
            
        Returns:
            Predicción como numpy array
            
        Raises:
            ModelError: Si hay error en la predicción
        """
        if input_data is None or input_data.size == 0:
            raise ValueError("input_data cannot be empty")
        
        try:
            self.eval()
            with torch.no_grad():
                # Validar forma de entrada
                if input_data.ndim == 1:
                    input_data = input_data.reshape(1, -1)
                
                if input_data.shape[1] != self.input_size:
                    raise ValueError(
                        f"Input size mismatch: expected {self.input_size}, "
                        f"got {input_data.shape[1]}"
                    )
                
                input_tensor = torch.FloatTensor(input_data)
                output = self.forward(input_tensor)
                return output.numpy()
        except Exception as e:
            logger.error(f"Error in PyTorch prediction: {e}", exc_info=True)
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def train_step(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        Un paso de entrenamiento (optimizado).
        
        Args:
            x_train: Tensor de datos de entrenamiento
            y_train: Tensor de etiquetas
            optimizer: Optimizador PyTorch
            criterion: Función de pérdida
            
        Returns:
            Valor de pérdida
            
        Raises:
            ModelError: Si hay error en el entrenamiento
        """
        if x_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        
        try:
            self.train()
            optimizer.zero_grad()
            output = self.forward(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            return loss.item()
        except Exception as e:
            logger.error(f"Error in training step: {e}", exc_info=True)
            raise ModelError(f"Training step failed: {str(e)}")


class AIModelManager:
    """
    Gestor de modelos de IA (optimizado).
    
    Gestiona múltiples modelos TensorFlow y PyTorch con validaciones y manejo de errores.
    """
    
    def __init__(self):
        """
        Inicializar gestor de modelos (optimizado).
        """
        self.tf_models: Dict[str, TensorFlowModel] = {}
        self.pytorch_models: Dict[str, PyTorchModel] = {}
        
        logger.info("AI Model Manager initialized")
    
    def load_tensorflow_model(
        self,
        name: str,
        model_path: Optional[str] = None,
        input_shape: Tuple[int, ...] = (32,),
        output_size: int = 32
    ) -> bool:
        """
        Cargar modelo TensorFlow (optimizado).
        
        Args:
            name: Nombre del modelo
            model_path: Ruta al modelo (opcional, crea uno por defecto si no se proporciona)
            input_shape: Forma de entrada (solo si se crea modelo por defecto)
            output_size: Tamaño de salida (solo si se crea modelo por defecto)
            
        Returns:
            True si se cargó exitosamente, False en caso contrario
        """
        if not name or not name.strip():
            logger.error("Model name cannot be empty")
            return False
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available")
            return False
        
        try:
            self.tf_models[name] = TensorFlowModel(
                model_path=model_path,
                input_shape=input_shape,
                output_size=output_size
            )
            logger.info(f"Loaded TensorFlow model: {name}")
            return True
        except Exception as e:
            logger.error(f"Error loading TensorFlow model '{name}': {e}", exc_info=True)
            return False
    
    def load_pytorch_model(
        self,
        name: str,
        model: Optional[PyTorchModel] = None,
        input_size: int = 32,
        hidden_size: int = 128,
        output_size: int = 32
    ) -> bool:
        """
        Cargar modelo PyTorch (optimizado).
        
        Args:
            name: Nombre del modelo
            model: Modelo PyTorch (opcional, crea uno por defecto si no se proporciona)
            input_size: Tamaño de entrada (solo si se crea modelo por defecto)
            hidden_size: Tamaño de capa oculta (solo si se crea modelo por defecto)
            output_size: Tamaño de salida (solo si se crea modelo por defecto)
            
        Returns:
            True si se cargó exitosamente, False en caso contrario
        """
        if not name or not name.strip():
            logger.error("Model name cannot be empty")
            return False
        
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            return False
        
        try:
            if model is None:
                model = PyTorchModel(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size
                )
            
            self.pytorch_models[name] = model
            logger.info(f"Loaded PyTorch model: {name}")
            return True
        except Exception as e:
            logger.error(f"Error loading PyTorch model '{name}': {e}", exc_info=True)
            return False
    
    def predict_with_tf(
        self,
        model_name: str,
        input_data: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Predecir usando modelo TensorFlow (optimizado).
        
        Args:
            model_name: Nombre del modelo
            input_data: Datos de entrada
            
        Returns:
            Predicción o None si hay error
        """
        if not model_name or model_name not in self.tf_models:
            logger.error(f"TensorFlow model '{model_name}' not found")
            return None
        
        try:
            return self.tf_models[model_name].predict(input_data)
        except Exception as e:
            logger.error(f"Error predicting with TensorFlow model '{model_name}': {e}", exc_info=True)
            return None
    
    def predict_with_pytorch(
        self,
        model_name: str,
        input_data: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Predecir usando modelo PyTorch (optimizado).
        
        Args:
            model_name: Nombre del modelo
            input_data: Datos de entrada
            
        Returns:
            Predicción o None si hay error
        """
        if not model_name or model_name not in self.pytorch_models:
            logger.error(f"PyTorch model '{model_name}' not found")
            return None
        
        try:
            return self.pytorch_models[model_name].predict(input_data)
        except Exception as e:
            logger.error(f"Error predicting with PyTorch model '{model_name}': {e}", exc_info=True)
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información de todos los modelos cargados (optimizado).
        
        Returns:
            Diccionario con información de modelos
        """
        return {
            "tensorflow_models": list(self.tf_models.keys()),
            "pytorch_models": list(self.pytorch_models.keys()),
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "pytorch_available": PYTORCH_AVAILABLE,
            "total_models": len(self.tf_models) + len(self.pytorch_models)
        }
    
    def unload_model(self, model_name: str, framework: str = "auto") -> bool:
        """
        Descargar modelo de memoria (optimizado).
        
        Args:
            model_name: Nombre del modelo a descargar
            framework: Framework ("tensorflow", "pytorch", o "auto" para ambos)
            
        Returns:
            True si se descargó exitosamente
        """
        if not model_name or not model_name.strip():
            logger.error("Model name cannot be empty")
            return False
        
        success = False
        
        if framework in ("auto", "tensorflow"):
            if model_name in self.tf_models:
                del self.tf_models[model_name]
                logger.info(f"Unloaded TensorFlow model: {model_name}")
                success = True
        
        if framework in ("auto", "pytorch"):
            if model_name in self.pytorch_models:
                del self.pytorch_models[model_name]
                logger.info(f"Unloaded PyTorch model: {model_name}")
                success = True
        
        if not success:
            logger.warning(f"Model '{model_name}' not found in {framework} models")
        
        return success
    
    def clear_all_models(self):
        """
        Limpiar todos los modelos de memoria (optimizado).
        """
        self.tf_models.clear()
        self.pytorch_models.clear()
        logger.info("All models cleared from memory")
    
    def has_model(self, model_name: str, framework: str = "auto") -> bool:
        """
        Verificar si un modelo está cargado (optimizado).
        
        Args:
            model_name: Nombre del modelo
            framework: Framework a verificar ("tensorflow", "pytorch", o "auto")
            
        Returns:
            True si el modelo está cargado
        """
        if framework in ("auto", "tensorflow"):
            if model_name in self.tf_models:
                return True
        
        if framework in ("auto", "pytorch"):
            if model_name in self.pytorch_models:
                return True
        
        return False
    
    def get_model_count(self) -> Dict[str, int]:
        """
        Obtener conteo de modelos por framework (optimizado).
        
        Returns:
            Dict con conteos de modelos
        """
        return {
            "tensorflow": len(self.tf_models),
            "pytorch": len(self.pytorch_models),
            "total": len(self.tf_models) + len(self.pytorch_models)
        }

