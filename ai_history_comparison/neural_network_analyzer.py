"""
Advanced Neural Network and Deep Learning Analysis System
Sistema avanzado de análisis de redes neuronales y deep learning
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkType(Enum):
    """Tipos de redes neuronales"""
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    RESNET = "resnet"
    ATTENTION = "attention"

class TaskType(Enum):
    """Tipos de tareas"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQUENCE_TO_SEQUENCE = "sequence_to_sequence"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"

class FrameworkType(Enum):
    """Tipos de frameworks"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    KERAS = "keras"

@dataclass
class NetworkArchitecture:
    """Arquitectura de red neuronal"""
    id: str
    network_type: NetworkType
    framework: FrameworkType
    layers: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    total_parameters: int
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TrainingResult:
    """Resultado de entrenamiento"""
    id: str
    architecture_id: str
    task_type: TaskType
    training_history: Dict[str, List[float]]
    final_metrics: Dict[str, float]
    training_time: float
    epochs_trained: int
    best_epoch: int
    model_path: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModelPrediction:
    """Predicción del modelo"""
    id: str
    model_id: str
    input_data: Any
    prediction: Any
    confidence: float
    processing_time: float
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedNeuralNetworkAnalyzer:
    """
    Analizador avanzado de redes neuronales y deep learning
    """
    
    def __init__(
        self,
        enable_tensorflow: bool = True,
        enable_pytorch: bool = True,
        models_directory: str = "models/neural_networks/",
        max_models: int = 50
    ):
        self.enable_tensorflow = enable_tensorflow and TENSORFLOW_AVAILABLE
        self.enable_pytorch = enable_pytorch and PYTORCH_AVAILABLE
        self.models_directory = models_directory
        self.max_models = max_models
        
        # Almacenamiento
        self.network_architectures: Dict[str, NetworkArchitecture] = {}
        self.training_results: Dict[str, TrainingResult] = {}
        self.trained_models: Dict[str, Any] = {}
        self.model_predictions: Dict[str, ModelPrediction] = {}
        
        # Configuración
        self.config = {
            "default_epochs": 100,
            "default_batch_size": 32,
            "default_learning_rate": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
            "model_checkpoint": True,
            "tensorboard_logging": True
        }
        
        # Inicializar frameworks
        self._initialize_frameworks()
        
        # Crear directorio de modelos
        import os
        os.makedirs(self.models_directory, exist_ok=True)
    
    def _initialize_frameworks(self):
        """Inicializar frameworks de deep learning"""
        if self.enable_tensorflow:
            # Configurar TensorFlow
            tf.config.set_visible_devices([], 'GPU')  # Usar CPU por defecto
            logger.info("TensorFlow inicializado")
        
        if self.enable_pytorch:
            # Configurar PyTorch
            torch.set_num_threads(1)  # Usar un solo hilo por defecto
            logger.info("PyTorch inicializado")
    
    async def create_network_architecture(
        self,
        network_type: NetworkType,
        framework: FrameworkType,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        custom_layers: Optional[List[Dict[str, Any]]] = None
    ) -> NetworkArchitecture:
        """
        Crear arquitectura de red neuronal
        
        Args:
            network_type: Tipo de red neuronal
            framework: Framework a usar
            input_shape: Forma de entrada
            output_shape: Forma de salida
            custom_layers: Capas personalizadas
            
        Returns:
            Arquitectura de red neuronal
        """
        try:
            architecture_id = f"arch_{network_type.value}_{framework.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Creating {network_type.value} architecture with {framework.value}")
            
            # Crear capas según el tipo de red
            if custom_layers:
                layers_config = custom_layers
            else:
                layers_config = self._get_default_layers(network_type, input_shape, output_shape)
            
            # Calcular parámetros totales
            total_parameters = self._calculate_parameters(layers_config, input_shape)
            
            # Crear arquitectura
            architecture = NetworkArchitecture(
                id=architecture_id,
                network_type=network_type,
                framework=framework,
                layers=layers_config,
                parameters={
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "total_layers": len(layers_config)
                },
                input_shape=input_shape,
                output_shape=output_shape,
                total_parameters=total_parameters
            )
            
            # Almacenar arquitectura
            self.network_architectures[architecture_id] = architecture
            
            logger.info(f"Architecture {architecture_id} created with {total_parameters:,} parameters")
            return architecture
            
        except Exception as e:
            logger.error(f"Error creating network architecture: {e}")
            raise
    
    def _get_default_layers(
        self,
        network_type: NetworkType,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...]
    ) -> List[Dict[str, Any]]:
        """Obtener capas por defecto según el tipo de red"""
        
        if network_type == NetworkType.FEEDFORWARD:
            return [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": output_shape[0], "activation": "softmax"}
            ]
        
        elif network_type == NetworkType.CONVOLUTIONAL:
            return [
                {"type": "conv2d", "filters": 32, "kernel_size": 3, "activation": "relu"},
                {"type": "maxpool2d", "pool_size": 2},
                {"type": "conv2d", "filters": 64, "kernel_size": 3, "activation": "relu"},
                {"type": "maxpool2d", "pool_size": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.5},
                {"type": "dense", "units": output_shape[0], "activation": "softmax"}
            ]
        
        elif network_type == NetworkType.LSTM:
            return [
                {"type": "lstm", "units": 128, "return_sequences": True},
                {"type": "dropout", "rate": 0.2},
                {"type": "lstm", "units": 64, "return_sequences": False},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": output_shape[0], "activation": "softmax"}
            ]
        
        elif network_type == NetworkType.GRU:
            return [
                {"type": "gru", "units": 128, "return_sequences": True},
                {"type": "dropout", "rate": 0.2},
                {"type": "gru", "units": 64, "return_sequences": False},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": output_shape[0], "activation": "softmax"}
            ]
        
        elif network_type == NetworkType.AUTOENCODER:
            # Encoder
            encoder_layers = [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 32, "activation": "relu"}  # Latent space
            ]
            # Decoder
            decoder_layers = [
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": input_shape[0], "activation": "sigmoid"}
            ]
            return encoder_layers + decoder_layers
        
        else:
            # Red por defecto
            return [
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dense", "units": output_shape[0], "activation": "softmax"}
            ]
    
    def _calculate_parameters(self, layers_config: List[Dict[str, Any]], input_shape: Tuple[int, ...]) -> int:
        """Calcular número total de parámetros"""
        try:
            # Estimación simple de parámetros
            total_params = 0
            current_shape = input_shape[0] if len(input_shape) == 1 else np.prod(input_shape)
            
            for layer in layers_config:
                layer_type = layer.get("type", "")
                
                if layer_type == "dense":
                    units = layer.get("units", 1)
                    total_params += current_shape * units + units  # weights + biases
                    current_shape = units
                
                elif layer_type in ["conv2d", "conv1d"]:
                    filters = layer.get("filters", 1)
                    kernel_size = layer.get("kernel_size", 3)
                    if layer_type == "conv2d":
                        total_params += current_shape * filters * kernel_size * kernel_size + filters
                    else:
                        total_params += current_shape * filters * kernel_size + filters
                    current_shape = filters
                
                elif layer_type in ["lstm", "gru"]:
                    units = layer.get("units", 1)
                    # LSTM/GRU parameters: 4 * (input_size * units + units * units + units)
                    total_params += 4 * (current_shape * units + units * units + units)
                    current_shape = units
                
                elif layer_type == "flatten":
                    current_shape = current_shape  # No change in parameters
            
            return total_params
            
        except Exception as e:
            logger.error(f"Error calculating parameters: {e}")
            return 0
    
    async def train_model(
        self,
        architecture_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: TaskType,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> TrainingResult:
        """
        Entrenar modelo de red neuronal
        
        Args:
            architecture_id: ID de la arquitectura
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            task_type: Tipo de tarea
            X_val: Datos de validación
            y_val: Etiquetas de validación
            epochs: Número de épocas
            batch_size: Tamaño del batch
            learning_rate: Tasa de aprendizaje
            
        Returns:
            Resultado del entrenamiento
        """
        try:
            if architecture_id not in self.network_architectures:
                raise ValueError(f"Architecture {architecture_id} not found")
            
            architecture = self.network_architectures[architecture_id]
            
            # Configurar parámetros
            epochs = epochs or self.config["default_epochs"]
            batch_size = batch_size or self.config["default_batch_size"]
            learning_rate = learning_rate or self.config["default_learning_rate"]
            
            logger.info(f"Training {architecture.network_type.value} model with {architecture.framework.value}")
            
            start_time = time.time()
            
            # Entrenar según el framework
            if architecture.framework == FrameworkType.TENSORFLOW and self.enable_tensorflow:
                model, history = await self._train_tensorflow_model(
                    architecture, X_train, y_train, X_val, y_val, task_type,
                    epochs, batch_size, learning_rate
                )
            elif architecture.framework == FrameworkType.PYTORCH and self.enable_pytorch:
                model, history = await self._train_pytorch_model(
                    architecture, X_train, y_train, X_val, y_val, task_type,
                    epochs, batch_size, learning_rate
                )
            else:
                raise ValueError(f"Framework {architecture.framework.value} not available")
            
            training_time = time.time() - start_time
            
            # Calcular métricas finales
            final_metrics = self._calculate_final_metrics(model, X_val, y_val, task_type)
            
            # Encontrar mejor época
            best_epoch = self._find_best_epoch(history)
            
            # Guardar modelo
            model_path = await self._save_model(model, architecture_id)
            
            # Crear resultado de entrenamiento
            result = TrainingResult(
                id=f"training_{architecture_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                architecture_id=architecture_id,
                task_type=task_type,
                training_history=history,
                final_metrics=final_metrics,
                training_time=training_time,
                epochs_trained=epochs,
                best_epoch=best_epoch,
                model_path=model_path
            )
            
            # Almacenar resultado y modelo
            self.training_results[result.id] = result
            self.trained_models[result.id] = model
            
            logger.info(f"Model trained successfully in {training_time:.2f} seconds")
            logger.info(f"Final metrics: {final_metrics}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    async def _train_tensorflow_model(
        self,
        architecture: NetworkArchitecture,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        task_type: TaskType,
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Tuple[Any, Dict[str, List[float]]]:
        """Entrenar modelo con TensorFlow"""
        try:
            # Crear modelo
            model = self._build_tensorflow_model(architecture, task_type)
            
            # Compilar modelo
            if task_type == TaskType.CLASSIFICATION:
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            elif task_type == TaskType.REGRESSION:
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=learning_rate),
                    loss='mse',
                    metrics=['mae']
                )
            else:
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=learning_rate),
                    loss='mse',
                    metrics=['mae']
                )
            
            # Configurar callbacks
            callbacks_list = []
            
            if self.config["early_stopping"]:
                early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config["early_stopping_patience"],
                    restore_best_weights=True
                )
                callbacks_list.append(early_stopping)
            
            if self.config["model_checkpoint"]:
                checkpoint = callbacks.ModelCheckpoint(
                    f"{self.models_directory}/best_model_{architecture.id}.h5",
                    monitor='val_loss',
                    save_best_only=True
                )
                callbacks_list.append(checkpoint)
            
            # Entrenar modelo
            if X_val is not None and y_val is not None:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    verbose=0
                )
            else:
                history = model.fit(
                    X_train, y_train,
                    validation_split=self.config["validation_split"],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks_list,
                    verbose=0
                )
            
            # Convertir historia a diccionario
            history_dict = {
                'loss': history.history['loss'],
                'val_loss': history.history.get('val_loss', []),
                'accuracy': history.history.get('accuracy', []),
                'val_accuracy': history.history.get('val_accuracy', [])
            }
            
            return model, history_dict
            
        except Exception as e:
            logger.error(f"Error training TensorFlow model: {e}")
            raise
    
    def _build_tensorflow_model(self, architecture: NetworkArchitecture, task_type: TaskType) -> Any:
        """Construir modelo TensorFlow"""
        try:
            model = models.Sequential()
            
            # Capa de entrada
            model.add(layers.Input(shape=architecture.input_shape))
            
            # Agregar capas según la configuración
            for layer_config in architecture.layers:
                layer_type = layer_config.get("type", "")
                
                if layer_type == "dense":
                    model.add(layers.Dense(
                        units=layer_config.get("units", 1),
                        activation=layer_config.get("activation", "relu")
                    ))
                
                elif layer_type == "dropout":
                    model.add(layers.Dropout(rate=layer_config.get("rate", 0.2)))
                
                elif layer_type == "conv2d":
                    model.add(layers.Conv2D(
                        filters=layer_config.get("filters", 32),
                        kernel_size=layer_config.get("kernel_size", 3),
                        activation=layer_config.get("activation", "relu")
                    ))
                
                elif layer_type == "maxpool2d":
                    model.add(layers.MaxPooling2D(
                        pool_size=layer_config.get("pool_size", 2)
                    ))
                
                elif layer_type == "flatten":
                    model.add(layers.Flatten())
                
                elif layer_type == "lstm":
                    model.add(layers.LSTM(
                        units=layer_config.get("units", 128),
                        return_sequences=layer_config.get("return_sequences", False)
                    ))
                
                elif layer_type == "gru":
                    model.add(layers.GRU(
                        units=layer_config.get("units", 128),
                        return_sequences=layer_config.get("return_sequences", False)
                    ))
            
            return model
            
        except Exception as e:
            logger.error(f"Error building TensorFlow model: {e}")
            raise
    
    async def _train_pytorch_model(
        self,
        architecture: NetworkArchitecture,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        task_type: TaskType,
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Tuple[Any, Dict[str, List[float]]]:
        """Entrenar modelo con PyTorch"""
        try:
            # Crear modelo
            model = self._build_pytorch_model(architecture, task_type)
            
            # Configurar optimizador y función de pérdida
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            if task_type == TaskType.CLASSIFICATION:
                criterion = nn.CrossEntropyLoss()
            elif task_type == TaskType.REGRESSION:
                criterion = nn.MSELoss()
            else:
                criterion = nn.MSELoss()
            
            # Convertir datos a tensores
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train) if task_type == TaskType.CLASSIFICATION else torch.FloatTensor(y_train)
            
            # Crear DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Historial de entrenamiento
            history = {
                'loss': [],
                'val_loss': [],
                'accuracy': [],
                'val_accuracy': []
            }
            
            # Entrenar modelo
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if task_type == TaskType.CLASSIFICATION:
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                # Calcular métricas de la época
                avg_loss = epoch_loss / len(train_loader)
                history['loss'].append(avg_loss)
                
                if task_type == TaskType.CLASSIFICATION:
                    accuracy = 100 * correct / total
                    history['accuracy'].append(accuracy)
                
                # Validación si hay datos de validación
                if X_val is not None and y_val is not None:
                    val_loss, val_accuracy = self._validate_pytorch_model(
                        model, X_val, y_val, criterion, task_type
                    )
                    history['val_loss'].append(val_loss)
                    if task_type == TaskType.CLASSIFICATION:
                        history['val_accuracy'].append(val_accuracy)
            
            return model, history
            
        except Exception as e:
            logger.error(f"Error training PyTorch model: {e}")
            raise
    
    def _build_pytorch_model(self, architecture: NetworkArchitecture, task_type: TaskType) -> Any:
        """Construir modelo PyTorch"""
        try:
            layers_list = []
            
            # Procesar capas
            input_size = architecture.input_shape[0] if len(architecture.input_shape) == 1 else np.prod(architecture.input_shape)
            
            for layer_config in architecture.layers:
                layer_type = layer_config.get("type", "")
                
                if layer_type == "dense":
                    units = layer_config.get("units", 1)
                    activation = layer_config.get("activation", "relu")
                    
                    layers_list.append(nn.Linear(input_size, units))
                    
                    if activation == "relu":
                        layers_list.append(nn.ReLU())
                    elif activation == "sigmoid":
                        layers_list.append(nn.Sigmoid())
                    elif activation == "tanh":
                        layers_list.append(nn.Tanh())
                    elif activation == "softmax":
                        layers_list.append(nn.Softmax(dim=1))
                    
                    input_size = units
                
                elif layer_type == "dropout":
                    rate = layer_config.get("rate", 0.2)
                    layers_list.append(nn.Dropout(rate))
                
                elif layer_type == "lstm":
                    units = layer_config.get("units", 128)
                    layers_list.append(nn.LSTM(
                        input_size=input_size,
                        hidden_size=units,
                        batch_first=True
                    ))
                    input_size = units
                
                elif layer_type == "gru":
                    units = layer_config.get("units", 128)
                    layers_list.append(nn.GRU(
                        input_size=input_size,
                        hidden_size=units,
                        batch_first=True
                    ))
                    input_size = units
            
            return nn.Sequential(*layers_list)
            
        except Exception as e:
            logger.error(f"Error building PyTorch model: {e}")
            raise
    
    def _validate_pytorch_model(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        criterion: Any,
        task_type: TaskType
    ) -> Tuple[float, float]:
        """Validar modelo PyTorch"""
        try:
            model.eval()
            
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val) if task_type == TaskType.CLASSIFICATION else torch.FloatTensor(y_val)
            
            with torch.no_grad():
                outputs = model(X_val_tensor)
                loss = criterion(outputs, y_val_tensor).item()
                
                if task_type == TaskType.CLASSIFICATION:
                    _, predicted = torch.max(outputs.data, 1)
                    accuracy = 100 * (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
                else:
                    accuracy = 0.0  # No aplicable para regresión
            
            return loss, accuracy
            
        except Exception as e:
            logger.error(f"Error validating PyTorch model: {e}")
            return 0.0, 0.0
    
    def _calculate_final_metrics(
        self,
        model: Any,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        task_type: TaskType
    ) -> Dict[str, float]:
        """Calcular métricas finales"""
        try:
            metrics = {}
            
            if X_val is not None and y_val is not None:
                # Hacer predicciones
                if hasattr(model, 'predict'):  # TensorFlow/Keras
                    predictions = model.predict(X_val, verbose=0)
                else:  # PyTorch
                    model.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.FloatTensor(X_val)
                        predictions = model(X_val_tensor).numpy()
                
                if task_type == TaskType.CLASSIFICATION:
                    # Métricas de clasificación
                    y_pred = np.argmax(predictions, axis=1)
                    y_true = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val
                    
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
                
                elif task_type == TaskType.REGRESSION:
                    # Métricas de regresión
                    mse = np.mean((predictions.flatten() - y_val.flatten()) ** 2)
                    mae = np.mean(np.abs(predictions.flatten() - y_val.flatten()))
                    r2 = 1 - (np.sum((y_val.flatten() - predictions.flatten()) ** 2) / 
                             np.sum((y_val.flatten() - np.mean(y_val.flatten())) ** 2))
                    
                    metrics['mse'] = mse
                    metrics['mae'] = mae
                    metrics['r2_score'] = r2
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}")
            return {}
    
    def _find_best_epoch(self, history: Dict[str, List[float]]) -> int:
        """Encontrar la mejor época"""
        try:
            if 'val_loss' in history and history['val_loss']:
                best_epoch = np.argmin(history['val_loss']) + 1
            elif 'loss' in history and history['loss']:
                best_epoch = np.argmin(history['loss']) + 1
            else:
                best_epoch = 1
            
            return best_epoch
            
        except Exception as e:
            logger.error(f"Error finding best epoch: {e}")
            return 1
    
    async def _save_model(self, model: Any, architecture_id: str) -> str:
        """Guardar modelo"""
        try:
            model_path = f"{self.models_directory}/model_{architecture_id}.h5"
            
            if hasattr(model, 'save'):  # TensorFlow/Keras
                model.save(model_path)
            else:  # PyTorch
                torch.save(model.state_dict(), model_path.replace('.h5', '.pth'))
                model_path = model_path.replace('.h5', '.pth')
            
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    async def predict(
        self,
        model_id: str,
        input_data: np.ndarray
    ) -> ModelPrediction:
        """
        Hacer predicción con modelo entrenado
        
        Args:
            model_id: ID del modelo entrenado
            input_data: Datos de entrada
            
        Returns:
            Predicción del modelo
        """
        try:
            if model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.trained_models[model_id]
            
            logger.info(f"Making prediction with model {model_id}")
            
            start_time = time.time()
            
            # Hacer predicción
            if hasattr(model, 'predict'):  # TensorFlow/Keras
                prediction = model.predict(input_data, verbose=0)
            else:  # PyTorch
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_data)
                    prediction = model(input_tensor).numpy()
            
            processing_time = time.time() - start_time
            
            # Calcular confianza (simplificado)
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                confidence = np.max(prediction, axis=1).mean()
            else:
                confidence = 1.0 - np.std(prediction) / (np.mean(prediction) + 1e-8)
            
            # Crear predicción
            model_prediction = ModelPrediction(
                id=f"pred_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_id=model_id,
                input_data=input_data.tolist(),
                prediction=prediction.tolist(),
                confidence=float(confidence),
                processing_time=processing_time
            )
            
            # Almacenar predicción
            self.model_predictions[model_prediction.id] = model_prediction
            
            logger.info(f"Prediction completed in {processing_time:.4f} seconds")
            return model_prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    async def compare_models(
        self,
        model_ids: List[str]
    ) -> Dict[str, Any]:
        """Comparar modelos entrenados"""
        try:
            if len(model_ids) < 2:
                raise ValueError("Se necesitan al menos 2 modelos para comparar")
            
            comparison = {
                "model_ids": model_ids,
                "models_found": 0,
                "comparison_results": {}
            }
            
            # Obtener resultados de entrenamiento
            training_results = []
            for model_id in model_ids:
                if model_id in self.training_results:
                    training_results.append(self.training_results[model_id])
                    comparison["models_found"] += 1
            
            if len(training_results) < 2:
                raise ValueError("No hay suficientes resultados de entrenamiento para comparar")
            
            # Comparar métricas
            metrics_comparison = {}
            for result in training_results:
                for metric, value in result.final_metrics.items():
                    if metric not in metrics_comparison:
                        metrics_comparison[metric] = []
                    metrics_comparison[metric].append({
                        "model_id": result.id,
                        "value": value
                    })
            
            # Encontrar mejor modelo por métrica
            best_models = {}
            for metric, values in metrics_comparison.items():
                best_value = max(values, key=lambda x: x["value"])
                best_models[metric] = {
                    "model_id": best_value["model_id"],
                    "value": best_value["value"]
                }
            
            comparison["comparison_results"] = {
                "metrics_comparison": metrics_comparison,
                "best_models": best_models,
                "total_models": len(training_results)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
    
    async def get_neural_network_summary(self) -> Dict[str, Any]:
        """Obtener resumen de redes neuronales"""
        try:
            return {
                "total_architectures": len(self.network_architectures),
                "total_trained_models": len(self.trained_models),
                "total_predictions": len(self.model_predictions),
                "frameworks_available": {
                    "tensorflow": self.enable_tensorflow,
                    "pytorch": self.enable_pytorch
                },
                "network_types": {
                    arch.network_type.value: len([a for a in self.network_architectures.values() if a.network_type == arch.network_type])
                    for arch in self.network_architectures.values()
                },
                "task_types": {
                    result.task_type.value: len([r for r in self.training_results.values() if r.task_type == result.task_type])
                    for result in self.training_results.values()
                },
                "average_training_time": np.mean([r.training_time for r in self.training_results.values()]) if self.training_results else 0,
                "total_parameters": sum(arch.total_parameters for arch in self.network_architectures.values()),
                "last_activity": max([arch.created_at for arch in self.network_architectures.values()]).isoformat() if self.network_architectures else None
            }
        except Exception as e:
            logger.error(f"Error getting neural network summary: {e}")
            return {}
    
    async def export_neural_network_data(self, filepath: str = None) -> str:
        """Exportar datos de redes neuronales"""
        try:
            if filepath is None:
                filepath = f"exports/neural_network_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            export_data = {
                "network_architectures": {
                    arch_id: {
                        "network_type": arch.network_type.value,
                        "framework": arch.framework.value,
                        "layers": arch.layers,
                        "parameters": arch.parameters,
                        "input_shape": arch.input_shape,
                        "output_shape": arch.output_shape,
                        "total_parameters": arch.total_parameters,
                        "created_at": arch.created_at.isoformat()
                    }
                    for arch_id, arch in self.network_architectures.items()
                },
                "training_results": {
                    result_id: {
                        "architecture_id": result.architecture_id,
                        "task_type": result.task_type.value,
                        "training_history": result.training_history,
                        "final_metrics": result.final_metrics,
                        "training_time": result.training_time,
                        "epochs_trained": result.epochs_trained,
                        "best_epoch": result.best_epoch,
                        "model_path": result.model_path,
                        "created_at": result.created_at.isoformat()
                    }
                    for result_id, result in self.training_results.items()
                },
                "model_predictions": {
                    pred_id: {
                        "model_id": pred.model_id,
                        "input_data": pred.input_data,
                        "prediction": pred.prediction,
                        "confidence": pred.confidence,
                        "processing_time": pred.processing_time,
                        "created_at": pred.created_at.isoformat()
                    }
                    for pred_id, pred in self.model_predictions.items()
                },
                "summary": await self.get_neural_network_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Neural network data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting neural network data: {e}")
            raise
























