"""
Deep Learning Models System
============================

Sistema de modelos de deep learning para movimiento y control de robots.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Dataset = None
    DataLoader = None

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipo de modelo."""
    TRAJECTORY_PREDICTOR = "trajectory_predictor"
    MOTION_CONTROLLER = "motion_controller"
    OBSTACLE_DETECTOR = "obstacle_detector"
    PATH_PLANNER = "path_planner"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


@dataclass
class ModelConfig:
    """Configuración de modelo."""
    model_id: str
    model_type: ModelType
    input_size: int
    output_size: int
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "relu"
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    use_gpu: bool = True
    mixed_precision: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: float = 0.001
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ModelCheckpoint:
    """Checkpoint de modelo."""
    checkpoint_id: str
    model_id: str
    epoch: int
    loss: float
    metrics: Dict[str, float]
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryPredictor(nn.Module):
    """
    Modelo de red neuronal para predecir trayectorias.
    
    Arquitectura: MLP con capas ocultas configurables.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        activation: str = "relu",
        dropout: float = 0.1
    ):
        """
        Inicializar modelo.
        
        Args:
            input_size: Tamaño de entrada (posición actual, velocidad, etc.)
            output_size: Tamaño de salida (posición futura, velocidad, etc.)
            hidden_sizes: Tamaños de capas ocultas
            activation: Función de activación
            dropout: Tasa de dropout
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # Construir capas
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos usando Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, input_size]
            
        Returns:
            Tensor de salida [batch_size, output_size]
        """
        return self.network(x)


class MotionController(nn.Module):
    """
    Controlador de movimiento basado en deep learning.
    
    Usa LSTM para modelar secuencias temporales.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Inicializar controlador.
        
        Args:
            input_size: Tamaño de entrada
            output_size: Tamaño de salida
            hidden_size: Tamaño de capa oculta LSTM
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # LSTM para secuencias temporales
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Capa de salida
        self.fc = nn.Linear(hidden_size, output_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_size]
            
        Returns:
            Tensor de salida [batch_size, output_size]
        """
        lstm_out, _ = self.lstm(x)
        # Usar la última salida de la secuencia
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class ObstacleDetector(nn.Module):
    """
    Detector de obstáculos usando CNN.
    
    Procesa datos de sensores (LIDAR, cámaras, etc.).
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,  # obstáculo / sin obstáculo
        conv_channels: List[int] = [32, 64, 128]
    ):
        """
        Inicializar detector.
        
        Args:
            input_channels: Canales de entrada
            num_classes: Número de clases
            conv_channels: Canales de convolución
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")
        
        # Capas convolucionales
        conv_layers = []
        prev_channels = input_channels
        
        for channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            prev_channels = channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(conv_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, channels, height, width]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        x = self.conv_layers(x)
        return self.fc(x)


class RobotDataset(Dataset):
    """Dataset para entrenamiento de modelos de robot."""
    
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        transform: Optional[Callable] = None
    ):
        """
        Inicializar dataset.
        
        Args:
            inputs: Datos de entrada
            targets: Datos objetivo
            transform: Transformación opcional
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for datasets")
        
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
    
    def __len__(self) -> int:
        """Tamaño del dataset."""
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtener item."""
        input_data = self.inputs[idx]
        target = self.targets[idx]
        
        if self.transform:
            input_data = self.transform(input_data)
        
        return input_data, target


class DeepLearningModelManager:
    """
    Gestor de modelos de deep learning.
    
    Maneja creación, entrenamiento y uso de modelos.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Deep learning features will be limited.")
        
        self.models: Dict[str, nn.Module] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.checkpoints: Dict[str, List[ModelCheckpoint]] = {}
        self.training_metrics: Dict[str, List[TrainingMetrics]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    
    def create_model(
        self,
        model_type: ModelType,
        input_size: int,
        output_size: int,
        config: Optional[ModelConfig] = None
    ) -> str:
        """
        Crear nuevo modelo.
        
        Args:
            model_type: Tipo de modelo
            input_size: Tamaño de entrada
            output_size: Tamaño de salida
            config: Configuración opcional
            
        Returns:
            ID del modelo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model creation")
        
        model_id = str(uuid.uuid4())
        
        if config is None:
            config = ModelConfig(
                model_id=model_id,
                model_type=model_type,
                input_size=input_size,
                output_size=output_size
            )
        
        # Crear modelo según tipo
        if model_type == ModelType.TRAJECTORY_PREDICTOR:
            model = TrajectoryPredictor(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=config.hidden_sizes,
                activation=config.activation,
                dropout=config.dropout
            )
        elif model_type == ModelType.MOTION_CONTROLLER:
            model = MotionController(
                input_size=input_size,
                output_size=output_size,
                hidden_size=config.hidden_sizes[0] if config.hidden_sizes else 128
            )
        elif model_type == ModelType.OBSTACLE_DETECTOR:
            model = ObstacleDetector(
                input_channels=input_size,
                num_classes=output_size
            )
        else:
            # Default: TrajectoryPredictor
            model = TrajectoryPredictor(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=config.hidden_sizes,
                activation=config.activation,
                dropout=config.dropout
            )
        
        model = model.to(self.device)
        self.models[model_id] = model
        self.configs[model_id] = config
        self.checkpoints[model_id] = []
        self.training_metrics[model_id] = []
        
        logger.info(f"Created model {model_id} of type {model_type.value}")
        
        return model_id
    
    def train_model(
        self,
        model_id: str,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        val_inputs: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> List[TrainingMetrics]:
        """
        Entrenar modelo.
        
        Args:
            model_id: ID del modelo
            train_inputs: Datos de entrenamiento
            train_targets: Objetivos de entrenamiento
            val_inputs: Datos de validación (opcional)
            val_targets: Objetivos de validación (opcional)
            loss_fn: Función de pérdida (opcional)
            optimizer: Optimizador (opcional)
            
        Returns:
            Lista de métricas de entrenamiento
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")
        
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.models[model_id]
        config = self.configs[model_id]
        
        # Dataset y DataLoader
        train_dataset = RobotDataset(train_inputs, train_targets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_inputs is not None and val_targets is not None:
            val_dataset = RobotDataset(val_inputs, val_targets)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False
            )
        
        # Loss y optimizer
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate
            )
        
        # Mixed precision
        scaler = None
        if config.mixed_precision and self.device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
        
        metrics = []
        
        # Entrenamiento
        for epoch in range(config.num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validación
            val_loss = None
            if val_loader:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
            
            # Guardar métricas
            metric = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=optimizer.param_groups[0]['lr']
            )
            metrics.append(metric)
            self.training_metrics[model_id].append(metric)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}" +
                    (f", Val Loss: {val_loss:.4f}" if val_loss else "")
                )
        
        return metrics
    
    def predict(
        self,
        model_id: str,
        inputs: np.ndarray
    ) -> np.ndarray:
        """
        Realizar predicción.
        
        Args:
            model_id: ID del modelo
            inputs: Datos de entrada
            
        Returns:
            Predicciones
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for prediction")
        
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.models[model_id]
        model.eval()
        
        inputs_tensor = torch.FloatTensor(inputs).to(self.device)
        
        with torch.no_grad():
            outputs = model(inputs_tensor)
        
        return outputs.cpu().numpy()
    
    def save_checkpoint(
        self,
        model_id: str,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> str:
        """
        Guardar checkpoint.
        
        Args:
            model_id: ID del modelo
            epoch: Época actual
            loss: Pérdida actual
            metrics: Métricas adicionales
            optimizer: Optimizador (opcional)
            
        Returns:
            ID del checkpoint
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        checkpoint_id = str(uuid.uuid4())
        
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            epoch=epoch,
            loss=loss,
            metrics=metrics,
            model_state=self.models[model_id].state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer else None
        )
        
        self.checkpoints[model_id].append(checkpoint)
        
        logger.info(f"Saved checkpoint {checkpoint_id} for model {model_id}")
        
        return checkpoint_id
    
    def load_checkpoint(
        self,
        model_id: str,
        checkpoint_id: str
    ) -> ModelCheckpoint:
        """
        Cargar checkpoint.
        
        Args:
            model_id: ID del modelo
            checkpoint_id: ID del checkpoint
            
        Returns:
            Checkpoint cargado
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        checkpoints = self.checkpoints.get(model_id, [])
        checkpoint = next(
            (c for c in checkpoints if c.checkpoint_id == checkpoint_id),
            None
        )
        
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        self.models[model_id].load_state_dict(checkpoint.model_state)
        
        logger.info(f"Loaded checkpoint {checkpoint_id} for model {model_id}")
        
        return checkpoint
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        model_types = {}
        for config in self.configs.values():
            model_types[config.model_type.value] = model_types.get(config.model_type.value, 0) + 1
        
        return {
            "total_models": len(self.models),
            "model_types": model_types,
            "total_checkpoints": sum(len(cps) for cps in self.checkpoints.values()),
            "device": str(self.device) if self.device else "N/A"
        }


# Instancia global
_dl_model_manager: Optional[DeepLearningModelManager] = None


def get_dl_model_manager() -> DeepLearningModelManager:
    """Obtener instancia global del gestor de modelos."""
    global _dl_model_manager
    if _dl_model_manager is None:
        _dl_model_manager = DeepLearningModelManager()
    return _dl_model_manager




