"""
Deep Learning Routing Module
============================

Módulo de enrutamiento basado en deep learning para predicción de tiempos,
costos y optimización de rutas usando PyTorch.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Importar optimizaciones extremas
try:
    from .routing_optimization import (
        OptimizedInferenceEngine,
        HardwareOptimizer,
        MemoryPoolOptimizer
    )
    EXTREME_OPTIMIZATION_AVAILABLE = True
except ImportError:
    EXTREME_OPTIMIZATION_AVAILABLE = False
    logger.warning("Extreme optimization modules not available")


@dataclass
class RouteFeatures:
    """Features para predicción de rutas."""
    distance: float
    time: float
    cost: float
    capacity: float
    current_load: float
    node_features: List[float] = field(default_factory=list)
    edge_features: List[float] = field(default_factory=list)
    temporal_features: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoutePredictionDataset(Dataset):
    """Dataset para entrenamiento del modelo de predicción."""
    
    def __init__(self, features: List[RouteFeatures], targets: List[Dict[str, float]]):
        """
        Inicializar dataset.
        
        Args:
            features: Lista de features de rutas
            targets: Lista de targets (tiempo, costo, etc.)
        """
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        
        # Construir vector de features
        feature_vector = np.array([
            feature.distance,
            feature.time,
            feature.cost,
            feature.capacity,
            feature.current_load,
            *feature.node_features,
            *feature.edge_features,
            *feature.temporal_features
        ], dtype=np.float32)
        
        # Targets
        target_vector = np.array([
            target.get("predicted_time", 0.0),
            target.get("predicted_cost", 0.0),
            target.get("predicted_load", 0.0),
            target.get("success_probability", 0.0)
        ], dtype=np.float32)
        
        return torch.FloatTensor(feature_vector), torch.FloatTensor(target_vector)


class RoutePredictionModel(nn.Module):
    """
    Modelo de deep learning para predicción de rutas.
    
    Arquitectura: MLP con capas residuales y atención.
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: List[int] = [128, 256, 128],
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        """
        Inicializar modelo.
        
        Args:
            input_dim: Dimensión de entrada
            hidden_dims: Dimensiones de capas ocultas
            dropout: Tasa de dropout
            use_attention: Usar mecanismo de atención
        """
        super(RoutePredictionModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention
        
        # Capa de entrada
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        
        # Capas ocultas con conexiones residuales
        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Mecanismo de atención (self-attention)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[-1],
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dims[-1])
        
        # Capa de salida (4 outputs: tiempo, costo, carga, probabilidad de éxito)
        self.output_layer = nn.Linear(hidden_dims[-1], 4)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos del modelo."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, input_dim]
            
        Returns:
            Tensor de salida [batch_size, 4]
        """
        # Capa de entrada
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Capas ocultas con conexiones residuales
        for i, (hidden, norm, dropout) in enumerate(
            zip(self.hidden_layers, self.norm_layers, self.dropout_layers)
        ):
            residual = x if i == 0 else None
            x = hidden(x)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
            
            # Conexión residual (si las dimensiones coinciden)
            if residual is not None and x.shape == residual.shape:
                x = x + residual
        
        # Mecanismo de atención
        if self.use_attention:
            # Reshape para atención: [batch_size, 1, hidden_dim]
            x_attn = x.unsqueeze(1)
            attn_out, _ = self.attention(x_attn, x_attn, x_attn)
            x = x + attn_out.squeeze(1)
            x = self.attention_norm(x)
        
        # Capa de salida
        output = self.output_layer(x)
        
        # Aplicar activaciones apropiadas
        # Tiempo y costo: ReLU (valores positivos)
        # Carga: Sigmoid (0-1)
        # Probabilidad: Sigmoid (0-1)
        output = torch.cat([
            F.relu(output[:, :2]),  # tiempo, costo
            torch.sigmoid(output[:, 2:])  # carga, probabilidad
        ], dim=1)
        
        return output


class DeepLearningRouter:
    """
    Enrutador basado en deep learning.
    
    Usa modelos de PyTorch para predecir tiempos, costos y optimizar rutas.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_mixed_precision: bool = True,
        use_extreme_optimization: bool = False
    ):
        """
        Inicializar enrutador de deep learning.
        
        Args:
            model_path: Ruta al modelo pre-entrenado (opcional)
            device: Dispositivo ('cuda', 'cpu', o None para auto-detectar)
            use_mixed_precision: Usar precisión mixta para entrenamiento
            use_extreme_optimization: Usar optimizaciones extremas para inferencia
        """
        # Detectar dispositivo
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self.use_extreme_optimization = use_extreme_optimization
        
        # Optimizar hardware si está disponible
        if EXTREME_OPTIMIZATION_AVAILABLE and self.use_extreme_optimization:
            HardwareOptimizer.auto_optimize()
            if self.device.type == "cuda":
                MemoryPoolOptimizer.optimize_memory_pool(device=self.device.index or 0)
        
        # Inicializar modelo
        self.model = RoutePredictionModel(
            input_dim=20,
            hidden_dims=[128, 256, 128],
            dropout=0.2,
            use_attention=True
        ).to(self.device)
        
        # Motor de inferencia optimizado (solo para inferencia)
        self.inference_engine: Optional[OptimizedInferenceEngine] = None
        if EXTREME_OPTIMIZATION_AVAILABLE and self.use_extreme_optimization:
            try:
                self.inference_engine = OptimizedInferenceEngine(
                    self.model,
                    device=str(self.device),
                    use_cache=True,
                    cache_size=10000
                )
                logger.info("Motor de inferencia extremo inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando motor de inferencia extremo: {e}")
                self.inference_engine = None
        
        # Optimizador y scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Scaler para mixed precision
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Criterio de pérdida
        self.criterion = nn.MSELoss()
        
        # Cargar modelo si existe
        if model_path:
            self.load_model(model_path)
        
        # Historial de entrenamiento
        self.training_history: List[Dict[str, float]] = []
        
        logger.info(f"DeepLearningRouter inicializado en dispositivo: {self.device}")
    
    def predict_route_metrics(
        self,
        features: RouteFeatures
    ) -> Dict[str, float]:
        """
        Predecir métricas de ruta.
        
        Args:
            features: Features de la ruta
            
        Returns:
            Diccionario con predicciones
        """
        # Construir vector de features
        feature_vector = np.array([
            features.distance,
            features.time,
            features.cost,
            features.capacity,
            features.current_load,
            *features.node_features[:10],  # Limitar a 10 features
            *features.edge_features[:5],   # Limitar a 5 features
            *features.temporal_features[:5]  # Limitar a 5 features
        ], dtype=np.float32)
        
        # Padding si es necesario
        if len(feature_vector) < 20:
            feature_vector = np.pad(
                feature_vector,
                (0, 20 - len(feature_vector)),
                mode='constant',
                constant_values=0.0
            )
        elif len(feature_vector) > 20:
            feature_vector = feature_vector[:20]
        
        # Convertir a tensor
        x = torch.FloatTensor(feature_vector).unsqueeze(0)
        
        # Usar motor de inferencia optimizado si está disponible
        if self.inference_engine is not None:
            predictions = self.inference_engine.predict(x.squeeze(0))
            predictions = predictions.unsqueeze(0) if predictions.dim() == 1 else predictions
            output = predictions.cpu().numpy()[0]
        else:
            # Fallback a método estándar
            x = x.to(self.device)
            self.model.eval()
            with torch.inference_mode():
                output = self.model(x)
                output = output.cpu().numpy()[0]
        
        return {
            "predicted_time": float(output[0]),
            "predicted_cost": float(output[1]),
            "predicted_load": float(output[2]),
            "success_probability": float(output[3])
        }
    
    def train(
        self,
        train_features: List[RouteFeatures],
        train_targets: List[Dict[str, float]],
        val_features: Optional[List[RouteFeatures]] = None,
        val_targets: Optional[List[Dict[str, float]]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 20
    ) -> Dict[str, Any]:
        """
        Entrenar modelo.
        
        Args:
            train_features: Features de entrenamiento
            train_targets: Targets de entrenamiento
            val_features: Features de validación (opcional)
            val_targets: Targets de validación (opcional)
            epochs: Número de épocas
            batch_size: Tamaño de batch
            early_stopping_patience: Paciencia para early stopping
            
        Returns:
            Historial de entrenamiento
        """
        # Crear datasets
        train_dataset = RoutePredictionDataset(train_features, train_targets)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Evitar problemas con multiprocessing
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        val_loader = None
        if val_features and val_targets:
            val_dataset = RoutePredictionDataset(val_features, val_targets)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == "cuda" else False
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Entrenamiento
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_features)
                        loss = self.criterion(outputs, batch_targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)
                    loss.backward()
                    self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validación
            val_loss = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        if self.use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(batch_features)
                                loss = self.criterion(outputs, batch_targets)
                        else:
                            outputs = self.model(batch_features)
                            loss = self.criterion(outputs, batch_targets)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping en época {epoch + 1}")
                        break
            else:
                self.scheduler.step(train_loss)
            
            # Guardar historial
            epoch_history = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            self.training_history.append(epoch_history)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Época {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}" +
                    (f" - Val Loss: {val_loss:.4f}" if val_loss else "")
                )
        
        return {
            "training_history": self.training_history,
            "best_val_loss": best_val_loss if val_loss else None
        }
    
    def save_model(self, path: str):
        """Guardar modelo."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_history": self.training_history
        }, path)
        logger.info(f"Modelo guardado en: {path}")
    
    def load_model(self, path: str):
        """Cargar modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        logger.info(f"Modelo cargado desde: {path}")

