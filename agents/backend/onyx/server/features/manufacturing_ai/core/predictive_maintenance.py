"""
Predictive Maintenance System
==============================

Sistema de mantenimiento predictivo usando deep learning.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class MaintenanceStatus(Enum):
    """Estado de mantenimiento."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class MaintenancePrediction:
    """Predicción de mantenimiento."""
    prediction_id: str
    equipment_id: str
    predicted_failure_date: Optional[str]
    failure_probability: float
    remaining_life: Optional[float]  # días
    recommended_action: str
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SensorData:
    """Datos de sensores."""
    equipment_id: str
    timestamp: str
    temperature: float = 0.0
    vibration: float = 0.0
    pressure: float = 0.0
    current: float = 0.0
    other_metrics: Dict[str, float] = field(default_factory=dict)


class PredictiveMaintenanceModel(nn.Module):
    """
    Modelo para mantenimiento predictivo.
    
    Usa CNN para análisis de señales de sensores.
    """
    
    def __init__(
        self,
        input_channels: int = 4,  # temperature, vibration, pressure, current
        num_classes: int = 4  # healthy, warning, critical, failure
    ):
        """
        Inicializar modelo.
        
        Args:
            input_channels: Canales de entrada
            num_classes: Número de clases
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        # CNN para señales temporales (1D)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Regresor para tiempo restante
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Días restantes
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch, channels, sequence_length]
            
        Returns:
            Tupla (clasificación, tiempo_restante)
        """
        # CNN
        features = self.conv_layers(x)
        
        # Clasificación
        classification = self.classifier(features)
        
        # Regresión (tiempo restante)
        remaining_life = self.regressor(features)
        
        return classification, remaining_life


class PredictiveMaintenanceSystem:
    """
    Sistema de mantenimiento predictivo.
    
    Predice fallas de equipos usando datos de sensores.
    """
    
    def __init__(self):
        """Inicializar sistema."""
        self.models: Dict[str, PredictiveMaintenanceModel] = {}
        self.predictions: Dict[str, MaintenancePrediction] = {}
        self.sensor_data: Dict[str, List[SensorData]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    
    def create_model(
        self,
        equipment_id: str,
        input_channels: int = 4
    ) -> str:
        """
        Crear modelo para equipo.
        
        Args:
            equipment_id: ID del equipo
            input_channels: Canales de entrada
            
        Returns:
            ID del modelo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        model = PredictiveMaintenanceModel(input_channels=input_channels)
        model = model.to(self.device)
        self.models[equipment_id] = model
        
        logger.info(f"Created predictive maintenance model for {equipment_id}")
        return equipment_id
    
    def add_sensor_data(self, sensor_data: SensorData):
        """
        Agregar datos de sensores.
        
        Args:
            sensor_data: Datos de sensores
        """
        if sensor_data.equipment_id not in self.sensor_data:
            self.sensor_data[sensor_data.equipment_id] = []
        
        self.sensor_data[sensor_data.equipment_id].append(sensor_data)
        
        # Mantener solo últimos 1000 puntos
        if len(self.sensor_data[sensor_data.equipment_id]) > 1000:
            self.sensor_data[sensor_data.equipment_id] = \
                self.sensor_data[sensor_data.equipment_id][-1000:]
    
    def predict_failure(
        self,
        equipment_id: str,
        sequence_length: int = 100
    ) -> MaintenancePrediction:
        """
        Predecir falla.
        
        Args:
            equipment_id: ID del equipo
            sequence_length: Longitud de secuencia
            
        Returns:
            Predicción de mantenimiento
        """
        if equipment_id not in self.models:
            raise ValueError(f"Model not found for equipment: {equipment_id}")
        
        if equipment_id not in self.sensor_data:
            raise ValueError(f"No sensor data for equipment: {equipment_id}")
        
        model = self.models[equipment_id]
        model.eval()
        
        # Obtener datos recientes
        recent_data = self.sensor_data[equipment_id][-sequence_length:]
        
        # Preparar input [channels, sequence_length]
        input_data = np.array([
            [d.temperature, d.vibration, d.pressure, d.current]
            for d in recent_data
        ]).T  # [4, sequence_length]
        
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            classification, remaining_life = model(input_tensor)
            
            probs = torch.softmax(classification, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            failure_prob = float(probs[0, 3].item())  # Probabilidad de failure
            remaining_days = float(remaining_life.item())
        
        # Determinar estado y acción
        if predicted_class == 0:
            status = MaintenanceStatus.HEALTHY
            action = "Continue monitoring"
        elif predicted_class == 1:
            status = MaintenanceStatus.WARNING
            action = "Schedule preventive maintenance"
        elif predicted_class == 2:
            status = MaintenanceStatus.CRITICAL
            action = "Schedule urgent maintenance"
        else:
            status = MaintenanceStatus.FAILURE
            action = "Immediate maintenance required"
        
        predicted_failure_date = None
        if remaining_days > 0:
            predicted_failure_date = (datetime.now() + timedelta(days=remaining_days)).isoformat()
        
        prediction = MaintenancePrediction(
            prediction_id=str(uuid.uuid4()),
            equipment_id=equipment_id,
            predicted_failure_date=predicted_failure_date,
            failure_probability=failure_prob,
            remaining_life=remaining_days if remaining_days > 0 else None,
            recommended_action=action,
            confidence=float(probs.max().item())
        )
        
        self.predictions[prediction.prediction_id] = prediction
        logger.info(f"Predicted maintenance for {equipment_id}: {status.value}")
        
        return prediction
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        status_counts = {}
        for pred in self.predictions.values():
            if pred.remaining_life:
                if pred.remaining_life < 7:
                    status = "critical"
                elif pred.remaining_life < 30:
                    status = "warning"
                else:
                    status = "healthy"
                status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_models": len(self.models),
            "total_predictions": len(self.predictions),
            "equipment_with_data": len(self.sensor_data),
            "status_counts": status_counts
        }


# Instancia global
_predictive_maintenance_system = None


def get_predictive_maintenance_system() -> PredictiveMaintenanceSystem:
    """Obtener instancia global."""
    global _predictive_maintenance_system
    if _predictive_maintenance_system is None:
        _predictive_maintenance_system = PredictiveMaintenanceSystem()
    return _predictive_maintenance_system

