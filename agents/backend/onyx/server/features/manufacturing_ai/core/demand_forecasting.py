"""
Demand Forecasting System
=========================

Sistema de predicción de demanda usando deep learning.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


@dataclass
class DemandForecast:
    """Pronóstico de demanda."""
    forecast_id: str
    product_id: str
    forecast_date: str
    predicted_demand: float
    confidence_interval: tuple  # (lower, upper)
    model_confidence: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DemandForecastingModel(nn.Module):
    """
    Modelo LSTM para predicción de demanda.
    
    Usa LSTM para modelar patrones temporales en demanda.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Inicializar modelo.
        
        Args:
            input_size: Tamaño de entrada (features)
            hidden_size: Tamaño de capa oculta
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Capas de salida
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Predicción de demanda
        )
        
        # Capa para intervalo de confianza
        self.confidence_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # lower, upper
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch, seq_len, input_size]
            
        Returns:
            Tupla (predicción, intervalo de confianza)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Predicción
        prediction = self.fc(last_output)
        
        # Intervalo de confianza
        confidence = self.confidence_fc(last_output)
        
        return prediction, confidence


class DemandForecastingSystem:
    """
    Sistema de predicción de demanda.
    
    Usa deep learning para predecir demanda futura.
    """
    
    def __init__(self):
        """Inicializar sistema."""
        self.models: Dict[str, DemandForecastingModel] = {}
        self.forecasts: Dict[str, DemandForecast] = {}
        self.historical_data: Dict[str, List[float]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    
    def create_model(
        self,
        product_id: str,
        input_size: int = 1,
        hidden_size: int = 64
    ) -> str:
        """
        Crear modelo para producto.
        
        Args:
            product_id: ID del producto
            input_size: Tamaño de entrada
            hidden_size: Tamaño oculto
            
        Returns:
            ID del modelo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        model = DemandForecastingModel(
            input_size=input_size,
            hidden_size=hidden_size
        )
        
        model = model.to(self.device)
        self.models[product_id] = model
        
        logger.info(f"Created demand forecasting model for {product_id}")
        return product_id
    
    def add_historical_data(
        self,
        product_id: str,
        data: List[float]
    ):
        """
        Agregar datos históricos.
        
        Args:
            product_id: ID del producto
            data: Datos históricos de demanda
        """
        if product_id not in self.historical_data:
            self.historical_data[product_id] = []
        
        self.historical_data[product_id].extend(data)
        logger.info(f"Added {len(data)} data points for {product_id}")
    
    def forecast(
        self,
        product_id: str,
        forecast_days: int = 30,
        sequence_length: int = 30
    ) -> DemandForecast:
        """
        Predecir demanda.
        
        Args:
            product_id: ID del producto
            forecast_days: Días a predecir
            sequence_length: Longitud de secuencia histórica
            
        Returns:
            Pronóstico de demanda
        """
        if product_id not in self.models:
            raise ValueError(f"Model not found for product: {product_id}")
        
        if product_id not in self.historical_data:
            raise ValueError(f"No historical data for product: {product_id}")
        
        model = self.models[product_id]
        model.eval()
        
        # Obtener datos históricos
        historical = self.historical_data[product_id][-sequence_length:]
        
        # Preparar input
        input_data = np.array(historical).reshape(1, sequence_length, 1)
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        
        with torch.no_grad():
            prediction, confidence = model(input_tensor)
            
            predicted_demand = float(prediction.item())
            conf_lower = float(prediction.item() - confidence[0, 0].item())
            conf_upper = float(prediction.item() + confidence[0, 1].item())
        
        forecast = DemandForecast(
            forecast_id=str(uuid.uuid4()),
            product_id=product_id,
            forecast_date=(datetime.now() + timedelta(days=forecast_days)).isoformat(),
            predicted_demand=predicted_demand,
            confidence_interval=(max(0, conf_lower), conf_upper),
            model_confidence=0.85
        )
        
        self.forecasts[forecast.forecast_id] = forecast
        logger.info(f"Forecasted demand for {product_id}: {predicted_demand:.2f}")
        
        return forecast
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_models": len(self.models),
            "total_forecasts": len(self.forecasts),
            "products_with_data": len(self.historical_data)
        }


# Instancia global
_demand_forecasting_system = None


def get_demand_forecasting_system() -> DemandForecastingSystem:
    """Obtener instancia global."""
    global _demand_forecasting_system
    if _demand_forecasting_system is None:
        _demand_forecasting_system = DemandForecastingSystem()
    return _demand_forecasting_system

