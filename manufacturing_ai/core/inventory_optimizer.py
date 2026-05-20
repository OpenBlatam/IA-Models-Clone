"""
Intelligent Inventory System
=============================

Sistema de inventario inteligente con predicción y optimización.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
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


class InventoryStatus(Enum):
    """Estado de inventario."""
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    OVERSTOCK = "overstock"


@dataclass
class InventoryItem:
    """Item de inventario."""
    item_id: str
    product_id: str
    current_quantity: float
    min_quantity: float
    max_quantity: float
    reorder_point: float
    lead_time: float  # días
    status: InventoryStatus = InventoryStatus.IN_STOCK
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class InventoryPrediction:
    """Predicción de inventario."""
    prediction_id: str
    item_id: str
    predicted_quantity: float
    predicted_date: str
    reorder_recommendation: float
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class InventoryPredictorModel(nn.Module):
    """
    Modelo para predicción de inventario.
    
    Usa LSTM para predecir niveles de inventario futuros.
    """
    
    def __init__(
        self,
        input_size: int = 3,  # quantity, demand, time_since_reorder
        hidden_size: int = 64,
        num_layers: int = 2
    ):
        """
        Inicializar modelo.
        
        Args:
            input_size: Tamaño de entrada
            hidden_size: Tamaño oculto
            num_layers: Número de capas
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor [batch, seq_len, input_size]
            
        Returns:
            Predicción [batch, 1]
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class IntelligentInventorySystem:
    """
    Sistema de inventario inteligente.
    
    Predice y optimiza niveles de inventario.
    """
    
    def __init__(self):
        """Inicializar sistema."""
        self.items: Dict[str, InventoryItem] = {}
        self.predictions: Dict[str, InventoryPrediction] = {}
        self.models: Dict[str, InventoryPredictorModel] = {}
        self.historical_data: Dict[str, List[float]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    
    def register_item(
        self,
        item_id: str,
        product_id: str,
        current_quantity: float,
        min_quantity: float,
        max_quantity: float,
        reorder_point: float,
        lead_time: float
    ):
        """
        Registrar item de inventario.
        
        Args:
            item_id: ID del item
            product_id: ID del producto
            current_quantity: Cantidad actual
            min_quantity: Cantidad mínima
            max_quantity: Cantidad máxima
            reorder_point: Punto de reorden
            lead_time: Tiempo de entrega (días)
        """
        status = InventoryStatus.IN_STOCK
        if current_quantity <= min_quantity:
            status = InventoryStatus.LOW_STOCK
        elif current_quantity == 0:
            status = InventoryStatus.OUT_OF_STOCK
        elif current_quantity >= max_quantity:
            status = InventoryStatus.OVERSTOCK
        
        item = InventoryItem(
            item_id=item_id,
            product_id=product_id,
            current_quantity=current_quantity,
            min_quantity=min_quantity,
            max_quantity=max_quantity,
            reorder_point=reorder_point,
            lead_time=lead_time,
            status=status
        )
        
        self.items[item_id] = item
        logger.info(f"Registered inventory item: {item_id}")
    
    def create_model(self, item_id: str) -> str:
        """Crear modelo para item."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        model = InventoryPredictorModel()
        model = model.to(self.device)
        self.models[item_id] = model
        
        return item_id
    
    def predict_inventory(
        self,
        item_id: str,
        forecast_days: int = 30,
        demand_forecast: Optional[float] = None
    ) -> InventoryPrediction:
        """
        Predecir inventario.
        
        Args:
            item_id: ID del item
            forecast_days: Días a predecir
            demand_forecast: Pronóstico de demanda (opcional)
            
        Returns:
            Predicción de inventario
        """
        if item_id not in self.items:
            raise ValueError(f"Item not found: {item_id}")
        
        item = self.items[item_id]
        
        # Calcular predicción
        if demand_forecast is None:
            # Usar modelo si está disponible
            if item_id in self.models and item_id in self.historical_data:
                model = self.models[item_id]
                historical = self.historical_data[item_id][-30:]
                
                # Preparar input
                input_data = np.array([
                    [q, 0.0, 0.0] for q in historical
                ]).reshape(1, len(historical), 3)
                input_tensor = torch.FloatTensor(input_data).to(self.device)
                
                model.eval()
                with torch.no_grad():
                    predicted = float(model(input_tensor).item())
            else:
                # Predicción simple
                predicted = item.current_quantity - (item.current_quantity * 0.1 * forecast_days / 30)
        else:
            predicted = item.current_quantity - (demand_forecast * forecast_days)
        
        predicted = max(0, predicted)
        
        # Recomendación de reorden
        reorder_qty = 0.0
        if predicted < item.reorder_point:
            reorder_qty = item.max_quantity - predicted
        
        prediction = InventoryPrediction(
            prediction_id=str(uuid.uuid4()),
            item_id=item_id,
            predicted_quantity=predicted,
            predicted_date=(datetime.now() + timedelta(days=forecast_days)).isoformat(),
            reorder_recommendation=reorder_qty,
            confidence=0.85
        )
        
        self.predictions[prediction.prediction_id] = prediction
        logger.info(f"Predicted inventory for {item_id}: {predicted:.2f}")
        
        return prediction
    
    def optimize_inventory_levels(self) -> Dict[str, Any]:
        """
        Optimizar niveles de inventario.
        
        Returns:
            Recomendaciones de optimización
        """
        recommendations = []
        
        for item_id, item in self.items.items():
            if item.status == InventoryStatus.LOW_STOCK:
                recommendations.append({
                    "item_id": item_id,
                    "action": "reorder",
                    "quantity": item.max_quantity - item.current_quantity,
                    "priority": "high"
                })
            elif item.status == InventoryStatus.OVERSTOCK:
                recommendations.append({
                    "item_id": item_id,
                    "action": "reduce",
                    "quantity": item.current_quantity - item.max_quantity,
                    "priority": "medium"
                })
        
        return {
            "recommendations": recommendations,
            "total_items": len(self.items),
            "low_stock_count": sum(1 for i in self.items.values() if i.status == InventoryStatus.LOW_STOCK),
            "overstock_count": sum(1 for i in self.items.values() if i.status == InventoryStatus.OVERSTOCK)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        status_counts = {}
        for item in self.items.values():
            status_counts[item.status.value] = status_counts.get(item.status.value, 0) + 1
        
        return {
            "total_items": len(self.items),
            "status_counts": status_counts,
            "total_predictions": len(self.predictions),
            "total_models": len(self.models)
        }


# Instancia global
_intelligent_inventory_system = None


def get_intelligent_inventory_system() -> IntelligentInventorySystem:
    """Obtener instancia global."""
    global _intelligent_inventory_system
    if _intelligent_inventory_system is None:
        _intelligent_inventory_system = IntelligentInventorySystem()
    return _intelligent_inventory_system

