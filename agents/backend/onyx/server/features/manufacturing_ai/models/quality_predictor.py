"""
Quality Predictor Model
=======================

Modelo de deep learning para predecir calidad de productos.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Importar arquitecturas avanzadas
try:
    from ..core.architecture.advanced_models import AdvancedQualityPredictor
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    AdvancedQualityPredictor = None

logger = logging.getLogger(__name__)


class QualityPredictor(nn.Module):
    """
    Modelo para predecir calidad de productos.
    
    Usa CNN para análisis de imágenes y MLP para características numéricas.
    """
    
    def __init__(
        self,
        image_input_size: int = 224,
        num_features: int = 10,
        num_classes: int = 3  # pass, warning, fail
    ):
        """
        Inicializar modelo.
        
        Args:
            image_input_size: Tamaño de imagen de entrada
            num_features: Número de características numéricas
            num_classes: Número de clases de calidad
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        # CNN para imágenes
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # MLP para características numéricas
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion y clasificación
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Tensor de imágenes [batch, 3, H, W]
            features: Tensor de características [batch, num_features]
            
        Returns:
            Logits [batch, num_classes]
        """
        # Encoder de imágenes
        image_features = self.image_encoder(images)
        
        # Encoder de características
        feature_emb = self.feature_encoder(features)
        
        # Fusion
        combined = torch.cat([image_features, feature_emb], dim=1)
        
        # Clasificación
        output = self.classifier(combined)
        
        return output


class QualityPredictorManager:
    """Gestor de modelos de predicción de calidad."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.models: Dict[str, QualityPredictor] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    
    def create_model(
        self,
        model_id: str,
        image_input_size: int = 224,
        num_features: int = 10,
        use_advanced: bool = False
    ) -> str:
        """
        Crear modelo.
        
        Args:
            model_id: ID del modelo
            image_input_size: Tamaño de imagen
            num_features: Número de características
            use_advanced: Usar arquitectura avanzada (con atención)
            
        Returns:
            ID del modelo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        if use_advanced and ADVANCED_AVAILABLE:
            model = AdvancedQualityPredictor(
                image_input_size=image_input_size,
                num_features=num_features
            )
            logger.info(f"Created advanced quality predictor model: {model_id}")
        else:
            model = QualityPredictor(
                image_input_size=image_input_size,
                num_features=num_features
            )
            logger.info(f"Created quality predictor model: {model_id}")
        
        model = model.to(self.device)
        self.models[model_id] = model
        
        return model_id
    
    def predict(
        self,
        model_id: str,
        images: np.ndarray,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predecir calidad.
        
        Args:
            model_id: ID del modelo
            images: Imágenes [N, H, W, 3]
            features: Características [N, num_features]
            
        Returns:
            Predicciones
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.models[model_id]
        model.eval()
        
        # Convertir a tensores
        images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            logits = model(images_tensor, features_tensor)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        return {
            "predictions": predictions.cpu().numpy().tolist(),
            "probabilities": probs.cpu().numpy().tolist(),
            "confidence": probs.max(dim=1)[0].cpu().numpy().tolist()
        }


# Instancia global
_quality_predictor_manager = None


def get_quality_predictor_manager() -> QualityPredictorManager:
    """Obtener instancia global."""
    global _quality_predictor_manager
    if _quality_predictor_manager is None:
        _quality_predictor_manager = QualityPredictorManager()
    return _quality_predictor_manager

