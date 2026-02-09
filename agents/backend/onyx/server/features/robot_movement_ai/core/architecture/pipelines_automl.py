"""
AutoML Module
==============

Sistema de AutoML para automatizar pipeline de machine learning.
Incluye auto-feature engineering, auto-model selection, y auto-tuning.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
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
class AutoMLConfig:
    """Configuración para AutoML."""
    max_models: int = 10
    max_training_time: float = 3600.0  # segundos
    metric: str = "accuracy"
    cv_folds: int = 5
    ensemble: bool = True
    feature_engineering: bool = True


class AutoFeatureEngineering:
    """
    Ingeniería automática de features.
    
    Genera features automáticamente a partir de datos raw.
    """
    
    def __init__(self):
        """Inicializar feature engineering automático."""
        self.transformations: List[Callable] = []
        logger.info("AutoFeatureEngineering initialized")
    
    def add_polynomial_features(
        self,
        data: np.ndarray,
        degree: int = 2
    ) -> np.ndarray:
        """
        Agregar features polinomiales.
        
        Args:
            data: Datos originales
            degree: Grado de polinomio
            
        Returns:
            Datos con features polinomiales
        """
        try:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            return poly.fit_transform(data)
        except ImportError:
            logger.warning("sklearn not available for polynomial features")
            return data
    
    def add_interaction_features(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Agregar features de interacción.
        
        Args:
            data: Datos originales
            
        Returns:
            Datos con features de interacción
        """
        n_features = data.shape[1]
        interaction_features = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_features.append(data[:, i] * data[:, j])
        
        if interaction_features:
            interaction_array = np.column_stack(interaction_features)
            return np.hstack([data, interaction_array])
        
        return data
    
    def auto_transform(
        self,
        data: np.ndarray,
        methods: List[str] = ["polynomial", "interaction"]
    ) -> np.ndarray:
        """
        Aplicar transformaciones automáticas.
        
        Args:
            data: Datos originales
            methods: Métodos a aplicar
            
        Returns:
            Datos transformados
        """
        transformed = data.copy()
        
        if "polynomial" in methods:
            transformed = self.add_polynomial_features(transformed, degree=2)
        
        if "interaction" in methods:
            transformed = self.add_interaction_features(transformed)
        
        logger.info(f"Feature engineering: {data.shape[1]} -> {transformed.shape[1]} features")
        return transformed


class AutoModelSelection:
    """
    Selección automática de modelos.
    
    Prueba múltiples arquitecturas y selecciona la mejor.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: AutoMLConfig
    ):
        """
        Inicializar selector.
        
        Args:
            input_size: Tamaño de entrada
            output_size: Tamaño de salida
            config: Configuración AutoML
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.models: List[Dict[str, Any]] = []
        logger.info("AutoModelSelection initialized")
    
    def create_model_candidates(self) -> List[nn.Module]:
        """
        Crear candidatos de modelos.
        
        Returns:
            Lista de modelos candidatos
        """
        candidates = []
        
        # Modelo simple (MLP pequeño)
        candidates.append(nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_size)
        ))
        
        # Modelo medio (MLP mediano)
        candidates.append(nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_size)
        ))
        
        # Modelo grande (MLP grande)
        candidates.append(nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_size)
        ))
        
        logger.info(f"Created {len(candidates)} model candidates")
        return candidates
    
    def evaluate_model(
        self,
        model: nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 10
    ) -> Dict[str, float]:
        """
        Evaluar modelo.
        
        Args:
            model: Modelo a evaluar
            train_data: Datos de entrenamiento
            val_data: Datos de validación
            epochs: Número de épocas
            
        Returns:
            Dict con métricas
        """
        from .pipelines_training import TrainingPipeline, TrainingConfig
        
        train_inputs, train_targets = train_data
        val_inputs, val_targets = val_data
        
        # Crear datasets
        from .pipelines_datasets import TrajectoryDataset
        train_dataset = TrajectoryDataset(train_inputs.numpy(), train_targets.numpy())
        val_dataset = TrajectoryDataset(val_inputs.numpy(), val_targets.numpy())
        
        # Configurar entrenamiento rápido
        config = TrainingConfig(
            batch_size=32,
            num_epochs=epochs,
            learning_rate=1e-3
        )
        
        # Entrenar
        pipeline = TrainingPipeline(model, config, train_dataset, val_dataset)
        history = pipeline.train()
        
        # Obtener mejor score
        best_val_loss = min(history.get("val_loss", [float('inf')]))
        
        return {
            "val_loss": best_val_loss,
            "final_val_loss": history.get("val_loss", [0])[-1] if history.get("val_loss") else 0
        }
    
    def select_best_model(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Seleccionar mejor modelo.
        
        Args:
            train_data: Datos de entrenamiento
            val_data: Datos de validación
            
        Returns:
            Mejor modelo y sus métricas
        """
        candidates = self.create_model_candidates()
        best_model = None
        best_score = float('inf')
        best_metrics = {}
        
        for i, model in enumerate(candidates):
            logger.info(f"Evaluating model {i+1}/{len(candidates)}")
            
            try:
                metrics = self.evaluate_model(model, train_data, val_data, epochs=5)
                score = metrics["val_loss"]
                
                self.models.append({
                    "model": model,
                    "metrics": metrics,
                    "score": score
                })
                
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_metrics = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating model {i+1}: {e}")
                continue
        
        logger.info(f"Best model selected with score: {best_score:.4f}")
        return best_model, best_metrics


class AutoMLPipeline:
    """
    Pipeline completo de AutoML.
    
    Automatiza todo el proceso de ML.
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        """
        Inicializar pipeline AutoML.
        
        Args:
            config: Configuración AutoML
        """
        self.config = config or AutoMLConfig()
        self.feature_engineer = AutoFeatureEngineering()
        logger.info("AutoMLPipeline initialized")
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Ejecutar pipeline AutoML completo.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Targets de entrenamiento
            X_val: Features de validación
            y_val: Targets de validación
            
        Returns:
            Dict con mejor modelo y resultados
        """
        # Feature engineering
        if self.config.feature_engineering:
            logger.info("Applying automatic feature engineering...")
            X_train = self.feature_engineer.auto_transform(X_train)
            X_val = self.feature_engineer.auto_transform(X_val)
        
        # Convertir a tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train) if y_train.ndim > 1 else torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val) if y_val.ndim > 1 else torch.LongTensor(y_val)
        
        # Model selection
        selector = AutoModelSelection(
            input_size=X_train.shape[1],
            output_size=y_train.shape[1] if y_train.ndim > 1 else len(np.unique(y_train)),
            config=self.config
        )
        
        best_model, metrics = selector.select_best_model(
            (X_train_tensor, y_train_tensor),
            (X_val_tensor, y_val_tensor)
        )
        
        return {
            "model": best_model,
            "metrics": metrics,
            "feature_count": X_train.shape[1],
            "models_tested": len(selector.models)
        }

