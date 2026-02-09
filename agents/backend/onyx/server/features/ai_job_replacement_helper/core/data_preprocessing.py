"""
Data Preprocessing Service - Preprocesamiento de datos
=======================================================

Sistema avanzado para preprocesar datos para entrenamiento de modelos.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available")


@dataclass
class PreprocessingConfig:
    """Configuración de preprocesamiento"""
    normalize: bool = True
    normalization_method: str = "standard"  # standard, minmax, robust
    handle_missing: bool = True
    missing_strategy: str = "mean"  # mean, median, mode, drop
    encode_categorical: bool = True
    feature_selection: bool = False
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42


@dataclass
class PreprocessedData:
    """Datos preprocesados"""
    train_data: Any
    val_data: Any
    test_data: Any
    scaler: Optional[Any] = None
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataPreprocessingService:
    """Servicio de preprocesamiento de datos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.scalers: Dict[str, Any] = {}
        logger.info("DataPreprocessingService initialized")
    
    def normalize_features(
        self,
        data: np.ndarray,
        method: str = "standard",
        fit: bool = True,
        scaler_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Any]:
        """Normalizar features"""
        if not SKLEARN_AVAILABLE:
            # Fallback normalization
            if method == "standard":
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                normalized = (data - mean) / (std + 1e-8)
                scaler = {"mean": mean, "std": std, "method": "standard"}
            elif method == "minmax":
                min_val = np.min(data, axis=0)
                max_val = np.max(data, axis=0)
                normalized = (data - min_val) / (max_val - min_val + 1e-8)
                scaler = {"min": min_val, "max": max_val, "method": "minmax"}
            else:
                normalized = data
                scaler = None
        else:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                return data, None
            
            if fit:
                normalized = scaler.fit_transform(data)
            else:
                normalized = scaler.transform(data)
        
        if scaler_id:
            self.scalers[scaler_id] = scaler
        
        return normalized, scaler
    
    def handle_missing_values(
        self,
        data: np.ndarray,
        strategy: str = "mean"
    ) -> np.ndarray:
        """Manejar valores faltantes"""
        if strategy == "mean":
            fill_value = np.nanmean(data, axis=0)
        elif strategy == "median":
            fill_value = np.nanmedian(data, axis=0)
        elif strategy == "mode":
            fill_value = []
            for col in range(data.shape[1]):
                col_data = data[:, col]
                col_data = col_data[~np.isnan(col_data)]
                if len(col_data) > 0:
                    fill_value.append(np.bincount(col_data.astype(int)).argmax())
                else:
                    fill_value.append(0)
            fill_value = np.array(fill_value)
        elif strategy == "drop":
            # Drop rows with any NaN
            return data[~np.isnan(data).any(axis=1)]
        else:
            fill_value = 0
        
        # Fill NaN values
        filled_data = data.copy()
        for col in range(data.shape[1]):
            mask = np.isnan(data[:, col])
            filled_data[mask, col] = fill_value[col] if isinstance(fill_value, np.ndarray) else fill_value
        
        return filled_data
    
    def split_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Dividir datos en train/val/test"""
        if SKLEARN_AVAILABLE and y is not None:
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Split train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size / (1 - test_size), random_state=random_state
            )
            
            return {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            }
        else:
            # Manual split
            n = len(X)
            test_start = int(n * (1 - test_size))
            val_start = int(test_start * (1 - val_size / (1 - test_size)))
            
            X_train = X[:val_start]
            X_val = X[val_start:test_start]
            X_test = X[test_start:]
            
            result = {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
            }
            
            if y is not None:
                y_train = y[:val_start]
                y_val = y[val_start:test_start]
                y_test = y[test_start:]
                result.update({
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test,
                })
            
            return result
    
    def create_dataloader(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Optional[Any]:
        """Crear DataLoader de PyTorch"""
        if not TORCH_AVAILABLE:
            return None
        
        X_tensor = torch.FloatTensor(X)
        
        if y is not None:
            if y.dtype == np.float64 or y.dtype == np.float32:
                y_tensor = torch.FloatTensor(y)
            else:
                y_tensor = torch.LongTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to >0 for multi-process loading
        )
        
        return dataloader
    
    def preprocess_dataset(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        config: Optional[PreprocessingConfig] = None
    ) -> PreprocessedData:
        """Preprocesar dataset completo"""
        if config is None:
            config = PreprocessingConfig()
        
        # Handle missing values
        if config.handle_missing:
            X = self.handle_missing_values(X, config.missing_strategy)
            if y is not None:
                y = y[~np.isnan(X).any(axis=1)]
                X = X[~np.isnan(X).any(axis=1)]
        
        # Normalize
        scaler = None
        if config.normalize:
            X, scaler = self.normalize_features(
                X,
                method=config.normalization_method,
                fit=True
            )
        
        # Split data
        splits = self.split_data(
            X, y,
            test_size=config.test_size,
            val_size=config.validation_size,
            random_state=config.random_state
        )
        
        # Create DataLoaders
        train_loader = self.create_dataloader(
            splits["X_train"],
            splits.get("y_train"),
            batch_size=32,
            shuffle=True
        )
        
        val_loader = self.create_dataloader(
            splits["X_val"],
            splits.get("y_val"),
            batch_size=32,
            shuffle=False
        )
        
        test_loader = self.create_dataloader(
            splits["X_test"],
            splits.get("y_test"),
            batch_size=32,
            shuffle=False
        )
        
        return PreprocessedData(
            train_data=train_loader,
            val_data=val_loader,
            test_data=test_loader,
            scaler=scaler,
            metadata={
                "train_size": len(splits["X_train"]),
                "val_size": len(splits["X_val"]),
                "test_size": len(splits["X_test"]),
                "normalization_method": config.normalization_method,
            }
        )




