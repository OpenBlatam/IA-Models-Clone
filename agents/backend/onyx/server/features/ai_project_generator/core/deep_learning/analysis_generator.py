"""
Analysis Generator - Generador de utilidades de análisis
========================================================

Genera utilidades para análisis de datos y modelos:
- Data analysis
- Model analysis
- Feature importance
- Statistical analysis
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AnalysisGenerator:
    """Generador de utilidades de análisis"""
    
    def __init__(self):
        """Inicializa el generador de análisis"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de análisis.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_dir = utils_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_data_analysis(analysis_dir, keywords, project_info)
        self._generate_model_analysis(analysis_dir, keywords, project_info)
        self._generate_analysis_init(analysis_dir, keywords)
    
    def _generate_analysis_init(
        self,
        analysis_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de análisis"""
        
        init_content = '''"""
Analysis Utilities Module
==========================

Utilidades para análisis de datos y modelos.
"""

from .data_analysis import (
    analyze_dataset,
    get_dataset_stats,
    visualize_data_distribution,
)
from .model_analysis import (
    analyze_model_predictions,
    get_feature_importance,
    analyze_model_errors,
)

__all__ = [
    "analyze_dataset",
    "get_dataset_stats",
    "visualize_data_distribution",
    "analyze_model_predictions",
    "get_feature_importance",
    "analyze_model_errors",
]
'''
        
        (analysis_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_data_analysis(
        self,
        analysis_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de análisis de datos"""
        
        analysis_content = '''"""
Data Analysis - Análisis de datos
===================================

Utilidades para analizar datasets y distribuciones.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn no disponible")

logger = logging.getLogger(__name__)


def analyze_dataset(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Analiza un dataset completo.
    
    Args:
        data: Datos a analizar
        labels: Etiquetas (opcional)
    
    Returns:
        Diccionario con análisis completo
    """
    analysis = {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "size_mb": data.nbytes / 1024**2,
    }
    
    # Estadísticas básicas
    if data.size > 0:
        analysis["stats"] = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
        }
        
        # Verificar valores especiales
        analysis["special_values"] = {
            "nan_count": int(np.isnan(data).sum()),
            "inf_count": int(np.isinf(data).sum()),
            "zero_count": int((data == 0).sum()),
        }
    
    # Análisis de labels si están disponibles
    if labels is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        analysis["labels"] = {
            "unique_count": len(unique_labels),
            "distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
            "imbalance_ratio": float(counts.max() / counts.min()) if len(counts) > 1 else 1.0,
        }
    
    return analysis


def get_dataset_stats(
    dataset,
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de un dataset PyTorch.
    
    Args:
        dataset: Dataset de PyTorch
    
    Returns:
        Diccionario con estadísticas
    """
    stats = {
        "size": len(dataset),
    }
    
    # Analizar primer elemento
    try:
        sample = dataset[0]
        if isinstance(sample, (tuple, list)):
            data, label = sample[0], sample[1]
        else:
            data, label = sample, None
        
        if hasattr(data, "shape"):
            stats["data_shape"] = list(data.shape)
            stats["data_dtype"] = str(data.dtype)
        
        if label is not None:
            stats["has_labels"] = True
            if hasattr(label, "shape"):
                stats["label_shape"] = list(label.shape)
    except Exception as e:
        logger.warning(f"Error analizando dataset: {e}")
    
    return stats


def visualize_data_distribution(
    data: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Visualiza distribución de datos.
    
    Args:
        data: Datos a visualizar
        save_path: Ruta donde guardar (opcional)
        show: Si mostrar el plot
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting no disponible")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histograma
    axes[0].hist(data.flatten(), bins=50, alpha=0.7)
    axes[0].set_title("Data Distribution")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    
    # Box plot
    axes[1].boxplot(data.flatten())
    axes[1].set_title("Data Box Plot")
    axes[1].set_ylabel("Value")
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot guardado en {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
'''
        
        (analysis_dir / "data_analysis.py").write_text(analysis_content, encoding="utf-8")
    
    def _generate_model_analysis(
        self,
        analysis_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de análisis de modelos"""
        
        model_analysis_content = '''"""
Model Analysis - Análisis de modelos
======================================

Utilidades para analizar predicciones y comportamiento de modelos.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging

try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


def analyze_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analiza predicciones del modelo.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        class_names: Nombres de clases (opcional)
    
    Returns:
        Diccionario con análisis
    """
    analysis = {
        "total_samples": len(y_true),
        "accuracy": float(np.mean(y_true == y_pred)),
    }
    
    if SKLEARN_AVAILABLE:
        try:
            # Classification report
            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
            )
            analysis["classification_report"] = report
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            analysis["confusion_matrix"] = cm.tolist()
        except Exception as e:
            logger.warning(f"Error generando reporte: {e}")
    
    # Análisis de errores
    errors = y_true != y_pred
    analysis["error_analysis"] = {
        "total_errors": int(errors.sum()),
        "error_rate": float(errors.mean()),
    }
    
    return analysis


def get_feature_importance(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    target: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Obtiene importancia de features (aproximada).
    
    Args:
        model: Modelo a analizar
        input_data: Datos de entrada
        target: Target (opcional, para gradientes)
    
    Returns:
        Diccionario con importancia de features
    """
    model.eval()
    input_data = input_data.clone().requires_grad_(True)
    
    try:
        output = model(input_data)
        
        if target is not None:
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
        else:
            output.sum().backward()
        
        # Importancia basada en gradientes
        gradients = input_data.grad.abs()
        importance = gradients.mean(dim=0).cpu().numpy()
        
        return {
            f"feature_{i}": float(importance[i])
            for i in range(len(importance))
        }
    except Exception as e:
        logger.warning(f"Error calculando importancia: {e}")
        return {}


def analyze_model_errors(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Analiza errores del modelo en un dataset.
    
    Args:
        model: Modelo a analizar
        dataloader: DataLoader con datos
        device: Dispositivo a usar
    
    Returns:
        Diccionario con análisis de errores
    """
    model.eval()
    all_predictions = []
    all_targets = []
    error_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch.to(device)
                targets = None
            
            outputs = model(inputs)
            
            if targets is not None:
                if outputs.dim() > 1:
                    preds = outputs.argmax(dim=1)
                else:
                    preds = (outputs > 0.5).long()
                
                all_predictions.append(preds.cpu())
                all_targets.append(targets.cpu())
                
                # Guardar muestras con error
                errors = preds != targets
                if errors.any():
                    error_indices = torch.where(errors)[0]
                    for idx in error_indices:
                        error_samples.append({
                            "batch_idx": batch_idx,
                            "sample_idx": int(idx),
                            "prediction": int(preds[idx]),
                            "target": int(targets[idx]),
                        })
    
    analysis = {
        "total_samples": len(all_predictions) * dataloader.batch_size if all_predictions else 0,
        "error_samples": error_samples[:10],  # Primeros 10 errores
    }
    
    if all_predictions and all_targets:
        all_preds = torch.cat(all_predictions)
        all_targs = torch.cat(all_targets)
        analysis["error_rate"] = float((all_preds != all_targs).mean().item())
    
    return analysis
'''
        
        (analysis_dir / "model_analysis.py").write_text(model_analysis_content, encoding="utf-8")

