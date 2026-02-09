"""
Evaluation Utilities
====================

Utilidades para evaluación de modelos.
"""

from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)


def generate_evaluation_code() -> str:
    """Genera código para evaluación de modelos."""
    return '''"""
Evaluation Script
=================

Script para evaluar modelos entrenados.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import logging
from pathlib import Path
import json
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluador de modelos."""
    
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            model: Modelo a evaluar
            device: Dispositivo a usar
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evalúa el modelo en un DataLoader.
        
        Args:
            dataloader: DataLoader con datos de evaluación
            criterion: Función de pérdida (opcional)
            return_predictions: Si retornar predicciones
        
        Returns:
            Diccionario con métricas
        """
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Mover batch a dispositivo
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Calcular pérdida si hay criterion
                if criterion is not None:
                    loss = criterion(outputs.logits, batch['labels'])
                    total_loss += loss.item()
                
                # Obtener predicciones
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calcular métricas
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'num_samples': len(all_labels)
        }
        
        if criterion is not None:
            results['loss'] = total_loss / len(dataloader)
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['labels'] = all_labels
        
        return results
    
    def evaluate_classification(
        self,
        dataloader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evalúa modelo de clasificación con métricas detalladas.
        
        Args:
            dataloader: DataLoader con datos
            class_names: Nombres de clases (opcional)
        
        Returns:
            Diccionario con métricas detalladas
        """
        results = self.evaluate(dataloader, return_predictions=True)
        
        # Matriz de confusión
        cm = confusion_matrix(results['labels'], results['predictions'])
        results['confusion_matrix'] = cm.tolist()
        
        # Reporte de clasificación
        if class_names:
            report = classification_report(
                results['labels'],
                results['predictions'],
                target_names=class_names,
                output_dict=True
            )
            results['classification_report'] = report
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Guarda resultados en archivo JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir numpy arrays a listas
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                clean_results[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                clean_results[k] = float(v)
            else:
                clean_results[k] = v
        
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    criterion: Optional[nn.Module] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Función helper para evaluar modelo.
    
    Args:
        model: Modelo a evaluar
        test_loader: DataLoader de test
        device: Dispositivo
        criterion: Función de pérdida
        save_path: Path para guardar resultados
    
    Returns:
        Diccionario con métricas
    """
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_loader, criterion)
    
    if save_path:
        evaluator.save_results(results, save_path)
    
    return results


if __name__ == "__main__":
    # Ejemplo de uso
    from model import TransformerModel
    
    # Cargar modelo
    model = TransformerModel()
    checkpoint = torch.load("checkpoints/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Crear dataloader (reemplazar con tu dataloader)
    from data import create_dataloader
    test_loader = create_dataloader(
        texts=["test text 1", "test text 2"],
        tokenizer=None,  # Reemplazar con tu tokenizer
        batch_size=32
    )
    
    # Evaluar
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        save_path="evaluation_results.json"
    )
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
'''

