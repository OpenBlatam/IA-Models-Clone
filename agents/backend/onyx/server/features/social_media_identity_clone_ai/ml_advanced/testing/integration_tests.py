"""
Tests de integración para el sistema completo
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging
import pytest

logger = logging.getLogger(__name__)


class IntegrationTester:
    """Tester de integración"""
    
    def __init__(self):
        pass
    
    def test_training_pipeline(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 2
    ) -> Dict[str, Any]:
        """
        Test completo de pipeline de entrenamiento
        
        Args:
            model: Modelo
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número de épocas
            
        Returns:
            Resultados del test
        """
        try:
            from ..training.trainer import Trainer
            from ..training.advanced_checkpointing import AdvancedCheckpointer
            
            # Setup
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            checkpointer = AdvancedCheckpointer()
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Entrenar
            def loss_fn(outputs, batch):
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                labels = batch.get("labels", batch.get("input_ids"))
                return criterion(logits, labels)
            
            result = trainer.train(
                num_epochs=num_epochs,
                optimizer=optimizer,
                loss_fn=loss_fn,
                checkpoint_dir="./test_checkpoints"
            )
            
            return {
                "success": True,
                "final_loss": result.get("best_val_loss", 0.0),
                "epochs_completed": result.get("epochs_trained", 0)
            }
            
        except Exception as e:
            logger.error(f"Error en test de pipeline: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_inference_pipeline(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Test de pipeline de inferencia"""
        try:
            model.eval()
            
            with torch.no_grad():
                outputs = model(**sample_input)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions = torch.softmax(logits, dim=-1)
            
            return {
                "success": True,
                "output_shape": list(predictions.shape),
                "has_valid_probs": torch.all((predictions >= 0) & (predictions <= 1))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_optimization_pipeline(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """Test de pipeline de optimización"""
        try:
            from ..optimization.quantization import ModelQuantizer
            from ..optimization.pruning import ModelPruner
            
            # Quantize
            quantizer = ModelQuantizer()
            quantized = quantizer.quantize_dynamic(model)
            
            # Prune
            pruner = ModelPruner()
            pruned = pruner.prune_structured(quantized, pruning_ratio=0.1)
            
            # Verificar que funciona
            sample_input = {"input_ids": torch.randint(0, 1000, (1, 10))}
            with torch.no_grad():
                _ = pruned(**sample_input)
            
            return {
                "success": True,
                "quantization_applied": True,
                "pruning_applied": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }




