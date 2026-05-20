"""
Batch Inference - Inferencia en Batch Optimizada
=================================================

Sistema de inferencia en batch altamente optimizado.
"""

import logging
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class OptimizedBatchInference:
    """Inferencia en batch optimizada"""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        batch_size: int = 32,
        use_amp: bool = True,
        num_workers: int = 4
    ):
        """
        Inicializar inferencia en batch
        
        Args:
            model: Modelo a usar
            device: Dispositivo
            batch_size: Tamaño de batch
            use_amp: Usar mixed precision
            num_workers: Número de workers
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.model = model.to(self.device)
        self.model.eval()
        
        # Compilar modelo si es posible
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Modelo compilado para batch inference")
            except Exception as e:
                logger.warning(f"No se pudo compilar: {e}")
        
        logger.info(f"Optimized Batch Inference inicializado (batch_size={batch_size})")
    
    @torch.inference_mode()
    def predict_batch(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        collate_fn: Optional[callable] = None
    ) -> List[torch.Tensor]:
        """
        Predecir batch de inputs
        
        Args:
            inputs: Lista de inputs
            collate_fn: Función de collate (opcional)
            
        Returns:
            Lista de predicciones
        """
        results = []
        
        # Procesar en batches
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            # Collate batch
            if collate_fn:
                batched_input = collate_fn(batch)
            else:
                batched_input = self._default_collate(batch)
            
            # Mover a dispositivo
            batched_input = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batched_input.items()
            }
            
            # Inferencia con mixed precision
            with torch.cuda.amp.autocast() if self.use_amp and self.device == "cuda" else torch.no_grad():
                outputs = self.model(**batched_input)
            
            # Separar resultados
            if isinstance(outputs, torch.Tensor):
                batch_results = [outputs[j] for j in range(outputs.size(0))]
            elif hasattr(outputs, "logits"):
                batch_results = [outputs.logits[j] for j in range(outputs.logits.size(0))]
            else:
                batch_results = [outputs] * len(batch)
            
            results.extend(batch_results)
        
        return results
    
    def _default_collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate por defecto"""
        collated = {}
        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            # Padding si es necesario
            max_len = max(t.size(0) if t.dim() > 0 else 1 for t in tensors)
            padded = []
            for t in tensors:
                if t.dim() == 1:
                    pad_size = max_len - t.size(0)
                    if pad_size > 0:
                        t = torch.nn.functional.pad(t, (0, pad_size))
                padded.append(t)
            collated[key] = torch.stack(padded)
        return collated
    
    def predict_dataloader(
        self,
        dataloader: DataLoader
    ) -> List[torch.Tensor]:
        """
        Predecir desde DataLoader
        
        Args:
            dataloader: DataLoader
            
        Returns:
            Lista de predicciones
        """
        all_predictions = []
        
        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.cuda.amp.autocast() if self.use_amp and self.device == "cuda" else torch.no_grad():
                outputs = self.model(**batch)
            
            if hasattr(outputs, "logits"):
                predictions = outputs.logits
            else:
                predictions = outputs
            
            all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions, dim=0)




