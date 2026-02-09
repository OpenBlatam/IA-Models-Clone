"""
Active Learning para selección inteligente de datos
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ActiveLearner:
    """Aprendizaje activo"""
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
    
    def uncertainty_sampling(
        self,
        unlabeled_data: List[Dict[str, Any]],
        num_samples: int = 10,
        method: str = "entropy"  # "entropy", "margin", "least_confidence"
    ) -> List[int]:
        """
        Selección basada en incertidumbre
        
        Args:
            unlabeled_data: Datos sin etiquetar
            num_samples: Número de muestras a seleccionar
            method: Método de selección
            
        Returns:
            Índices de muestras seleccionadas
        """
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for data in unlabeled_data:
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in data.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=-1)
                
                if method == "entropy":
                    # Entropía
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    uncertainty = entropy.item()
                elif method == "margin":
                    # Margen entre top-2
                    sorted_probs, _ = torch.sort(probs, descending=True)
                    margin = sorted_probs[0, 0] - sorted_probs[0, 1]
                    uncertainty = (1 - margin).item()
                elif method == "least_confidence":
                    # Menor confianza
                    max_prob = torch.max(probs, dim=-1)[0]
                    uncertainty = (1 - max_prob).item()
                else:
                    uncertainty = 0.0
                
                uncertainties.append(uncertainty)
        
        # Seleccionar top-k más inciertos
        indices = np.argsort(uncertainties)[::-1][:num_samples]
        return indices.tolist()
    
    def diversity_sampling(
        self,
        unlabeled_data: List[Dict[str, Any]],
        labeled_data: List[Dict[str, Any]],
        num_samples: int = 10
    ) -> List[int]:
        """
        Selección basada en diversidad
        
        Args:
            unlabeled_data: Datos sin etiquetar
            labeled_data: Datos etiquetados
            num_samples: Número de muestras
            
        Returns:
            Índices seleccionados
        """
        # Calcular embeddings
        unlabeled_embeddings = self._get_embeddings(unlabeled_data)
        labeled_embeddings = self._get_embeddings(labeled_data)
        
        # Seleccionar muestras más diferentes a las etiquetadas
        selected_indices = []
        
        for _ in range(num_samples):
            if not selected_indices:
                # Primera muestra: más diferente a todas las etiquetadas
                distances = []
                for unlabeled_emb in unlabeled_embeddings:
                    min_dist = min([
                        np.linalg.norm(unlabeled_emb - labeled_emb)
                        for labeled_emb in labeled_embeddings
                    ])
                    distances.append(min_dist)
                selected_indices.append(np.argmax(distances))
            else:
                # Siguientes: diferentes a etiquetadas Y seleccionadas
                distances = []
                for i, unlabeled_emb in enumerate(unlabeled_embeddings):
                    if i in selected_indices:
                        continue
                    min_dist = min([
                        np.linalg.norm(unlabeled_emb - labeled_emb)
                        for labeled_emb in labeled_embeddings
                    ] + [
                        np.linalg.norm(unlabeled_emb - unlabeled_embeddings[j])
                        for j in selected_indices
                    ])
                    distances.append((i, min_dist))
                
                if distances:
                    selected_indices.append(max(distances, key=lambda x: x[1])[0])
        
        return selected_indices[:num_samples]
    
    def _get_embeddings(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Obtiene embeddings de datos"""
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for item in data:
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in item.items()}
                outputs = self.model(**inputs)
                
                # Obtener embeddings (última capa oculta)
                if hasattr(outputs, 'hidden_states'):
                    emb = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
                else:
                    # Fallback: usar logits
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    emb = logits.cpu().numpy()
                
                embeddings.append(emb.flatten())
        
        return np.array(embeddings)




