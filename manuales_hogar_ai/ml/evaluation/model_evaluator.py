"""
Evaluador de Modelos
====================

Evaluación completa de modelos de generación.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from .metrics import calculate_bleu, calculate_rouge, calculate_bertscore

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluador de modelos de generación."""
    
    def __init__(self):
        """Inicializar evaluador."""
        self._logger = logger
    
    def evaluate(
        self,
        model,
        test_data: List[Dict[str, str]],
        metrics: List[str] = ["bleu", "rouge", "bertscore"],
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluar modelo en dataset de prueba.
        
        Args:
            model: Modelo a evaluar
            test_data: Lista de {input, reference}
            metrics: Métricas a calcular
            batch_size: Tamaño de batch
        
        Returns:
            Diccionario con resultados
        """
        try:
            references = [item["reference"] for item in test_data]
            candidates = []
            
            # Generar predicciones
            self._logger.info(f"Generando predicciones para {len(test_data)} ejemplos...")
            for item in tqdm(test_data, desc="Evaluando"):
                try:
                    generated = model.generate(item["input"])
                    candidates.append(generated)
                except Exception as e:
                    self._logger.warning(f"Error generando para ejemplo: {str(e)}")
                    candidates.append("")
            
            # Calcular métricas
            results = {}
            
            if "bleu" in metrics:
                bleu_scores = []
                for ref, cand in zip(references, candidates):
                    bleu = calculate_bleu(ref, cand)
                    bleu_scores.append(bleu)
                results["bleu"] = {
                    "mean": float(np.mean(bleu_scores)),
                    "std": float(np.std(bleu_scores)),
                    "scores": bleu_scores
                }
            
            if "rouge" in metrics:
                rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
                for ref, cand in zip(references, candidates):
                    rouge = calculate_rouge(ref, cand)
                    for key in rouge_scores:
                        rouge_scores[key].append(rouge[key])
                
                results["rouge"] = {
                    key: {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores))
                    }
                    for key, scores in rouge_scores.items()
                }
            
            if "bertscore" in metrics:
                bertscore = calculate_bertscore(references, candidates)
                results["bertscore"] = bertscore
            
            # Métricas adicionales
            results["length_ratio"] = {
                "mean": float(np.mean([
                    len(cand.split()) / len(ref.split()) if len(ref.split()) > 0 else 0
                    for ref, cand in zip(references, candidates)
                ]))
            }
            
            results["num_examples"] = len(test_data)
            results["num_successful"] = sum(1 for c in candidates if c)
            
            self._logger.info("Evaluación completada")
            return results
        
        except Exception as e:
            self._logger.error(f"Error en evaluación: {str(e)}")
            raise




