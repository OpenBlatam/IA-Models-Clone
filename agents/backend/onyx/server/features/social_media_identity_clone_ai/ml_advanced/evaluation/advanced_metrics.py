"""
Métricas avanzadas de evaluación (BLEU, ROUGE, etc.)
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Métricas avanzadas para evaluación"""
    
    def __init__(self):
        pass
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
        n_gram: int = 4
    ) -> Dict[str, float]:
        """
        Calcula BLEU score
        
        Args:
            predictions: Lista de predicciones
            references: Lista de listas de referencias
            n_gram: N-gram máximo
            
        Returns:
            BLEU scores
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            smoothing = SmoothingFunction().method1
            bleu_scores = []
            
            for pred, refs in zip(predictions, references):
                # Convertir a listas de palabras
                pred_tokens = pred.split()
                ref_tokens_list = [ref.split() for ref in refs]
                
                # Calcular BLEU
                score = sentence_bleu(
                    ref_tokens_list,
                    pred_tokens,
                    smoothing_function=smoothing
                )
                bleu_scores.append(score)
            
            return {
                "bleu_mean": float(np.mean(bleu_scores)),
                "bleu_std": float(np.std(bleu_scores)),
                "bleu_scores": bleu_scores
            }
        except ImportError:
            logger.warning("nltk no instalado, usando cálculo básico")
            return {"bleu_mean": 0.0, "bleu_std": 0.0}
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calcula ROUGE scores (simplificado)
        
        Args:
            predictions: Lista de predicciones
            references: Lista de referencias
            
        Returns:
            ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            return {
                "rouge1": float(np.mean(rouge1_scores)),
                "rouge2": float(np.mean(rouge2_scores)),
                "rougeL": float(np.mean(rougeL_scores))
            }
        except ImportError:
            logger.warning("rouge_score no instalado, usando cálculo básico")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_perplexity(
        self,
        log_probs: np.ndarray,
        lengths: Optional[np.ndarray] = None
    ) -> float:
        """
        Calcula perplexity
        
        Args:
            log_probs: Log probabilidades [batch_size, seq_len]
            lengths: Longitudes reales de secuencias
            
        Returns:
            Perplexity
        """
        if lengths is not None:
            # Solo considerar tokens válidos
            total_log_prob = 0.0
            total_tokens = 0
            
            for i, length in enumerate(lengths):
                total_log_prob += log_probs[i, :length].sum()
                total_tokens += length
            
            if total_tokens > 0:
                avg_log_prob = total_log_prob / total_tokens
                perplexity = np.exp(-avg_log_prob)
            else:
                perplexity = float('inf')
        else:
            avg_log_prob = log_probs.mean()
            perplexity = np.exp(-avg_log_prob)
        
        return float(perplexity)
    
    def calculate_diversity(
        self,
        texts: List[str],
        n_gram: int = 2
    ) -> Dict[str, float]:
        """
        Calcula diversidad de n-grams
        
        Args:
            texts: Lista de textos
            n_gram: N-gram a usar
            
        Returns:
            Métricas de diversidad
        """
        all_ngrams = set()
        total_ngrams = 0
        
        for text in texts:
            words = text.split()
            for i in range(len(words) - n_gram + 1):
                ngram = tuple(words[i:i+n_gram])
                all_ngrams.add(ngram)
                total_ngrams += 1
        
        unique_ngrams = len(all_ngrams)
        diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        
        return {
            "unique_ngrams": unique_ngrams,
            "total_ngrams": total_ngrams,
            "diversity": diversity
        }




