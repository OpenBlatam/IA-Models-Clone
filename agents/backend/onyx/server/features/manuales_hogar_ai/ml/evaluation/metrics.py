"""
Métricas de Evaluación
======================

Métricas para evaluar calidad de generación.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    BLEU_AVAILABLE = True
    ROUGE_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    ROUGE_AVAILABLE = False
    logging.warning("NLTK o rouge-score no disponibles. Instalar con: pip install nltk rouge-score")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("BERTScore no disponible. Instalar con: pip install bert-score")

logger = logging.getLogger(__name__)


def calculate_bleu(
    reference: str,
    candidate: str,
    smoothing: bool = True
) -> float:
    """
    Calcular BLEU score.
    
    Args:
        reference: Texto de referencia
        candidate: Texto generado
        smoothing: Usar smoothing
    
    Returns:
        BLEU score (0-1)
    """
    if not BLEU_AVAILABLE:
        logger.warning("BLEU no disponible")
        return 0.0
    
    try:
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if smoothing:
            smoothing_func = SmoothingFunction().method1
            score = sentence_bleu(
                [ref_tokens],
                cand_tokens,
                smoothing_function=smoothing_func
            )
        else:
            score = sentence_bleu([ref_tokens], cand_tokens)
        
        return float(score)
    
    except Exception as e:
        logger.error(f"Error calculando BLEU: {str(e)}")
        return 0.0


def calculate_rouge(
    reference: str,
    candidate: str,
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
) -> Dict[str, float]:
    """
    Calcular ROUGE scores.
    
    Args:
        reference: Texto de referencia
        candidate: Texto generado
        rouge_types: Tipos de ROUGE a calcular
    
    Returns:
        Diccionario con scores
    """
    if not ROUGE_AVAILABLE:
        logger.warning("ROUGE no disponible")
        return {r: 0.0 for r in rouge_types}
    
    try:
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            rouge_type: scores[rouge_type].fmeasure
            for rouge_type in rouge_types
        }
    
    except Exception as e:
        logger.error(f"Error calculando ROUGE: {str(e)}")
        return {r: 0.0 for r in rouge_types}


def calculate_bertscore(
    references: List[str],
    candidates: List[str],
    lang: str = "es"
) -> Dict[str, Any]:
    """
    Calcular BERTScore.
    
    Args:
        references: Lista de textos de referencia
        candidates: Lista de textos generados
        lang: Idioma
    
    Returns:
        Diccionario con scores
    """
    if not BERTSCORE_AVAILABLE:
        logger.warning("BERTScore no disponible")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    try:
        P, R, F1 = bert_score(
            candidates,
            references,
            lang=lang,
            verbose=False
        )
        
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item())
        }
    
    except Exception as e:
        logger.error(f"Error calculando BERTScore: {str(e)}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}




