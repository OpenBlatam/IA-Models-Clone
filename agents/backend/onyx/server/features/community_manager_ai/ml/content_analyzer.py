"""
Content Analyzer - Analizador de Contenido con Deep Learning
============================================================

Análisis avanzado de contenido usando transformers.
"""

import logging
import torch
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import numpy as np

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analizador de contenido usando modelos transformer"""
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[str] = None
    ):
        """
        Inicializar analizador de contenido
        
        Args:
            model_name: Nombre del modelo pre-entrenado
            device: Dispositivo (cuda/cpu), auto-detecta si es None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)
            self.model.eval()
            
            # Optimizar para inferencia rápida
            if hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Modelo compilado con torch.compile")
                except Exception as e:
                    logger.warning(f"No se pudo compilar modelo: {e}")
            
            # Pipeline para análisis de sentimiento
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Content Analyzer inicializado con {model_name} en {self.device}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.model = None
            self.tokenizer = None
            self.sentiment_pipeline = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analizar sentimiento del texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con análisis de sentimiento
        """
        if not self.sentiment_pipeline:
            return {"label": "neutral", "score": 0.5, "error": "Modelo no disponible"}
        
        try:
            # Truncar si es muy largo
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.sentiment_pipeline(text)[0]
            
            return {
                "label": result["label"],
                "score": result["score"],
                "text_length": len(text)
            }
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {e}")
            return {"label": "error", "score": 0.0, "error": str(e)}
    
    def analyze_content_quality(
        self,
        content: str,
        platform: str
    ) -> Dict[str, Any]:
        """
        Analizar calidad del contenido
        
        Args:
            content: Contenido a analizar
            platform: Plataforma objetivo
            
        Returns:
            Dict con análisis de calidad
        """
        analysis = {
            "length": len(content),
            "word_count": len(content.split()),
            "sentiment": self.analyze_sentiment(content),
            "has_hashtags": "#" in content,
            "has_mentions": "@" in content,
            "has_links": "http" in content.lower(),
            "platform": platform
        }
        
        # Análisis específico por plataforma
        platform_limits = {
            "twitter": 280,
            "facebook": 5000,
            "instagram": 2200,
            "linkedin": 3000
        }
        
        limit = platform_limits.get(platform.lower(), 5000)
        analysis["within_limit"] = len(content) <= limit
        analysis["limit"] = limit
        analysis["usage_percentage"] = (len(content) / limit * 100) if limit > 0 else 0
        
        return analysis
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analizar múltiples textos en lote
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de análisis
        """
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results

