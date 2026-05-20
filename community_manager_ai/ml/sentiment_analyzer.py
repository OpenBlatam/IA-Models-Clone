"""
Sentiment Analyzer - Analizador de Sentimiento Avanzado
========================================================

Análisis de sentimiento usando modelos transformer fine-tuned.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)
import numpy as np

logger = logging.getLogger(__name__)


class SentimentAnalyzer(nn.Module):
    """Analizador de sentimiento con arquitectura personalizada"""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        device: Optional[str] = None
    ):
        """
        Inicializar analizador de sentimiento
        
        Args:
            model_name: Nombre del modelo base
            num_labels: Número de clases (3: positivo, negativo, neutro)
            device: Dispositivo
        """
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            
            # Capa de clasificación personalizada
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.backbone.config.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_labels)
            )
            
            self.num_labels = num_labels
            self.to(self.device)
            self.eval()
            
            logger.info(f"Sentiment Analyzer inicializado en {self.device}")
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            self.backbone = None
            self.tokenizer = None
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: IDs de tokens
            attention_mask: Máscara de atención
            
        Returns:
            Logits de clasificación
        """
        if self.backbone is None:
            raise ValueError("Modelo no inicializado")
        
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(pooled_output)
        return logits
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predecir sentimiento
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict con predicción
        """
        if not self.tokenizer or not self.backbone:
            return {"label": "neutral", "confidence": 0.33, "error": "Modelo no disponible"}
        
        try:
            # Tokenizar
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Inferencia
            with torch.no_grad():
                logits = self.forward(**inputs)
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            labels = ["negative", "neutral", "positive"]
            label = labels[predicted_class] if predicted_class < len(labels) else "neutral"
            
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    label: prob.item() for label, prob in zip(labels, probabilities[0])
                }
            }
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return {"label": "error", "confidence": 0.0, "error": str(e)}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analizar lote de textos
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de predicciones
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results




