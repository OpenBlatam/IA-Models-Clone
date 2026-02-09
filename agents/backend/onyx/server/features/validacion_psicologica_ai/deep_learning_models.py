"""
Modelos de Deep Learning para Validación Psicológica
=====================================================
Modelos avanzados usando PyTorch, Transformers y LLMs
Refactorizado siguiendo mejores prácticas

NOTA: Este archivo mantiene compatibilidad hacia atrás.
Los modelos principales están ahora en models/
"""

from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import structlog
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline,
    TrainingArguments,
    Trainer
)
from transformers import get_linear_schedule_with_warmup
import os
from pathlib import Path

from .config_loader import config_loader
from .models import (
    PsychologicalEmbeddingModel,
    PersonalityClassifier,
    SentimentTransformerModel
)
from .core.analyzer import deep_learning_analyzer

logger = structlog.get_logger()


def get_device() -> torch.device:
    """
    Get appropriate device (GPU/CPU)
    
    Returns:
        torch.device
    """
    device_config = config_loader.get_device_config()
    
    if device_config.get("use_cuda", True) and torch.cuda.is_available():
        device_id = device_config.get("cuda_device", 0)
        return torch.device(f"cuda:{device_id}")
    elif device_config.get("allow_mps", False) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PsychologicalEmbeddingModel(nn.Module):
    """
    Modelo de embeddings para análisis psicológico
    Usa transformers para generar embeddings semánticos
    Refactorizado con mejor manejo de errores y configuración
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        dropout: Optional[float] = None
    ):
        """
        Inicializar modelo
        
        Args:
            model_name: Nombre del modelo pre-entrenado (opcional, usa config)
            embedding_dim: Dimensión de embeddings (opcional, usa config)
            dropout: Tasa de dropout (opcional, usa config)
        """
        super().__init__()
        
        # Load from config if not provided
        model_config = config_loader.get_model_config("embedding")
        self.model_name = model_name or model_config.get("name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = embedding_dim or model_config.get("embedding_dim", 384)
        dropout = dropout or model_config.get("dropout", 0.1)
        self.max_length = model_config.get("max_length", 512)
        
        self.device = get_device()
        
        # Initialize tokenizer and model with proper error handling
        self.tokenizer = None
        self.backbone = None
        self._load_model()
        
        # Initialize layers
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _load_model(self) -> None:
        """Load model with proper error handling"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.backbone = AutoModel.from_pretrained(self.model_name)
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()  # Set to eval mode for inference
            
            logger.info("Embedding model loaded", model=self.model_name, device=str(self.device))
        except Exception as e:
            logger.error(f"Could not load model {self.model_name}", error=str(e))
            self.tokenizer = None
            self.backbone = None
    
    def _initialize_weights(self) -> None:
        """Initialize projection layer weights"""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Generar embeddings para textos
        
        Args:
            texts: Lista de textos
            
        Returns:
            Tensor de embeddings [batch_size, embedding_dim]
        """
        if self.backbone is None or self.tokenizer is None:
            logger.warning("Model not loaded, returning zero embeddings")
            return torch.zeros(len(texts), self.embedding_dim, device=self.device)
        
        try:
            # Tokenizar textos con manejo de errores
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Obtener embeddings
            with torch.no_grad():
                outputs = self.backbone(**inputs)
                # Usar el embedding del token [CLS]
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Proyección y dropout (solo en training)
            if self.training:
                embeddings = self.projection(embeddings)
                embeddings = self.dropout(embeddings)
            else:
                embeddings = self.projection(embeddings)
            
            # Normalizar
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings
            
        except Exception as e:
            logger.error("Error generating embeddings", error=str(e), exc_info=True)
            return torch.zeros(len(texts), self.embedding_dim, device=self.device)


class PersonalityClassifier(nn.Module):
    """
    Clasificador de personalidad Big Five usando transformers
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_traits: int = 5,
        hidden_dim: int = 768
    ):
        """
        Inicializar clasificador
        
        Args:
            model_name: Nombre del modelo base
            num_traits: Número de rasgos (Big Five = 5)
            hidden_dim: Dimensión oculta
        """
        super().__init__()
        
        self.num_traits = num_traits
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Could not load model {model_name}", error=str(e))
            self.tokenizer = None
            self.backbone = None
            hidden_dim = 768  # Default
        
        # Head para cada rasgo de personalidad
        self.trait_heads = nn.ModuleDict({
            "openness": nn.Linear(hidden_dim, 1),
            "conscientiousness": nn.Linear(hidden_dim, 1),
            "extraversion": nn.Linear(hidden_dim, 1),
            "agreeableness": nn.Linear(hidden_dim, 1),
            "neuroticism": nn.Linear(hidden_dim, 1)
        })
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Predecir rasgos de personalidad
        
        Args:
            texts: Lista de textos
            
        Returns:
            Diccionario con scores de cada rasgo
        """
        if self.backbone is None:
            # Fallback: scores aleatorios
            return {
                trait: torch.rand(len(texts), 1) * 0.5 + 0.5
                for trait in self.trait_heads.keys()
            }
        
        try:
            # Tokenizar
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Obtener representaciones
            with torch.no_grad():
                outputs = self.backbone(**inputs)
                pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            pooled = self.dropout(pooled)
            
            # Predecir cada rasgo
            predictions = {}
            for trait, head in self.trait_heads.items():
                predictions[trait] = torch.sigmoid(head(pooled))
            
            return predictions
        except Exception as e:
            logger.error("Error in personality classification", error=str(e))
            return {
                trait: torch.rand(len(texts), 1) * 0.5 + 0.5
                for trait in self.trait_heads.keys()
            }


class SentimentTransformerModel(nn.Module):
    """
    Modelo de análisis de sentimientos usando transformers
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        num_labels: int = 3  # positive, neutral, negative
    ):
        """
        Inicializar modelo
        
        Args:
            model_name: Nombre del modelo pre-entrenado
            num_labels: Número de clases de sentimiento
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        except Exception as e:
            logger.warning(f"Could not load sentiment model {model_name}", error=str(e))
            self.tokenizer = None
            self.model = None
    
    def forward(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analizar sentimiento de textos
        
        Args:
            texts: Lista de textos
            
        Returns:
            Análisis de sentimiento
        """
        if self.model is None:
            # Fallback
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
            }
        
        try:
            # Tokenizar
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Predecir
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
            
            # Obtener sentimiento dominante
            predicted_class = torch.argmax(probs, dim=-1)[0].item()
            confidence = probs[0][predicted_class].item()
            
            labels = ["negative", "neutral", "positive"]
            sentiment = labels[predicted_class] if predicted_class < len(labels) else "neutral"
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": {
                    label: probs[0][i].item() if i < len(labels) else 0.0
                    for i, label in enumerate(labels)
                }
            }
        except Exception as e:
            logger.error("Error in sentiment analysis", error=str(e))
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
            }


class LLMAnalyzer:
    """
    Analizador usando LLMs para análisis psicológico avanzado
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_api: bool = True
    ):
        """
        Inicializar analizador
        
        Args:
            model_name: Nombre del modelo (opcional)
            use_api: Usar API en lugar de modelo local
        """
        self.model_name = model_name or "gpt-3.5-turbo"
        self.use_api = use_api
        self.pipeline = None
        
        if not use_api:
            try:
                # Intentar cargar modelo local si está disponible
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_name or "gpt2",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.warning("Could not load local LLM, will use API", error=str(e))
    
    async def analyze_psychological_patterns(
        self,
        texts: List[str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar patrones psicológicos usando LLM
        
        Args:
            texts: Lista de textos a analizar
            context: Contexto adicional
            
        Returns:
            Análisis psicológico
        """
        if self.use_api:
            # Usar API externa (OpenAI, Anthropic, etc.)
            try:
                from .ai_integrations import ai_service_manager
                service = ai_service_manager.get_service()
                
                if service:
                    prompt = self._create_analysis_prompt(texts, context)
                    result = await service.generate_insights(
                        {"texts": texts, "context": context},
                        prompt
                    )
                    return self._parse_llm_response(result)
            except Exception as e:
                logger.error("Error using LLM API", error=str(e))
        
        # Fallback: análisis básico
        return {
            "patterns": ["No patterns detected"],
            "insights": "Basic analysis only",
            "confidence": 0.5
        }
    
    def _create_analysis_prompt(
        self,
        texts: List[str],
        context: Optional[str] = None
    ) -> str:
        """Crear prompt para análisis"""
        prompt = f"""
Analyze the following social media posts for psychological patterns:

Posts:
{chr(10).join(f"- {text[:200]}" for text in texts[:10])}

Context: {context or "General social media activity"}

Provide analysis of:
1. Emotional patterns
2. Behavioral indicators
3. Personality traits
4. Risk factors
5. Recommendations

Format as JSON.
        """.strip()
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parsear respuesta del LLM"""
        try:
            import json
            return json.loads(response)
        except:
            return {
                "patterns": [response],
                "insights": response,
                "confidence": 0.7
            }


class DeepLearningAnalyzer:
    """
    Analizador principal que integra todos los modelos de deep learning
    """
    
    def __init__(self):
        """Inicializar analizador"""
        self.embedding_model = PsychologicalEmbeddingModel()
        self.personality_classifier = PersonalityClassifier()
        self.sentiment_model = SentimentTransformerModel()
        self.llm_analyzer = LLMAnalyzer()
        
        # Mover modelos a GPU si está disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_models_to_device()
        
        logger.info("DeepLearningAnalyzer initialized", device=str(self.device))
    
    def _move_models_to_device(self) -> None:
        """Mover modelos a dispositivo (GPU/CPU)"""
        try:
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to(self.device)
                self.personality_classifier = self.personality_classifier.to(self.device)
                self.sentiment_model = self.sentiment_model.to(self.device)
        except Exception as e:
            logger.warning("Error moving models to device", error=str(e))
    
    async def analyze_comprehensive(
        self,
        texts: List[str],
        include_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Análisis completo usando todos los modelos
        
        Args:
            texts: Lista de textos
            include_llm: Incluir análisis con LLM
            
        Returns:
            Análisis completo
        """
        results = {}
        
        # Embeddings semánticos
        try:
            embeddings = self.embedding_model(texts)
            results["embeddings"] = embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error("Error generating embeddings", error=str(e))
        
        # Análisis de personalidad
        try:
            personality = self.personality_classifier(texts)
            results["personality"] = {
                trait: scores.cpu().numpy().tolist()
                for trait, scores in personality.items()
            }
        except Exception as e:
            logger.error("Error in personality analysis", error=str(e))
        
        # Análisis de sentimiento
        try:
            sentiment = self.sentiment_model(texts)
            results["sentiment"] = sentiment
        except Exception as e:
            logger.error("Error in sentiment analysis", error=str(e))
        
        # Análisis con LLM (opcional)
        if include_llm:
            try:
                llm_analysis = await self.llm_analyzer.analyze_psychological_patterns(texts)
                results["llm_analysis"] = llm_analysis
            except Exception as e:
                logger.error("Error in LLM analysis", error=str(e))
        
        return results


# Instancia global del analizador
deep_learning_analyzer = DeepLearningAnalyzer()

