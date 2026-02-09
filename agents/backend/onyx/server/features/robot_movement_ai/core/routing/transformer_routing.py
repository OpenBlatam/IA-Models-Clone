"""
Transformer-based Routing Module
=================================

Módulo de enrutamiento basado en transformers para análisis contextual
y toma de decisiones inteligentes.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline
)

logger = logging.getLogger(__name__)


@dataclass
class ContextualRouteInfo:
    """Información contextual de ruta."""
    route_description: str
    constraints: List[str]
    preferences: List[str]
    historical_data: Dict[str, Any]
    environmental_factors: Dict[str, Any]


class TransformerRouteAnalyzer:
    """
    Analizador de rutas basado en transformers.
    
    Usa modelos de lenguaje para entender contexto y generar
    recomendaciones inteligentes de rutas.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_llm: bool = False
    ):
        """
        Inicializar analizador.
        
        Args:
            model_name: Nombre del modelo transformer
            device: Dispositivo ('cuda', 'cpu', o None para auto-detectar)
            use_llm: Usar LLM para análisis avanzado
        """
        # Detectar dispositivo
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        self.model_name = model_name
        self.use_llm = use_llm
        
        try:
            # Cargar tokenizer y modelo
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            logger.info(f"TransformerRouteAnalyzer inicializado con modelo: {model_name}")
        except Exception as e:
            logger.warning(f"Error cargando modelo transformer: {e}")
            self.tokenizer = None
            self.model = None
        
        # Pipeline de LLM (opcional)
        self.llm_pipeline = None
        if use_llm:
            try:
                # Usar un modelo más pequeño para análisis
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model="gpt2",  # Modelo base, puede cambiarse
                    device=0 if self.device.type == "cuda" else -1,
                    max_length=200
                )
                logger.info("Pipeline de LLM inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando LLM pipeline: {e}")
    
    def encode_route_context(
        self,
        context: ContextualRouteInfo
    ) -> torch.Tensor:
        """
        Codificar contexto de ruta en embeddings.
        
        Args:
            context: Información contextual
            
        Returns:
            Tensor de embeddings
        """
        if not self.model or not self.tokenizer:
            # Fallback a embeddings simples
            return torch.zeros(384)  # Dimensión típica de MiniLM
        
        # Construir texto de contexto
        context_text = f"""
        Route Description: {context.route_description}
        Constraints: {', '.join(context.constraints)}
        Preferences: {', '.join(context.preferences)}
        Historical Performance: {context.historical_data}
        Environmental Factors: {context.environmental_factors}
        """
        
        # Tokenizar y codificar
        inputs = self.tokenizer(
            context_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Obtener embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Usar el embedding del token [CLS] o promedio de embeddings
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return embeddings.squeeze(0).cpu()
    
    def analyze_route_similarity(
        self,
        route1: ContextualRouteInfo,
        route2: ContextualRouteInfo
    ) -> float:
        """
        Analizar similitud entre dos rutas usando embeddings.
        
        Args:
            route1: Primera ruta
            route2: Segunda ruta
            
        Returns:
            Score de similitud (0-1)
        """
        if not self.model or not self.tokenizer:
            return 0.5  # Fallback
        
        # Codificar rutas
        emb1 = self.encode_route_context(route1)
        emb2 = self.encode_route_context(route2)
        
        # Calcular similitud coseno
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()
        
        # Normalizar a [0, 1]
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def generate_route_recommendation(
        self,
        context: ContextualRouteInfo,
        available_routes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generar recomendación de ruta usando análisis contextual.
        
        Args:
            context: Contexto de la ruta
            available_routes: Lista de rutas disponibles
            
        Returns:
            Recomendación con score y explicación
        """
        if not self.model or not self.tokenizer:
            # Fallback: retornar primera ruta
            return {
                "recommended_route": available_routes[0] if available_routes else None,
                "score": 0.5,
                "explanation": "Análisis básico (modelo no disponible)"
            }
        
        # Codificar contexto
        context_embedding = self.encode_route_context(context)
        
        # Analizar cada ruta disponible
        route_scores = []
        for route in available_routes:
            # Crear contexto de ruta
            route_context = ContextualRouteInfo(
                route_description=route.get("description", ""),
                constraints=route.get("constraints", []),
                preferences=route.get("preferences", []),
                historical_data=route.get("historical_data", {}),
                environmental_factors=route.get("environmental_factors", {})
            )
            
            # Codificar ruta
            route_embedding = self.encode_route_context(route_context)
            
            # Calcular similitud
            similarity = torch.nn.functional.cosine_similarity(
                context_embedding.unsqueeze(0),
                route_embedding.unsqueeze(0)
            ).item()
            
            # Normalizar
            score = (similarity + 1) / 2
            
            route_scores.append({
                "route": route,
                "score": score
            })
        
        # Ordenar por score
        route_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Generar explicación usando LLM si está disponible
        explanation = self._generate_explanation(context, route_scores[0] if route_scores else None)
        
        return {
            "recommended_route": route_scores[0]["route"] if route_scores else None,
            "score": route_scores[0]["score"] if route_scores else 0.0,
            "all_scores": route_scores,
            "explanation": explanation
        }
    
    def _generate_explanation(
        self,
        context: ContextualRouteInfo,
        best_route: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generar explicación de la recomendación.
        
        Args:
            context: Contexto
            best_route: Mejor ruta encontrada
            
        Returns:
            Explicación en texto
        """
        if not best_route:
            return "No se encontraron rutas adecuadas."
        
        if self.llm_pipeline:
            try:
                prompt = f"""
                Based on the following route context:
                Description: {context.route_description}
                Constraints: {', '.join(context.constraints)}
                Preferences: {', '.join(context.preferences)}
                
                Explain why this route is recommended:
                Route: {best_route.get('route', {}).get('description', 'N/A')}
                Score: {best_route.get('score', 0.0):.2f}
                
                Explanation:
                """
                
                result = self.llm_pipeline(
                    prompt,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                explanation = result[0]["generated_text"].split("Explanation:")[-1].strip()
                return explanation
            except Exception as e:
                logger.warning(f"Error generando explicación con LLM: {e}")
        
        # Fallback: explicación basada en reglas
        score = best_route.get("score", 0.0)
        if score > 0.8:
            return f"Ruta altamente recomendada (score: {score:.2f}) basada en similitud contextual y preferencias."
        elif score > 0.6:
            return f"Ruta recomendada (score: {score:.2f}) que cumple con la mayoría de los criterios."
        else:
            return f"Ruta seleccionada (score: {score:.2f}) como mejor opción disponible."
    
    def analyze_route_constraints(
        self,
        context: ContextualRouteInfo
    ) -> Dict[str, Any]:
        """
        Analizar y clasificar restricciones de ruta.
        
        Args:
            context: Contexto de la ruta
            
        Returns:
            Análisis de restricciones
        """
        if not self.model or not self.tokenizer:
            return {
                "constraint_count": len(context.constraints),
                "severity": "unknown",
                "feasible": True
            }
        
        # Codificar restricciones
        constraint_text = " ".join(context.constraints)
        inputs = self.tokenizer(
            constraint_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            constraint_embedding = outputs.last_hidden_state[:, 0, :].cpu()
        
        # Análisis básico
        constraint_count = len(context.constraints)
        severity = "low" if constraint_count < 3 else "medium" if constraint_count < 5 else "high"
        
        return {
            "constraint_count": constraint_count,
            "severity": severity,
            "feasible": constraint_count < 10,
            "embedding": constraint_embedding.numpy().tolist()
        }




