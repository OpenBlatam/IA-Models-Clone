"""
Sistema de recomendaciones basado en ML
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict


@dataclass
class MLRecommendation:
    """Recomendación basada en ML"""
    product_id: str
    confidence: float
    reasoning: str
    features_used: List[str]
    model_version: str
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_id": self.product_id,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "features_used": self.features_used,
            "model_version": self.model_version,
            "created_at": self.created_at
        }


class MLRecommender:
    """Sistema de recomendaciones basado en ML"""
    
    def __init__(self):
        """Inicializa el recomendador"""
        self.user_features: Dict[str, np.ndarray] = {}
        self.product_features: Dict[str, np.ndarray] = {}
        self.interaction_matrix: Dict[Tuple[str, str], float] = {}  # (user_id, product_id) -> rating
    
    def generate_ml_recommendations(self, user_id: str,
                                   analysis_features: np.ndarray,
                                   top_k: int = 10) -> List[MLRecommendation]:
        """
        Genera recomendaciones basadas en ML
        
        Args:
            user_id: ID del usuario
            analysis_features: Features del análisis
            top_k: Número de recomendaciones
            
        Returns:
            Lista de recomendaciones
        """
        # Combinar features de usuario y análisis
        if user_id in self.user_features:
            user_features = np.concatenate([self.user_features[user_id], analysis_features])
        else:
            user_features = analysis_features
        
        # Calcular scores para todos los productos
        product_scores = []
        
        for product_id, product_features in self.product_features.items():
            # Calcular similitud (cosine similarity)
            similarity = self._cosine_similarity(user_features, product_features)
            
            # Ajustar con interacciones previas
            interaction_score = self.interaction_matrix.get((user_id, product_id), 0.5)
            
            # Score final
            final_score = similarity * 0.7 + interaction_score * 0.3
            
            product_scores.append({
                "product_id": product_id,
                "score": final_score,
                "similarity": similarity
            })
        
        # Ordenar y obtener top K
        product_scores.sort(key=lambda x: x["score"], reverse=True)
        top_products = product_scores[:top_k]
        
        # Generar recomendaciones
        recommendations = []
        for item in top_products:
            recommendation = MLRecommendation(
                product_id=item["product_id"],
                confidence=float(item["score"]),
                reasoning=f"Similitud: {item['similarity']:.2f}, Interacción: {self.interaction_matrix.get((user_id, item['product_id']), 0.5):.2f}",
                features_used=["user_features", "analysis_features", "product_features"],
                model_version="1.0"
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def update_user_features(self, user_id: str, features: np.ndarray):
        """Actualiza features de usuario"""
        self.user_features[user_id] = features
    
    def update_product_features(self, product_id: str, features: np.ndarray):
        """Actualiza features de producto"""
        self.product_features[product_id] = features
    
    def record_interaction(self, user_id: str, product_id: str, rating: float):
        """Registra interacción usuario-producto"""
        self.interaction_matrix[(user_id, product_id)] = rating
    
    def get_recommendation_explanation(self, user_id: str, product_id: str) -> Dict:
        """Obtiene explicación de una recomendación"""
        if user_id not in self.user_features or product_id not in self.product_features:
            return {"error": "Datos insuficientes"}
        
        user_features = self.user_features[user_id]
        product_features = self.product_features[product_id]
        
        similarity = self._cosine_similarity(user_features, product_features)
        interaction = self.interaction_matrix.get((user_id, product_id), None)
        
        return {
            "product_id": product_id,
            "similarity_score": float(similarity),
            "interaction_score": float(interaction) if interaction else None,
            "feature_importance": self._calculate_feature_importance(user_features, product_features)
        }
    
    def _calculate_feature_importance(self, user_features: np.ndarray,
                                     product_features: np.ndarray) -> List[float]:
        """Calcula importancia de features"""
        # Diferencia absoluta normalizada
        diff = np.abs(user_features - product_features)
        importance = 1.0 / (1.0 + diff)
        return importance.tolist()

