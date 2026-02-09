"""
Sistema de comparación de productos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class ProductComparison:
    """Comparación de productos"""
    id: str
    user_id: str
    products: List[Dict]  # Lista de productos a comparar
    comparison_metrics: Dict
    winner: Optional[str] = None
    recommendations: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "products": self.products,
            "comparison_metrics": self.comparison_metrics,
            "winner": self.winner,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class ProductComparisonSystem:
    """Sistema de comparación de productos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.comparisons: Dict[str, List[ProductComparison]] = {}  # user_id -> [comparisons]
    
    def compare_products(self, user_id: str, products: List[Dict],
                        comparison_criteria: List[str] = None) -> ProductComparison:
        """
        Compara productos
        
        Args:
            user_id: ID del usuario
            products: Lista de productos con sus datos
            comparison_criteria: Criterios de comparación (precio, efectividad, etc.)
        """
        if comparison_criteria is None:
            comparison_criteria = ["price", "effectiveness", "ingredients", "reviews"]
        
        # Calcular métricas de comparación
        comparison_metrics = self._calculate_comparison_metrics(
            products, comparison_criteria
        )
        
        # Determinar ganador
        winner = self._determine_winner(products, comparison_metrics)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(products, comparison_metrics, winner)
        
        comparison = ProductComparison(
            id=str(uuid.uuid4()),
            user_id=user_id,
            products=products,
            comparison_metrics=comparison_metrics,
            winner=winner,
            recommendations=recommendations
        )
        
        if user_id not in self.comparisons:
            self.comparisons[user_id] = []
        
        self.comparisons[user_id].append(comparison)
        return comparison
    
    def _calculate_comparison_metrics(self, products: List[Dict],
                                     criteria: List[str]) -> Dict:
        """Calcula métricas de comparación"""
        metrics = {}
        
        for criterion in criteria:
            if criterion == "price":
                prices = [p.get("price", 0) for p in products]
                metrics["price"] = {
                    "values": prices,
                    "lowest": min(prices) if prices else 0,
                    "highest": max(prices) if prices else 0,
                    "average": sum(prices) / len(prices) if prices else 0
                }
            
            elif criterion == "effectiveness":
                effectiveness_scores = [p.get("effectiveness_score", 0) for p in products]
                metrics["effectiveness"] = {
                    "values": effectiveness_scores,
                    "highest": max(effectiveness_scores) if effectiveness_scores else 0,
                    "average": sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0
                }
            
            elif criterion == "ingredients":
                ingredient_counts = [len(p.get("ingredients", [])) for p in products]
                metrics["ingredients"] = {
                    "counts": ingredient_counts,
                    "most": max(ingredient_counts) if ingredient_counts else 0,
                    "least": min(ingredient_counts) if ingredient_counts else 0
                }
            
            elif criterion == "reviews":
                review_scores = [p.get("average_rating", 0) for p in products]
                metrics["reviews"] = {
                    "ratings": review_scores,
                    "highest": max(review_scores) if review_scores else 0,
                    "average": sum(review_scores) / len(review_scores) if review_scores else 0
                }
        
        return metrics
    
    def _determine_winner(self, products: List[Dict], metrics: Dict) -> Optional[str]:
        """Determina el producto ganador"""
        if not products:
            return None
        
        scores = []
        
        for i, product in enumerate(products):
            score = 0.0
            
            # Score por efectividad
            if "effectiveness" in metrics:
                eff_score = product.get("effectiveness_score", 0)
                max_eff = metrics["effectiveness"]["highest"]
                if max_eff > 0:
                    score += (eff_score / max_eff) * 0.4
            
            # Score por reviews
            if "reviews" in metrics:
                review_score = product.get("average_rating", 0)
                max_review = metrics["reviews"]["highest"]
                if max_review > 0:
                    score += (review_score / max_review) * 0.3
            
            # Score por precio (más barato = mejor)
            if "price" in metrics:
                price = product.get("price", float('inf'))
                min_price = metrics["price"]["lowest"]
                if price > 0 and min_price > 0:
                    score += (min_price / price) * 0.2
            
            # Score por ingredientes
            if "ingredients" in metrics:
                ing_count = len(product.get("ingredients", []))
                max_ing = metrics["ingredients"]["most"]
                if max_ing > 0:
                    score += (ing_count / max_ing) * 0.1
            
            scores.append((i, score))
        
        # Ordenar por score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores:
            winner_idx = scores[0][0]
            return products[winner_idx].get("id") or products[winner_idx].get("name")
        
        return None
    
    def _generate_recommendations(self, products: List[Dict], metrics: Dict,
                                 winner: Optional[str]) -> List[str]:
        """Genera recomendaciones basadas en comparación"""
        recommendations = []
        
        if winner:
            recommendations.append(f"Producto recomendado: {winner}")
        
        # Recomendación por precio
        if "price" in metrics:
            price_diff = metrics["price"]["highest"] - metrics["price"]["lowest"]
            if price_diff > 0:
                recommendations.append(
                    f"Diferencia de precio: ${price_diff:.2f} entre productos"
                )
        
        # Recomendación por efectividad
        if "effectiveness" in metrics:
            eff_diff = metrics["effectiveness"]["highest"] - metrics["effectiveness"]["average"]
            if eff_diff > 10:
                recommendations.append(
                    "Hay un producto significativamente más efectivo"
                )
        
        return recommendations
    
    def get_user_comparisons(self, user_id: str) -> List[ProductComparison]:
        """Obtiene comparaciones del usuario"""
        return self.comparisons.get(user_id, [])






