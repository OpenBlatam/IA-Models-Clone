from typing import List, Optional

from ..entities import Analysis, User, Product, Recommendation
from ..interfaces import IRecommendationService


class RecommendationService(IRecommendationService):
    
    def __init__(
        self,
        product_repository=None
    ):
        self.product_repository = product_repository
    
    async def generate_recommendations(
        self,
        analysis: Analysis,
        user: Optional[User] = None
    ) -> List[Recommendation]:
        if not analysis.is_completed():
            return []
        
        recommendations = []
        
        if analysis.metrics:
            if analysis.metrics.hydration_score < 50:
                recommendations.append(Recommendation(
                    product_id="hydrating-serum-001",
                    product_name="Hydrating Serum",
                    category="serum",
                    priority=1,
                    reason="Low hydration score detected",
                    confidence=0.85,
                    usage_frequency="daily"
                ))
            
            if analysis.metrics.wrinkles_score < 50:
                recommendations.append(Recommendation(
                    product_id="anti-aging-cream-001",
                    product_name="Anti-Aging Cream",
                    category="cream",
                    priority=2,
                    reason="Wrinkles detected",
                    confidence=0.80,
                    usage_frequency="nightly"
                ))
        
        if analysis.conditions:
            for condition in analysis.conditions:
                if condition.name.lower() == "acne":
                    recommendations.append(Recommendation(
                        product_id="acne-treatment-001",
                        product_name="Acne Treatment",
                        category="treatment",
                        priority=1,
                        reason=f"{condition.name} detected",
                        confidence=condition.confidence,
                        usage_frequency="daily"
                    ))
        
        if user and user.skin_type and self.product_repository:
            try:
                products = await self.product_repository.get_by_category("cleanser", limit=5)
                suitable_products = [
                    p for p in products 
                    if p.is_suitable_for_skin_type(user.skin_type)
                ][:2]
                
                for product in suitable_products:
                    recommendations.append(Recommendation(
                        product_id=product.id,
                        product_name=product.name,
                        category=product.category,
                        priority=3,
                        reason=f"Suitable for {user.skin_type.value} skin",
                        confidence=0.75,
                        usage_frequency="daily"
                    ))
            except Exception:
                pass
        
        return sorted(recommendations, key=lambda r: r.priority)
    
    async def get_recommendations_for_user(
        self,
        user_id: str
    ) -> List[Recommendation]:
        return []















