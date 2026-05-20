"""
Sistema de reseñas y ratings
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import statistics


@dataclass
class Review:
    """Reseña de producto"""
    id: str
    user_id: str
    product_id: str
    product_name: str
    rating: int  # 1-5
    title: str
    comment: str
    pros: List[str] = None
    cons: List[str] = None
    verified_purchase: bool = False
    helpful_count: int = 0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.pros is None:
            self.pros = []
        if self.cons is None:
            self.cons = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "rating": self.rating,
            "title": self.title,
            "comment": self.comment,
            "pros": self.pros,
            "cons": self.cons,
            "verified_purchase": self.verified_purchase,
            "helpful_count": self.helpful_count,
            "created_at": self.created_at
        }


@dataclass
class ProductRating:
    """Rating agregado de producto"""
    product_id: str
    product_name: str
    average_rating: float
    total_reviews: int
    rating_distribution: Dict[int, int]  # rating -> count
    verified_reviews_count: int
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "average_rating": self.average_rating,
            "total_reviews": self.total_reviews,
            "rating_distribution": self.rating_distribution,
            "verified_reviews_count": self.verified_reviews_count,
            "created_at": self.created_at
        }


class ReviewsRatingsSystem:
    """Sistema de reseñas y ratings"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reviews: Dict[str, List[Review]] = {}  # product_id -> [reviews]
        self.user_reviews: Dict[str, List[Review]] = {}  # user_id -> [reviews]
    
    def create_review(self, user_id: str, product_id: str, product_name: str,
                     rating: int, title: str, comment: str,
                     pros: Optional[List[str]] = None,
                     cons: Optional[List[str]] = None,
                     verified_purchase: bool = False) -> Review:
        """Crea una reseña"""
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        review = Review(
            id=str(uuid.uuid4()),
            user_id=user_id,
            product_id=product_id,
            product_name=product_name,
            rating=rating,
            title=title,
            comment=comment,
            pros=pros or [],
            cons=cons or [],
            verified_purchase=verified_purchase
        )
        
        # Agregar a reviews del producto
        if product_id not in self.reviews:
            self.reviews[product_id] = []
        self.reviews[product_id].append(review)
        
        # Agregar a reviews del usuario
        if user_id not in self.user_reviews:
            self.user_reviews[user_id] = []
        self.user_reviews[user_id].append(review)
        
        return review
    
    def get_product_reviews(self, product_id: str, limit: int = 50,
                          sort_by: str = "newest") -> List[Review]:
        """Obtiene reseñas de un producto"""
        product_reviews = self.reviews.get(product_id, [])
        
        # Ordenar
        if sort_by == "newest":
            product_reviews.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "oldest":
            product_reviews.sort(key=lambda x: x.created_at)
        elif sort_by == "highest_rating":
            product_reviews.sort(key=lambda x: x.rating, reverse=True)
        elif sort_by == "lowest_rating":
            product_reviews.sort(key=lambda x: x.rating)
        elif sort_by == "most_helpful":
            product_reviews.sort(key=lambda x: x.helpful_count, reverse=True)
        
        return product_reviews[:limit]
    
    def get_product_rating(self, product_id: str) -> Optional[ProductRating]:
        """Obtiene rating agregado de un producto"""
        product_reviews = self.reviews.get(product_id, [])
        
        if not product_reviews:
            return None
        
        # Calcular promedio
        ratings = [r.rating for r in product_reviews]
        average_rating = statistics.mean(ratings)
        
        # Distribución de ratings
        rating_distribution = {}
        for i in range(1, 6):
            rating_distribution[i] = sum(1 for r in ratings if r == i)
        
        # Reviews verificadas
        verified_count = sum(1 for r in product_reviews if r.verified_purchase)
        
        product_name = product_reviews[0].product_name if product_reviews else "Unknown"
        
        return ProductRating(
            product_id=product_id,
            product_name=product_name,
            average_rating=average_rating,
            total_reviews=len(product_reviews),
            rating_distribution=rating_distribution,
            verified_reviews_count=verified_count
        )
    
    def mark_review_helpful(self, review_id: str, product_id: str) -> bool:
        """Marca una reseña como útil"""
        product_reviews = self.reviews.get(product_id, [])
        
        for review in product_reviews:
            if review.id == review_id:
                review.helpful_count += 1
                return True
        
        return False
    
    def get_user_reviews(self, user_id: str) -> List[Review]:
        """Obtiene reseñas del usuario"""
        return self.user_reviews.get(user_id, [])
    
    def get_top_rated_products(self, limit: int = 10) -> List[ProductRating]:
        """Obtiene productos mejor calificados"""
        all_ratings = []
        
        for product_id in self.reviews.keys():
            rating = self.get_product_rating(product_id)
            if rating:
                all_ratings.append(rating)
        
        # Ordenar por rating promedio
        all_ratings.sort(key=lambda x: x.average_rating, reverse=True)
        
        return all_ratings[:limit]






