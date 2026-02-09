"""
Sistema de Marketplace

Proporciona:
- Marketplace de canciones
- Compra/venta de canciones
- Sistema de licencias
- Ratings y reviews
- Búsqueda de productos
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class LicenseType(Enum):
    """Tipos de licencia"""
    FREE = "free"
    PERSONAL = "personal"
    COMMERCIAL = "commercial"
    EXCLUSIVE = "exclusive"


class ListingStatus(Enum):
    """Estado de publicación"""
    DRAFT = "draft"
    ACTIVE = "active"
    SOLD = "sold"
    INACTIVE = "inactive"


@dataclass
class MarketplaceListing:
    """Publicación en marketplace"""
    listing_id: str
    song_id: str
    seller_id: str
    title: str
    description: str
    price: float
    license_type: LicenseType
    status: ListingStatus = ListingStatus.DRAFT
    tags: List[str] = field(default_factory=list)
    preview_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    views: int = 0
    purchases: int = 0
    rating: float = 0.0
    review_count: int = 0


@dataclass
class Purchase:
    """Compra"""
    purchase_id: str
    listing_id: str
    buyer_id: str
    seller_id: str
    price: float
    license_type: LicenseType
    transaction_date: datetime = field(default_factory=datetime.now)
    status: str = "completed"


@dataclass
class Review:
    """Review de producto"""
    review_id: str
    listing_id: str
    user_id: str
    rating: int  # 1-5
    comment: str
    created_at: datetime = field(default_factory=datetime.now)


class MarketplaceService:
    """Servicio de marketplace"""
    
    def __init__(self):
        self.listings: Dict[str, MarketplaceListing] = {}
        self.purchases: Dict[str, Purchase] = {}
        self.reviews: Dict[str, List[Review]] = defaultdict(list)
        logger.info("MarketplaceService initialized")
    
    def create_listing(
        self,
        listing_id: str,
        song_id: str,
        seller_id: str,
        title: str,
        description: str,
        price: float,
        license_type: LicenseType,
        tags: Optional[List[str]] = None,
        preview_url: Optional[str] = None
    ) -> MarketplaceListing:
        """
        Crea una publicación en el marketplace
        
        Args:
            listing_id: ID único de la publicación
            song_id: ID de la canción
            seller_id: ID del vendedor
            title: Título
            description: Descripción
            price: Precio
            license_type: Tipo de licencia
            tags: Tags (opcional)
            preview_url: URL de preview (opcional)
        
        Returns:
            MarketplaceListing
        """
        listing = MarketplaceListing(
            listing_id=listing_id,
            song_id=song_id,
            seller_id=seller_id,
            title=title,
            description=description,
            price=price,
            license_type=license_type,
            tags=tags or [],
            preview_url=preview_url
        )
        
        self.listings[listing_id] = listing
        logger.info(f"Listing created: {listing_id}")
        return listing
    
    def update_listing(
        self,
        listing_id: str,
        seller_id: str,
        **updates
    ) -> bool:
        """
        Actualiza una publicación
        
        Args:
            listing_id: ID de la publicación
            seller_id: ID del vendedor
            **updates: Campos a actualizar
        
        Returns:
            True si se actualizó exitosamente
        """
        if listing_id not in self.listings:
            return False
        
        listing = self.listings[listing_id]
        if listing.seller_id != seller_id:
            return False
        
        # Actualizar campos permitidos
        allowed_fields = ["title", "description", "price", "tags", "preview_url", "status"]
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(listing, field, value)
        
        listing.updated_at = datetime.now()
        logger.info(f"Listing updated: {listing_id}")
        return True
    
    def purchase(
        self,
        purchase_id: str,
        listing_id: str,
        buyer_id: str
    ) -> Optional[Purchase]:
        """
        Realiza una compra
        
        Args:
            purchase_id: ID único de la compra
            listing_id: ID de la publicación
            buyer_id: ID del comprador
        
        Returns:
            Purchase o None si falla
        """
        if listing_id not in self.listings:
            return None
        
        listing = self.listings[listing_id]
        
        if listing.status != ListingStatus.ACTIVE:
            return None
        
        if listing.seller_id == buyer_id:
            return None  # No puede comprar su propia canción
        
        # Crear compra
        purchase = Purchase(
            purchase_id=purchase_id,
            listing_id=listing_id,
            buyer_id=buyer_id,
            seller_id=listing.seller_id,
            price=listing.price,
            license_type=listing.license_type
        )
        
        self.purchases[purchase_id] = purchase
        
        # Actualizar estadísticas
        listing.purchases += 1
        
        # Si es exclusivo, marcar como vendido
        if listing.license_type == LicenseType.EXCLUSIVE:
            listing.status = ListingStatus.SOLD
        
        logger.info(f"Purchase completed: {purchase_id}")
        return purchase
    
    def add_review(
        self,
        review_id: str,
        listing_id: str,
        user_id: str,
        rating: int,
        comment: str
    ) -> Review:
        """
        Agrega un review
        
        Args:
            review_id: ID único del review
            listing_id: ID de la publicación
            user_id: ID del usuario
            rating: Rating (1-5)
            comment: Comentario
        
        Returns:
            Review
        """
        review = Review(
            review_id=review_id,
            listing_id=listing_id,
            user_id=user_id,
            rating=rating,
            comment=comment
        )
        
        self.reviews[listing_id].append(review)
        
        # Actualizar rating promedio
        if listing_id in self.listings:
            listing = self.listings[listing_id]
            reviews = self.reviews[listing_id]
            listing.rating = sum(r.rating for r in reviews) / len(reviews)
            listing.review_count = len(reviews)
        
        logger.info(f"Review added: {review_id}")
        return review
    
    def search_listings(
        self,
        query: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        license_type: Optional[LicenseType] = None,
        tags: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        limit: int = 20
    ) -> List[MarketplaceListing]:
        """
        Busca publicaciones
        
        Args:
            query: Búsqueda de texto
            min_price: Precio mínimo
            max_price: Precio máximo
            license_type: Tipo de licencia
            tags: Tags
            min_rating: Rating mínimo
            limit: Límite de resultados
        
        Returns:
            Lista de publicaciones
        """
        results = []
        
        for listing in self.listings.values():
            if listing.status != ListingStatus.ACTIVE:
                continue
            
            # Filtros
            if min_price and listing.price < min_price:
                continue
            if max_price and listing.price > max_price:
                continue
            if license_type and listing.license_type != license_type:
                continue
            if min_rating and listing.rating < min_rating:
                continue
            if tags and not any(tag in listing.tags for tag in tags):
                continue
            if query and query.lower() not in listing.title.lower() and query.lower() not in listing.description.lower():
                continue
            
            results.append(listing)
        
        # Ordenar por relevancia (simplificado: por rating y purchases)
        results.sort(key=lambda x: (x.rating, x.purchases), reverse=True)
        
        return results[:limit]
    
    def get_listing(self, listing_id: str) -> Optional[MarketplaceListing]:
        """Obtiene una publicación"""
        listing = self.listings.get(listing_id)
        if listing:
            listing.views += 1
        return listing
    
    def get_user_listings(self, user_id: str) -> List[MarketplaceListing]:
        """Obtiene publicaciones de un usuario"""
        return [
            listing for listing in self.listings.values()
            if listing.seller_id == user_id
        ]
    
    def get_user_purchases(self, user_id: str) -> List[Purchase]:
        """Obtiene compras de un usuario"""
        return [
            purchase for purchase in self.purchases.values()
            if purchase.buyer_id == user_id
        ]


# Instancia global
_marketplace_service: Optional[MarketplaceService] = None


def get_marketplace_service() -> MarketplaceService:
    """Obtiene la instancia global del servicio de marketplace"""
    global _marketplace_service
    if _marketplace_service is None:
        _marketplace_service = MarketplaceService()
    return _marketplace_service

