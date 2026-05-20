"""
Marketplace - Sistema de marketplace
=====================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ListingStatus(str, Enum):
    """Estados de listing"""
    DRAFT = "draft"
    ACTIVE = "active"
    SOLD = "sold"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class Marketplace:
    """Sistema de marketplace"""
    
    def __init__(self):
        self.listings: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.reviews: Dict[str, List[Dict[str, Any]]] = {}
        self.categories: List[str] = ["prototypes", "materials", "templates", "services"]
    
    def create_listing(self, listing_id: str, seller_id: str, title: str,
                     description: str, price: float, category: str,
                     item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crea un listing"""
        listing = {
            "id": listing_id,
            "seller_id": seller_id,
            "title": title,
            "description": description,
            "price": price,
            "category": category,
            "item_data": item_data,
            "status": ListingStatus.DRAFT.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "views": 0,
            "favorites": 0
        }
        
        self.listings[listing_id] = listing
        
        logger.info(f"Listing creado: {listing_id} por {seller_id}")
        return listing
    
    def publish_listing(self, listing_id: str) -> Dict[str, Any]:
        """Publica un listing"""
        listing = self.listings.get(listing_id)
        if not listing:
            raise ValueError(f"Listing no encontrado: {listing_id}")
        
        listing["status"] = ListingStatus.ACTIVE.value
        listing["published_at"] = datetime.now().isoformat()
        listing["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Listing publicado: {listing_id}")
        return listing
    
    def search_listings(self, query: Optional[str] = None,
                       category: Optional[str] = None,
                       min_price: Optional[float] = None,
                       max_price: Optional[float] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
        """Busca listings"""
        results = [
            l for l in self.listings.values()
            if l["status"] == ListingStatus.ACTIVE.value
        ]
        
        if query:
            query_lower = query.lower()
            results = [
                l for l in results
                if query_lower in l["title"].lower() or query_lower in l["description"].lower()
            ]
        
        if category:
            results = [l for l in results if l["category"] == category]
        
        if min_price:
            results = [l for l in results if l["price"] >= min_price]
        
        if max_price:
            results = [l for l in results if l["price"] <= max_price]
        
        # Ordenar por relevancia/views
        results.sort(key=lambda x: x["views"], reverse=True)
        
        return results[:limit]
    
    def create_order(self, order_id: str, buyer_id: str, listing_id: str,
                    quantity: int = 1) -> Dict[str, Any]:
        """Crea una orden"""
        listing = self.listings.get(listing_id)
        if not listing:
            raise ValueError(f"Listing no encontrado: {listing_id}")
        
        if listing["status"] != ListingStatus.ACTIVE.value:
            raise ValueError(f"Listing no está disponible: {listing_id}")
        
        order = {
            "id": order_id,
            "buyer_id": buyer_id,
            "seller_id": listing["seller_id"],
            "listing_id": listing_id,
            "quantity": quantity,
            "total_price": listing["price"] * quantity,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        self.orders[order_id] = order
        
        logger.info(f"Orden creada: {order_id}")
        return order
    
    def add_review(self, listing_id: str, user_id: str, rating: float,
                  comment: str) -> Dict[str, Any]:
        """Agrega una reseña"""
        if listing_id not in self.reviews:
            self.reviews[listing_id] = []
        
        review = {
            "user_id": user_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now().isoformat()
        }
        
        self.reviews[listing_id].append(review)
        
        # Actualizar rating promedio del listing
        listing = self.listings.get(listing_id)
        if listing:
            all_ratings = [r["rating"] for r in self.reviews[listing_id]]
            listing["average_rating"] = sum(all_ratings) / len(all_ratings) if all_ratings else 0
            listing["review_count"] = len(all_ratings)
        
        logger.info(f"Reseña agregada para listing: {listing_id}")
        return review
    
    def get_listing_details(self, listing_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene detalles de un listing"""
        listing = self.listings.get(listing_id)
        if not listing:
            return None
        
        # Incrementar views
        listing["views"] += 1
        
        # Agregar reviews
        listing["reviews"] = self.reviews.get(listing_id, [])
        
        return listing
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del marketplace"""
        active_listings = sum(
            1 for l in self.listings.values()
            if l["status"] == ListingStatus.ACTIVE.value
        )
        
        total_orders = len(self.orders)
        completed_orders = sum(
            1 for o in self.orders.values()
            if o["status"] == "completed"
        )
        
        return {
            "total_listings": len(self.listings),
            "active_listings": active_listings,
            "total_orders": total_orders,
            "completed_orders": completed_orders,
            "categories": self.categories
        }




