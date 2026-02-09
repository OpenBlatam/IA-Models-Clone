"""
Marketplace Service - Marketplace de diseños
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ListingStatus(str, Enum):
    """Estados de listing"""
    DRAFT = "draft"
    ACTIVE = "active"
    SOLD = "sold"
    ARCHIVED = "archived"


class MarketplaceService:
    """Servicio para marketplace de diseños"""
    
    def __init__(self):
        self.listings: Dict[str, Dict[str, Any]] = {}
        self.purchases: Dict[str, List[Dict[str, Any]]] = {}
        self.reviews: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_listing(
        self,
        store_id: str,
        seller_id: str,
        price: float,
        title: str,
        description: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Crear listing en marketplace"""
        
        listing_id = f"listing_{len(self.listings) + 1}"
        
        listing = {
            "listing_id": listing_id,
            "store_id": store_id,
            "seller_id": seller_id,
            "title": title,
            "description": description,
            "price": price,
            "category": category or "general",
            "tags": tags or [],
            "status": ListingStatus.DRAFT.value,
            "views": 0,
            "favorites": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.listings[listing_id] = listing
        
        return listing
    
    def publish_listing(self, listing_id: str) -> bool:
        """Publicar listing"""
        listing = self.listings.get(listing_id)
        
        if not listing:
            return False
        
        listing["status"] = ListingStatus.ACTIVE.value
        listing["published_at"] = datetime.now().isoformat()
        listing["updated_at"] = datetime.now().isoformat()
        
        return True
    
    def get_listings(
        self,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "recent"  # "recent", "price_low", "price_high", "popular"
    ) -> List[Dict[str, Any]]:
        """Obtener listings filtrados"""
        listings = [
            l for l in self.listings.values()
            if l["status"] == ListingStatus.ACTIVE.value
        ]
        
        # Filtrar
        if category:
            listings = [l for l in listings if l["category"] == category]
        
        if min_price:
            listings = [l for l in listings if l["price"] >= min_price]
        
        if max_price:
            listings = [l for l in listings if l["price"] <= max_price]
        
        if tags:
            listings = [
                l for l in listings
                if any(tag in l.get("tags", []) for tag in tags)
            ]
        
        # Ordenar
        if sort_by == "price_low":
            listings.sort(key=lambda x: x["price"])
        elif sort_by == "price_high":
            listings.sort(key=lambda x: x["price"], reverse=True)
        elif sort_by == "popular":
            listings.sort(key=lambda x: x.get("views", 0) + x.get("favorites", 0), reverse=True)
        else:  # recent
            listings.sort(key=lambda x: x.get("published_at", x["created_at"]), reverse=True)
        
        return listings
    
    def purchase_listing(
        self,
        listing_id: str,
        buyer_id: str
    ) -> Dict[str, Any]:
        """Comprar listing"""
        listing = self.listings.get(listing_id)
        
        if not listing:
            raise ValueError(f"Listing {listing_id} no encontrado")
        
        if listing["status"] != ListingStatus.ACTIVE.value:
            raise ValueError(f"Listing no está disponible para compra")
        
        if listing["seller_id"] == buyer_id:
            raise ValueError("No puedes comprar tu propio listing")
        
        purchase_id = f"purchase_{len(self.purchases.get(buyer_id, [])) + 1}"
        
        purchase = {
            "purchase_id": purchase_id,
            "listing_id": listing_id,
            "store_id": listing["store_id"],
            "seller_id": listing["seller_id"],
            "buyer_id": buyer_id,
            "price": listing["price"],
            "purchased_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        if buyer_id not in self.purchases:
            self.purchases[buyer_id] = []
        
        self.purchases[buyer_id].append(purchase)
        
        # Actualizar listing
        listing["status"] = ListingStatus.SOLD.value
        listing["sold_at"] = datetime.now().isoformat()
        listing["buyer_id"] = buyer_id
        
        return purchase
    
    def add_review(
        self,
        listing_id: str,
        reviewer_id: str,
        rating: int,
        comment: str
    ) -> Dict[str, Any]:
        """Agregar review a listing"""
        if rating < 1 or rating > 5:
            raise ValueError("Rating debe estar entre 1 y 5")
        
        review = {
            "review_id": f"rev_{len(self.reviews.get(listing_id, [])) + 1}",
            "listing_id": listing_id,
            "reviewer_id": reviewer_id,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now().isoformat()
        }
        
        if listing_id not in self.reviews:
            self.reviews[listing_id] = []
        
        self.reviews[listing_id].append(review)
        
        # Actualizar rating promedio del listing
        self._update_listing_rating(listing_id)
        
        return review
    
    def _update_listing_rating(self, listing_id: str):
        """Actualizar rating promedio"""
        reviews = self.reviews.get(listing_id, [])
        
        if reviews:
            avg_rating = sum(r["rating"] for r in reviews) / len(reviews)
            if listing_id in self.listings:
                self.listings[listing_id]["average_rating"] = round(avg_rating, 2)
                self.listings[listing_id]["reviews_count"] = len(reviews)
    
    def get_listing_details(self, listing_id: str) -> Optional[Dict[str, Any]]:
        """Obtener detalles de listing"""
        listing = self.listings.get(listing_id)
        
        if not listing:
            return None
        
        # Agregar reviews
        listing["reviews"] = self.reviews.get(listing_id, [])
        
        return listing
    
    def favorite_listing(self, listing_id: str, user_id: str) -> bool:
        """Agregar a favoritos"""
        listing = self.listings.get(listing_id)
        
        if not listing:
            return False
        
        listing["favorites"] = listing.get("favorites", 0) + 1
        return True
    
    def view_listing(self, listing_id: str) -> bool:
        """Registrar vista"""
        listing = self.listings.get(listing_id)
        
        if not listing:
            return False
        
        listing["views"] = listing.get("views", 0) + 1
        return True




