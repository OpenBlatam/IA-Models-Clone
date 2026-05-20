"""
Wardrobe Manager
================

Gestión de vestimenta y estilo para artistas.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DressCode(Enum):
    """Códigos de vestimenta."""
    FORMAL = "formal"
    BUSINESS_CASUAL = "business_casual"
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    BLACK_TIE = "black_tie"
    COCKTAIL = "cocktail"
    STREETWEAR = "streetwear"
    ARTISTIC = "artistic"
    CUSTOM = "custom"


class Season(Enum):
    """Estaciones."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all_season"


@dataclass
class WardrobeItem:
    """Item del guardarropa."""
    id: str
    name: str
    category: str  # e.g., "shirt", "pants", "shoes", "accessories"
    color: str
    brand: Optional[str] = None
    size: Optional[str] = None
    season: Season = Season.ALL_SEASON
    dress_codes: List[DressCode] = None
    notes: Optional[str] = None
    image_url: Optional[str] = None
    last_worn: Optional[datetime] = None
    
    def __post_init__(self):
        if self.dress_codes is None:
            self.dress_codes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['season'] = self.season.value
        data['dress_codes'] = [dc.value for dc in self.dress_codes]
        if self.last_worn:
            data['last_worn'] = self.last_worn.isoformat()
        return data


@dataclass
class Outfit:
    """Outfit completo."""
    id: str
    name: str
    items: List[str]  # IDs de WardrobeItem
    dress_code: DressCode
    occasion: str
    season: Season
    notes: Optional[str] = None
    image_url: Optional[str] = None
    last_worn: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['dress_code'] = self.dress_code.value
        data['season'] = self.season.value
        if self.last_worn:
            data['last_worn'] = self.last_worn.isoformat()
        return data


@dataclass
class WardrobeRecommendation:
    """Recomendación de vestimenta."""
    id: str
    event_id: Optional[str]
    occasion: str
    dress_code: DressCode
    recommended_outfit_id: Optional[str] = None
    recommended_items: List[str] = None  # IDs de items
    reasoning: str = ""
    weather_considerations: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.recommended_items is None:
            self.recommended_items = []
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['dress_code'] = self.dress_code.value
        data['created_at'] = self.created_at.isoformat()
        return data


class WardrobeManager:
    """Gestor de vestimenta para artistas."""
    
    def __init__(self, artist_id: str):
        """
        Inicializar gestor de vestimenta.
        
        Args:
            artist_id: ID del artista
        """
        self.artist_id = artist_id
        self.items: Dict[str, WardrobeItem] = {}
        self.outfits: Dict[str, Outfit] = {}
        self.recommendations: List[WardrobeRecommendation] = []
        self._logger = logger
    
    def add_item(self, item: WardrobeItem) -> WardrobeItem:
        """
        Agregar item al guardarropa.
        
        Args:
            item: Item a agregar
        
        Returns:
            Item agregado
        """
        if item.id in self.items:
            raise ValueError(f"Item with id {item.id} already exists")
        
        self.items[item.id] = item
        self._logger.info(f"Added wardrobe item {item.id} for artist {self.artist_id}")
        return item
    
    def get_item(self, item_id: str) -> Optional[WardrobeItem]:
        """
        Obtener item por ID.
        
        Args:
            item_id: ID del item
        
        Returns:
            Item o None si no existe
        """
        return self.items.get(item_id)
    
    def update_item(self, item_id: str, **updates) -> WardrobeItem:
        """
        Actualizar item.
        
        Args:
            item_id: ID del item
            **updates: Campos a actualizar
        
        Returns:
            Item actualizado
        """
        if item_id not in self.items:
            raise ValueError(f"Item {item_id} not found")
        
        item = self.items[item_id]
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        self._logger.info(f"Updated wardrobe item {item_id} for artist {self.artist_id}")
        return item
    
    def delete_item(self, item_id: str) -> bool:
        """
        Eliminar item.
        
        Args:
            item_id: ID del item
        
        Returns:
            True si se eliminó, False si no existía
        """
        if item_id in self.items:
            del self.items[item_id]
            self._logger.info(f"Deleted wardrobe item {item_id} for artist {self.artist_id}")
            return True
        return False
    
    def get_items_by_category(self, category: str) -> List[WardrobeItem]:
        """
        Obtener items por categoría.
        
        Args:
            category: Categoría del item
        
        Returns:
            Lista de items de la categoría
        """
        return [
            item for item in self.items.values()
            if item.category.lower() == category.lower()
        ]
    
    def get_items_by_dress_code(self, dress_code: DressCode) -> List[WardrobeItem]:
        """
        Obtener items por código de vestimenta.
        
        Args:
            dress_code: Código de vestimenta
        
        Returns:
            Lista de items apropiados para el código
        """
        return [
            item for item in self.items.values()
            if dress_code in item.dress_codes
        ]
    
    def add_outfit(self, outfit: Outfit) -> Outfit:
        """
        Agregar outfit.
        
        Args:
            outfit: Outfit a agregar
        
        Returns:
            Outfit agregado
        """
        if outfit.id in self.outfits:
            raise ValueError(f"Outfit with id {outfit.id} already exists")
        
        # Verificar que todos los items existan
        for item_id in outfit.items:
            if item_id not in self.items:
                raise ValueError(f"Item {item_id} not found in wardrobe")
        
        self.outfits[outfit.id] = outfit
        self._logger.info(f"Added outfit {outfit.id} for artist {self.artist_id}")
        return outfit
    
    def get_outfit(self, outfit_id: str) -> Optional[Outfit]:
        """
        Obtener outfit por ID.
        
        Args:
            outfit_id: ID del outfit
        
        Returns:
            Outfit o None si no existe
        """
        return self.outfits.get(outfit_id)
    
    def get_outfits_by_dress_code(self, dress_code: DressCode) -> List[Outfit]:
        """
        Obtener outfits por código de vestimenta.
        
        Args:
            dress_code: Código de vestimenta
        
        Returns:
            Lista de outfits apropiados
        """
        return [
            outfit for outfit in self.outfits.values()
            if outfit.dress_code == dress_code
        ]
    
    def create_recommendation(
        self,
        occasion: str,
        dress_code: DressCode,
        event_id: Optional[str] = None,
        reasoning: str = "",
        weather_considerations: Optional[str] = None
    ) -> WardrobeRecommendation:
        """
        Crear recomendación de vestimenta.
        
        Args:
            occasion: Ocasión del evento
            dress_code: Código de vestimenta requerido
            event_id: ID del evento relacionado (opcional)
            reasoning: Razón de la recomendación
            weather_considerations: Consideraciones del clima
        
        Returns:
            Recomendación creada
        """
        recommendation_id = f"rec_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Buscar outfits apropiados
        appropriate_outfits = self.get_outfits_by_dress_code(dress_code)
        recommended_outfit_id = appropriate_outfits[0].id if appropriate_outfits else None
        
        # Buscar items apropiados
        appropriate_items = self.get_items_by_dress_code(dress_code)
        recommended_items = [item.id for item in appropriate_items[:5]]  # Top 5
        
        recommendation = WardrobeRecommendation(
            id=recommendation_id,
            event_id=event_id,
            occasion=occasion,
            dress_code=dress_code,
            recommended_outfit_id=recommended_outfit_id,
            recommended_items=recommended_items,
            reasoning=reasoning,
            weather_considerations=weather_considerations
        )
        
        self.recommendations.append(recommendation)
        self._logger.info(f"Created wardrobe recommendation {recommendation_id} for artist {self.artist_id}")
        return recommendation
    
    def get_recommendations_for_event(self, event_id: str) -> List[WardrobeRecommendation]:
        """
        Obtener recomendaciones para un evento.
        
        Args:
            event_id: ID del evento
        
        Returns:
            Lista de recomendaciones
        """
        return [
            rec for rec in self.recommendations
            if rec.event_id == event_id
        ]
    
    def mark_item_worn(self, item_id: str) -> WardrobeItem:
        """
        Marcar item como usado.
        
        Args:
            item_id: ID del item
        
        Returns:
            Item actualizado
        """
        if item_id not in self.items:
            raise ValueError(f"Item {item_id} not found")
        
        self.items[item_id].last_worn = datetime.now()
        self._logger.info(f"Marked item {item_id} as worn for artist {self.artist_id}")
        return self.items[item_id]
    
    def mark_outfit_worn(self, outfit_id: str) -> Outfit:
        """
        Marcar outfit como usado.
        
        Args:
            outfit_id: ID del outfit
        
        Returns:
            Outfit actualizado
        """
        if outfit_id not in self.outfits:
            raise ValueError(f"Outfit {outfit_id} not found")
        
        self.outfits[outfit_id].last_worn = datetime.now()
        # Marcar todos los items del outfit como usados
        for item_id in self.outfits[outfit_id].items:
            if item_id in self.items:
                self.items[item_id].last_worn = datetime.now()
        
        self._logger.info(f"Marked outfit {outfit_id} as worn for artist {self.artist_id}")
        return self.outfits[outfit_id]
    
    def get_all_items(self) -> List[WardrobeItem]:
        """Obtener todos los items."""
        return list(self.items.values())
    
    def get_all_outfits(self) -> List[Outfit]:
        """Obtener todos los outfits."""
        return list(self.outfits.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "artist_id": self.artist_id,
            "items": [item.to_dict() for item in self.get_all_items()],
            "outfits": [outfit.to_dict() for outfit in self.get_all_outfits()],
            "total_items": len(self.items),
            "total_outfits": len(self.outfits),
            "total_recommendations": len(self.recommendations)
        }




