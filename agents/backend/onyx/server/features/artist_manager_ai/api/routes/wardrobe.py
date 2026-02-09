"""
Wardrobe API Routes
===================

Endpoints para gestión de vestimenta.
"""

import os
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

router = APIRouter(prefix="/wardrobe", tags=["wardrobe"])


class WardrobeItemCreate(BaseModel):
    name: str
    category: str
    color: str
    brand: Optional[str] = None
    size: Optional[str] = None
    season: str = "all_season"
    dress_codes: List[str] = None
    notes: Optional[str] = None
    image_url: Optional[str] = None


class OutfitCreate(BaseModel):
    name: str
    items: List[str]  # IDs de items
    dress_code: str
    occasion: str
    season: str = "all_season"
    notes: Optional[str] = None
    image_url: Optional[str] = None


def get_artist_manager(artist_id: str):
    """Dependency para obtener ArtistManager."""
    from ...core.artist_manager import ArtistManager
    from ...core.wardrobe_manager import WardrobeItem, Outfit, DressCode, Season
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    manager = ArtistManager(artist_id=artist_id, openrouter_api_key=openrouter_key)
    return manager, WardrobeItem, Outfit, DressCode, Season


@router.post("/{artist_id}/items", response_model=Dict[str, Any])
async def create_item(artist_id: str, item: WardrobeItemCreate):
    """Crear nuevo item de vestimenta."""
    try:
        manager, WardrobeItem, _, DressCode, Season = get_artist_manager(artist_id)
        
        import uuid
        item_id = str(uuid.uuid4())
        
        season = Season(item.season) if item.season in [s.value for s in Season] else Season.ALL_SEASON
        dress_codes = [DressCode(dc) for dc in (item.dress_codes or []) if dc in [d.value for d in DressCode]]
        
        wardrobe_item = WardrobeItem(
            id=item_id,
            name=item.name,
            category=item.category,
            color=item.color,
            brand=item.brand,
            size=item.size,
            season=season,
            dress_codes=dress_codes,
            notes=item.notes,
            image_url=item.image_url
        )
        
        created_item = manager.wardrobe.add_item(wardrobe_item)
        return created_item.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/items", response_model=List[Dict[str, Any]])
async def get_items(
    artist_id: str,
    category: Optional[str] = None,
    dress_code: Optional[str] = None
):
    """Obtener items de vestimenta."""
    try:
        manager, _, _, DressCode, _ = get_artist_manager(artist_id)
        
        if category:
            items = manager.wardrobe.get_items_by_category(category)
        elif dress_code:
            dc = DressCode(dress_code)
            items = manager.wardrobe.get_items_by_dress_code(dc)
        else:
            items = manager.wardrobe.get_all_items()
        
        return [item.to_dict() for item in items]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/items/{item_id}", response_model=Dict[str, Any])
async def get_item(artist_id: str, item_id: str):
    """Obtener item específico."""
    try:
        manager, _, _, _, _ = get_artist_manager(artist_id)
        item = manager.wardrobe.get_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{artist_id}/items/{item_id}", response_model=Dict[str, Any])
async def update_item(artist_id: str, item_id: str, item_update: Dict[str, Any]):
    """Actualizar item."""
    try:
        manager, _, _, _, _ = get_artist_manager(artist_id)
        updated_item = manager.wardrobe.update_item(item_id, **item_update)
        return updated_item.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{artist_id}/items/{item_id}")
async def delete_item(artist_id: str, item_id: str):
    """Eliminar item."""
    try:
        manager, _, _, _, _ = get_artist_manager(artist_id)
        success = manager.wardrobe.delete_item(item_id)
        if not success:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"status": "deleted", "item_id": item_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artist_id}/outfits", response_model=Dict[str, Any])
async def create_outfit(artist_id: str, outfit: OutfitCreate):
    """Crear nuevo outfit."""
    try:
        manager, _, Outfit, DressCode, Season = get_artist_manager(artist_id)
        
        import uuid
        outfit_id = str(uuid.uuid4())
        
        dress_code = DressCode(outfit.dress_code) if outfit.dress_code in [dc.value for dc in DressCode] else DressCode.CASUAL
        season = Season(outfit.season) if outfit.season in [s.value for s in Season] else Season.ALL_SEASON
        
        outfit_obj = Outfit(
            id=outfit_id,
            name=outfit.name,
            items=outfit.items,
            dress_code=dress_code,
            occasion=outfit.occasion,
            season=season,
            notes=outfit.notes,
            image_url=outfit.image_url
        )
        
        created_outfit = manager.wardrobe.add_outfit(outfit_obj)
        return created_outfit.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/outfits", response_model=List[Dict[str, Any]])
async def get_outfits(
    artist_id: str,
    dress_code: Optional[str] = None
):
    """Obtener outfits."""
    try:
        manager, _, _, DressCode, _ = get_artist_manager(artist_id)
        
        if dress_code:
            dc = DressCode(dress_code)
            outfits = manager.wardrobe.get_outfits_by_dress_code(dc)
        else:
            outfits = manager.wardrobe.get_all_outfits()
        
        return [outfit.to_dict() for outfit in outfits]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artist_id}/items/{item_id}/mark-worn", response_model=Dict[str, Any])
async def mark_item_worn(artist_id: str, item_id: str):
    """Marcar item como usado."""
    try:
        manager, _, _, _, _ = get_artist_manager(artist_id)
        item = manager.wardrobe.mark_item_worn(item_id)
        return item.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artist_id}/outfits/{outfit_id}/mark-worn", response_model=Dict[str, Any])
async def mark_outfit_worn(artist_id: str, outfit_id: str):
    """Marcar outfit como usado."""
    try:
        manager, _, _, _, _ = get_artist_manager(artist_id)
        outfit = manager.wardrobe.mark_outfit_worn(outfit_id)
        return outfit.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




