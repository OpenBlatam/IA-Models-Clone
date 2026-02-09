"""
API de Marketplace

Endpoints para:
- Crear/actualizar publicaciones
- Buscar publicaciones
- Comprar canciones
- Agregar reviews
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.marketplace import (
    get_marketplace_service,
    LicenseType,
    ListingStatus
)
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/marketplace",
    tags=["marketplace"]
)


@router.post("/listings")
async def create_listing(
    song_id: str = Body(..., description="ID de la canción"),
    title: str = Body(..., description="Título"),
    description: str = Body(..., description="Descripción"),
    price: float = Body(..., description="Precio"),
    license_type: str = Body(..., description="Tipo de licencia"),
    tags: Optional[List[str]] = Body(None, description="Tags"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea una publicación en el marketplace.
    """
    try:
        import uuid
        listing_id = str(uuid.uuid4())
        seller_id = current_user.get("user_id", "unknown")
        
        try:
            license_enum = LicenseType(license_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid license type: {license_type}"
            )
        
        service = get_marketplace_service()
        listing = service.create_listing(
            listing_id=listing_id,
            song_id=song_id,
            seller_id=seller_id,
            title=title,
            description=description,
            price=price,
            license_type=license_enum,
            tags=tags
        )
        
        return {
            "listing_id": listing.listing_id,
            "song_id": listing.song_id,
            "title": listing.title,
            "price": listing.price,
            "license_type": listing.license_type.value,
            "status": listing.status.value
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating listing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating listing: {str(e)}"
        )


@router.get("/listings/search")
async def search_listings(
    query: Optional[str] = Query(None, description="Búsqueda de texto"),
    min_price: Optional[float] = Query(None, description="Precio mínimo"),
    max_price: Optional[float] = Query(None, description="Precio máximo"),
    license_type: Optional[str] = Query(None, description="Tipo de licencia"),
    tags: Optional[str] = Query(None, description="Tags (comma-separated)"),
    min_rating: Optional[float] = Query(None, description="Rating mínimo"),
    limit: int = Query(20, description="Límite de resultados")
) -> Dict[str, Any]:
    """
    Busca publicaciones en el marketplace.
    """
    try:
        service = get_marketplace_service()
        
        license_enum = None
        if license_type:
            try:
                license_enum = LicenseType(license_type)
            except ValueError:
                pass
        
        tags_list = None
        if tags:
            tags_list = [t.strip() for t in tags.split(",")]
        
        results = service.search_listings(
            query=query,
            min_price=min_price,
            max_price=max_price,
            license_type=license_enum,
            tags=tags_list,
            min_rating=min_rating,
            limit=limit
        )
        
        return {
            "listings": [
                {
                    "listing_id": l.listing_id,
                    "title": l.title,
                    "description": l.description,
                    "price": l.price,
                    "license_type": l.license_type.value,
                    "rating": l.rating,
                    "review_count": l.review_count,
                    "tags": l.tags
                }
                for l in results
            ],
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching listings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching listings: {str(e)}"
        )


@router.post("/listings/{listing_id}/purchase")
async def purchase_listing(
    listing_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Compra una publicación.
    """
    try:
        import uuid
        purchase_id = str(uuid.uuid4())
        buyer_id = current_user.get("user_id", "unknown")
        
        service = get_marketplace_service()
        purchase = service.purchase(purchase_id, listing_id, buyer_id)
        
        if not purchase:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Purchase failed"
            )
        
        return {
            "purchase_id": purchase.purchase_id,
            "listing_id": purchase.listing_id,
            "price": purchase.price,
            "status": purchase.status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error purchasing listing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error purchasing listing: {str(e)}"
        )


@router.post("/listings/{listing_id}/reviews")
async def add_review(
    listing_id: str,
    rating: int = Body(..., ge=1, le=5, description="Rating (1-5)"),
    comment: str = Body(..., description="Comentario"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Agrega un review a una publicación.
    """
    try:
        import uuid
        review_id = str(uuid.uuid4())
        user_id = current_user.get("user_id", "unknown")
        
        service = get_marketplace_service()
        review = service.add_review(review_id, listing_id, user_id, rating, comment)
        
        return {
            "review_id": review.review_id,
            "rating": review.rating,
            "comment": review.comment
        }
    except Exception as e:
        logger.error(f"Error adding review: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding review: {str(e)}"
        )

