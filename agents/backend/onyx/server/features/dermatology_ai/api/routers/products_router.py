"""
Products Router - Handles product-related endpoints
"""

from fastapi import APIRouter, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json

from ...api.services_locator import get_service
from ...utils.logger import logger

router = APIRouter(prefix="/dermatology", tags=["products"])


@router.post("/products/search")
async def search_products(
    query: str = Form(...),
    category: Optional[str] = Form(None),
    limit: int = Form(20)
):
    """Busca productos"""
    try:
        product_database = get_service("product_database")
        products = product_database.search_products(query, category, limit)
        return JSONResponse(content={
            "success": True,
            "products": [p.to_dict() if hasattr(p, 'to_dict') else p for p in products]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/products/{product_id}")
async def get_product(product_id: str):
    """Obtiene información de un producto"""
    try:
        product_database = get_service("product_database")
        product = product_database.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Producto no encontrado")
        return JSONResponse(content={
            "success": True,
            "product": product.to_dict() if hasattr(product, 'to_dict') else product
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/products/track")
async def track_product(
    user_id: str = Form(...),
    product_id: str = Form(...),
    usage_date: str = Form(...),
    amount_used: Optional[float] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Registra uso de producto"""
    try:
        product_tracker = get_service("product_tracker")
        usage = product_tracker.track_usage(user_id, product_id, usage_date, amount_used, notes)
        return JSONResponse(content={"success": True, "usage": usage.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/products/insights/{user_id}/{product_id}")
async def get_product_insights(user_id: str, product_id: str):
    """Obtiene insights de uso de producto"""
    try:
        product_tracker = get_service("product_tracker")
        insights = product_tracker.get_product_insights(user_id, product_id)
        return JSONResponse(content={"success": True, "insights": insights.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/products/compare")
async def compare_products(
    product_ids: str = Form(...)
):
    """Compara productos"""
    try:
        product_comparison = get_service("product_comparison")
        ids_list = json.loads(product_ids)
        comparison = product_comparison.compare_products(ids_list)
        return JSONResponse(content={
            "success": True,
            "comparison": comparison.to_dict() if hasattr(comparison, 'to_dict') else comparison
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-compatibility/register")
async def register_product_compatibility(
    product_id: str = Form(...),
    name: str = Form(...),
    ingredients: str = Form(...)
):
    """Registra producto para análisis de compatibilidad"""
    try:
        product_compatibility = get_service("product_compatibility")
        product = product_compatibility.register_product(product_id, name, json.loads(ingredients))
        return JSONResponse(content={"success": True, "product": product.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/product-compatibility/check")
async def check_product_compatibility(
    product_ids: str = Form(...)
):
    """Verifica compatibilidad entre productos"""
    try:
        product_compatibility = get_service("product_compatibility")
        ids_list = json.loads(product_ids)
        report = product_compatibility.check_compatibility(ids_list)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")




