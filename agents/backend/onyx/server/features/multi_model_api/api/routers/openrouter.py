"""
OpenRouter router for Multi-Model API
Handles OpenRouter-specific endpoints
"""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from ...integrations.openrouter import get_openrouter_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multi-model/openrouter", tags=["OpenRouter"])


@router.get("/models")
async def list_openrouter_models(
    provider: Optional[str] = Query(None, description="Filter by provider (e.g., 'openai', 'anthropic')"),
    search: Optional[str] = Query(None, description="Search models by name")
):
    """
    List all available OpenRouter models
    
    Returns:
    - List of all available models from OpenRouter
    - Model details including pricing, context length, etc.
    - Filtered by provider or search term if provided
    
    Args:
        provider: Optional provider filter (e.g., 'openai', 'anthropic')
        search: Optional search term to filter models by name
    """
    try:
        client = get_openrouter_client()
        models = await client.list_models()
        
        if not models:
            return {
                "models": [],
                "total": 0,
                "message": "No models available. Check OPENROUTER_API_KEY configuration."
            }
        
        filtered_models = models
        
        if provider:
            filtered_models = [
                m for m in filtered_models 
                if m.get("id", "").startswith(f"{provider}/")
            ]
        
        if search:
            search_lower = search.lower()
            filtered_models = [
                m for m in filtered_models
                if search_lower in m.get("id", "").lower() or 
                   search_lower in m.get("name", "").lower()
            ]
        
        formatted_models = []
        for model in filtered_models[:100]:  # Limit to 100 results
            formatted_models.append({
                "id": model.get("id"),
                "name": model.get("name"),
                "description": model.get("description"),
                "context_length": model.get("context_length"),
                "pricing": model.get("pricing", {}),
                "architecture": model.get("architecture", {}),
                "top_provider": model.get("top_provider", {}),
                "permission": model.get("permission")
            })
        
        return {
            "models": formatted_models,
            "total": len(formatted_models),
            "filtered": len(formatted_models) < len(models),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing OpenRouter models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list OpenRouter models: {str(e)}"
        )




