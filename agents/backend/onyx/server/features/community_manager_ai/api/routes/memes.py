"""
Memes API Routes
================

Endpoints para gestión de memes.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/memes", tags=["memes"])


class MemeCreate(BaseModel):
    caption: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None


class MemeResponse(BaseModel):
    meme_id: str
    image_path: str
    caption: str
    tags: List[str]
    category: str


def get_meme_manager():
    """Dependency para obtener MemeManager"""
    from ...services.meme_manager import MemeManager
    return MemeManager()


@router.post("/", response_model=MemeResponse)
async def create_meme(
    meme: MemeCreate,
    file: UploadFile = File(...),
    manager = Depends(get_meme_manager)
):
    """Subir un nuevo meme"""
    try:
        # Guardar archivo temporalmente
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        meme_id = manager.add_meme(
            image_path=tmp_path,
            caption=meme.caption,
            tags=meme.tags or [],
            category=meme.category
        )
        
        meme_data = manager.get_meme(meme_id)
        
        # Limpiar archivo temporal
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return MemeResponse(
            meme_id=meme_id,
            image_path=meme_data.get("image_path", ""),
            caption=meme_data.get("caption", ""),
            tags=meme_data.get("tags", []),
            category=meme_data.get("category", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[dict])
async def get_memes(
    category: Optional[str] = None,
    tags: Optional[str] = None,
    query: Optional[str] = None,
    manager = Depends(get_meme_manager)
):
    """Obtener memes"""
    try:
        tag_list = tags.split(",") if tags else None
        memes = manager.search_memes(
            query=query,
            category=category,
            tags=tag_list
        )
        return memes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/random", response_model=dict)
async def get_random_meme(
    category: Optional[str] = None,
    manager = Depends(get_meme_manager)
):
    """Obtener un meme aleatorio"""
    try:
        meme = manager.get_random_meme(category=category)
        if not meme:
            raise HTTPException(status_code=404, detail="No se encontraron memes")
        return meme
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories", response_model=List[str])
async def get_categories(manager = Depends(get_meme_manager)):
    """Obtener lista de categorías"""
    try:
        return manager.get_categories()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{meme_id}")
async def delete_meme(
    meme_id: str,
    manager = Depends(get_meme_manager)
):
    """Eliminar un meme"""
    try:
        success = manager.delete_meme(meme_id)
        if not success:
            raise HTTPException(status_code=404, detail="Meme no encontrado")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




