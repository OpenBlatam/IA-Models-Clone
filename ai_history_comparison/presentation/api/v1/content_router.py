"""
Content Router - Router de Contenido
Router para endpoints relacionados con contenido
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel, Field

from ....application.use_cases import ManageContentUseCase
from ....application.dto import ContentDTO
from ....application.validators import ContentValidator
from ....core.dependencies import get_content_use_case
from ....core.exceptions import ValidationError, NotFoundError

# Router
router = APIRouter(prefix="/content", tags=["Content Management"])

# Modelos de request/response
class CreateContentRequest(BaseModel):
    """Request para crear contenido"""
    content: str = Field(..., min_length=1, max_length=100000, description="Contenido a analizar")
    title: Optional[str] = Field(None, max_length=200, description="Título del contenido")
    description: Optional[str] = Field(None, max_length=1000, description="Descripción del contenido")
    content_type: str = Field(default="text", description="Tipo de contenido")
    model_version: Optional[str] = Field(None, description="Versión del modelo")
    model_provider: Optional[str] = Field(None, description="Proveedor del modelo")
    tags: List[str] = Field(default_factory=list, description="Tags del contenido")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos adicionales")

class UpdateContentRequest(BaseModel):
    """Request para actualizar contenido"""
    content: Optional[str] = Field(None, min_length=1, max_length=100000, description="Nuevo contenido")
    title: Optional[str] = Field(None, max_length=200, description="Nuevo título")
    description: Optional[str] = Field(None, max_length=1000, description="Nueva descripción")
    tags: Optional[List[str]] = Field(None, description="Nuevos tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Nuevos metadatos")

class ContentResponse(BaseModel):
    """Response de contenido"""
    success: bool = Field(True, description="Indica si la operación fue exitosa")
    data: ContentDTO = Field(..., description="Datos del contenido")
    message: str = Field("Success", description="Mensaje de respuesta")

class ContentListResponse(BaseModel):
    """Response de lista de contenidos"""
    success: bool = Field(True, description="Indica si la operación fue exitosa")
    data: List[ContentDTO] = Field(..., description="Lista de contenidos")
    total: int = Field(..., description="Total de contenidos")
    page: int = Field(..., description="Página actual")
    size: int = Field(..., description="Tamaño de página")
    message: str = Field("Success", description="Mensaje de respuesta")

# Endpoints
@router.post("/", response_model=ContentResponse, summary="Crear contenido")
async def create_content(
    request: CreateContentRequest,
    use_case: ManageContentUseCase = Depends(get_content_use_case)
):
    """
    Crear nuevo contenido
    
    - **content**: Contenido a analizar (requerido)
    - **title**: Título del contenido (opcional)
    - **description**: Descripción del contenido (opcional)
    - **content_type**: Tipo de contenido (default: text)
    - **model_version**: Versión del modelo (opcional)
    - **model_provider**: Proveedor del modelo (opcional)
    - **tags**: Tags del contenido (opcional)
    - **metadata**: Metadatos adicionales (opcional)
    """
    try:
        content_data = request.dict()
        content = await use_case.create_content(content_data)
        
        return ContentResponse(
            data=ContentDTO.from_entity(content),
            message="Content created successfully"
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create content: {str(e)}")

@router.get("/{content_id}", response_model=ContentResponse, summary="Obtener contenido")
async def get_content(
    content_id: str = Path(..., description="ID del contenido"),
    use_case: ManageContentUseCase = Depends(get_content_use_case)
):
    """
    Obtener contenido por ID
    
    - **content_id**: ID único del contenido
    """
    try:
        content = await use_case.get_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return ContentResponse(
            data=ContentDTO.from_entity(content),
            message="Content retrieved successfully"
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Content not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get content: {str(e)}")

@router.put("/{content_id}", response_model=ContentResponse, summary="Actualizar contenido")
async def update_content(
    content_id: str = Path(..., description="ID del contenido"),
    request: UpdateContentRequest = ...,
    use_case: ManageContentUseCase = Depends(get_content_use_case)
):
    """
    Actualizar contenido existente
    
    - **content_id**: ID único del contenido
    - **content**: Nuevo contenido (opcional)
    - **title**: Nuevo título (opcional)
    - **description**: Nueva descripción (opcional)
    - **tags**: Nuevos tags (opcional)
    - **metadata**: Nuevos metadatos (opcional)
    """
    try:
        update_data = request.dict(exclude_unset=True)
        content = await use_case.update_content(content_id, update_data)
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return ContentResponse(
            data=ContentDTO.from_entity(content),
            message="Content updated successfully"
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Content not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update content: {str(e)}")

@router.delete("/{content_id}", summary="Eliminar contenido")
async def delete_content(
    content_id: str = Path(..., description="ID del contenido"),
    use_case: ManageContentUseCase = Depends(get_content_use_case)
):
    """
    Eliminar contenido
    
    - **content_id**: ID único del contenido
    """
    try:
        success = await use_case.delete_content(content_id)
        if not success:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {"success": True, "message": "Content deleted successfully"}
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Content not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete content: {str(e)}")

@router.get("/", response_model=ContentListResponse, summary="Listar contenidos")
async def list_contents(
    page: int = Query(1, ge=1, description="Número de página"),
    size: int = Query(20, ge=1, le=100, description="Tamaño de página"),
    content_type: Optional[str] = Query(None, description="Filtrar por tipo de contenido"),
    status: Optional[str] = Query(None, description="Filtrar por estado"),
    search: Optional[str] = Query(None, description="Buscar en contenido"),
    use_case: ManageContentUseCase = Depends(get_content_use_case)
):
    """
    Listar contenidos con paginación y filtros
    
    - **page**: Número de página (default: 1)
    - **size**: Tamaño de página (default: 20, max: 100)
    - **content_type**: Filtrar por tipo de contenido (opcional)
    - **status**: Filtrar por estado (opcional)
    - **search**: Buscar en contenido (opcional)
    """
    try:
        filters = {
            "content_type": content_type,
            "status": status,
            "search": search
        }
        
        contents, total = await use_case.list_contents(
            page=page,
            size=size,
            filters=filters
        )
        
        return ContentListResponse(
            data=[ContentDTO.from_entity(content) for content in contents],
            total=total,
            page=page,
            size=size,
            message="Contents retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list contents: {str(e)}")

@router.get("/{content_id}/summary", summary="Obtener resumen de contenido")
async def get_content_summary(
    content_id: str = Path(..., description="ID del contenido"),
    max_length: int = Query(100, ge=10, le=1000, description="Longitud máxima del resumen"),
    use_case: ManageContentUseCase = Depends(get_content_use_case)
):
    """
    Obtener resumen del contenido
    
    - **content_id**: ID único del contenido
    - **max_length**: Longitud máxima del resumen (default: 100)
    """
    try:
        content = await use_case.get_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        summary = content.get_summary(max_length)
        
        return {
            "success": True,
            "data": {
                "content_id": content_id,
                "summary": summary,
                "word_count": content.word_count,
                "character_count": content.character_count,
                "max_length": max_length
            },
            "message": "Content summary retrieved successfully"
        }
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Content not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get content summary: {str(e)}")

@router.post("/{content_id}/tags", summary="Agregar tag a contenido")
async def add_content_tag(
    content_id: str = Path(..., description="ID del contenido"),
    tag: str = Query(..., min_length=1, max_length=50, description="Tag a agregar"),
    use_case: ManageContentUseCase = Depends(get_content_use_case)
):
    """
    Agregar tag a contenido
    
    - **content_id**: ID único del contenido
    - **tag**: Tag a agregar
    """
    try:
        content = await use_case.add_content_tag(content_id, tag)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "success": True,
            "data": ContentDTO.from_entity(content),
            "message": "Tag added successfully"
        }
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Content not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add tag: {str(e)}")

@router.delete("/{content_id}/tags/{tag}", summary="Eliminar tag de contenido")
async def remove_content_tag(
    content_id: str = Path(..., description="ID del contenido"),
    tag: str = Path(..., description="Tag a eliminar"),
    use_case: ManageContentUseCase = Depends(get_content_use_case)
):
    """
    Eliminar tag de contenido
    
    - **content_id**: ID único del contenido
    - **tag**: Tag a eliminar
    """
    try:
        content = await use_case.remove_content_tag(content_id, tag)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "success": True,
            "data": ContentDTO.from_entity(content),
            "message": "Tag removed successfully"
        }
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Content not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove tag: {str(e)}")







