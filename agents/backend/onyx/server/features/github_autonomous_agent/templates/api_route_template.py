"""
Template para crear nuevos endpoints de API.

Uso:
1. Copia este archivo a api/routes/nombre_ruta.py
2. Reemplaza 'Template' con el nombre de tu ruta
3. Implementa los endpoints necesarios
4. Registra el router en main.py
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel

from api.dependencies import get_current_user  # Ajustar según tu sistema de auth
from core.exceptions import CustomException  # Ajustar según tus excepciones

# Crear router
router = APIRouter(prefix="/template", tags=["template"])


# ============================================================================
# Schemas (Pydantic Models)
# ============================================================================

class TemplateCreate(BaseModel):
    """Schema para crear un template."""
    name: str
    description: Optional[str] = None
    # Agregar más campos según necesidad


class TemplateUpdate(BaseModel):
    """Schema para actualizar un template."""
    name: Optional[str] = None
    description: Optional[str] = None
    # Agregar más campos según necesidad


class TemplateResponse(BaseModel):
    """Schema de respuesta para template."""
    id: str
    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/", response_model=List[TemplateResponse])
async def list_templates(
    skip: int = 0,
    limit: int = 100,
    # current_user = Depends(get_current_user),  # Descomentar si necesitas auth
):
    """
    Lista todos los templates.
    
    Args:
        skip: Número de items a saltar
        limit: Número máximo de items a retornar
        
    Returns:
        Lista de templates
    """
    # TODO: Implementar lógica
    return []


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    # current_user = Depends(get_current_user),
):
    """
    Obtiene un template por ID.
    
    Args:
        template_id: ID del template
        
    Returns:
        Template encontrado
        
    Raises:
        HTTPException: Si el template no existe
    """
    # TODO: Implementar lógica
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Template not found"
    )


@router.post("/", response_model=TemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_template(
    template: TemplateCreate,
    # current_user = Depends(get_current_user),
):
    """
    Crea un nuevo template.
    
    Args:
        template: Datos del template a crear
        
    Returns:
        Template creado
    """
    # TODO: Implementar lógica
    pass


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    template: TemplateUpdate,
    # current_user = Depends(get_current_user),
):
    """
    Actualiza un template existente.
    
    Args:
        template_id: ID del template
        template: Datos a actualizar
        
    Returns:
        Template actualizado
        
    Raises:
        HTTPException: Si el template no existe
    """
    # TODO: Implementar lógica
    pass


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: str,
    # current_user = Depends(get_current_user),
):
    """
    Elimina un template.
    
    Args:
        template_id: ID del template a eliminar
        
    Raises:
        HTTPException: Si el template no existe
    """
    # TODO: Implementar lógica
    pass




