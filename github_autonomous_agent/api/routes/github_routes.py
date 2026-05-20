"""
GitHub Routes - Rutas para interacción con GitHub.
"""

from fastapi import APIRouter, Depends
from typing import Optional

from api.utils import handle_api_errors, validate_github_token
from api.validators import validate_repository
from api.dependencies import (
    get_get_repository_info_use_case,
    get_clone_repository_use_case
)
from api.schemas import RepositoryInfoResponse
from application.use_cases.github_use_cases import (
    GetRepositoryInfoUseCase,
    CloneRepositoryUseCase
)
from config.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/repository/{owner}/{repo}", response_model=RepositoryInfoResponse)
@handle_api_errors
async def get_repository_info(
    owner: str,
    repo: str,
    use_case: GetRepositoryInfoUseCase = Depends(get_get_repository_info_use_case)
):
    """
    Obtener información de un repositorio de GitHub.
    
    Args:
        owner: Propietario del repositorio
        repo: Nombre del repositorio
        
    Returns:
        RepositoryInfoResponse: Información del repositorio
        
    Raises:
        HTTPException: Si el repositorio no existe o hay un error al obtenerlo
    """
    validate_github_token()
    
    # Validar parámetros
    validated_owner, validated_repo = validate_repository(owner, repo)
    
    logger.info(f"Obteniendo información del repositorio {validated_owner}/{validated_repo}")
    
    repo_info = use_case.execute(validated_owner, validated_repo)
    
    logger.debug(f"Información del repositorio {validated_owner}/{validated_repo} obtenida exitosamente")
    return RepositoryInfoResponse(**repo_info)


@router.post("/repository/{owner}/{repo}/clone")
@handle_api_errors
async def clone_repository(
    owner: str,
    repo: str,
    local_path: Optional[str] = None,
    use_case: CloneRepositoryUseCase = Depends(get_clone_repository_use_case)
):
    """
    Clonar un repositorio de GitHub localmente.
    
    Args:
        owner: Propietario del repositorio
        repo: Nombre del repositorio
        local_path: Ruta local donde clonar (opcional, usa la configuración por defecto si no se proporciona)
        
    Returns:
        dict: Mensaje de confirmación y ruta local del repositorio clonado
        
    Raises:
        HTTPException: Si hay un error al clonar el repositorio
    """
    validate_github_token()
    
    # Validar parámetros
    validated_owner, validated_repo = validate_repository(owner, repo)
    
    logger.info(f"Clonando repositorio {validated_owner}/{validated_repo} a {local_path or 'ruta por defecto'}")
    
    cloned_path = use_case.execute(validated_owner, validated_repo, local_path)
    
    logger.info(f"Repositorio {validated_owner}/{validated_repo} clonado exitosamente en {cloned_path}")
    return {
        "message": "Repositorio clonado exitosamente",
        "local_path": cloned_path
    }
