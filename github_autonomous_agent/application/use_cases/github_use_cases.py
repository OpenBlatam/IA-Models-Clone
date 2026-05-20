"""
GitHub Use Cases

Use cases for GitHub operations with improved validation and error handling.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from config.logging_config import get_logger
from core.github_client import GitHubClient
from core.exceptions import GitHubClientError
from core.validators import RepositoryValidator
from config.settings import settings

logger = get_logger(__name__)


class GetRepositoryInfoUseCase:
    """Use case for getting repository information."""
    
    def __init__(self, github_client: GitHubClient):
        """
        Initialize use case.
        
        Args:
            github_client: GitHub client instance
        """
        self.github_client = github_client
    
    def execute(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Get repository information with improved validation.
        
        Args:
            owner: Repository owner
            repo: Repository name
        
        Returns:
            Repository information dictionary
        
        Raises:
            GitHubClientError: If repository not found or access denied
            ValueError: If owner or repo is invalid
        """
        # Validar repositorio
        try:
            validator = RepositoryValidator(owner=owner, repo=repo)
            validated_owner = validator.owner
            validated_repo = validator.repo
        except ValueError as e:
            logger.warning(f"Validación de repositorio falló: {e}")
            raise ValueError(f"Información de repositorio inválida: {e}") from e
        
        logger.info(f"Obteniendo información del repositorio: {validated_owner}/{validated_repo}")
        
        try:
            repo_info = self.github_client.get_repository_info(validated_owner, validated_repo)
            logger.info(
                f"✅ Información del repositorio obtenida: {validated_owner}/{validated_repo} "
                f"(stars: {repo_info.get('stars', 0)}, language: {repo_info.get('language', 'N/A')})"
            )
            return repo_info
        except GitHubClientError:
            # Re-raise GitHub client errors as-is
            raise
        except Exception as e:
            logger.error(
                f"Error inesperado al obtener información del repositorio "
                f"{validated_owner}/{validated_repo}: {e}",
                exc_info=True
            )
            raise GitHubClientError(
                message="Failed to get repository info",
                owner=validated_owner,
                repo=validated_repo,
                original_error=e
            ) from e


class CloneRepositoryUseCase:
    """Use case for cloning a repository."""
    
    def __init__(self, github_client: GitHubClient):
        """
        Initialize use case.
        
        Args:
            github_client: GitHub client instance
        """
        self.github_client = github_client
    
    def execute(
        self,
        owner: str,
        repo: str,
        local_path: Optional[str] = None
    ) -> str:
        """
        Clone a repository locally with improved validation.
        
        Args:
            owner: Repository owner
            repo: Repository name
            local_path: Optional local path (uses settings.STORAGE_PATH if not provided)
        
        Returns:
            Path to cloned repository
        
        Raises:
            GitHubClientError: If clone fails
            ValueError: If owner, repo, or local_path is invalid
        """
        # Validar repositorio
        try:
            validator = RepositoryValidator(owner=owner, repo=repo)
            validated_owner = validator.owner
            validated_repo = validator.repo
        except ValueError as e:
            logger.warning(f"Validación de repositorio falló: {e}")
            raise ValueError(f"Información de repositorio inválida: {e}") from e
        
        # Validar y normalizar local_path
        if not local_path:
            local_path = settings.STORAGE_PATH
        elif not isinstance(local_path, str) or not local_path.strip():
            raise ValueError("local_path debe ser un string no vacío")
        else:
            local_path = local_path.strip()
        
        logger.info(
            f"Clonando repositorio: {validated_owner}/{validated_repo} "
            f"a {local_path}"
        )
        
        try:
            cloned_path = self.github_client.clone_repository(
                validated_owner,
                validated_repo,
                local_path
            )
            logger.info(
                f"✅ Repositorio clonado exitosamente: {validated_owner}/{validated_repo} "
                f"en {cloned_path}"
            )
            return cloned_path
        except GitHubClientError:
            # Re-raise GitHub client errors as-is
            raise
        except Exception as e:
            logger.error(
                f"Error inesperado al clonar repositorio "
                f"{validated_owner}/{validated_repo}: {e}",
                exc_info=True
            )
            raise GitHubClientError(
                message="Failed to clone repository",
                owner=validated_owner,
                repo=validated_repo,
                details={"local_path": local_path},
                original_error=e
            ) from e

