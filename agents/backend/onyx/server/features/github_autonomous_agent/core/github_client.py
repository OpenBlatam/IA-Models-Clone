"""
GitHub Client - Cliente para interactuar con la API de GitHub.
"""

from typing import Optional, Dict, Any, List
from github import Github
from github.Repository import Repository
from github.GithubException import GithubException

from config.logging_config import get_logger
from config.settings import settings
from core.exceptions import GitHubClientError
from core.utils import handle_github_exception
from core.constants import GitConfig
from core.retry_utils import retry_on_github_error

logger = get_logger(__name__)


class GitHubClient:
    """Cliente para interactuar con la API de GitHub."""

    def __init__(self, token: Optional[str] = None):
        """
        Inicializar cliente de GitHub.

        Args:
            token: Token de autenticación de GitHub (opcional, usa el de settings si no se proporciona)
        """
        self.token = token or settings.GITHUB_TOKEN
        if not self.token:
            raise GitHubClientError("GitHub token es requerido")
        self.github = Github(self.token)


    @retry_on_github_error(max_attempts=3)
    @handle_github_exception
    def get_repository(self, owner: str, repo: str) -> Repository:
        """
        Obtener un repositorio con reintentos automáticos.

        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio

        Returns:
            Objeto Repository de GitHub
            
        Raises:
            GitHubClientError: Si no se puede obtener el repositorio
        """
        try:
            return self.github.get_repo(f"{owner}/{repo}")
        except GithubException as e:
            logger.error(f"Error al obtener repositorio {owner}/{repo}: {e}")
            raise GitHubClientError(f"Error al obtener repositorio: {e}") from e

    def clone_repository(self, owner: str, repo: str, local_path: str) -> str:
        """
        Clonar un repositorio localmente con validaciones mejoradas.

        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio
            local_path: Ruta local donde clonar

        Returns:
            Ruta al repositorio clonado
            
        Raises:
            GitHubClientError: Si no se puede clonar el repositorio
        """
        import subprocess
        import os
        from pathlib import Path

        if not owner or not repo:
            raise GitHubClientError("El propietario y nombre del repositorio son requeridos")
        
        if not local_path:
            raise GitHubClientError("La ruta local es requerida")

        repo_url = f"https://{self.token}@github.com/{owner}/{repo}.git"
        local_repo_path = Path(local_path) / repo

        # Crear directorio padre si no existe
        local_repo_path.parent.mkdir(parents=True, exist_ok=True)

        if local_repo_path.exists():
            logger.info(f"Repositorio ya existe en {local_repo_path}")
            return str(local_repo_path)

        try:
            result = subprocess.run(
                ["git", "clone", repo_url, str(local_repo_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos timeout
            )
            logger.info(f"Repositorio clonado exitosamente en {local_repo_path}")
            return str(local_repo_path)
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout al clonar repositorio {owner}/{repo}"
            logger.error(error_msg)
            raise GitHubClientError(error_msg)
        except subprocess.CalledProcessError as e:
            error_msg = f"Error al clonar repositorio {owner}/{repo}: {e.stderr if e.stderr else str(e)}"
            logger.error(error_msg)
            raise GitHubClientError(error_msg) from e
        except FileNotFoundError:
            error_msg = "Git no está instalado o no está en el PATH"
            logger.error(error_msg)
            raise GitHubClientError(error_msg)

    @retry_on_github_error(max_attempts=3)
    @handle_github_exception
    def create_branch(self, repo: Repository, branch_name: str, base_branch: str = None) -> bool:
        """
        Crear una nueva rama con reintentos automáticos.

        Args:
            repo: Repositorio de GitHub
            branch_name: Nombre de la nueva rama
            base_branch: Rama base (default: GitConfig.DEFAULT_BASE_BRANCH)

        Returns:
            True si se creó exitosamente
            
        Raises:
            GitHubClientError: Si no se puede crear la rama
        """
        if not branch_name or not branch_name.strip():
            raise GitHubClientError("El nombre de la rama no puede estar vacío")
        
        branch_name = branch_name.strip()
        
        # Validar formato del nombre de rama
        if not all(c.isalnum() or c in ['-', '_', '/'] for c in branch_name):
            raise GitHubClientError(f"Nombre de rama inválido: {branch_name}")
        
        if base_branch is None:
            base_branch = GitConfig.DEFAULT_BASE_BRANCH
        
        try:
            base_ref = repo.get_git_ref(f"heads/{base_branch}")
            repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=base_ref.object.sha
            )
            logger.info(f"Rama {branch_name} creada exitosamente desde {base_branch}")
            return True
        except GithubException as e:
            logger.error(f"Error al crear rama {branch_name}: {e}")
            raise GitHubClientError(f"Error al crear rama: {e}") from e

    @retry_on_github_error(max_attempts=3)
    @handle_github_exception
    def create_file(
        self,
        repo: Repository,
        path: str,
        content: str,
        message: str,
        branch: str = None
    ) -> bool:
        """
        Crear un archivo en el repositorio con reintentos automáticos.

        Args:
            repo: Repositorio de GitHub
            path: Ruta del archivo
            content: Contenido del archivo
            message: Mensaje de commit
            branch: Rama donde crear el archivo (default: GitConfig.DEFAULT_BASE_BRANCH)

        Returns:
            True si se creó exitosamente
            
        Raises:
            GitHubClientError: Si no se puede crear el archivo
        """
        if not path or not path.strip():
            raise GitHubClientError("La ruta del archivo no puede estar vacía")
        
        if not message or not message.strip():
            raise GitHubClientError("El mensaje de commit no puede estar vacío")
        
        path = path.strip()
        message = message.strip()
        
        if branch is None:
            branch = GitConfig.DEFAULT_BASE_BRANCH
        
        try:
            repo.create_file(
                path=path,
                message=message,
                content=content,
                branch=branch
            )
            logger.info(f"Archivo {path} creado exitosamente en rama {branch}")
            return True
        except GithubException as e:
            logger.error(f"Error al crear archivo {path} en rama {branch}: {e}")
            raise GitHubClientError(f"Error al crear archivo: {e}") from e

    @retry_on_github_error(max_attempts=3)
    @handle_github_exception
    def update_file(
        self,
        repo: Repository,
        path: str,
        content: str,
        message: str,
        branch: str = None
    ) -> bool:
        """
        Actualizar un archivo en el repositorio con reintentos automáticos.

        Args:
            repo: Repositorio de GitHub
            path: Ruta del archivo
            content: Nuevo contenido del archivo
            message: Mensaje de commit
            branch: Rama donde actualizar el archivo (default: GitConfig.DEFAULT_BASE_BRANCH)

        Returns:
            True si se actualizó exitosamente
            
        Raises:
            GitHubClientError: Si no se puede actualizar el archivo
        """
        if not path or not path.strip():
            raise GitHubClientError("La ruta del archivo no puede estar vacía")
        
        if not message or not message.strip():
            raise GitHubClientError("El mensaje de commit no puede estar vacío")
        
        path = path.strip()
        message = message.strip()
        
        if branch is None:
            branch = GitConfig.DEFAULT_BASE_BRANCH
        
        try:
            file = repo.get_contents(path, ref=branch)
            repo.update_file(
                path=path,
                message=message,
                content=content,
                sha=file.sha,
                branch=branch
            )
            logger.info(f"Archivo {path} actualizado exitosamente en rama {branch}")
            return True
        except GithubException as e:
            logger.error(f"Error al actualizar archivo {path} en rama {branch}: {e}")
            raise GitHubClientError(f"Error al actualizar archivo: {e}") from e

    @retry_on_github_error(max_attempts=3)
    @handle_github_exception
    def create_pull_request(
        self,
        repo: Repository,
        title: str,
        body: str,
        head: str,
        base: str = None
    ) -> Dict[str, Any]:
        """
        Crear un pull request con reintentos automáticos.

        Args:
            repo: Repositorio de GitHub
            title: Título del PR
            body: Descripción del PR
            head: Rama origen
            base: Rama destino (default: GitConfig.DEFAULT_BASE_BRANCH)

        Returns:
            Información del PR creado
            
        Raises:
            GitHubClientError: Si no se puede crear el pull request
        """
        if not title or not title.strip():
            raise GitHubClientError("El título del pull request no puede estar vacío")
        
        if not head or not head.strip():
            raise GitHubClientError("La rama origen (head) no puede estar vacía")
        
        title = title.strip()
        head = head.strip()
        
        if base is None:
            base = GitConfig.DEFAULT_BASE_BRANCH
        
        if head == base:
            raise GitHubClientError(f"La rama origen ({head}) no puede ser igual a la rama destino ({base})")
        
        try:
            pr = repo.create_pull(
                title=title,
                body=body or "",
                head=head,
                base=base
            )
            logger.info(f"Pull request #{pr.number} '{title}' creado exitosamente: {head} -> {base}")
            return {
                "number": pr.number,
                "title": pr.title,
                "url": pr.html_url,
                "state": pr.state
            }
        except GithubException as e:
            logger.error(f"Error al crear pull request de {head} a {base}: {e}")
            raise GitHubClientError(f"Error al crear pull request: {e}") from e

    @retry_on_github_error(max_attempts=3)
    @handle_github_exception
    def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Obtener información de un repositorio con reintentos automáticos.

        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio

        Returns:
            Diccionario con información del repositorio
            
        Raises:
            GitHubClientError: Si no se puede obtener la información del repositorio
        """
        try:
            repository = self.get_repository(owner, repo)
            return {
                "name": repository.name,
                "full_name": repository.full_name,
                "description": repository.description,
                "url": repository.html_url,
                "default_branch": repository.default_branch,
                "language": repository.language,
                "stars": repository.stargazers_count,
                "forks": repository.forks_count,
                "is_private": repository.private,
                "created_at": repository.created_at.isoformat() if repository.created_at else None,
                "updated_at": repository.updated_at.isoformat() if repository.updated_at else None,
            }
        except Exception as e:
            logger.error(f"Error al obtener información del repositorio {owner}/{repo}: {e}")
            raise GitHubClientError(f"Error al obtener información del repositorio: {e}") from e

