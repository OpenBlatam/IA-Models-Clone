"""
Git Integration - Integración con Git para aplicar mejoras
==========================================================
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)


class GitIntegration:
    """
    Integración con Git para aplicar mejoras directamente a repositorios.
    """
    
    def __init__(self, work_dir: Optional[str] = None):
        """
        Inicializar integración con Git.
        
        Args:
            work_dir: Directorio de trabajo (opcional)
        """
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def clone_repository(self, repo_url: str, branch: str = "main") -> str:
        """
        Clona un repositorio.
        
        Args:
            repo_url: URL del repositorio
            branch: Rama a clonar
            
        Returns:
            Ruta al repositorio clonado
        """
        try:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = self.work_dir / repo_name
            
            if repo_path.exists():
                logger.info(f"Repositorio ya existe: {repo_path}")
                return str(repo_path)
            
            # Clonar repositorio
            cmd = ["git", "clone", "-b", branch, repo_url, str(repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"Repositorio clonado: {repo_path}")
            return str(repo_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error clonando repositorio: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    def create_branch(self, repo_path: str, branch_name: str) -> bool:
        """
        Crea una nueva rama.
        
        Args:
            repo_path: Ruta al repositorio
            branch_name: Nombre de la rama
            
        Returns:
            True si se creó exitosamente
        """
        try:
            cmd = ["git", "-C", repo_path, "checkout", "-b", branch_name]
            subprocess.run(cmd, check=True, capture_output=True)
            
            logger.info(f"Rama creada: {branch_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creando rama: {e.stderr}")
            return False
    
    def apply_improvements(
        self,
        repo_path: str,
        improvements: List[Dict[str, Any]],
        commit_message: str = "Apply code improvements from research papers"
    ) -> Dict[str, Any]:
        """
        Aplica mejoras a archivos en el repositorio.
        
        Args:
            repo_path: Ruta al repositorio
            improvements: Lista de mejoras [{"file_path": "...", "improved_code": "..."}]
            commit_message: Mensaje de commit
            
        Returns:
            Resultado de la operación
        """
        try:
            applied_files = []
            
            for improvement in improvements:
                file_path = improvement.get("file_path")
                improved_code = improvement.get("improved_code")
                
                if not file_path or not improved_code:
                    continue
                
                full_path = Path(repo_path) / file_path
                
                if not full_path.exists():
                    logger.warning(f"Archivo no existe: {full_path}")
                    continue
                
                # Escribir código mejorado
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(improved_code)
                
                applied_files.append(file_path)
            
            # Agregar archivos al staging
            for file_path in applied_files:
                cmd = ["git", "-C", repo_path, "add", file_path]
                subprocess.run(cmd, check=True, capture_output=True)
            
            # Commit
            cmd = ["git", "-C", repo_path, "commit", "-m", commit_message]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Commit creado: {commit_message}")
                return {
                    "success": True,
                    "files_applied": applied_files,
                    "commit_message": commit_message
                }
            else:
                logger.warning(f"No se pudo crear commit: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "files_applied": applied_files
                }
                
        except Exception as e:
            logger.error(f"Error aplicando mejoras: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def push_branch(self, repo_path: str, branch_name: str, remote: str = "origin") -> bool:
        """
        Hace push de una rama.
        
        Args:
            repo_path: Ruta al repositorio
            branch_name: Nombre de la rama
            remote: Remote a usar
            
        Returns:
            True si se hizo push exitosamente
        """
        try:
            cmd = ["git", "-C", repo_path, "push", "-u", remote, branch_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Rama pusheada: {branch_name}")
                return True
            else:
                logger.error(f"Error haciendo push: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def create_pull_request_info(
        self,
        repo_path: str,
        branch_name: str,
        title: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Genera información para crear un Pull Request.
        
        Args:
            repo_path: Ruta al repositorio
            branch_name: Nombre de la rama
            title: Título del PR
            description: Descripción del PR
            
        Returns:
            Información del PR
        """
        try:
            # Obtener remote URL
            cmd = ["git", "-C", repo_path, "remote", "get-url", "origin"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            remote_url = result.stdout.strip()
            
            # Extraer owner/repo
            if "github.com" in remote_url:
                parts = remote_url.replace(".git", "").split("/")
                owner = parts[-2]
                repo = parts[-1]
                
                pr_info = {
                    "title": title,
                    "description": description,
                    "branch": branch_name,
                    "base": "main",
                    "repo": f"{owner}/{repo}",
                    "pr_url": f"https://github.com/{owner}/{repo}/compare/{branch_name}",
                    "create_command": f"gh pr create --title '{title}' --body '{description}' --base main --head {branch_name}"
                }
                
                return pr_info
            
            return {
                "title": title,
                "description": description,
                "branch": branch_name,
                "error": "No se pudo determinar información del repositorio"
            }
            
        except Exception as e:
            logger.error(f"Error generando info de PR: {e}")
            return {
                "error": str(e)
            }
    
    def cleanup(self, repo_path: Optional[str] = None):
        """Limpia directorios temporales"""
        try:
            if repo_path:
                if Path(repo_path).exists():
                    shutil.rmtree(repo_path)
            else:
                if self.work_dir.exists():
                    shutil.rmtree(self.work_dir)
        except Exception as e:
            logger.warning(f"Error limpiando: {e}")




