"""
Git Manager
===========

Sistema de gestión de Git siguiendo las reglas específicas de Devin:
- Nunca hacer force push
- Nunca usar `git add .`
- Usar gh cli para operaciones de GitHub
- No cambiar git config a menos que se solicite
- Formato de branch: devin/{timestamp}-{feature-name}
- Cuando se crea un PR, actualizar el mismo PR en iteraciones
- Pedir ayuda si CI no pasa después del tercer intento
"""

import asyncio
import logging
import subprocess
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GitOperation:
    """Operación de Git"""
    operation_type: str
    command: str
    success: bool = False
    output: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "operation_type": self.operation_type,
            "command": self.command,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class BranchInfo:
    """Información de branch"""
    name: str
    base: str
    created_at: datetime = field(default_factory=datetime.now)
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "name": self.name,
            "base": self.base,
            "created_at": self.created_at.isoformat(),
            "pr_number": self.pr_number,
            "pr_url": self.pr_url
        }


class GitManager:
    """
    Gestor de Git siguiendo las reglas de Devin.
    
    Implementa todas las reglas específicas de Devin para operaciones Git.
    """
    
    DEFAULT_USERNAME = "Devin AI"
    DEFAULT_EMAIL = "devin-ai-integration[bot]@users.noreply.github.com"
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar gestor de Git.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.operations: List[GitOperation] = []
        self.branches: Dict[str, BranchInfo] = {}
        self.current_pr: Optional[Dict[str, Any]] = None
        self.ci_attempts: Dict[str, int] = {}
        logger.info("🔀 Git manager initialized")
    
    async def create_branch(
        self,
        feature_name: str,
        base_branch: str = "main",
        custom_format: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Crear branch siguiendo el formato de Devin.
        
        Formato: devin/{timestamp}-{feature-name}
        
        Args:
            feature_name: Nombre de la feature.
            base_branch: Branch base (default: main).
            custom_format: Formato personalizado (opcional).
        
        Returns:
            Tupla (éxito, nombre_branch, error).
        """
        try:
            if custom_format:
                branch_name = custom_format
            else:
                timestamp = int(datetime.now().timestamp())
                safe_feature = re.sub(r'[^a-zA-Z0-9-]', '-', feature_name.lower())
                branch_name = f"devin/{timestamp}-{safe_feature}"
            
            command = f"git checkout -b {branch_name}"
            if base_branch != "main":
                command = f"git checkout {base_branch} && git checkout -b {branch_name}"
            
            result = await self._run_git_command(command)
            
            if result["success"]:
                branch_info = BranchInfo(
                    name=branch_name,
                    base=base_branch
                )
                self.branches[branch_name] = branch_info
                logger.info(f"✅ Created branch: {branch_name}")
                return True, branch_name, None
            else:
                return False, None, result.get("error")
        
        except Exception as e:
            logger.error(f"Error creating branch: {e}", exc_info=True)
            return False, None, str(e)
    
    async def add_files(self, files: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Agregar archivos específicos a staging.
        
        NUNCA usar `git add .` según las reglas de Devin.
        
        Args:
            files: Lista de archivos a agregar.
        
        Returns:
            Tupla (éxito, error).
        """
        if not files:
            return False, "No files specified"
        
        try:
            files_str = " ".join(f'"{f}"' for f in files)
            command = f"git add {files_str}"
            
            result = await self._run_git_command(command, operation_type="add")
            
            if result["success"]:
                logger.info(f"✅ Added {len(files)} files to staging")
                return True, None
            else:
                return False, result.get("error")
        
        except Exception as e:
            logger.error(f"Error adding files: {e}", exc_info=True)
            return False, str(e)
    
    async def commit(
        self,
        message: str,
        files: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Hacer commit.
        
        Args:
            message: Mensaje del commit.
            files: Archivos específicos (opcional, si no se proporciona usa staged).
        
        Returns:
            Tupla (éxito, error).
        """
        try:
            if files:
                success, error = await self.add_files(files)
                if not success:
                    return False, error
            
            command = f'git commit -m "{message}"'
            result = await self._run_git_command(command, operation_type="commit")
            
            if result["success"]:
                logger.info(f"✅ Committed: {message[:50]}")
                return True, None
            else:
                return False, result.get("error")
        
        except Exception as e:
            logger.error(f"Error committing: {e}", exc_info=True)
            return False, str(e)
    
    async def push(
        self,
        branch: Optional[str] = None,
        force: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Hacer push.
        
        NUNCA hacer force push según las reglas de Devin.
        
        Args:
            branch: Branch a pushear (opcional).
            force: Si hacer force push (siempre False según reglas).
        
        Returns:
            Tupla (éxito, error).
        """
        if force:
            logger.warning("⚠️ Force push requested but blocked by Devin rules")
            return False, "Force push is not allowed. Ask user for help if push fails."
        
        try:
            if branch:
                command = f"git push origin {branch}"
            else:
                command = "git push"
            
            result = await self._run_git_command(command, operation_type="push")
            
            if result["success"]:
                logger.info(f"✅ Pushed branch: {branch or 'current'}")
                return True, None
            else:
                error_msg = result.get("error", "Unknown error")
                if "force" in error_msg.lower() or "non-fast-forward" in error_msg.lower():
                    return False, "Push failed. Do not force push. Ask user for help."
                return False, error_msg
        
        except Exception as e:
            logger.error(f"Error pushing: {e}", exc_info=True)
            return False, str(e)
    
    async def create_pr(
        self,
        title: str,
        body: str,
        base: str = "main",
        head: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Crear PR usando gh cli.
        
        Args:
            title: Título del PR.
            body: Cuerpo del PR.
            base: Branch base.
            head: Branch head (opcional).
        
        Returns:
            Tupla (éxito, pr_info, error).
        """
        try:
            if not head:
                result = await self._run_git_command("git branch --show-current")
                if not result["success"]:
                    return False, None, "Could not determine current branch"
                head = result["output"].strip()
            
            command = f'gh pr create --title "{title}" --body "{body}" --base {base} --head {head}'
            result = await self._run_git_command(command, operation_type="create_pr")
            
            if result["success"]:
                pr_info = self._parse_pr_info(result["output"])
                self.current_pr = pr_info
                
                if head in self.branches:
                    self.branches[head].pr_number = pr_info.get("number")
                    self.branches[head].pr_url = pr_info.get("url")
                
                logger.info(f"✅ Created PR: {pr_info.get('number')}")
                return True, pr_info, None
            else:
                return False, None, result.get("error")
        
        except Exception as e:
            logger.error(f"Error creating PR: {e}", exc_info=True)
            return False, None, str(e)
    
    async def update_pr(
        self,
        pr_number: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Actualizar PR existente.
        
        Según las reglas de Devin:
        - Cuando el usuario hace follow-up y ya existe un PR, actualizar el mismo PR
        
        Args:
            pr_number: Número del PR (opcional, usa current_pr si no se proporciona).
        
        Returns:
            Tupla (éxito, error).
        """
        try:
            if not pr_number and self.current_pr:
                pr_number = self.current_pr.get("number")
            
            if not pr_number:
                return False, "No PR number available"
            
            command = f"git push"
            result = await self._run_git_command(command, operation_type="update_pr")
            
            if result["success"]:
                logger.info(f"✅ Updated PR: {pr_number}")
                return True, None
            else:
                return False, result.get("error")
        
        except Exception as e:
            logger.error(f"Error updating PR: {e}", exc_info=True)
            return False, str(e)
    
    async def check_ci_status(
        self,
        pr_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Verificar estado de CI.
        
        Args:
            pr_number: Número del PR (opcional).
        
        Returns:
            Estado de CI.
        """
        try:
            if not pr_number and self.current_pr:
                pr_number = self.current_pr.get("number")
            
            if not pr_number:
                return {
                    "success": False,
                    "error": "No PR number available"
                }
            
            command = f"gh pr checks {pr_number}"
            result = await self._run_git_command(command, operation_type="check_ci")
            
            if result["success"]:
                ci_status = self._parse_ci_status(result["output"])
                
                pr_key = f"pr_{pr_number}"
                if ci_status.get("status") != "success":
                    self.ci_attempts[pr_key] = self.ci_attempts.get(pr_key, 0) + 1
                    
                    if self.ci_attempts[pr_key] >= 3:
                        return {
                            "success": False,
                            "status": ci_status.get("status"),
                            "attempts": self.ci_attempts[pr_key],
                            "should_ask_help": True,
                            "message": "CI has not passed after 3 attempts. Ask user for help."
                        }
                
                return {
                    "success": True,
                    "status": ci_status.get("status"),
                    "attempts": self.ci_attempts.get(pr_key, 0),
                    "details": ci_status
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error")
                }
        
        except Exception as e:
            logger.error(f"Error checking CI status: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_git_command(
        self,
        command: str,
        operation_type: str = "git"
    ) -> Dict[str, Any]:
        """Ejecutar comando de git"""
        try:
            process = await asyncio.create_subprocess_exec(
                *command.split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root)
            )
            
            stdout, stderr = await process.communicate()
            
            output = stdout.decode('utf-8', errors='ignore')
            error = stderr.decode('utf-8', errors='ignore')
            
            operation = GitOperation(
                operation_type=operation_type,
                command=command,
                success=process.returncode == 0,
                output=output if process.returncode == 0 else None,
                error=error if process.returncode != 0 else None
            )
            
            self.operations.append(operation)
            
            return {
                "success": process.returncode == 0,
                "output": output,
                "error": error
            }
        
        except Exception as e:
            logger.error(f"Error running git command: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_pr_info(self, output: str) -> Dict[str, Any]:
        """Parsear información de PR del output de gh"""
        pr_info = {}
        
        url_match = re.search(r'https://github\.com/[^/]+/[^/]+/pull/(\d+)', output)
        if url_match:
            pr_info["number"] = int(url_match.group(1))
            pr_info["url"] = url_match.group(0)
        
        return pr_info
    
    def _parse_ci_status(self, output: str) -> Dict[str, Any]:
        """Parsear estado de CI del output"""
        status = {
            "status": "unknown",
            "checks": []
        }
        
        if "passing" in output.lower() or "success" in output.lower():
            status["status"] = "success"
        elif "failing" in output.lower() or "failure" in output.lower():
            status["status"] = "failure"
        elif "pending" in output.lower():
            status["status"] = "pending"
        
        return status
    
    def get_current_branch(self) -> Optional[str]:
        """Obtener branch actual"""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Error getting current branch: {e}")
        return None
    
    def get_operations(self) -> List[Dict[str, Any]]:
        """Obtener todas las operaciones"""
        return [op.to_dict() for op in self.operations]
    
    def get_branches(self) -> List[Dict[str, Any]]:
        """Obtener todos los branches"""
        return [b.to_dict() for b in self.branches.values()]

