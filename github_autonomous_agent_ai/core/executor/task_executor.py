"""
Ejecutor de tareas
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.github.client import GitHubClient
from core.queue.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Ejecutor de tareas de GitHub"""
    
    def __init__(self, github_client: Optional[GitHubClient] = None, task_queue: Optional[TaskQueue] = None):
        """
        Inicializar ejecutor
        
        Args:
            github_client: Cliente de GitHub
            task_queue: Cola de tareas
        """
        self.github_client = github_client or GitHubClient()
        self.task_queue = task_queue or TaskQueue()
        self.is_running = False
        self.active_tasks: Dict[str, asyncio.Task] = {}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar una tarea
        
        Args:
            task: Datos de la tarea
            
        Returns:
            Resultado de la ejecución
        """
        task_id = task["id"]
        logger.info(f"Ejecutando tarea {task_id}: {task.get('instruction', '')[:50]}...")
        
        try:
            self.task_queue.update_task(task_id, {
                "status": "running",
                "started_at": datetime.now()
            })
            
            owner, repo = self.github_client.parse_repo_url(task["github_repo"])
            instruction = task["instruction"]
            
            result = await self._process_instruction(owner, repo, instruction, task)
            
            self.task_queue.update_task(task_id, {
                "status": "completed",
                "completed_at": datetime.now(),
                "result": result
            })
            
            logger.info(f"Tarea {task_id} completada")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error ejecutando tarea {task_id}: {error_msg}")
            
            self.task_queue.update_task(task_id, {
                "status": "failed",
                "completed_at": datetime.now(),
                "error": error_msg
            })
            
            raise
    
    async def _process_instruction(
        self,
        owner: str,
        repo: str,
        instruction: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesar una instrucción
        
        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio
            instruction: Instrucción a ejecutar
            task: Datos completos de la tarea
            
        Returns:
            Resultado del procesamiento
        """
        instruction_lower = instruction.lower()
        
        if "read" in instruction_lower or "obtener" in instruction_lower or "ver" in instruction_lower:
            return await self._handle_read_instruction(owner, repo, instruction)
        
        elif "create" in instruction_lower or "crear" in instruction_lower or "agregar" in instruction_lower:
            return await self._handle_create_instruction(owner, repo, instruction)
        
        elif "update" in instruction_lower or "actualizar" in instruction_lower or "modificar" in instruction_lower:
            return await self._handle_update_instruction(owner, repo, instruction)
        
        elif "delete" in instruction_lower or "eliminar" in instruction_lower:
            return await self._handle_delete_instruction(owner, repo, instruction)
        
        elif "branch" in instruction_lower or "rama" in instruction_lower:
            return await self._handle_branch_instruction(owner, repo, instruction)
        
        elif "pull request" in instruction_lower or "pr" in instruction_lower:
            return await self._handle_pr_instruction(owner, repo, instruction)
        
        else:
            return {
                "type": "generic",
                "message": f"Instrucción procesada: {instruction}",
                "repo": f"{owner}/{repo}",
                "status": "completed"
            }
    
    async def _handle_read_instruction(self, owner: str, repo: str, instruction: str) -> Dict[str, Any]:
        """Manejar instrucción de lectura"""
        repo_info = self.github_client.get_repo_info(owner, repo)
        files = self.github_client.list_files(owner, repo)
        
        return {
            "type": "read",
            "repo_info": {
                "name": repo_info["name"],
                "description": repo_info.get("description"),
                "stars": repo_info.get("stargazers_count", 0),
                "language": repo_info.get("language")
            },
            "files_count": len(files),
            "files": files[:10]
        }
    
    async def _handle_create_instruction(self, owner: str, repo: str, instruction: str) -> Dict[str, Any]:
        """Manejar instrucción de creación"""
        import re
        
        file_match = re.search(r'file[:\s]+([^\s]+)', instruction, re.IGNORECASE)
        content_match = re.search(r'content[:\s]+(.+)', instruction, re.IGNORECASE)
        
        if file_match and content_match:
            file_path = file_match.group(1)
            content = content_match.group(1)
            message = f"Create {file_path}"
            
            result = self.github_client.create_file(owner, repo, file_path, content, message)
            
            return {
                "type": "create",
                "file_path": file_path,
                "commit_sha": result.get("commit", {}).get("sha"),
                "status": "created"
            }
        
        return {
            "type": "create",
            "message": "No se pudo parsear la instrucción de creación",
            "status": "failed"
        }
    
    async def _handle_update_instruction(self, owner: str, repo: str, instruction: str) -> Dict[str, Any]:
        """Manejar instrucción de actualización"""
        return await self._handle_create_instruction(owner, repo, instruction)
    
    async def _handle_delete_instruction(self, owner: str, repo: str, instruction: str) -> Dict[str, Any]:
        """Manejar instrucción de eliminación"""
        return {
            "type": "delete",
            "message": "Funcionalidad de eliminación pendiente de implementar",
            "status": "pending"
        }
    
    async def _handle_branch_instruction(self, owner: str, repo: str, instruction: str) -> Dict[str, Any]:
        """Manejar instrucción de rama"""
        import re
        
        branch_match = re.search(r'branch[:\s]+([^\s]+)', instruction, re.IGNORECASE)
        
        if branch_match:
            branch_name = branch_match.group(1)
            result = self.github_client.create_branch(owner, repo, branch_name)
            
            return {
                "type": "branch",
                "branch_name": branch_name,
                "ref": result.get("ref"),
                "status": "created"
            }
        
        return {
            "type": "branch",
            "message": "No se pudo parsear el nombre de la rama",
            "status": "failed"
        }
    
    async def _handle_pr_instruction(self, owner: str, repo: str, instruction: str) -> Dict[str, Any]:
        """Manejar instrucción de pull request"""
        import re
        
        title_match = re.search(r'title[:\s]+([^\n]+)', instruction, re.IGNORECASE)
        head_match = re.search(r'head[:\s]+([^\s]+)', instruction, re.IGNORECASE)
        
        if title_match and head_match:
            title = title_match.group(1)
            head = head_match.group(1)
            body = f"Pull request creado automáticamente: {title}"
            
            result = self.github_client.create_pull_request(owner, repo, title, body, head)
            
            return {
                "type": "pull_request",
                "pr_number": result.get("number"),
                "pr_url": result.get("html_url"),
                "status": "created"
            }
        
        return {
            "type": "pull_request",
            "message": "No se pudo parsear la instrucción de PR",
            "status": "failed"
        }
    
    async def run_continuous(self):
        """Ejecutar tareas continuamente"""
        self.is_running = True
        logger.info("Iniciando ejecución continua de tareas")
        
        while self.is_running:
            try:
                task = self.task_queue.get_next_task()
                
                if task:
                    task_id = task["id"]
                    if task_id not in self.active_tasks:
                        self.active_tasks[task_id] = asyncio.create_task(
                            self.execute_task(task)
                        )
                else:
                    await asyncio.sleep(1)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error en loop de ejecución: {e}")
                await asyncio.sleep(1)
    
    def stop(self):
        """Detener ejecución"""
        self.is_running = False
        logger.info("Deteniendo ejecución de tareas")
        
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
        
        self.active_tasks.clear()

