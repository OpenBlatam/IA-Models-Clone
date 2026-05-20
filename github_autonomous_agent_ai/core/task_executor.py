"""
Task Executor
=============

Ejecutor de tareas que procesa las instrucciones en repositorios de GitHub.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .github_client import GitHubClient

logger = logging.getLogger(__name__)

# Importar cliente de LLM si está disponible
try:
    from .llm.deepseek_client import DeepSeekClient
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("Cliente de LLM no disponible")


class TaskExecutor:
    """Ejecutor de tareas."""
    
    def __init__(self, github_client: GitHubClient, max_concurrent: int = 3):
        self.github_client = github_client
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Inicializar cliente de LLM si está disponible
        self.llm_client = None
        if LLM_AVAILABLE:
            try:
                self.llm_client = DeepSeekClient()
                if self.llm_client.enabled:
                    logger.info("✅ Cliente de DeepSeek LLM inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar DeepSeek: {e}")
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar una tarea."""
        async with self.semaphore:
            task_id = task["id"]
            repository = task["repository"]
            instruction = task["instruction"]
            
            logger.info(f"🔄 Ejecutando tarea {task_id} en {repository}")
            
            try:
                owner, repo = repository.split("/", 1)
                
                result = await self._process_instruction(
                    owner=owner,
                    repo=repo,
                    instruction=instruction,
                    metadata=task.get("metadata", {})
                )
                
                logger.info(f"✅ Tarea {task_id} completada exitosamente")
                
                from .task_queue import TaskQueue
                queue = TaskQueue()
                await queue.initialize()
                await queue.complete_task(task_id, result=result)
                
                return {
                    "success": True,
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"❌ Error en tarea {task_id}: {e}", exc_info=True)
                
                from .task_queue import TaskQueue
                queue = TaskQueue()
                await queue.initialize()
                await queue.complete_task(task_id, error=str(e))
                
                return {
                    "success": False,
                    "error": str(e)
                }
                
    async def _process_instruction(
        self,
        owner: str,
        repo: str,
        instruction: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Procesar una instrucción."""
        logger.info(f"📝 Procesando instrucción: {instruction[:100]}...")
        
        repo_info = await self.github_client.get_repository(owner, repo)
        
        result = {
            "repository": f"{owner}/{repo}",
            "instruction": instruction,
            "processed_at": datetime.utcnow().isoformat(),
            "actions": [],
            "llm_used": False
        }
        
        # Intentar usar LLM primero si está disponible
        if self.llm_client and self.llm_client.enabled:
            try:
                logger.info("🤖 Usando DeepSeek para procesar instrucción...")
                llm_plan = await self.llm_client.process_instruction(
                    instruction=instruction,
                    repository=f"{owner}/{repo}",
                    context={
                        "repo_info": repo_info,
                        "metadata": metadata
                    }
                )
                
                if llm_plan.get("llm_used"):
                    result["llm_used"] = True
                    result["llm_plan"] = llm_plan
                    
                    # Ejecutar el plan generado por el LLM
                    await self._execute_llm_plan(owner, repo, llm_plan, result)
                    return result
                else:
                    logger.info("LLM no pudo procesar, usando método básico")
            except Exception as e:
                logger.warning(f"Error usando LLM, cayendo a método básico: {e}")
        
        # Método básico (fallback)
        if "read" in instruction.lower() or "leer" in instruction.lower():
            if "file" in instruction.lower() or "archivo" in instruction.lower():
                file_path = metadata.get("file_path", "README.md")
                try:
                    content = await self.github_client.get_file_content(
                        owner, repo, file_path
                    )
                    result["actions"].append({
                        "type": "read_file",
                        "file": file_path,
                        "content_length": len(content)
                    })
                except Exception as e:
                    logger.warning(f"No se pudo leer el archivo {file_path}: {e}")
                    
        elif "create" in instruction.lower() or "crear" in instruction.lower():
            if "file" in instruction.lower() or "archivo" in instruction.lower():
                file_path = metadata.get("file_path", "new_file.txt")
                file_content = metadata.get("file_content", "# New File\n\nCreated by GitHub Autonomous Agent")
                message = metadata.get("commit_message", f"Create {file_path}")
                
                await self.github_client.create_file(
                    owner, repo, file_path, file_content, message
                )
                
                result["actions"].append({
                    "type": "create_file",
                    "file": file_path,
                    "message": message
                })
                
        elif "pull request" in instruction.lower() or "pr" in instruction.lower():
            title = metadata.get("pr_title", "Changes by Autonomous Agent")
            body = metadata.get("pr_body", instruction)
            head = metadata.get("head_branch", "feature/autonomous-changes")
            base = metadata.get("base_branch", "main")
            
            pr = await self.github_client.create_pull_request(
                owner, repo, title, body, head, base
            )
            
            result["actions"].append({
                "type": "create_pull_request",
                "pr_number": pr.get("number"),
                "pr_url": pr.get("html_url")
            })
            
        else:
            result["actions"].append({
                "type": "instruction_processed",
                "note": "Instrucción procesada, se requiere implementación específica"
            })
            
        return result
    
    async def _execute_llm_plan(
        self,
        owner: str,
        repo: str,
        plan: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Ejecutar un plan generado por el LLM."""
        action = plan.get("action", "").lower()
        
        try:
            if action == "create_file":
                file_path = plan.get("file_path", "new_file.txt")
                content = plan.get("content", "")
                commit_message = plan.get("commit_message", f"Create {file_path}")
                
                await self.github_client.create_file(
                    owner, repo, file_path, content, commit_message
                )
                
                result["actions"].append({
                    "type": "create_file",
                    "file": file_path,
                    "message": commit_message,
                    "reasoning": plan.get("reasoning", "")
                })
                
            elif action == "update_file":
                file_path = plan.get("file_path")
                content = plan.get("content", "")
                commit_message = plan.get("commit_message", f"Update {file_path}")
                
                # Obtener contenido actual primero
                try:
                    current_content = await self.github_client.get_file_content(owner, repo, file_path)
                    # Aquí podrías hacer un merge inteligente, por ahora reemplazamos
                    await self.github_client.update_file(
                        owner, repo, file_path, content, commit_message
                    )
                except Exception:
                    # Si el archivo no existe, crearlo
                    await self.github_client.create_file(
                        owner, repo, file_path, content, commit_message
                    )
                
                result["actions"].append({
                    "type": "update_file",
                    "file": file_path,
                    "message": commit_message,
                    "reasoning": plan.get("reasoning", "")
                })
                
            elif action == "delete_file":
                file_path = plan.get("file_path")
                commit_message = plan.get("commit_message", f"Delete {file_path}")
                
                await self.github_client.delete_file(owner, repo, file_path, commit_message)
                
                result["actions"].append({
                    "type": "delete_file",
                    "file": file_path,
                    "message": commit_message,
                    "reasoning": plan.get("reasoning", "")
                })
                
            elif action == "read_file":
                file_path = plan.get("file_path", "README.md")
                
                content = await self.github_client.get_file_content(owner, repo, file_path)
                
                result["actions"].append({
                    "type": "read_file",
                    "file": file_path,
                    "content_length": len(content),
                    "content_preview": content[:500] if len(content) > 500 else content,
                    "reasoning": plan.get("reasoning", "")
                })
                
            elif action == "create_branch":
                branch_name = plan.get("branch_name")
                
                await self.github_client.create_branch(owner, repo, branch_name)
                
                result["actions"].append({
                    "type": "create_branch",
                    "branch": branch_name,
                    "reasoning": plan.get("reasoning", "")
                })
                
            elif action == "create_pr":
                pr_title = plan.get("pr_title", "Changes by Autonomous Agent")
                pr_body = plan.get("pr_body", plan.get("reasoning", ""))
                head = plan.get("branch_name", "feature/autonomous-changes")
                base = plan.get("base_branch", "main")
                
                pr = await self.github_client.create_pull_request(
                    owner, repo, pr_title, pr_body, head, base
                )
                
                result["actions"].append({
                    "type": "create_pull_request",
                    "pr_number": pr.get("number"),
                    "pr_url": pr.get("html_url"),
                    "reasoning": plan.get("reasoning", "")
                })
                
            else:
                result["actions"].append({
                    "type": "unknown_action",
                    "action": action,
                    "note": "Acción no reconocida del plan LLM",
                    "plan": plan
                })
                
        except Exception as e:
            logger.error(f"Error ejecutando plan LLM: {e}", exc_info=True)
            result["actions"].append({
                "type": "error",
                "error": str(e),
                "plan": plan
            })

