"""
Task Processor - Procesador de instrucciones y tareas.
"""

import uuid
from datetime import datetime
from config.logging_config import get_logger
from typing import Dict, Any, Optional, Tuple

from core.github_client import GitHubClient
from core.storage import TaskStorage
from core.utils import parse_instruction_params
from core.exceptions import TaskProcessingError, InstructionParseError
from core.constants import TaskStatus, SuccessMessages, GitConfig, ErrorMessages

logger = get_logger(__name__)


class TaskProcessor:
    """Procesador de tareas e instrucciones."""

    def __init__(
        self, 
        github_client: GitHubClient, 
        storage: TaskStorage,
        llm_service: Optional[Any] = None
    ):
        """
        Inicializar procesador de tareas.

        Args:
            github_client: Cliente de GitHub
            storage: Almacenamiento de tareas
            llm_service: Servicio LLM opcional para procesamiento con IA
        """
        self.github_client = github_client
        self.storage = storage
        self.llm_service = llm_service

    async def process_instruction(
        self,
        repository_owner: str,
        repository_name: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Procesar una instrucción y crear una tarea.

        Args:
            repository_owner: Propietario del repositorio
            repository_name: Nombre del repositorio
            instruction: Instrucción a ejecutar
            metadata: Metadatos adicionales (opcional)

        Returns:
            Diccionario con información de la tarea creada
            
        Raises:
            InstructionParseError: Si la instrucción es inválida
        """
        if not instruction or not instruction.strip():
            raise InstructionParseError("La instrucción no puede estar vacía")
        
        if not repository_owner or not repository_name:
            raise InstructionParseError("El propietario y nombre del repositorio son requeridos")
        
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "repository_owner": repository_owner,
            "repository_name": repository_name,
            "instruction": instruction.strip(),
            "status": TaskStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        await self.storage.save_task(task)
        logger.info(f"Tarea {task_id} creada para repositorio {repository_owner}/{repository_name}")

        return task

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar una tarea.

        Args:
            task: Diccionario con información de la tarea

        Returns:
            Diccionario con el resultado de la ejecución
        """
        task_id = task["id"]
        repository_owner = task["repository_owner"]
        repository_name = task["repository_name"]
        instruction = task["instruction"]

        logger.info(f"Ejecutando tarea {task_id}: {instruction}")

        await self.storage.update_task_status(task_id, TaskStatus.RUNNING)
        await self.storage.save_log(task_id, f"Iniciando ejecución: {instruction}")
        
        # Broadcast WebSocket update
        try:
            from api.routes.websocket_routes import broadcast_task_update
            updated_task = await self.storage.get_task(task_id)
            if updated_task:
                await broadcast_task_update(updated_task)
        except Exception as e:
            logger.debug(f"WebSocket broadcast failed (non-critical): {e}")

        try:
            repo = self.github_client.get_repository(repository_owner, repository_name)
            
            result = await self._execute_instruction(
                repo=repo,
                instruction=instruction,
                task_id=task_id
            )

            await self.storage.update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                result=result
            )
            await self.storage.save_log(task_id, SuccessMessages.TASK_COMPLETED)
            
            # Broadcast WebSocket update
            try:
                from api.routes.websocket_routes import broadcast_task_update
                updated_task = await self.storage.get_task(task_id)
                if updated_task:
                    await broadcast_task_update(updated_task)
            except Exception as e:
                logger.debug(f"WebSocket broadcast failed (non-critical): {e}")

            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": SuccessMessages.TASK_COMPLETED
            }

        except TaskProcessingError as e:
            error_message = str(e)
            logger.error(f"Error al procesar tarea {task_id}: {error_message}")
            await self.storage.save_log(task_id, f"Error de procesamiento: {error_message}", level="ERROR")

            await self.storage.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=error_message
            )
            
            # Broadcast WebSocket update
            try:
                from api.routes.websocket_routes import broadcast_task_update
                updated_task = await self.storage.get_task(task_id)
                if updated_task:
                    await broadcast_task_update(updated_task)
            except Exception as ws_error:
                logger.debug(f"WebSocket broadcast failed (non-critical): {ws_error}")

            return {
                "success": False,
                "task_id": task_id,
                "error": error_message,
                "error_type": "TaskProcessingError"
            }
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error inesperado al ejecutar tarea {task_id}: {error_message}", exc_info=True)
            await self.storage.save_log(task_id, f"Error inesperado: {error_message}", level="ERROR")

            await self.storage.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error=error_message
            )
            
            # Broadcast WebSocket update
            try:
                from api.routes.websocket_routes import broadcast_task_update
                updated_task = await self.storage.get_task(task_id)
                if updated_task:
                    await broadcast_task_update(updated_task)
            except Exception as ws_error:
                logger.debug(f"WebSocket broadcast failed (non-critical): {ws_error}")

            return {
                "success": False,
                "task_id": task_id,
                "error": error_message,
                "error_type": type(e).__name__
            }

    async def _execute_instruction(
        self,
        repo,
        instruction: str,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Ejecutar una instrucción específica.

        Args:
            repo: Repositorio de GitHub
            instruction: Instrucción a ejecutar
            task_id: ID de la tarea

        Returns:
            Diccionario con el resultado
        """
        instruction_lower = instruction.lower()

        # Si la instrucción requiere procesamiento con IA, usar LLM
        if self.llm_service and self._should_use_llm(instruction):
            return await self._handle_llm_instruction(repo, instruction, task_id)
        
        if "create file" in instruction_lower or "crear archivo" in instruction_lower:
            return await self._handle_create_file(repo, instruction, task_id)
        elif "update file" in instruction_lower or "actualizar archivo" in instruction_lower:
            return await self._handle_update_file(repo, instruction, task_id)
        elif "create branch" in instruction_lower or "crear rama" in instruction_lower:
            return await self._handle_create_branch(repo, instruction, task_id)
        elif "create pr" in instruction_lower or "crear pr" in instruction_lower or "pull request" in instruction_lower:
            return await self._handle_create_pr(repo, instruction, task_id)
        else:
            return {
                "message": f"Instrucción procesada: {instruction}",
                "type": "generic",
                "instruction": instruction
            }
    
    def _should_use_llm(self, instruction: str) -> bool:
        """
        Determinar si una instrucción debe procesarse con LLM.
        
        Args:
            instruction: Instrucción a evaluar
            
        Returns:
            True si debe usar LLM, False en caso contrario
        """
        instruction_lower = instruction.lower()
        
        # Palabras clave que indican que se necesita procesamiento con IA
        llm_keywords = [
            "analizar", "analyze", "generar", "generate", "sugerir", "suggest",
            "revisar", "review", "mejorar", "improve", "optimizar", "optimize",
            "explicar", "explain", "documentar", "document", "refactorizar", "refactor",
            "crear código", "create code", "escribir código", "write code",
            "comentar", "comment", "testear", "test", "debuggear", "debug"
        ]
        
        return any(keyword in instruction_lower for keyword in llm_keywords)
    
    async def _handle_llm_instruction(
        self,
        repo,
        instruction: str,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Manejar instrucción usando modelos LLM en paralelo.
        
        Args:
            repo: Repositorio de GitHub
            instruction: Instrucción a procesar
            task_id: ID de la tarea
            
        Returns:
            Diccionario con el resultado de múltiples modelos
        """
        await self.storage.save_log(task_id, "Procesando instrucción con modelos LLM en paralelo")
        
        try:
            # Obtener contexto del repositorio
            repo_info = {
                "name": repo.full_name,
                "description": repo.description or "",
                "language": repo.language or "unknown"
            }
            
            # Crear prompt mejorado con contexto
            system_prompt = f"""Eres un asistente experto en desarrollo de software y GitHub.
Tienes acceso al repositorio: {repo_info['name']}
Lenguaje principal: {repo_info['language']}
Descripción: {repo_info['description']}

Tu tarea es ayudar a procesar instrucciones relacionadas con este repositorio.
Proporciona respuestas claras, técnicas y útiles."""
            
            # Generar respuestas de múltiples modelos en paralelo
            logger.info(f"Tarea {task_id}: Generando respuestas con múltiples modelos LLM")
            responses = await self.llm_service.generate_parallel(
                prompt=instruction,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Procesar respuestas
            results = {}
            for model, response in responses.items():
                if response.error:
                    logger.warning(f"Tarea {task_id}: Error en modelo {model}: {response.error}")
                    results[model] = {
                        "success": False,
                        "error": response.error
                    }
                else:
                    results[model] = {
                        "success": True,
                        "content": response.content,
                        "latency_ms": response.latency_ms,
                        "usage": response.usage
                    }
                    logger.info(f"Tarea {task_id}: Modelo {model} completó en {response.latency_ms:.2f}ms")
            
            # Guardar todas las respuestas
            await self.storage.save_log(
                task_id, 
                f"Respuestas generadas por {len(responses)} modelos en paralelo"
            )
            
            return {
                "type": "llm_parallel",
                "instruction": instruction,
                "models_responses": results,
                "total_models": len(responses),
                "successful_models": sum(1 for r in results.values() if r.get("success", False))
            }
            
        except Exception as e:
            error_msg = f"Error al procesar con LLM: {str(e)}"
            logger.error(f"Tarea {task_id}: {error_msg}", exc_info=True)
            raise TaskProcessingError(error_msg) from e

    def _validate_file_params(self, params: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Validar y preparar parámetros de archivo.
        
        Args:
            params: Diccionario con parámetros parseados
            
        Returns:
            Tupla con (file_path, content, branch)
            
        Raises:
            InstructionParseError: Si los parámetros son inválidos
        """
        file_path = params.get("file_path")
        content = params.get("content", "")
        branch = params.get("branch") or GitConfig.DEFAULT_BASE_BRANCH
        
        if not file_path:
            raise InstructionParseError(ErrorMessages.INVALID_FILE_PATH)
        
        return file_path, content, branch

    async def _handle_create_file(self, repo, instruction: str, task_id: str) -> Dict[str, Any]:
        """
        Manejar creación de archivo.
        
        Args:
            repo: Repositorio de GitHub
            instruction: Instrucción a procesar
            task_id: ID de la tarea
            
        Returns:
            Diccionario con el resultado de la operación
            
        Raises:
            InstructionParseError: Si los parámetros son inválidos
            TaskProcessingError: Si hay un error al crear el archivo
        """
        await self.storage.save_log(task_id, "Procesando creación de archivo")
        
        try:
            params = parse_instruction_params(instruction)
            file_path, content, branch = self._validate_file_params(params)
            
            logger.info(f"Tarea {task_id}: Creando archivo {file_path} en rama {branch}")
            await self.storage.save_log(task_id, f"Creando archivo: {file_path} en rama {branch}")

            success = self.github_client.create_file(
                repo=repo,
                path=file_path,
                content=content,
                message=f"Create {file_path} - Task {task_id}",
                branch=branch
            )

            result = {
                "type": "create_file",
                "file_path": file_path,
                "branch": branch,
                "success": success
            }
            
            await self.storage.save_log(task_id, f"Archivo {file_path} creado exitosamente")
            return result
        except InstructionParseError:
            raise
        except Exception as e:
            error_msg = f"Error al crear archivo: {str(e)}"
            logger.error(f"Tarea {task_id}: {error_msg}", exc_info=True)
            raise TaskProcessingError(error_msg) from e

    async def _handle_update_file(self, repo, instruction: str, task_id: str) -> Dict[str, Any]:
        """
        Manejar actualización de archivo.
        
        Args:
            repo: Repositorio de GitHub
            instruction: Instrucción a procesar
            task_id: ID de la tarea
            
        Returns:
            Diccionario con el resultado de la operación
            
        Raises:
            InstructionParseError: Si los parámetros son inválidos
            TaskProcessingError: Si hay un error al actualizar el archivo
        """
        await self.storage.save_log(task_id, "Procesando actualización de archivo")
        
        try:
            params = parse_instruction_params(instruction)
            file_path, content, branch = self._validate_file_params(params)
            
            logger.info(f"Tarea {task_id}: Actualizando archivo {file_path} en rama {branch}")
            await self.storage.save_log(task_id, f"Actualizando archivo: {file_path} en rama {branch}")

            success = self.github_client.update_file(
                repo=repo,
                path=file_path,
                content=content,
                message=f"Update {file_path} - Task {task_id}",
                branch=branch
            )

            result = {
                "type": "update_file",
                "file_path": file_path,
                "branch": branch,
                "success": success
            }
            
            await self.storage.save_log(task_id, f"Archivo {file_path} actualizado exitosamente")
            return result
        except InstructionParseError:
            raise
        except Exception as e:
            error_msg = f"Error al actualizar archivo: {str(e)}"
            logger.error(f"Tarea {task_id}: {error_msg}", exc_info=True)
            raise TaskProcessingError(error_msg) from e

    async def _handle_create_branch(self, repo, instruction: str, task_id: str) -> Dict[str, Any]:
        """
        Manejar creación de rama.
        
        Args:
            repo: Repositorio de GitHub
            instruction: Instrucción a procesar
            task_id: ID de la tarea
            
        Returns:
            Diccionario con el resultado de la operación
            
        Raises:
            InstructionParseError: Si los parámetros son inválidos
            TaskProcessingError: Si hay un error al crear la rama
        """
        await self.storage.save_log(task_id, "Procesando creación de rama")
        
        try:
            params = parse_instruction_params(instruction)
            branch_name = params.get("branch_name")
            base_branch = params.get("base_branch", GitConfig.DEFAULT_BASE_BRANCH)

            if not branch_name or not branch_name.strip():
                raise InstructionParseError(ErrorMessages.INVALID_BRANCH_NAME)
            
            branch_name = branch_name.strip()
            
            if not base_branch:
                base_branch = GitConfig.DEFAULT_BASE_BRANCH
            
            logger.info(f"Tarea {task_id}: Creando rama {branch_name} desde {base_branch}")
            await self.storage.save_log(task_id, f"Creando rama: {branch_name} desde {base_branch}")

            success = self.github_client.create_branch(
                repo=repo,
                branch_name=branch_name,
                base_branch=base_branch
            )

            result = {
                "type": "create_branch",
                "branch_name": branch_name,
                "base_branch": base_branch,
                "success": success
            }
            
            await self.storage.save_log(task_id, f"Rama {branch_name} creada exitosamente")
            return result
        except InstructionParseError:
            raise
        except Exception as e:
            error_msg = f"Error al crear rama: {str(e)}"
            logger.error(f"Tarea {task_id}: {error_msg}", exc_info=True)
            raise TaskProcessingError(error_msg) from e

    async def _handle_create_pr(self, repo, instruction: str, task_id: str) -> Dict[str, Any]:
        """
        Manejar creación de pull request.
        
        Args:
            repo: Repositorio de GitHub
            instruction: Instrucción a procesar
            task_id: ID de la tarea
            
        Returns:
            Diccionario con el resultado de la operación
            
        Raises:
            InstructionParseError: Si los parámetros son inválidos
            TaskProcessingError: Si hay un error al crear el pull request
        """
        await self.storage.save_log(task_id, "Procesando creación de pull request")
        
        try:
            params = parse_instruction_params(instruction)
            title = params.get("title") or f"Auto PR - Task {task_id}"
            body = params.get("body", instruction)
            head = params.get("head", GitConfig.DEFAULT_BASE_BRANCH)
            base = params.get("base", GitConfig.DEFAULT_BASE_BRANCH)
            
            if not title or not title.strip():
                title = f"Auto PR - Task {task_id}"
            
            title = title.strip()

            logger.info(f"Tarea {task_id}: Creando PR '{title}' de {head} a {base}")
            await self.storage.save_log(task_id, f"Creando PR: {title} ({head} -> {base})")

            pr_info = self.github_client.create_pull_request(
                repo=repo,
                title=title,
                body=body or "",
                head=head,
                base=base
            )

            result = {
                "type": "create_pr",
                "pr_info": pr_info,
                "success": pr_info is not None
            }
            
            if pr_info:
                await self.storage.save_log(
                    task_id, 
                    f"PR #{pr_info.get('number')} creado exitosamente: {pr_info.get('url')}"
                )
            
            return result
        except InstructionParseError:
            raise
        except Exception as e:
            error_msg = f"Error al crear pull request: {str(e)}"
            logger.error(f"Tarea {task_id}: {error_msg}", exc_info=True)
            raise TaskProcessingError(error_msg) from e

