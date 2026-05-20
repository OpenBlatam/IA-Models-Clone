"""
Devin Command System
====================

Sistema de comandos estructurado que simula los comandos disponibles
en el prompt de Devin, permitiendo al agente ejecutar acciones
de manera similar a Devin.
"""

import asyncio
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Categorías de comandos"""
    REASONING = "reasoning"
    SHELL = "shell"
    EDITOR = "editor"
    SEARCH = "search"
    LSP = "lsp"
    BROWSER = "browser"
    DEPLOYMENT = "deployment"
    USER_INTERACTION = "user_interaction"
    MISC = "misc"
    PLAN = "plan"


@dataclass
class CommandResult:
    """Resultado de ejecución de comando"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PlanStep:
    """Paso de un plan"""
    id: str
    description: str
    status: str = "pending"
    dependencies: List[str] = field(default_factory=list)
    estimated_time: Optional[float] = None
    actual_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "dependencies": self.dependencies,
            "estimated_time": self.estimated_time,
            "actual_time": self.actual_time,
            "error": self.error
        }


@dataclass
class Plan:
    """Plan de trabajo"""
    id: str
    title: str
    description: str
    steps: List[PlanStep] = field(default_factory=list)
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_step(
        self,
        description: str,
        dependencies: Optional[List[str]] = None,
        estimated_time: Optional[float] = None
    ) -> PlanStep:
        """Agregar paso al plan"""
        step_id = f"step_{len(self.steps) + 1}"
        step = PlanStep(
            id=step_id,
            description=description,
            dependencies=dependencies or [],
            estimated_time=estimated_time
        )
        self.steps.append(step)
        self.updated_at = datetime.now()
        return step
    
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Obtener paso por ID"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class DevinCommandExecutor:
    """
    Ejecutor de comandos estilo Devin.
    
    Proporciona una interfaz estructurada para ejecutar comandos
    similares a los disponibles en el prompt de Devin.
    """
    
    def __init__(self, agent: Optional[Any] = None) -> None:
        """
        Inicializar ejecutor de comandos.
        
        Args:
            agent: Instancia del agente (opcional).
        """
        self.agent = agent
        self.command_history: List[CommandResult] = []
        self.plans: Dict[str, Plan] = {}
        self.command_handlers: Dict[str, Callable] = {}
        
        self._register_default_handlers()
        logger.info("🔧 Devin command executor initialized")
    
    def _register_default_handlers(self) -> None:
        """Registrar manejadores de comandos por defecto"""
        # Reasoning commands
        self.register_handler("reasoning", self._handle_reasoning)
        
        # Shell commands
        self.register_handler("shell", self._handle_shell)
        self.register_handler("view_shell", self._handle_view_shell)
        self.register_handler("write_to_shell_process", self._handle_write_to_shell)
        self.register_handler("kill_shell_process", self._handle_kill_shell)
        
        # Editor commands
        self.register_handler("open_file", self._handle_open_file)
        self.register_handler("str_replace", self._handle_str_replace)
        self.register_handler("create_file", self._handle_create_file)
        self.register_handler("insert", self._handle_insert)
        self.register_handler("remove_str", self._handle_remove_str)
        self.register_handler("find_and_edit", self._handle_find_and_edit)
        self.register_handler("undo_edit", self._handle_undo_edit)
        
        # Search commands
        self.register_handler("search", self._handle_search)
        self.register_handler("find_filecontent", self._handle_find_filecontent)
        self.register_handler("find_filename", self._handle_find_filename)
        self.register_handler("semantic_search", self._handle_semantic_search)
        
        # LSP commands
        self.register_handler("go_to_definition", self._handle_go_to_definition)
        self.register_handler("go_to_references", self._handle_go_to_references)
        self.register_handler("hover_symbol", self._handle_hover_symbol)
        
        # Browser commands
        self.register_handler("navigate_browser", self._handle_navigate_browser)
        self.register_handler("view_browser", self._handle_view_browser)
        self.register_handler("click_browser", self._handle_click_browser)
        self.register_handler("type_browser", self._handle_type_browser)
        self.register_handler("restart_browser", self._handle_restart_browser)
        self.register_handler("move_mouse", self._handle_move_mouse)
        self.register_handler("press_key_browser", self._handle_press_key_browser)
        self.register_handler("browser_console", self._handle_browser_console)
        self.register_handler("select_option_browser", self._handle_select_option_browser)
        
        # Deployment commands
        self.register_handler("deploy_frontend", self._handle_deploy_frontend)
        self.register_handler("deploy_backend", self._handle_deploy_backend)
        self.register_handler("expose_port", self._handle_expose_port)
        
        # User interaction commands
        self.register_handler("wait", self._handle_wait)
        self.register_handler("message_user", self._handle_message_user)
        self.register_handler("list_secrets", self._handle_list_secrets)
        self.register_handler("report_environment_issue", self._handle_report_environment_issue)
        
        # Git commands
        self.register_handler("git_view_pr", self._handle_git_view_pr)
        self.register_handler("gh_pr_checklist", self._handle_gh_pr_checklist)
        
        # Plan commands
        self.register_handler("plan", self._handle_plan)
        self.register_handler("suggest_plan", self._handle_suggest_plan)
    
    def register_handler(
        self,
        command_name: str,
        handler: Callable[..., CommandResult]
    ) -> None:
        """
        Registrar manejador de comando.
        
        Args:
            command_name: Nombre del comando.
            handler: Función manejadora.
        """
        self.command_handlers[command_name] = handler
        logger.debug(f"Registered handler for command: {command_name}")
    
    async def execute_command(
        self,
        command_name: str,
        **kwargs
    ) -> CommandResult:
        """
        Ejecutar comando.
        
        Args:
            command_name: Nombre del comando.
            **kwargs: Argumentos del comando.
        
        Returns:
            Resultado de la ejecución.
        """
        start_time = datetime.now()
        
        try:
            if command_name not in self.command_handlers:
                return CommandResult(
                    success=False,
                    error=f"Unknown command: {command_name}"
                )
            
            handler = self.command_handlers[command_name]
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**kwargs)
            else:
                result = handler(**kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            self.command_history.append(result)
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result = CommandResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
            self.command_history.append(result)
            logger.error(f"Error executing command {command_name}: {e}", exc_info=True)
            return result
    
    def _handle_reasoning(self, content: str) -> CommandResult:
        """
        Manejar comando de razonamiento.
        
        Args:
            content: Contenido del razonamiento.
        
        Returns:
            Resultado del comando.
        """
        if not self.agent or not self.agent.devin:
            return CommandResult(
                success=False,
                error="Devin persona not available"
            )
        
        context = self.agent.devin.start_reasoning()
        context.add_observation(content)
        
        return CommandResult(
            success=True,
            output="Reasoning context created",
            metadata={"context_id": len(self.agent.devin.reasoning_contexts) - 1}
        )
    
    async def _handle_shell(
        self,
        command: str,
        exec_dir: Optional[str] = None
    ) -> CommandResult:
        """
        Manejar comando shell.
        
        Args:
            command: Comando a ejecutar.
            exec_dir: Directorio de ejecución.
        
        Returns:
            Resultado del comando.
        """
        try:
            from .command_executor import CommandExecutor
            
            executor = CommandExecutor(working_dir=exec_dir)
            result = await executor.execute(command, command_type="shell")
            
            if result.get("success", False):
                return CommandResult(
                    success=True,
                    output=result.get("output", ""),
                    metadata={"command_type": "shell"}
                )
            else:
                return CommandResult(
                    success=False,
                    error=result.get("error", "Unknown error")
                )
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e)
            )
    
    def _handle_search(
        self,
        query: str,
        path: Optional[str] = None,
        search_type: str = "semantic"
    ) -> CommandResult:
        """
        Manejar búsqueda.
        
        Args:
            query: Consulta de búsqueda.
            path: Ruta donde buscar.
            search_type: Tipo de búsqueda (semantic, content, filename).
        
        Returns:
            Resultado del comando.
        """
        try:
            if search_type == "semantic" and self.agent and self.agent.code_understanding:
                structure = self.agent.code_understanding.analyze_codebase_structure()
                return CommandResult(
                    success=True,
                    output=f"Found {structure['total_files']} files, {structure['total_functions']} functions, {structure['total_classes']} classes",
                    metadata={"structure": structure}
                )
            else:
                return CommandResult(
                    success=False,
                    error=f"Search type {search_type} not implemented"
                )
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e)
            )
    
    def _handle_plan(
        self,
        action: str,
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        step_description: Optional[str] = None
    ) -> CommandResult:
        """
        Manejar comandos de planificación.
        
        Args:
            action: Acción (create, add_step, get, list).
            plan_id: ID del plan.
            title: Título del plan.
            description: Descripción del plan.
            step_description: Descripción del paso.
        
        Returns:
            Resultado del comando.
        """
        try:
            if action == "create":
                if not title or not description:
                    return CommandResult(
                        success=False,
                        error="Title and description required for plan creation"
                    )
                
                plan_id = plan_id or f"plan_{datetime.now().timestamp()}"
                plan = Plan(
                    id=plan_id,
                    title=title,
                    description=description
                )
                self.plans[plan_id] = plan
                
                return CommandResult(
                    success=True,
                    output=f"Plan '{title}' created with ID: {plan_id}",
                    metadata={"plan_id": plan_id}
                )
            
            elif action == "add_step":
                if not plan_id or not step_description:
                    return CommandResult(
                        success=False,
                        error="Plan ID and step description required"
                    )
                
                if plan_id not in self.plans:
                    return CommandResult(
                        success=False,
                        error=f"Plan {plan_id} not found"
                    )
                
                plan = self.plans[plan_id]
                step = plan.add_step(step_description)
                
                return CommandResult(
                    success=True,
                    output=f"Step added to plan {plan_id}: {step.id}",
                    metadata={"step_id": step.id, "plan_id": plan_id}
                )
            
            elif action == "get":
                if not plan_id:
                    return CommandResult(
                        success=False,
                        error="Plan ID required"
                    )
                
                if plan_id not in self.plans:
                    return CommandResult(
                        success=False,
                        error=f"Plan {plan_id} not found"
                    )
                
                plan = self.plans[plan_id]
                return CommandResult(
                    success=True,
                    output=plan.to_dict(),
                    metadata={"plan": plan.to_dict()}
                )
            
            elif action == "list":
                return CommandResult(
                    success=True,
                    output=f"Found {len(self.plans)} plans",
                    metadata={"plans": [plan.to_dict() for plan in self.plans.values()]}
                )
            
            else:
                return CommandResult(
                    success=False,
                    error=f"Unknown plan action: {action}"
                )
                
        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e)
            )
    
    def suggest_plan(self, plan_id: str) -> CommandResult:
        """
        Sugerir plan (similar a <suggest_plan/>).
        
        Args:
            plan_id: ID del plan a sugerir.
        
        Returns:
            Resultado del comando.
        """
        if plan_id not in self.plans:
            return CommandResult(
                success=False,
                error=f"Plan {plan_id} not found"
            )
        
        plan = self.plans[plan_id]
        plan.status = "ready"
        plan.updated_at = datetime.now()
        
        return CommandResult(
            success=True,
            output=f"Plan '{plan.title}' is ready for execution",
            metadata={"plan": plan.to_dict()}
        )
    
    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de comandos.
        
        Args:
            limit: Número máximo de comandos a retornar.
        
        Returns:
            Lista de comandos ejecutados.
        """
        return [cmd.to_dict() for cmd in self.command_history[-limit:]]
    
    def get_plans(self) -> List[Dict[str, Any]]:
        """Obtener todos los planes"""
        return [plan.to_dict() for plan in self.plans.values()]
    
    # Shell command handlers
    async def _handle_view_shell(self, shell_id: str = "default") -> CommandResult:
        """Ver salida de shell"""
        try:
            if not self.agent:
                return CommandResult(success=False, error="Agent not available")
            return CommandResult(
                success=True,
                output=f"Shell {shell_id} output retrieved",
                metadata={"shell_id": shell_id}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_write_to_shell(
        self,
        shell_id: str = "default",
        content: str = "",
        press_enter: bool = True
    ) -> CommandResult:
        """Escribir a proceso shell"""
        try:
            return CommandResult(
                success=True,
                output=f"Content written to shell {shell_id}",
                metadata={"shell_id": shell_id, "content_length": len(content)}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_kill_shell(self, shell_id: str) -> CommandResult:
        """Terminar proceso shell"""
        try:
            return CommandResult(
                success=True,
                output=f"Shell {shell_id} terminated",
                metadata={"shell_id": shell_id}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    # Editor command handlers
    async def _handle_open_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False
    ) -> CommandResult:
        """Abrir archivo"""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return CommandResult(
                    success=False,
                    error=f"File not found: {path}"
                )
            
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            if start_line is not None:
                start_idx = max(0, start_line - 1)
                end_idx = end_line if end_line is not None else len(lines)
                content = '\n'.join(lines[start_idx:end_idx])
            
            return CommandResult(
                success=True,
                output=content,
                metadata={
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_lines": len(lines)
                }
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_str_replace(
        self,
        path: str,
        old_str: str,
        new_str: str,
        sudo: bool = False,
        many: bool = False
    ) -> CommandResult:
        """Reemplazar string en archivo"""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return CommandResult(
                    success=False,
                    error=f"File not found: {path}"
                )
            
            content = file_path.read_text(encoding='utf-8')
            
            if many:
                new_content = content.replace(old_str, new_str)
            else:
                if old_str not in content:
                    return CommandResult(
                        success=False,
                        error=f"String not found in file: {path}"
                    )
                new_content = content.replace(old_str, new_str, 1)
            
            file_path.write_text(new_content, encoding='utf-8')
            
            return CommandResult(
                success=True,
                output=f"File {path} updated",
                metadata={"path": path, "replacements": content.count(old_str) if many else 1}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_create_file(
        self,
        path: str,
        content: str = "",
        sudo: bool = False
    ) -> CommandResult:
        """Crear archivo"""
        try:
            file_path = Path(path)
            if file_path.exists():
                return CommandResult(
                    success=False,
                    error=f"File already exists: {path}"
                )
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            
            return CommandResult(
                success=True,
                output=f"File {path} created",
                metadata={"path": path, "content_length": len(content)}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_insert(
        self,
        path: str,
        insert_line: int,
        content: str,
        sudo: bool = False
    ) -> CommandResult:
        """Insertar contenido en línea"""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return CommandResult(
                    success=False,
                    error=f"File not found: {path}"
                )
            
            lines = file_path.read_text(encoding='utf-8').split('\n')
            lines.insert(insert_line - 1, content)
            file_path.write_text('\n'.join(lines), encoding='utf-8')
            
            return CommandResult(
                success=True,
                output=f"Content inserted at line {insert_line} in {path}",
                metadata={"path": path, "insert_line": insert_line}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_remove_str(
        self,
        path: str,
        content: str,
        sudo: bool = False,
        many: bool = False
    ) -> CommandResult:
        """Remover string de archivo"""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return CommandResult(
                    success=False,
                    error=f"File not found: {path}"
                )
            
            file_content = file_path.read_text(encoding='utf-8')
            
            if many:
                new_content = file_content.replace(content, '')
            else:
                if content not in file_content:
                    return CommandResult(
                        success=False,
                        error=f"String not found in file: {path}"
                    )
                new_content = file_content.replace(content, '', 1)
            
            file_path.write_text(new_content, encoding='utf-8')
            
            return CommandResult(
                success=True,
                output=f"String removed from {path}",
                metadata={"path": path}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_find_and_edit(
        self,
        dir: str,
        regex: str,
        exclude_file_glob: Optional[str] = None,
        file_extension_glob: Optional[str] = None,
        description: str = ""
    ) -> CommandResult:
        """Buscar y editar múltiples archivos"""
        try:
            import re
            from pathlib import Path
            
            search_dir = Path(dir)
            if not search_dir.exists():
                return CommandResult(
                    success=False,
                    error=f"Directory not found: {dir}"
                )
            
            pattern = re.compile(regex)
            matches = []
            
            for file_path in search_dir.rglob('*'):
                if file_path.is_file():
                    if exclude_file_glob and file_path.match(exclude_file_glob):
                        continue
                    if file_extension_glob and not file_path.match(file_extension_glob):
                        continue
                    
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if pattern.search(content):
                            matches.append(str(file_path))
                    except Exception:
                        continue
            
            return CommandResult(
                success=True,
                output=f"Found {len(matches)} files matching pattern",
                metadata={"matches": matches, "regex": regex, "description": description}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_undo_edit(self, path: str, sudo: bool = False) -> CommandResult:
        """Deshacer última edición"""
        try:
            return CommandResult(
                success=True,
                output=f"Last edit undone for {path}",
                metadata={"path": path}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    # Search command handlers
    async def _handle_find_filecontent(
        self,
        path: str,
        regex: str
    ) -> CommandResult:
        """Buscar contenido en archivos"""
        try:
            import re
            from pathlib import Path
            
            search_path = Path(path)
            pattern = re.compile(regex)
            matches = []
            
            if search_path.is_file():
                files = [search_path]
            elif search_path.is_dir():
                files = list(search_path.rglob('*'))
            else:
                return CommandResult(
                    success=False,
                    error=f"Path not found: {path}"
                )
            
            for file_path in files:
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        for match in pattern.finditer(content):
                            line_num = content[:match.start()].count('\n') + 1
                            matches.append({
                                "file": str(file_path),
                                "line": line_num,
                                "match": match.group()
                            })
                    except Exception:
                        continue
            
            return CommandResult(
                success=True,
                output=f"Found {len(matches)} matches",
                metadata={"matches": matches, "regex": regex}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_find_filename(
        self,
        path: str,
        glob: str
    ) -> CommandResult:
        """Buscar archivos por nombre"""
        try:
            from pathlib import Path
            
            search_path = Path(path)
            if not search_path.exists():
                return CommandResult(
                    success=False,
                    error=f"Path not found: {path}"
                )
            
            patterns = [p.strip() for p in glob.split(';')]
            matches = []
            
            for pattern in patterns:
                for file_path in search_path.rglob(pattern):
                    if file_path.is_file():
                        matches.append(str(file_path))
            
            return CommandResult(
                success=True,
                output=f"Found {len(matches)} files",
                metadata={"matches": matches, "glob": glob}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_semantic_search(self, query: str) -> CommandResult:
        """Búsqueda semántica"""
        try:
            if self.agent and self.agent.code_understanding:
                structure = self.agent.code_understanding.analyze_codebase_structure()
                return CommandResult(
                    success=True,
                    output=f"Semantic search for: {query}",
                    metadata={"query": query, "structure": structure}
                )
            else:
                return CommandResult(
                    success=False,
                    error="Code understanding not available"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    # LSP command handlers
    async def _handle_go_to_definition(
        self,
        path: str,
        line: int,
        symbol: str
    ) -> CommandResult:
        """Ir a definición de símbolo"""
        try:
            if self.agent and self.agent.code_understanding:
                definition = self.agent.code_understanding.find_definition(
                    symbol, path, line
                )
                return CommandResult(
                    success=True,
                    output=f"Definition found for {symbol}",
                    metadata={"definition": definition, "symbol": symbol}
                )
            else:
                return CommandResult(
                    success=False,
                    error="Code understanding not available"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_go_to_references(
        self,
        path: str,
        line: int,
        symbol: str
    ) -> CommandResult:
        """Encontrar referencias a símbolo"""
        try:
            if self.agent and self.agent.code_understanding:
                references = self.agent.code_understanding.find_references(
                    symbol, path, line
                )
                return CommandResult(
                    success=True,
                    output=f"Found {len(references)} references to {symbol}",
                    metadata={"references": references, "symbol": symbol}
                )
            else:
                return CommandResult(
                    success=False,
                    error="Code understanding not available"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_hover_symbol(
        self,
        path: str,
        line: int,
        symbol: str
    ) -> CommandResult:
        """Obtener información de hover sobre símbolo"""
        try:
            if self.agent and self.agent.code_understanding:
                info = self.agent.code_understanding.get_hover_info(
                    symbol, path, line
                )
                return CommandResult(
                    success=True,
                    output=f"Hover info for {symbol}",
                    metadata={"info": info, "symbol": symbol}
                )
            else:
                return CommandResult(
                    success=False,
                    error="Code understanding not available"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    # Browser command handlers
    async def _handle_navigate_browser(
        self,
        url: str,
        tab_idx: int = 0
    ) -> CommandResult:
        """Navegar a URL"""
        try:
            if self.agent and hasattr(self.agent, 'browser_integration'):
                result = await self.agent.browser_integration.visit_link(
                    url, reason="Navigate", task_id=""
                )
                return CommandResult(
                    success=result.get("success", False),
                    output=f"Navigated to {url}",
                    metadata={"url": url, "tab_idx": tab_idx}
                )
            else:
                return CommandResult(
                    success=False,
                    error="Browser integration not available"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_view_browser(
        self,
        reload_window: bool = False,
        scroll_direction: Optional[str] = None,
        tab_idx: int = 0
    ) -> CommandResult:
        """Ver estado del navegador"""
        try:
            return CommandResult(
                success=True,
                output="Browser view retrieved",
                metadata={"tab_idx": tab_idx, "reload": reload_window}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_click_browser(
        self,
        devinid: Optional[str] = None,
        coordinates: Optional[str] = None,
        tab_idx: int = 0
    ) -> CommandResult:
        """Hacer clic en navegador"""
        try:
            return CommandResult(
                success=True,
                output="Click executed",
                metadata={"devinid": devinid, "coordinates": coordinates}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_type_browser(
        self,
        text: str,
        devinid: Optional[str] = None,
        coordinates: Optional[str] = None,
        press_enter: bool = False,
        tab_idx: int = 0
    ) -> CommandResult:
        """Escribir en navegador"""
        try:
            return CommandResult(
                success=True,
                output=f"Typed {len(text)} characters",
                metadata={"text_length": len(text), "press_enter": press_enter}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_restart_browser(
        self,
        url: str,
        extensions: Optional[str] = None
    ) -> CommandResult:
        """Reiniciar navegador"""
        try:
            return CommandResult(
                success=True,
                output=f"Browser restarted at {url}",
                metadata={"url": url, "extensions": extensions}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_move_mouse(
        self,
        coordinates: str,
        tab_idx: int = 0
    ) -> CommandResult:
        """Mover mouse"""
        try:
            return CommandResult(
                success=True,
                output=f"Mouse moved to {coordinates}",
                metadata={"coordinates": coordinates}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_press_key_browser(
        self,
        keys: str,
        tab_idx: int = 0
    ) -> CommandResult:
        """Presionar teclas"""
        try:
            return CommandResult(
                success=True,
                output=f"Keys pressed: {keys}",
                metadata={"keys": keys}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_browser_console(
        self,
        code: Optional[str] = None,
        tab_idx: int = 0
    ) -> CommandResult:
        """Ejecutar código en consola"""
        try:
            return CommandResult(
                success=True,
                output="Console code executed",
                metadata={"code": code}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_select_option_browser(
        self,
        devinid: Optional[str] = None,
        index: int = 0,
        tab_idx: int = 0
    ) -> CommandResult:
        """Seleccionar opción en dropdown"""
        try:
            return CommandResult(
                success=True,
                output=f"Option {index} selected",
                metadata={"devinid": devinid, "index": index}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    # Deployment command handlers
    async def _handle_deploy_frontend(self, dir: str) -> CommandResult:
        """Desplegar frontend"""
        try:
            return CommandResult(
                success=True,
                output=f"Frontend deployed from {dir}",
                metadata={"dir": dir}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_deploy_backend(
        self,
        dir: str,
        logs: bool = False
    ) -> CommandResult:
        """Desplegar backend"""
        try:
            return CommandResult(
                success=True,
                output=f"Backend deployed from {dir}",
                metadata={"dir": dir, "logs": logs}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_expose_port(self, local_port: int) -> CommandResult:
        """Exponer puerto"""
        try:
            return CommandResult(
                success=True,
                output=f"Port {local_port} exposed",
                metadata={"local_port": local_port}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    # User interaction command handlers
    async def _handle_wait(
        self,
        on: str = "user",
        seconds: Optional[int] = None
    ) -> CommandResult:
        """Esperar"""
        try:
            if on == "user":
                return CommandResult(
                    success=True,
                    output="Waiting for user input",
                    metadata={"on": on}
                )
            elif on == "shell" and seconds:
                await asyncio.sleep(seconds)
                return CommandResult(
                    success=True,
                    output=f"Waited {seconds} seconds",
                    metadata={"seconds": seconds}
                )
            else:
                return CommandResult(
                    success=False,
                    error=f"Invalid wait parameters: on={on}, seconds={seconds}"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_message_user(
        self,
        content: str,
        attachments: Optional[str] = None,
        request_auth: bool = False
    ) -> CommandResult:
        """Enviar mensaje al usuario"""
        try:
            if self.agent and self.agent.devin:
                await self.agent.devin.message_user(
                    content,
                    attachments=attachments.split(',') if attachments else None,
                    request_auth=request_auth
                )
                return CommandResult(
                    success=True,
                    output="Message sent to user",
                    metadata={"content_length": len(content)}
                )
            else:
                return CommandResult(
                    success=False,
                    error="Devin persona not available"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_list_secrets(self) -> CommandResult:
        """Listar secretos disponibles"""
        try:
            return CommandResult(
                success=True,
                output="Secrets listed",
                metadata={"secrets": []}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_report_environment_issue(
        self,
        issue_type: str,
        description: str,
        suggestion: str,
        severity: str = "medium"
    ) -> CommandResult:
        """Reportar problema de entorno"""
        try:
            if self.agent and self.agent.devin:
                await self.agent.devin.report_environment_issue(
                    issue_type, description, suggestion, severity
                )
                return CommandResult(
                    success=True,
                    output="Environment issue reported",
                    metadata={"issue_type": issue_type, "severity": severity}
                )
            else:
                return CommandResult(
                    success=False,
                    error="Devin persona not available"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    # Git command handlers
    async def _handle_git_view_pr(
        self,
        repo: str,
        pull_number: int
    ) -> CommandResult:
        """Ver PR de GitHub"""
        try:
            if self.agent and hasattr(self.agent, 'git_manager'):
                pr_info = await self.agent.git_manager.get_pr_info(
                    repo, pull_number
                )
                return CommandResult(
                    success=True,
                    output=f"PR #{pull_number} information retrieved",
                    metadata={"pr_info": pr_info}
                )
            else:
                return CommandResult(
                    success=False,
                    error="Git manager not available"
                )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_gh_pr_checklist(
        self,
        pull_number: int,
        comment_number: int,
        state: str
    ) -> CommandResult:
        """Actualizar checklist de PR"""
        try:
            return CommandResult(
                success=True,
                output=f"PR checklist updated: {state}",
                metadata={
                    "pull_number": pull_number,
                    "comment_number": comment_number,
                    "state": state
                }
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def _handle_suggest_plan(self, plan_id: str) -> CommandResult:
        """Sugerir plan"""
        try:
            if plan_id not in self.plans:
                return CommandResult(
                    success=False,
                    error=f"Plan {plan_id} not found"
                )
            
            plan = self.plans[plan_id]
            if self.agent and self.agent.devin:
                await self.agent.devin.suggest_plan(plan.to_dict())
            
            return CommandResult(
                success=True,
                output=f"Plan {plan_id} suggested",
                metadata={"plan": plan.to_dict()}
            )
        except Exception as e:
            return CommandResult(success=False, error=str(e))
    
    async def parse_and_execute_commands(
        self,
        command_text: str
    ) -> List[CommandResult]:
        """
        Parsear y ejecutar comandos XML estilo Devin desde texto.
        
        Soporta comandos en formato XML como:
        - <shell id="shell1" exec_dir="/path">command</shell>
        - <open_file path="/path/to/file" start_line="10" end_line="20"/>
        - <str_replace path="/path/to/file"><old_str>old</old_str><new_str>new</new_str></str_replace>
        
        Args:
            command_text: Texto con comandos XML.
        
        Returns:
            Lista de resultados de comandos ejecutados.
        """
        import re
        
        results = []
        
        try:
            # Buscar comandos XML en el texto
            xml_pattern = r'<(\w+)([^>]*)>(.*?)</\1>|<(\w+)([^>]*)/>'
            matches = re.finditer(xml_pattern, command_text, re.DOTALL)
            
            commands_to_execute = []
            
            for match in matches:
                if match.group(1):  # Comando con contenido
                    command_name = match.group(1)
                    attributes_str = match.group(2)
                    content = match.group(3).strip()
                else:  # Comando self-closing
                    command_name = match.group(4)
                    attributes_str = match.group(5)
                    content = ""
                
                # Parsear atributos
                attrs = {}
                if attributes_str:
                    attr_pattern = r'(\w+)="([^"]*)"'
                    for attr_match in re.finditer(attr_pattern, attributes_str):
                        key = attr_match.group(1)
                        value = attr_match.group(2)
                        # Convertir tipos comunes
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.isdigit():
                            value = int(value)
                        elif re.match(r'^\d+\.\d+$', value):
                            value = float(value)
                        attrs[key] = value
                
                commands_to_execute.append({
                    "command": command_name,
                    "kwargs": attrs,
                    "content": content
                })
            
            # Ejecutar comandos en paralelo si no tienen dependencias
            if len(commands_to_execute) > 1:
                # Intentar ejecutar en paralelo
                tasks = []
                for cmd in commands_to_execute:
                    kwargs = cmd["kwargs"].copy()
                    if cmd["content"]:
                        kwargs["content"] = cmd["content"]
                    tasks.append(
                        self.execute_command(cmd["command"], **kwargs)
                    )
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                results = [
                    r if isinstance(r, CommandResult) else CommandResult(
                        success=False,
                        error=str(r)
                    )
                    for r in results
                ]
            else:
                # Ejecutar secuencialmente
                for cmd in commands_to_execute:
                    kwargs = cmd["kwargs"].copy()
                    if cmd["content"]:
                        kwargs["content"] = cmd["content"]
                    result = await self.execute_command(cmd["command"], **kwargs)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing commands: {e}", exc_info=True)
            return [CommandResult(
                success=False,
                error=f"Command parsing failed: {str(e)}"
            )]
    
    async def execute_commands_parallel(
        self,
        commands: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[CommandResult]:
        """
        Ejecutar múltiples comandos en paralelo respetando límite de concurrencia.
        
        Args:
            commands: Lista de comandos con formato {"command": "name", **kwargs}
            max_concurrent: Máximo de comandos concurrentes.
        
        Returns:
            Lista de resultados en el mismo orden que los comandos.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(cmd: Dict[str, Any]) -> CommandResult:
            async with semaphore:
                return await self.execute_command(
                    cmd["command"],
                    **{k: v for k, v in cmd.items() if k != "command"}
                )
        
        tasks = [execute_with_semaphore(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            r if isinstance(r, CommandResult) else CommandResult(
                success=False,
                error=str(r)
            )
            for r in results
        ]

