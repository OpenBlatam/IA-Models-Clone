"""
Command Executor - Ejecuta comandos reales
==========================================

Ejecuta comandos Python, shell, o llama a APIs con manejo robusto
de errores y timeouts.
"""

import asyncio
import logging
import subprocess
import sys
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Tipos de comandos soportados"""
    AUTO = "auto"
    PYTHON = "python"
    SHELL = "shell"
    API = "api"


@dataclass
class ExecutionResult:
    """
    Resultado de ejecución de comando.
    
    Attributes:
        success: True si la ejecución fue exitosa.
        output: Salida del comando (si fue exitoso).
        error: Mensaje de error (si falló).
        execution_time: Tiempo de ejecución en segundos.
        command_type: Tipo de comando ejecutado.
    """
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    command_type: str = "auto"


class CommandExecutor:
    """
    Ejecutor de comandos con soporte para múltiples tipos.
    
    Soporta ejecución de:
    - Código Python
    - Comandos shell
    - Llamadas API HTTP/HTTPS
    """
    
    def __init__(
        self,
        timeout: float = 300.0,
        working_dir: Optional[str] = None
    ) -> None:
        """
        Inicializar ejecutor de comandos.
        
        Args:
            timeout: Timeout en segundos para cada comando (default: 300.0).
            working_dir: Directorio de trabajo (default: directorio actual).
        
        Raises:
            ValueError: Si el timeout es inválido.
        """
        if timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {timeout}")
        
        self.timeout: float = timeout
        self.working_dir: Path = (
            Path(working_dir) if working_dir else Path.cwd()
        )
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    async def execute(
        self,
        command: str,
        command_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Ejecutar un comando.
        
        Args:
            command: Comando a ejecutar.
            command_type: Tipo de comando (auto, python, shell, api).
        
        Returns:
            Dict con resultado de ejecución.
        
        Raises:
            ValueError: Si el comando está vacío o el tipo es inválido.
        """
        if not command or not command.strip():
            raise ValueError("Command cannot be empty")
        
        start_time = datetime.now()
        detected_type = command_type
        
        try:
            # Detectar tipo de comando automáticamente
            if command_type == "auto":
                detected_type = self._detect_command_type(command)
            
            logger.info(
                f"🔨 Executing {detected_type} command: {command[:100]}..."
            )
            
            # Ejecutar según el tipo
            result = await self._execute_by_type(command, detected_type)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                success=True,
                output=result,
                execution_time=execution_time,
                command_type=detected_type
            ).__dict__
        
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Command timeout after {self.timeout}s"
            logger.error(f"⏱️ {error_msg}")
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                command_type=detected_type
            ).__dict__
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"❌ Execution error: {error_msg}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                command_type=detected_type
            ).__dict__
    
    def _detect_command_type(self, command: str) -> str:
        """
        Detectar tipo de comando automáticamente.
        
        Args:
            command: Comando a analizar.
        
        Returns:
            Tipo de comando detectado.
        """
        command = command.strip()
        
        # Python code (prefijos explícitos)
        if command.startswith(("python:", "py:")):
            return CommandType.PYTHON.value
        
        # Shell command (prefijos explícitos)
        if command.startswith(("shell:", "sh:")):
            return CommandType.SHELL.value
        
        # API call (URLs)
        if command.startswith(("http://", "https://")):
            return CommandType.API.value
        
        # Detectar código Python por contenido
        python_keywords = ["import ", "def ", "class ", "print(", "="]
        if any(keyword in command for keyword in python_keywords):
            return CommandType.PYTHON.value
        
        # Por defecto shell
        return CommandType.SHELL.value
    
    async def _execute_by_type(
        self,
        command: str,
        command_type: str
    ) -> str:
        """
        Ejecutar comando según su tipo.
        
        Args:
            command: Comando a ejecutar.
            command_type: Tipo de comando.
        
        Returns:
            Salida del comando.
        
        Raises:
            ValueError: Si el tipo de comando es inválido.
            RuntimeError: Si la ejecución falla.
        """
        if command_type == CommandType.PYTHON.value:
            return await self._execute_python(command)
        elif command_type == CommandType.SHELL.value:
            return await self._execute_shell(command)
        elif command_type == CommandType.API.value:
            return await self._execute_api_call(command)
        else:
            raise ValueError(f"Unknown command type: {command_type}")
    
    async def _execute_python(self, code: str) -> str:
        """
        Ejecutar código Python.
        
        Args:
            code: Código Python a ejecutar.
        
        Returns:
            Salida estándar del código.
        
        Raises:
            RuntimeError: Si la ejecución falla.
        """
        # Limpiar prefijos
        code = code.replace("python:", "").replace("py:", "").strip()
        
        if not code:
            raise ValueError("Python code cannot be empty")
        
        # Crear script temporal
        timestamp = datetime.now().timestamp()
        script_file = self.working_dir / f"temp_script_{timestamp}.py"
        
        try:
            # Escribir código
            script_file.write_text(code, encoding='utf-8')
            
            # Ejecutar
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            if process.returncode != 0:
                error_output = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(
                    f"Python execution failed (code {process.returncode}): "
                    f"{error_output}"
                )
            
            return stdout.decode('utf-8', errors='ignore')
        
        except asyncio.TimeoutError:
            raise RuntimeError(f"Python execution timeout after {self.timeout}s")
        
        finally:
            # Limpiar archivo
            if script_file.exists():
                try:
                    script_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp script: {e}")
    
    async def _execute_shell(self, command: str) -> str:
        """
        Ejecutar comando shell.
        
        Args:
            command: Comando shell a ejecutar.
        
        Returns:
            Salida estándar del comando.
        
        Raises:
            RuntimeError: Si la ejecución falla.
        """
        # Limpiar prefijos
        command = command.replace("shell:", "").replace("sh:", "").strip()
        
        if not command:
            raise ValueError("Shell command cannot be empty")
        
        # Determinar shell según OS
        if sys.platform == "win32":
            shell_cmd = ["cmd", "/c", command]
        else:
            shell_cmd = ["/bin/bash", "-c", command]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            if process.returncode != 0:
                error_output = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(
                    f"Shell execution failed (code {process.returncode}): "
                    f"{error_output}"
                )
            
            return stdout.decode('utf-8', errors='ignore')
        
        except asyncio.TimeoutError:
            raise RuntimeError(f"Shell execution timeout after {self.timeout}s")
    
    async def _execute_api_call(self, url: str) -> str:
        """
        Ejecutar llamada API HTTP/HTTPS.
        
        Args:
            url: URL a llamar.
        
        Returns:
            Contenido de la respuesta.
        
        Raises:
            RuntimeError: Si la llamada falla.
            ImportError: Si no hay librerías HTTP disponibles.
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        # Intentar usar httpx (más moderno)
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        
        except ImportError:
            # Fallback a requests si httpx no está disponible
            try:
                import requests
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except ImportError:
                raise ImportError(
                    "No HTTP library available. Install 'httpx' or 'requests'"
                )
        
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}") from e
