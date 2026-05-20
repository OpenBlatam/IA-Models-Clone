"""
Command Executor - Ejecuta comandos reales
==========================================

Ejecuta comandos Python, shell, o llama a APIs con detección automática
de tipo de comando y manejo robusto de errores.
"""

import asyncio
import logging
import subprocess
import sys
import os
import tempfile
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from datetime import datetime

from .observability import observe_async

logger = logging.getLogger(__name__)

CommandType = Literal["auto", "python", "shell", "api"]


class CommandExecutor:
    """
    Ejecutor de comandos con soporte para múltiples tipos.
    
    Soporta:
    - Código Python (ejecutado en subproceso)
    - Comandos shell (bash/cmd según OS)
    - Llamadas HTTP/HTTPS
    
    Attributes:
        timeout: Tiempo máximo de ejecución en segundos
        working_dir: Directorio de trabajo para comandos
    """
    
    def __init__(
        self,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None
    ) -> None:
        """
        Inicializar ejecutor de comandos.
        
        Args:
            timeout: Tiempo máximo de ejecución en segundos (default: desde constantes)
            working_dir: Directorio de trabajo (default: directorio actual)
        """
        if timeout is None:
            from .constants import DEFAULT_COMMAND_TIMEOUT
            timeout = DEFAULT_COMMAND_TIMEOUT
        
        self.timeout = timeout
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    @observe_async(operation_name="command_execution", log_args=False, track_metrics=True)
    async def execute(
        self,
        command: str,
        command_type: CommandType = "auto"
    ) -> Dict[str, Any]:
        """
        Ejecutar un comando
        
        Args:
            command: Comando a ejecutar
            command_type: Tipo de comando (auto, python, shell, api)
        
        Returns:
            Dict con resultado de ejecución
        """
        start_time = datetime.now()
        
        try:
            # Detectar tipo de comando automáticamente
            if command_type == "auto":
                command_type = self._detect_command_type(command)
            
            logger.info(f"🔨 Executing {command_type} command: {command[:100]}...")
            
            # Ejecutar según el tipo
            if command_type == "python":
                result = await self._execute_python(command)
            elif command_type == "shell":
                result = await self._execute_shell(command)
            elif command_type == "api":
                result = await self._execute_api_call(command)
            else:
                # Por defecto, intentar como Python
                result = await self._execute_python(command)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "output": result,
                "execution_time": execution_time,
                "command_type": command_type
            }
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Command timeout after {self.timeout}s"
            logger.error(f"⏱️ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "command_type": command_type
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"❌ Execution error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "command_type": command_type
            }
    
    def _detect_command_type(self, command: str) -> CommandType:
        """
        Detectar tipo de comando automáticamente basado en su contenido.
        
        Args:
            command: Comando a analizar
            
        Returns:
            Tipo de comando detectado: "python", "shell", o "api"
        """
        command = command.strip()
        
        # Python code
        if command.startswith("python:") or command.startswith("py:"):
            return "python"
        
        # Shell command
        if command.startswith("shell:") or command.startswith("sh:"):
            return "shell"
        
        # API call
        if command.startswith("http://") or command.startswith("https://"):
            return "api"
        
        # Si contiene imports o es código Python
        if any(keyword in command for keyword in ["import ", "def ", "class ", "print(", "="]):
            return "python"
        
        # Por defecto shell
        return "shell"
    
    async def _execute_python(self, code: str) -> str:
        """
        Ejecutar código Python en un subproceso aislado.
        
        Args:
            code: Código Python a ejecutar
            
        Returns:
            Salida estándar del código ejecutado
            
        Raises:
            Exception: Si el código falla o retorna código de error
        """
        # Limpiar prefijos
        code = code.replace("python:", "").replace("py:", "").strip()
        
        # Crear archivo temporal seguro
        import uuid
        temp_dir = tempfile.gettempdir()
        unique_id = uuid.uuid4().hex[:12]
        script_file = Path(temp_dir) / f"cursor_agent_{unique_id}.py"
        
        try:
            # Escribir código con permisos restrictivos
            script_file.write_text(code, encoding='utf-8')
            os.chmod(script_file, 0o600)  # Solo lectura/escritura para el propietario
            
            # Ejecutar con entorno limitado
            env = os.environ.copy()
            # Remover variables de entorno sensibles
            from .constants import SENSITIVE_ENV_VARS
            for key in list(env.keys()):
                if any(sensitive in key.upper() for sensitive in SENSITIVE_ENV_VARS):
                    del env[key]
            
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir),
                env=env
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            # Limpiar archivo
            try:
                if script_file.exists():
                    script_file.unlink()
            except Exception:
                pass  # Ignorar errores de limpieza
            
            if process.returncode != 0:
                error_output = stderr.decode('utf-8', errors='ignore')
                # Sanitizar mensaje de error
                from .security import SecurityValidator
                validator = SecurityValidator()
                safe_error = validator.sanitize_error_message(error_output)
                raise Exception(f"Python execution failed: {safe_error}")
            
            return stdout.decode('utf-8', errors='ignore')
            
        except Exception as e:
            # Limpiar archivo en caso de error
            try:
                if script_file.exists():
                    script_file.unlink()
            except Exception:
                pass
            raise
    
    async def _execute_shell(self, command: str) -> str:
        """
        Ejecutar comando shell (bash en Linux/Mac, cmd en Windows).
        
        Args:
            command: Comando shell a ejecutar
            
        Returns:
            Salida estándar del comando
            
        Raises:
            Exception: Si el comando falla o retorna código de error
        """
        # Limpiar prefijos
        command = command.replace("shell:", "").replace("sh:", "").strip()
        
        # Determinar shell según OS
        if sys.platform == "win32":
            shell_cmd = ["cmd", "/c", command]
        else:
            shell_cmd = ["/bin/bash", "-c", command]
        
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
            raise Exception(f"Shell execution failed: {error_output}")
        
        return stdout.decode('utf-8', errors='ignore')
    
    async def _execute_api_call(self, url: str) -> str:
        """
        Ejecutar llamada HTTP GET a una URL con validación de seguridad y circuit breaker.
        
        Args:
            url: URL a llamar
            
        Returns:
            Contenido de la respuesta HTTP
            
        Raises:
            ValueError: Si la URL no es válida o no está permitida
            ImportError: Si no hay librerías HTTP disponibles
            httpx.HTTPStatusError: Si la respuesta HTTP tiene error
            Exception: Si el circuit breaker está abierto
        """
        # Validar URL antes de hacer la llamada
        from .security import SecurityValidator
        validator = SecurityValidator()
        
        is_valid, error_msg = validator.validate_url(url)
        if not is_valid:
            raise ValueError(error_msg or "Invalid URL")
        
        # Obtener o crear circuit breaker para esta URL
        from .circuit_breaker import CircuitBreaker
        from urllib.parse import urlparse
        
        parsed_url = urlparse(url)
        circuit_key = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if not hasattr(self, '_circuit_breakers'):
            self._circuit_breakers = {}
        
        if circuit_key not in self._circuit_breakers:
            from .constants import (
                DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
            )
            self._circuit_breakers[circuit_key] = CircuitBreaker(
                failure_threshold=DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                recovery_timeout=DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                name=f"http_{parsed_url.netloc}"
            )
        
        circuit_breaker = self._circuit_breakers[circuit_key]
        
        async def _make_request() -> str:
            """Función interna para hacer la petición HTTP"""
            try:
                import httpx
                from .constants import DEFAULT_HTTP_TIMEOUT, MAX_RESPONSE_SIZE
                
                async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                    response = await client.get(url, follow_redirects=True)
                    response.raise_for_status()
                    
                    content_length = len(response.content)
                    if content_length > MAX_RESPONSE_SIZE:
                        raise ValueError(
                            f"Response too large: {content_length} bytes "
                            f"(max: {MAX_RESPONSE_SIZE} bytes)"
                        )
                    
                    return response.text
            except httpx.HTTPStatusError as e:
                raise Exception(f"HTTP error {e.response.status_code}: {e.response.text[:200]}")
            except httpx.TimeoutException:
                raise Exception(f"Request timeout after {DEFAULT_HTTP_TIMEOUT}s")
            except Exception as e:
                raise Exception(f"HTTP request failed: {str(e)}")
        
        return await circuit_breaker.call(_make_request)


