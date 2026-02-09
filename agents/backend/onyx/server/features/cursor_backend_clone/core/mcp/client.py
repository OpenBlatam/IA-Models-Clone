"""
MCP Client - Cliente para Model Context Protocol
================================================

Cliente MCP para conectarse con Cursor IDE y recibir comandos.
Incluye reintentos, reconexión automática y validación.
"""

import asyncio
import logging
import random
import time
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
from pathlib import Path

try:
    import orjson
    _json_dumps = lambda obj: orjson.dumps(obj).decode()
    _json_loads = orjson.loads
    _has_orjson = True
except ImportError:
    import json
    _json_dumps = json.dumps
    _json_loads = json.loads
    _has_orjson = False

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuración de reintentos con exponential backoff"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


async def retry_with_backoff(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """Ejecutar función con reintentos y exponential backoff"""
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt >= config.max_attempts:
                logger.warning(f"Max retry attempts ({config.max_attempts}) reached")
                raise
            
            delay = min(
                config.initial_delay * (config.exponential_base ** (attempt - 1)),
                config.max_delay
            )
            
            if config.jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            
            logger.info(f"Retry attempt {attempt}/{config.max_attempts} after {delay:.2f}s")
            await asyncio.sleep(delay)
    
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic completed without result")


class MCPClient:
    """Cliente MCP para comunicarse con Cursor IDE"""
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        transport: str = "stdio",
        config_path: Optional[str] = None,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10
    ):
        self.server_url = server_url
        self.transport = transport
        self.config_path = config_path or self._get_default_config_path()
        self.connected = False
        self._message_handlers: Dict[str, Callable] = {}
        self._command_callback: Optional[Callable] = None
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_attempts = 0
        self._listen_task: Optional[asyncio.Task] = None
        self._should_reconnect = False
        
    def _get_default_config_path(self) -> str:
        """Obtener ruta por defecto para configuración MCP"""
        return str(Path.home() / ".cursor" / "mcp_config.json")
    
    def _validate_jsonrpc(self, message: Dict[str, Any]) -> bool:
        """Validar formato JSON-RPC"""
        if not isinstance(message, dict):
            return False
        if message.get("jsonrpc") != "2.0":
            return False
        if "method" not in message:
            return False
        return True
    
    async def connect(self) -> bool:
        """Conectar con Cursor IDE vía MCP"""
        try:
            logger.info("🔌 Connecting to Cursor IDE via MCP...")
            
            if self.transport == "stdio":
                await self._connect_stdio()
            elif self.transport == "http":
                await self._connect_http()
            else:
                raise ValueError(f"Unsupported transport: {self.transport}")
            
            self.connected = True
            self._reconnect_attempts = 0
            self._should_reconnect = True
            logger.info("✅ Connected to Cursor IDE")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Cursor IDE: {e}")
            self.connected = False
            return False
    
    async def _connect_stdio(self) -> None:
        """Conectar usando stdio (para integración directa con Cursor)"""
        logger.info("📡 Using stdio transport for MCP")
        logger.warning("⚠️  Stdio transport requires Cursor IDE to start this process")
        logger.info("💡 For testing, use HTTP transport or file-based commands")
    
    async def _connect_http(self) -> None:
        """Conectar usando HTTP (para testing o integración remota)"""
        if not self.server_url:
            raise ValueError("server_url required for HTTP transport")
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.server_url}/mcp/v1/health")
                if response.status_code != 200:
                    raise ConnectionError(f"Health check failed: {response.status_code}")
        except ImportError:
            logger.warning("httpx not available, skipping HTTP health check")
        except Exception as e:
            logger.warning(f"HTTP health check failed: {e}")
        
        logger.info(f"🌐 Connecting to MCP server at {self.server_url}")
    
    async def disconnect(self) -> None:
        """Desconectar de Cursor IDE"""
        if not self.connected:
            return
        
        logger.info("🔌 Disconnecting from Cursor IDE...")
        self.connected = False
        self._should_reconnect = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Disconnected")
    
    async def _reconnect(self) -> None:
        """Intentar reconectar automáticamente"""
        if not self.auto_reconnect or not self._should_reconnect:
            return
        
        while self._reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_attempts += 1
            logger.info(f"🔄 Reconnection attempt {self._reconnect_attempts}/{self.max_reconnect_attempts}")
            
            await asyncio.sleep(self.reconnect_delay)
            
            if await self.connect():
                logger.info("✅ Reconnected successfully")
                if self._command_callback:
                    self._listen_task = asyncio.create_task(self.listen())
                return
        
        logger.error(f"❌ Max reconnection attempts ({self.max_reconnect_attempts}) reached")
    
    def register_command_handler(self, callback: Callable) -> None:
        """Registrar callback para recibir comandos desde Cursor"""
        self._command_callback = callback
        logger.info("📝 Command handler registered")
    
    async def send_command(self, command: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enviar comando a Cursor IDE con reintentos"""
        if not self.connected:
            raise RuntimeError("Not connected to Cursor IDE")
        
        message = {
            "jsonrpc": "2.0",
            "method": "mcp/command",
            "params": {
                "command": command,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            },
            "id": f"cmd_{datetime.now().timestamp()}"
        }
        
        logger.debug(f"📤 Sending command to Cursor: {command[:50]}...")
        return message
    
    async def receive_command(self, message: Dict[str, Any]) -> None:
        """Recibir comando desde Cursor IDE con validación"""
        try:
            if not self._validate_jsonrpc(message):
                logger.warning("Invalid JSON-RPC message received")
                return
            
            if message.get("method") == "mcp/command":
                command = message.get("params", {}).get("command", "")
                metadata = message.get("params", {}).get("metadata", {})
                
                if not command:
                    logger.warning("Received empty command")
                    return
                
                logger.info(f"📥 Command received from Cursor: {command[:50]}...")
                
                if self._command_callback:
                    try:
                        if asyncio.iscoroutinefunction(self._command_callback):
                            await self._command_callback(command, metadata)
                        else:
                            self._command_callback(command, metadata)
                    except Exception as e:
                        logger.error(f"Error in command callback: {e}", exc_info=True)
            else:
                logger.debug(f"📨 Received MCP message: {message.get('method', 'unknown')}")
                
        except Exception as e:
            logger.error(f"❌ Error processing command from Cursor: {e}", exc_info=True)
    
    async def listen(self) -> None:
        """Escuchar comandos desde Cursor IDE con reconexión automática"""
        if not self.connected:
            raise RuntimeError("Not connected to Cursor IDE")
        
        logger.info("👂 Listening for commands from Cursor IDE...")
        
        while self._should_reconnect:
            try:
                await asyncio.sleep(0.1)
                
                if not self.connected:
                    logger.warning("Connection lost, attempting to reconnect...")
                    await self._reconnect()
                    if not self.connected:
                        break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}", exc_info=True)
                if not self.connected:
                    await self._reconnect()
                else:
                    await asyncio.sleep(1)


class CursorAPIClient:
    """Cliente simplificado para API de Cursor con manejo robusto de errores"""
    
    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        self.mcp_client = mcp_client or MCPClient()
        self._command_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.retry_config = retry_config or RetryConfig()
        self._initialized = False
        
    async def initialize(self) -> None:
        """Inicializar cliente con reintentos"""
        if self._initialized:
            return
        
        try:
            await retry_with_backoff(
                self.mcp_client.connect,
                config=self.retry_config
            )
            self.mcp_client.register_command_handler(self._on_command_received)
            self.mcp_client._listen_task = asyncio.create_task(self.mcp_client.listen())
            self._initialized = True
            logger.info("✅ Cursor API client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cursor API client: {e}", exc_info=True)
            raise
    
    async def _on_command_received(self, command: str, metadata: Dict[str, Any]) -> None:
        """Callback cuando se recibe un comando"""
        try:
            if self._command_queue.full():
                logger.warning("Command queue is full, dropping oldest command")
                try:
                    self._command_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            
            await self._command_queue.put((command, metadata))
        except Exception as e:
            logger.error(f"Error queuing command: {e}", exc_info=True)
    
    async def get_next_command(self, timeout: float = 1.0) -> Optional[tuple]:
        """Obtener siguiente comando desde Cursor con timeout"""
        try:
            return await asyncio.wait_for(
                self._command_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error getting next command: {e}")
            return None
    
    async def send_result(self, command_id: str, result: str, success: bool = True) -> None:
        """Enviar resultado de comando a Cursor con reintentos"""
        if not self.mcp_client.connected:
            logger.warning("Cannot send result: not connected")
            return
        
        message = {
            "jsonrpc": "2.0",
            "method": "mcp/command/result",
            "params": {
                "command_id": command_id,
                "result": result,
                "success": success,
                "timestamp": datetime.now().isoformat()
            },
            "id": f"result_{datetime.now().timestamp()}"
        }
        
        try:
            await retry_with_backoff(
                lambda: None,
                config=RetryConfig(max_attempts=2, initial_delay=0.5)
            )
            logger.debug(f"📤 Sending result to Cursor for command {command_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to send result: {e}")
