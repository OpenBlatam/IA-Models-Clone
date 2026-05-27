import socket
import json
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OSNexus")

class SysCallError(Exception):
    pass

class OSEnvironment:
    """
    User Space API (libc) para el AI-OS (TruthGPT).
    Envuelve los Sockets TCP asíncronos del Kernel Rust en una API limpia.
    """
    def __init__(self, host: str = '127.0.0.1', port: int = 50051):
        self.host = host
        self.port = port
        self.pid = "python_brain"

    def _syscall(self, call_type: str, payload: str, priority: int = 5) -> Dict[str, Any]:
        request = {
            "pid": self.pid,
            "priority": priority,
            "call_type": call_type,
            "payload": payload
        }
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.settimeout(5.0) 
                s.connect((self.host, self.port))
                s.sendall(json.dumps(request).encode('utf-8'))
                
                data = s.recv(4096)
                if not data:
                    raise SysCallError("Empty response from Kernel")

                response = json.loads(data.decode('utf-8'))
                
                if response.get("status") == "error":
                    raise SysCallError(response.get("message", "Unknown error"))
                if response.get("status") == "circuit_broken":
                    raise SysCallError(f"CIRCUIT_BROKEN: {response.get('message')}")
                    
                return response

            except ConnectionRefusedError:
                raise SysCallError(f"Connection Refused en {self.host}:{self.port}. ¿Está el nexus_daemon corriendo?")
            except socket.timeout:
                raise SysCallError("Timeout esperando respuesta del Kernel.")
            except Exception as e:
                if isinstance(e, SysCallError):
                    raise
                raise SysCallError(f"Unexpected error: {e}")

    # =========================================================================
    # VFS (Virtual File System) SysCalls
    # =========================================================================
    
    def vfs_read(self, path: str) -> str:
        """Lee un archivo del sandbox seguro (VFS Jail)."""
        res = self._syscall("SYS_FILE_READ", path, priority=3)
        return res.get("message", "")

    def vfs_write(self, path: str, content: str) -> None:
        """Escribe un archivo en el sandbox seguro (VFS Jail)."""
        payload = json.dumps({"path": path, "content": content})
        self._syscall("SYS_FILE_WRITE", payload, priority=3)

    # =========================================================================
    # MM (Memory Management - DashMap L1 Cache) SysCalls
    # =========================================================================

    def mem_read(self, key: str) -> str:
        """Lee una variable de la caché hiper-rápida L1."""
        res = self._syscall("SYS_MEM_READ", key, priority=2)
        return res.get("message", "")

    def mem_write(self, key: str, value: str) -> None:
        """Escribe una variable en la caché L1 (compartida por todos los hilos)."""
        payload = json.dumps({"key": key, "value": value})
        self._syscall("SYS_MEM_WRITE", payload, priority=2)

    # =========================================================================
    # IPC (Inter-Process Communication) SysCalls
    # =========================================================================
    
    def ipc_send(self, to_agent: str, message: str) -> None:
        """Envía un mensaje hiper-rápido a la bandeja de entrada de otro agente usando L1 Cache."""
        key = f"ipc:inbox:{to_agent}"
        # Se podría implementar un append, pero por ahora sobreescribe o asume lectura destructiva
        self.mem_write(key, message)
        
    def ipc_read(self, my_agent_name: str) -> str:
        """Lee la bandeja de entrada de este agente desde la caché L1."""
        key = f"ipc:inbox:{my_agent_name}"
        return self.mem_read(key)

    # =========================================================================
    # NETWORK OFFLOADING SysCalls
    # =========================================================================

    def http_get(self, url: str) -> str:
        """Delega una petición HTTP GET al Kernel de Rust (descarga ultra rápida asíncrona)."""
        res = self._syscall("SYS_HTTP_GET", url, priority=4)
        return res.get("message", "")

    # =========================================================================
    # SCHEDULER & EXEC SysCalls
    # =========================================================================

    def execute_code(self, python_code: str) -> str:
        """Ejecuta código de forma nativa e independiente (aislado)."""
        res = self._syscall("SYS_EXEC", python_code, priority=5)
        return res.get("message", "")

    def kill_process(self, target_pid: str) -> None:
        """Asesina la ejecución de un sub-agente usando su PID."""
        self._syscall("SYS_KILL", target_pid, priority=1)

    def ping(self) -> str:
        """Prueba de vida del Kernel Ring 0."""
        res = self._syscall("SYS_PING", "ping", priority=1)
        return res.get("message", "")

# Exportamos una instancia global estándar (como la librería os de Python)
sys = OSEnvironment()
