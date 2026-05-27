import logging
from typing import Dict, Any, Type
from agents.razonamiento_planificacion.tools.base import BaseTool, ToolResult
import agents.os_nexus as os_nexus

logger = logging.getLogger(__name__)

class NexusTool(BaseTool):
    """
    Herramienta que delega la ejecución de código o acceso al sistema
    directamente al Nexus Kernel (Daemon de Rust) para máxima seguridad y concurrencia.
    """
    
    name: str = "nexus_kernel"
    description: str = (
        "Útil para ejecutar código en un sandbox seguro o interactuar con el AI-OS Kernel. "
        "Uso: Envía 'SYS_EXEC' para ejecutar código Python/Bash asilado, "
        "'SYS_MEM_WRITE'/'SYS_MEM_READ' para leer/escribir en la caché L1 ultrarrápida, "
        "'SYS_FILE_WRITE'/'SYS_FILE_READ' para gestionar archivos dentro de una Jaula Virtual (VFS Sandbox) segura, "
        "o 'SYS_KILL' pasándole un PID para asesinar un proceso rebelde."
    )
    
    def __init__(self):
        super().__init__()
        self.sys = os_nexus.sys
        logger.info("NexusTool inicializada con la API os_nexus (libc).")

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        call_type = params.get("call_type", "SYS_PING")
        payload = params.get("payload", "")
        
        try:
            logger.info(f"Delegando SysCall al Kernel vía os_nexus: {call_type}")
            
            # Mapeamos los SysCalls al wrapper de alto nivel "libc"
            output = ""
            if call_type == "SYS_PING":
                output = self.sys.ping()
            elif call_type == "SYS_EXEC":
                output = self.sys.execute_code(payload)
            elif call_type == "SYS_MEM_WRITE":
                # Esperamos payload en formato JSON string: {"key":"...", "value":"..."}
                import json
                try:
                    p = json.loads(payload)
                    self.sys.mem_write(p["key"], p["value"])
                    output = "L1 Cache write successful"
                except Exception as e:
                    return ToolResult(success=False, error=f"Invalid payload format for MEM_WRITE: {e}")
            elif call_type == "SYS_MEM_READ":
                output = self.sys.mem_read(payload)
            elif call_type == "SYS_FILE_WRITE":
                import json
                try:
                    p = json.loads(payload)
                    self.sys.vfs_write(p["path"], p["content"])
                    output = "File written safely in VFS Jail"
                except Exception as e:
                    return ToolResult(success=False, error=f"Invalid payload format for FILE_WRITE: {e}")
            elif call_type == "SYS_FILE_READ":
                output = self.sys.vfs_read(payload)
            elif call_type == "SYS_KILL":
                self.sys.kill_process(payload)
                output = f"SIGKILL sent to PID {payload}"
            else:
                return ToolResult(success=False, error=f"SysCall {call_type} no soportado.")
            
            return ToolResult(success=True, output=output)

        except os_nexus.SysCallError as e:
            error_msg = str(e)
            if "CIRCUIT_BROKEN" in error_msg:
                return ToolResult(
                    success=False, 
                    error=f"CRITICAL KERNEL INTERVENTION: {error_msg}\n[SYSTEM ACTION REQUIRED]: Stop trying this approach immediately and generate a new strategy."
                )
            return ToolResult(success=False, error=error_msg)
            
        except Exception as e:
            return ToolResult(success=False, error=f"Fallo crítico ejecutando SysCall: {str(e)}")

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "call_type": {
                    "type": "string",
                    "description": "El tipo de SysCall ('SYS_EXEC', 'SYS_MEM_READ', 'SYS_MEM_WRITE', 'SYS_FILE_READ', 'SYS_FILE_WRITE', 'SYS_KILL', 'SYS_PING')."
                },
                "payload": {
                    "type": "string",
                    "description": "El código, path, PID o contenido JSON a procesar. Para escrituras, usa un string JSON válido: {'key':'k','value':'v'} o {'path':'p','content':'c'}."
                }
            },
            "required": ["call_type", "payload"]
        }
