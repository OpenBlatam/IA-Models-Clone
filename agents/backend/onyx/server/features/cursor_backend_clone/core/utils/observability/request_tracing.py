"""
Request Tracing - Sistema de tracing de requests
=================================================

Sistema para rastrear requests a través del sistema con correlation IDs
y logging estructurado.
"""

import uuid
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Context variable para correlation ID
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


@dataclass
class TraceInfo:
    """Información de tracing de un request"""
    request_id: str
    start_time: datetime
    method: str
    path: str
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    status_code: Optional[int] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestTracer:
    """
    Tracer de requests con correlation IDs.
    
    Permite rastrear requests a través del sistema
    con IDs de correlación únicos.
    """
    
    def __init__(self, max_traces: int = 10000):
        self.traces: Dict[str, TraceInfo] = {}
        self.max_traces = max_traces
    
    def start_trace(
        self,
        method: str,
        path: str,
        request_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Iniciar trace de un request.
        
        Args:
            method: Método HTTP
            path: Path del request
            request_id: ID del request (se genera si no se proporciona)
            client_ip: IP del cliente
            user_agent: User agent
            **metadata: Metadata adicional
            
        Returns:
            Request ID
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        trace = TraceInfo(
            request_id=request_id,
            start_time=datetime.now(),
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent,
            metadata=metadata
        )
        
        self.traces[request_id] = trace
        
        # Limpiar traces antiguos si excede el límite
        if len(self.traces) > self.max_traces:
            # Eliminar los más antiguos
            sorted_traces = sorted(
                self.traces.items(),
                key=lambda x: x[1].start_time
            )
            for old_id, _ in sorted_traces[:len(self.traces) - self.max_traces]:
                del self.traces[old_id]
        
        # Establecer en context variable
        request_id_var.set(request_id)
        
        logger.debug(
            f"🔍 Trace started: {request_id} - {method} {path}",
            extra={"request_id": request_id, "method": method, "path": path}
        )
        
        return request_id
    
    def end_trace(
        self,
        request_id: str,
        status_code: int,
        **metadata
    ) -> None:
        """
        Finalizar trace de un request.
        
        Args:
            request_id: ID del request
            status_code: Código de estado HTTP
            **metadata: Metadata adicional
        """
        if request_id not in self.traces:
            logger.warning(f"Trace {request_id} not found")
            return
        
        trace = self.traces[request_id]
        trace.status_code = status_code
        trace.duration = (datetime.now() - trace.start_time).total_seconds()
        trace.metadata.update(metadata)
        
        logger.info(
            f"✅ Trace completed: {request_id} - {trace.method} {trace.path} - "
            f"{status_code} ({trace.duration:.3f}s)",
            extra={
                "request_id": request_id,
                "method": trace.method,
                "path": trace.path,
                "status_code": status_code,
                "duration": trace.duration
            }
        )
    
    def get_trace(self, request_id: str) -> Optional[TraceInfo]:
        """Obtener información de trace"""
        return self.traces.get(request_id)
    
    def get_current_request_id(self) -> Optional[str]:
        """Obtener request ID actual del contexto"""
        return request_id_var.get()
    
    def clear_trace(self, request_id: str) -> None:
        """Limpiar trace"""
        if request_id in self.traces:
            del self.traces[request_id]


def get_request_id() -> Optional[str]:
    """
    Obtener request ID actual del contexto.
    
    Returns:
        Request ID o None si no hay contexto
    """
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """
    Establecer request ID en el contexto.
    
    Args:
        request_id: ID del request
    """
    request_id_var.set(request_id)




