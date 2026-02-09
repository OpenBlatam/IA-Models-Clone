"""
Tracing System
===============

Sistema de tracing distribuido.
"""

import logging
import uuid
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """Contexto de trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


class TracingSystem:
    """
    Sistema de tracing.
    
    Gestiona tracing distribuido con contexto.
    """
    
    def __init__(self):
        """Inicializar sistema de tracing."""
        self.active_traces: Dict[str, TraceContext] = {}
        self.trace_stack: List[TraceContext] = []
    
    def start_trace(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        baggage: Optional[Dict[str, str]] = None
    ) -> TraceContext:
        """
        Iniciar trace.
        
        Args:
            operation: Nombre de la operación
            trace_id: ID del trace (opcional)
            baggage: Baggage items
            
        Returns:
            Contexto de trace
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        parent_context = self.trace_stack[-1] if self.trace_stack else None
        
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_context.span_id if parent_context else None,
            baggage=baggage or {}
        )
        
        if parent_context:
            context.baggage.update(parent_context.baggage)
        
        self.active_traces[trace_id] = context
        self.trace_stack.append(context)
        
        logger.debug(f"Started trace: {operation} ({trace_id})")
        
        return context
    
    def end_trace(self, trace_id: str) -> Optional[TraceContext]:
        """Finalizar trace."""
        if trace_id in self.active_traces:
            context = self.active_traces[trace_id]
            if context in self.trace_stack:
                self.trace_stack.remove(context)
            del self.active_traces[trace_id]
            return context
        return None
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Obtener contexto actual."""
        return self.trace_stack[-1] if self.trace_stack else None
    
    def inject_context(self, context: TraceContext) -> Dict[str, str]:
        """
        Inyectar contexto para propagación.
        
        Args:
            context: Contexto de trace
            
        Returns:
            Headers para propagación
        """
        return {
            "X-Trace-ID": context.trace_id,
            "X-Span-ID": context.span_id,
            "X-Parent-Span-ID": context.parent_span_id or "",
            **{f"X-Baggage-{k}": v for k, v in context.baggage.items()}
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """
        Extraer contexto de headers.
        
        Args:
            headers: Headers HTTP
            
        Returns:
            Contexto de trace o None
        """
        trace_id = headers.get("X-Trace-ID")
        span_id = headers.get("X-Span-ID")
        parent_span_id = headers.get("X-Parent-Span-ID")
        
        if not trace_id or not span_id:
            return None
        
        baggage = {}
        for key, value in headers.items():
            if key.startswith("X-Baggage-"):
                baggage_key = key.replace("X-Baggage-", "")
                baggage[baggage_key] = value
        
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id if parent_span_id else None,
            baggage=baggage
        )
        
        return context
    
    @contextmanager
    def trace(self, operation: str, trace_id: Optional[str] = None):
        """
        Context manager para trace.
        
        Args:
            operation: Nombre de la operación
            trace_id: ID del trace (opcional)
        """
        context = self.start_trace(operation, trace_id)
        try:
            yield context
        finally:
            self.end_trace(context.trace_id)


# Instancia global
_tracing_system: Optional[TracingSystem] = None


def get_tracing_system() -> TracingSystem:
    """Obtener instancia global del sistema de tracing."""
    global _tracing_system
    if _tracing_system is None:
        _tracing_system = TracingSystem()
    return _tracing_system






