"""
Span Manager - Gestión de spans de tracing
"""
from typing import Dict, List, Optional
import uuid


class SpanManager:
    """Gestor de spans de tracing"""
    
    def __init__(self):
        self.active_spans: Dict[str, dict] = {}
        self.span_stack: List[str] = []
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None) -> str:
        """Inicia un nuevo span"""
        span_id = str(uuid.uuid4())
        parent_id = parent_span_id or (self.span_stack[-1] if self.span_stack else None)
        
        span = {
            'span_id': span_id,
            'parent_span_id': parent_id,
            'operation': operation_name,
            'start_time': None,  # Se establecerá al finalizar
        }
        
        self.active_spans[span_id] = span
        self.span_stack.append(span_id)
        return span_id
    
    def finish_span(self, span_id: str, status: str = 'ok'):
        """Finaliza un span"""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span['status'] = status
            if span_id in self.span_stack:
                self.span_stack.remove(span_id)
            return span
        return None
    
    def get_active_spans(self) -> List[dict]:
        """Obtiene todos los spans activos"""
        return list(self.active_spans.values())

