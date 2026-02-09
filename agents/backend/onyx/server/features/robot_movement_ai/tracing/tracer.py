"""
Tracer - Tracer principal para distributed tracing
"""
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import time
import uuid


class Tracer:
    """Tracer para distributed tracing"""
    
    def __init__(self):
        self.spans: Dict[str, Dict] = {}
    
    @asynccontextmanager
    async def trace(self, operation_name: str, **kwargs):
        """Crea un span de tracing"""
        span_id = str(uuid.uuid4())
        trace_id = kwargs.get('trace_id', str(uuid.uuid4()))
        
        span = {
            'span_id': span_id,
            'trace_id': trace_id,
            'operation': operation_name,
            'start_time': time.time(),
            'tags': kwargs.get('tags', {}),
        }
        
        self.spans[span_id] = span
        
        try:
            yield span
        finally:
            span['duration'] = time.time() - span['start_time']
            span['status'] = kwargs.get('status', 'ok')
            # Enviar span a sistema de observabilidad

