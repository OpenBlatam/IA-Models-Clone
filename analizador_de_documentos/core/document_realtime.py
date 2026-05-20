"""
Document Realtime - Análisis en Tiempo Real
===========================================

Sistema de análisis en tiempo real con streaming.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class RealtimeAnalysisEvent:
    """Evento de análisis en tiempo real."""
    event_type: str  # 'start', 'progress', 'complete', 'error'
    document_id: str
    progress: float = 0.0
    message: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class RealtimeAnalyzer:
    """Analizador en tiempo real."""
    
    def __init__(self, analyzer):
        """Inicializar analizador."""
        self.analyzer = analyzer
        self.active_analyses: Dict[str, Any] = {}
        self.event_history: Dict[str, deque] = {}
        self.max_history = 100
    
    async def analyze_realtime(
        self,
        document_id: str,
        content: str,
        tasks: Optional[List[str]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None
    ) -> AsyncGenerator[RealtimeAnalysisEvent, None]:
        """
        Analizar documento en tiempo real con streaming de eventos.
        
        Args:
            document_id: ID del documento
            content: Contenido del documento
            tasks: Tareas de análisis
            on_progress: Callback de progreso
        
        Yields:
            RealtimeAnalysisEvent con eventos en tiempo real
        """
        self.active_analyses[document_id] = {
            "start_time": time.time(),
            "status": "processing"
        }
        
        if document_id not in self.event_history:
            self.event_history[document_id] = deque(maxlen=self.max_history)
        
        try:
            # Evento de inicio
            start_event = RealtimeAnalysisEvent(
                event_type="start",
                document_id=document_id,
                progress=0.0,
                message="Iniciando análisis..."
            )
            self.event_history[document_id].append(start_event)
            yield start_event
            
            if on_progress:
                on_progress(0.0, "Iniciando análisis...")
            
            # Progreso: Preprocesamiento
            await asyncio.sleep(0.1)  # Simular trabajo
            progress_event = RealtimeAnalysisEvent(
                event_type="progress",
                document_id=document_id,
                progress=20.0,
                message="Preprocesando documento..."
            )
            self.event_history[document_id].append(progress_event)
            yield progress_event
            
            if on_progress:
                on_progress(20.0, "Preprocesando documento...")
            
            # Progreso: Análisis básico
            await asyncio.sleep(0.1)
            progress_event = RealtimeAnalysisEvent(
                event_type="progress",
                document_id=document_id,
                progress=50.0,
                message="Analizando contenido..."
            )
            self.event_history[document_id].append(progress_event)
            yield progress_event
            
            if on_progress:
                on_progress(50.0, "Analizando contenido...")
            
            # Análisis real
            result = await self.analyzer.analyze_document(
                document_content=content,
                tasks=tasks
            )
            
            # Progreso: Post-procesamiento
            await asyncio.sleep(0.1)
            progress_event = RealtimeAnalysisEvent(
                event_type="progress",
                document_id=document_id,
                progress=80.0,
                message="Finalizando análisis..."
            )
            self.event_history[document_id].append(progress_event)
            yield progress_event
            
            if on_progress:
                on_progress(80.0, "Finalizando análisis...")
            
            # Evento de completado
            complete_event = RealtimeAnalysisEvent(
                event_type="complete",
                document_id=document_id,
                progress=100.0,
                message="Análisis completado",
                data={
                    "result": result.__dict__ if hasattr(result, '__dict__') else result,
                    "processing_time": time.time() - self.active_analyses[document_id]["start_time"]
                }
            )
            self.event_history[document_id].append(complete_event)
            yield complete_event
            
            if on_progress:
                on_progress(100.0, "Análisis completado")
            
            self.active_analyses[document_id]["status"] = "completed"
            
        except Exception as e:
            error_event = RealtimeAnalysisEvent(
                event_type="error",
                document_id=document_id,
                message=f"Error: {str(e)}",
                data={"error": str(e)}
            )
            self.event_history[document_id].append(error_event)
            yield error_event
            
            self.active_analyses[document_id]["status"] = "error"
            logger.error(f"Error en análisis en tiempo real: {e}")
        
        finally:
            if document_id in self.active_analyses:
                del self.active_analyses[document_id]
    
    def get_event_history(self, document_id: str) -> List[RealtimeAnalysisEvent]:
        """Obtener historial de eventos."""
        if document_id in self.event_history:
            return list(self.event_history[document_id])
        return []
    
    def get_active_analyses(self) -> Dict[str, Any]:
        """Obtener análisis activos."""
        return {
            doc_id: {
                "status": info["status"],
                "elapsed_time": time.time() - info["start_time"]
            }
            for doc_id, info in self.active_analyses.items()
        }


__all__ = [
    "RealtimeAnalyzer",
    "RealtimeAnalysisEvent"
]
















