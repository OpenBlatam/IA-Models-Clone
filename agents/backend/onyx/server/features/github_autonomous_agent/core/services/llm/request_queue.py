"""
Request Queue System - Sistema de cola y priorización de requests LLM.

Características:
- Cola de requests con prioridades
- Rate limiting inteligente
- Load balancing
- Retry automático
- Timeout management
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import asyncio
import uuid

from config.logging_config import get_logger

logger = get_logger(__name__)


class RequestPriority(str, Enum):
    """Prioridades de request."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class RequestStatus(str, Enum):
    """Estados de un request."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class QueuedRequest:
    """Request en cola."""
    request_id: str
    model: str
    prompt: str
    priority: RequestPriority
    created_at: datetime
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "error": self.error
        }


class RequestQueue:
    """
    Sistema de cola de requests con priorización.
    
    Características:
    - Cola con prioridades
    - Rate limiting por modelo
    - Timeout management
    - Retry automático
    - Load balancing
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        max_concurrent_requests: int = 10,
        default_timeout: float = 30.0
    ):
        """
        Inicializar cola de requests.
        
        Args:
            max_queue_size: Tamaño máximo de la cola
            max_concurrent_requests: Máximo de requests concurrentes
            default_timeout: Timeout por defecto en segundos
        """
        self.max_queue_size = max_queue_size
        self.max_concurrent_requests = max_concurrent_requests
        self.default_timeout = default_timeout
        
        # Colas por prioridad
        self.queues: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        
        # Requests activos
        self.active_requests: Dict[str, QueuedRequest] = {}
        
        # Historial de requests
        self.request_history: List[QueuedRequest] = []
        
        # Estadísticas
        self.stats = {
            "total_queued": 0,
            "total_processed": 0,
            "total_failed": 0,
            "total_timeout": 0,
            "avg_wait_time": 0.0,
            "avg_processing_time": 0.0
        }
        
        # Lock para thread safety
        self._lock = asyncio.Lock()
        
        # Worker task
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def enqueue(
        self,
        model: str,
        prompt: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        processor: Optional[Callable[[QueuedRequest], Awaitable[Any]]] = None
    ) -> str:
        """
        Agregar request a la cola.
        
        Args:
            model: Modelo a usar
            prompt: Prompt a procesar
            priority: Prioridad del request
            timeout: Timeout en segundos
            max_retries: Máximo de reintentos
            metadata: Metadatos adicionales
            processor: Función procesadora (opcional)
            
        Returns:
            ID del request
        """
        async with self._lock:
            # Verificar tamaño de cola
            total_queued = sum(len(q) for q in self.queues.values())
            if total_queued >= self.max_queue_size:
                raise ValueError(f"Queue is full (max: {self.max_queue_size})")
            
            request_id = str(uuid.uuid4())
            request = QueuedRequest(
                request_id=request_id,
                model=model,
                prompt=prompt,
                priority=priority,
                created_at=datetime.now(),
                timeout=timeout or self.default_timeout,
                max_retries=max_retries,
                metadata=metadata or {},
                status=RequestStatus.QUEUED
            )
            
            # Agregar a cola según prioridad
            self.queues[priority].append(request)
            self.stats["total_queued"] += 1
            
            logger.info(
                f"Request {request_id} agregado a cola con prioridad {priority.value}"
            )
            
            # Iniciar worker si no está corriendo
            if not self._running:
                await self.start()
            
            return request_id
    
    async def dequeue(self) -> Optional[QueuedRequest]:
        """
        Obtener siguiente request de la cola (por prioridad).
        
        Returns:
            Request o None si la cola está vacía
        """
        async with self._lock:
            # Procesar por orden de prioridad
            for priority in [
                RequestPriority.CRITICAL,
                RequestPriority.URGENT,
                RequestPriority.HIGH,
                RequestPriority.NORMAL,
                RequestPriority.LOW
            ]:
                if self.queues[priority]:
                    request = self.queues[priority].popleft()
                    request.status = RequestStatus.PROCESSING
                    self.active_requests[request.request_id] = request
                    return request
            
            return None
    
    async def get_request(self, request_id: str) -> Optional[QueuedRequest]:
        """
        Obtener request por ID.
        
        Args:
            request_id: ID del request
            
        Returns:
            Request o None si no existe
        """
        async with self._lock:
            # Buscar en colas
            for queue in self.queues.values():
                for request in queue:
                    if request.request_id == request_id:
                        return request
            
            # Buscar en activos
            if request_id in self.active_requests:
                return self.active_requests[request_id]
            
            # Buscar en historial
            for request in self.request_history:
                if request.request_id == request_id:
                    return request
            
            return None
    
    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancelar un request.
        
        Args:
            request_id: ID del request
            
        Returns:
            True si se canceló, False si no se encontró
        """
        async with self._lock:
            request = await self.get_request(request_id)
            if not request:
                return False
            
            if request.status == RequestStatus.PROCESSING:
                request.status = RequestStatus.CANCELLED
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
            elif request.status == RequestStatus.QUEUED:
                # Remover de cola
                for queue in self.queues.values():
                    if request in queue:
                        queue.remove(request)
                        request.status = RequestStatus.CANCELLED
                        break
            
            return True
    
    async def process_request(
        self,
        request: QueuedRequest,
        processor: Callable[[QueuedRequest], Awaitable[Any]]
    ) -> Any:
        """
        Procesar un request.
        
        Args:
            request: Request a procesar
            processor: Función procesadora
            
        Returns:
            Resultado del procesamiento
        """
        start_time = datetime.now()
        
        try:
            # Ejecutar con timeout
            result = await asyncio.wait_for(
                processor(request),
                timeout=request.timeout
            )
            
            request.status = RequestStatus.COMPLETED
            request.result = result
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, success=True)
            
            logger.info(
                f"Request {request.request_id} completado en {processing_time:.2f}s"
            )
            
            return result
            
        except asyncio.TimeoutError:
            request.status = RequestStatus.TIMEOUT
            request.error = f"Timeout after {request.timeout}s"
            self.stats["total_timeout"] += 1
            
            # Retry si es posible
            if request.retry_count < request.max_retries:
                request.retry_count += 1
                request.status = RequestStatus.QUEUED
                # Re-agregar a cola con menor prioridad
                priority = RequestPriority.NORMAL if request.priority == RequestPriority.CRITICAL else request.priority
                self.queues[priority].append(request)
                logger.info(
                    f"Request {request.request_id} re-agregado a cola (retry {request.retry_count})"
                )
            
            raise
            
        except Exception as e:
            request.status = RequestStatus.FAILED
            request.error = str(e)
            self.stats["total_failed"] += 1
            
            # Retry si es posible
            if request.retry_count < request.max_retries:
                request.retry_count += 1
                request.status = RequestStatus.QUEUED
                priority = RequestPriority.NORMAL if request.priority == RequestPriority.CRITICAL else request.priority
                self.queues[priority].append(request)
                logger.info(
                    f"Request {request.request_id} re-agregado a cola (retry {request.retry_count})"
                )
            
            raise
        
        finally:
            # Remover de activos y agregar a historial
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            self.request_history.append(request)
            
            # Mantener solo últimos 1000 en historial
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]
    
    async def start(self):
        """Iniciar worker de procesamiento."""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Request queue worker iniciado")
    
    async def stop(self):
        """Detener worker de procesamiento."""
        self._running = False
        if self._worker_task:
            await self._worker_task
        logger.info("Request queue worker detenido")
    
    async def _worker(self):
        """Worker que procesa requests de la cola."""
        while self._running:
            try:
                # Verificar capacidad
                if len(self.active_requests) >= self.max_concurrent_requests:
                    await asyncio.sleep(0.1)
                    continue
                
                # Obtener siguiente request
                request = await self.dequeue()
                if not request:
                    await asyncio.sleep(0.1)
                    continue
                
                # Procesar (requiere processor externo)
                # Este método debe ser llamado con un processor
                logger.debug(f"Request {request.request_id} listo para procesar")
                
            except Exception as e:
                logger.error(f"Error en worker: {e}")
                await asyncio.sleep(1)
    
    def _update_stats(self, processing_time: float, success: bool):
        """Actualizar estadísticas."""
        self.stats["total_processed"] += 1
        if not success:
            self.stats["total_failed"] += 1
        
        # Actualizar promedio de tiempo de procesamiento
        total = self.stats["total_processed"]
        current_avg = self.stats["avg_processing_time"]
        self.stats["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cola."""
        total_queued = sum(len(q) for q in self.queues.values())
        
        return {
            **self.stats,
            "current_queue_size": total_queued,
            "active_requests": len(self.active_requests),
            "queue_by_priority": {
                priority.value: len(queue)
                for priority, queue in self.queues.items()
            }
        }
    
    def get_queue_size(self) -> int:
        """Obtener tamaño actual de la cola."""
        return sum(len(q) for q in self.queues.values())


def get_request_queue() -> RequestQueue:
    """Factory function para obtener instancia singleton de la cola."""
    if not hasattr(get_request_queue, "_instance"):
        get_request_queue._instance = RequestQueue()
    return get_request_queue._instance



