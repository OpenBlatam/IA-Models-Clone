"""
Real-time Text Processing System for AI History Comparison
Sistema de procesamiento de texto en tiempo real para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import deque, Counter
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingPriority(Enum):
    """Prioridades de procesamiento"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ProcessingStatus(Enum):
    """Estados de procesamiento"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingTask:
    """Tarea de procesamiento"""
    id: str
    text: str
    document_id: str
    priority: ProcessingPriority
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None

@dataclass
class ProcessingMetrics:
    """Métricas de procesamiento"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    avg_processing_time: float = 0.0
    throughput_per_minute: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class RealTimeTextProcessor:
    """
    Procesador de texto en tiempo real
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        processing_timeout: int = 300,
        enable_metrics: bool = True
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout
        self.enable_metrics = enable_metrics
        
        # Colas de procesamiento por prioridad
        self.task_queues = {
            ProcessingPriority.CRITICAL: queue.PriorityQueue(),
            ProcessingPriority.HIGH: queue.PriorityQueue(),
            ProcessingPriority.NORMAL: queue.PriorityQueue(),
            ProcessingPriority.LOW: queue.PriorityQueue()
        }
        
        # Almacenamiento de tareas
        self.tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)  # Mantener últimas 1000 tareas
        
        # Pool de workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.workers = []
        self.is_running = False
        
        # Métricas
        self.metrics = ProcessingMetrics()
        self.processing_times = deque(maxlen=100)  # Últimos 100 tiempos de procesamiento
        
        # Callbacks y hooks
        self.pre_processing_hooks: List[Callable] = []
        self.post_processing_hooks: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        # Configuración
        self.config = {
            "batch_size": 10,
            "retry_delay": 5,  # segundos
            "metrics_update_interval": 60,  # segundos
            "cleanup_interval": 300  # segundos
        }
    
    async def start(self):
        """Iniciar el procesador"""
        if self.is_running:
            logger.warning("Processor is already running")
            return
        
        self.is_running = True
        logger.info(f"Starting text processor with {self.max_workers} workers")
        
        # Iniciar workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Iniciar tareas de mantenimiento
        if self.enable_metrics:
            asyncio.create_task(self._metrics_updater())
        
        asyncio.create_task(self._cleanup_old_tasks())
        
        logger.info("Text processor started successfully")
    
    async def stop(self):
        """Detener el procesador"""
        if not self.is_running:
            return
        
        logger.info("Stopping text processor...")
        self.is_running = False
        
        # Cancelar workers
        for worker in self.workers:
            worker.cancel()
        
        # Esperar a que terminen
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Cerrar executor
        self.executor.shutdown(wait=True)
        
        logger.info("Text processor stopped")
    
    async def submit_task(
        self,
        text: str,
        document_id: str,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Enviar tarea de procesamiento
        
        Args:
            text: Texto a procesar
            document_id: ID del documento
            priority: Prioridad de procesamiento
            callback: Función callback opcional
            
        Returns:
            ID de la tarea
        """
        if not self.is_running:
            raise RuntimeError("Processor is not running")
        
        # Verificar límite de cola
        total_pending = sum(q.qsize() for q in self.task_queues.values())
        if total_pending >= self.max_queue_size:
            raise RuntimeError(f"Queue is full ({total_pending}/{self.max_queue_size})")
        
        # Crear tarea
        task_id = f"{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        task = ProcessingTask(
            id=task_id,
            text=text,
            document_id=document_id,
            priority=priority,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
            callback=callback
        )
        
        # Agregar a cola
        self.task_queues[priority].put((priority.value, task))
        self.tasks[task_id] = task
        
        # Actualizar métricas
        self.metrics.total_tasks += 1
        self.metrics.pending_tasks += 1
        
        logger.info(f"Task {task_id} submitted with priority {priority.name}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Obtener estado de una tarea"""
        return self.tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancelar una tarea"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
            return False
        
        task.status = ProcessingStatus.CANCELLED
        task.completed_at = datetime.now()
        
        # Actualizar métricas
        if task.status == ProcessingStatus.PENDING:
            self.metrics.pending_tasks -= 1
        elif task.status == ProcessingStatus.PROCESSING:
            self.metrics.processing_tasks -= 1
        
        logger.info(f"Task {task_id} cancelled")
        return True
    
    async def get_metrics(self) -> ProcessingMetrics:
        """Obtener métricas de procesamiento"""
        return self.metrics
    
    async def get_queue_status(self) -> Dict[str, int]:
        """Obtener estado de las colas"""
        return {
            priority.name: queue.qsize()
            for priority, queue in self.task_queues.items()
        }
    
    async def _worker(self, worker_name: str):
        """Worker para procesar tareas"""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Obtener siguiente tarea (por prioridad)
                task = await self._get_next_task()
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Procesar tarea
                await self._process_task(task, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _get_next_task(self) -> Optional[ProcessingTask]:
        """Obtener siguiente tarea por prioridad"""
        # Buscar tareas en orden de prioridad
        for priority in [ProcessingPriority.CRITICAL, ProcessingPriority.HIGH, 
                        ProcessingPriority.NORMAL, ProcessingPriority.LOW]:
            try:
                if not self.task_queues[priority].empty():
                    _, task = self.task_queues[priority].get_nowait()
                    return task
            except queue.Empty:
                continue
        
        return None
    
    async def _process_task(self, task: ProcessingTask, worker_name: str):
        """Procesar una tarea individual"""
        start_time = datetime.now()
        
        try:
            # Actualizar estado
            task.status = ProcessingStatus.PROCESSING
            task.started_at = start_time
            
            # Actualizar métricas
            self.metrics.pending_tasks -= 1
            self.metrics.processing_tasks += 1
            
            logger.info(f"Worker {worker_name} processing task {task.id}")
            
            # Ejecutar hooks pre-procesamiento
            for hook in self.pre_processing_hooks:
                await hook(task)
            
            # Procesar texto
            result = await self._process_text(task.text, task.document_id)
            
            # Actualizar tarea
            task.status = ProcessingStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Ejecutar hooks post-procesamiento
            for hook in self.post_processing_hooks:
                await hook(task)
            
            # Ejecutar callback si existe
            if task.callback:
                try:
                    await task.callback(task)
                except Exception as e:
                    logger.error(f"Callback error for task {task.id}: {e}")
            
            # Actualizar métricas
            self.metrics.completed_tasks += 1
            self.metrics.processing_tasks -= 1
            
            # Calcular tiempo de procesamiento
            processing_time = (task.completed_at - task.started_at).total_seconds()
            self.processing_times.append(processing_time)
            
            # Mover a tareas completadas
            self.completed_tasks.append(task)
            
            logger.info(f"Task {task.id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            # Manejar error
            await self._handle_task_error(task, e, worker_name)
    
    async def _process_text(self, text: str, document_id: str) -> Dict[str, Any]:
        """Procesar texto (implementación base)"""
        # Esta es una implementación básica
        # En una implementación real, se integraría con el NLP engine
        
        # Simular procesamiento
        await asyncio.sleep(0.1)
        
        # Análisis básico
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(text.split('.'))
        
        # Detectar idioma básico
        language = "es" if any(word in text.lower() for word in ["el", "la", "de", "que", "y"]) else "en"
        
        # Análisis de sentimiento básico
        positive_words = ["bueno", "excelente", "genial", "perfecto", "good", "excellent", "great", "perfect"]
        negative_words = ["malo", "terrible", "horrible", "pésimo", "bad", "terrible", "awful", "horrible"]
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "document_id": document_id,
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "language": language,
            "sentiment": sentiment,
            "processing_timestamp": datetime.now().isoformat()
        }
    
    async def _handle_task_error(self, task: ProcessingTask, error: Exception, worker_name: str):
        """Manejar error en tarea"""
        logger.error(f"Worker {worker_name} failed to process task {task.id}: {error}")
        
        # Actualizar tarea
        task.error = str(error)
        task.retry_count += 1
        
        # Actualizar métricas
        self.metrics.processing_tasks -= 1
        
        # Ejecutar manejadores de error
        for handler in self.error_handlers:
            try:
                await handler(task, error)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")
        
        # Reintentar si es posible
        if task.retry_count < task.max_retries:
            task.status = ProcessingStatus.PENDING
            task.started_at = None
            task.error = None
            
            # Reagregar a cola con delay
            await asyncio.sleep(self.config["retry_delay"])
            self.task_queues[task.priority].put((task.priority.value, task))
            
            self.metrics.pending_tasks += 1
            logger.info(f"Task {task.id} retrying ({task.retry_count}/{task.max_retries})")
        else:
            # Marcar como fallida
            task.status = ProcessingStatus.FAILED
            task.completed_at = datetime.now()
            
            self.metrics.failed_tasks += 1
            logger.error(f"Task {task.id} failed after {task.max_retries} retries")
    
    async def _metrics_updater(self):
        """Actualizar métricas periódicamente"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["metrics_update_interval"])
                
                # Calcular métricas
                if self.processing_times:
                    self.metrics.avg_processing_time = np.mean(self.processing_times)
                
                # Calcular throughput
                recent_tasks = [
                    task for task in self.completed_tasks
                    if task.completed_at and (datetime.now() - task.completed_at).total_seconds() < 60
                ]
                self.metrics.throughput_per_minute = len(recent_tasks)
                
                # Calcular tasa de error
                if self.metrics.total_tasks > 0:
                    self.metrics.error_rate = self.metrics.failed_tasks / self.metrics.total_tasks
                
                self.metrics.last_updated = datetime.now()
                
                logger.debug(f"Metrics updated: {self.metrics}")
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
    
    async def _cleanup_old_tasks(self):
        """Limpiar tareas antiguas"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["cleanup_interval"])
                
                # Limpiar tareas completadas antiguas
                cutoff_time = datetime.now() - timedelta(hours=24)
                old_tasks = [
                    task_id for task_id, task in self.tasks.items()
                    if task.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]
                    and task.completed_at and task.completed_at < cutoff_time
                ]
                
                for task_id in old_tasks:
                    del self.tasks[task_id]
                
                if old_tasks:
                    logger.info(f"Cleaned up {len(old_tasks)} old tasks")
                
            except Exception as e:
                logger.error(f"Error cleaning up tasks: {e}")
    
    def add_pre_processing_hook(self, hook: Callable):
        """Agregar hook pre-procesamiento"""
        self.pre_processing_hooks.append(hook)
    
    def add_post_processing_hook(self, hook: Callable):
        """Agregar hook post-procesamiento"""
        self.post_processing_hooks.append(hook)
    
    def add_error_handler(self, handler: Callable):
        """Agregar manejador de errores"""
        self.error_handlers.append(handler)
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de procesamiento"""
        return {
            "metrics": {
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "pending_tasks": self.metrics.pending_tasks,
                "processing_tasks": self.metrics.processing_tasks,
                "avg_processing_time": self.metrics.avg_processing_time,
                "throughput_per_minute": self.metrics.throughput_per_minute,
                "error_rate": self.metrics.error_rate
            },
            "queue_status": await self.get_queue_status(),
            "worker_count": self.max_workers,
            "is_running": self.is_running,
            "last_updated": self.metrics.last_updated.isoformat()
        }
    
    async def batch_process(
        self,
        texts: List[Tuple[str, str]],  # (text, document_id)
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> List[str]:
        """Procesar múltiples textos en lote"""
        task_ids = []
        
        for text, document_id in texts:
            try:
                task_id = await self.submit_task(text, document_id, priority)
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Failed to submit task for document {document_id}: {e}")
        
        return task_ids
    
    async def wait_for_completion(self, task_ids: List[str], timeout: int = 300) -> Dict[str, ProcessingTask]:
        """Esperar a que las tareas se completen"""
        start_time = datetime.now()
        results = {}
        
        while task_ids and (datetime.now() - start_time).total_seconds() < timeout:
            completed_tasks = []
            
            for task_id in task_ids:
                task = self.tasks.get(task_id)
                if task and task.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
                    results[task_id] = task
                    completed_tasks.append(task_id)
            
            # Remover tareas completadas
            for task_id in completed_tasks:
                task_ids.remove(task_id)
            
            if task_ids:
                await asyncio.sleep(1)
        
        return results



























