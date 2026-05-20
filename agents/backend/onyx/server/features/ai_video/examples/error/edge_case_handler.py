from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import threading
import time
import psutil
import os
import gc
from typing import (
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import logging
import weakref
from collections import defaultdict, deque
import numpy as np
from contextlib import asynccontextmanager, contextmanager
import signal
import sys
from .error_handling import (
                import torch
from typing import Any, List, Dict, Optional
"""
🔍 EDGE CASE HANDLER - BOUNDARY CONDITIONS & RESOURCE MANAGEMENT
===============================================================

Manejador de casos edge para el AI Video System.
Incluye validación de límites, manejo de recursos, casos extremos,
y prevención de condiciones de carrera.
"""

    Any, Dict, List, Optional, Union, Callable, Tuple, 
    Set, TypeVar, Generic, Protocol
)

    AIVideoError, ErrorCategory, ErrorSeverity, 
    MemoryError, SystemError, ValidationError
)

# =============================================================================
# EDGE CASE TYPES
# =============================================================================

class EdgeCaseType(Enum):
    """Tipos de casos edge."""
    RESOURCE_LIMIT = auto()
    BOUNDARY_CONDITION = auto()
    RACE_CONDITION = auto()
    DATA_CORRUPTION = auto()
    SYSTEM_OVERLOAD = auto()
    TIMEOUT = auto()
    DEADLOCK = auto()
    MEMORY_LEAK = auto()
    FILE_SYSTEM = auto()
    NETWORK = auto()

class ResourceType(Enum):
    """Tipos de recursos del sistema."""
    CPU = auto()
    MEMORY = auto()
    GPU = auto()
    DISK = auto()
    NETWORK = auto()
    FILE_DESCRIPTORS = auto()
    THREADS = auto()
    PROCESSES = auto()

# =============================================================================
# RESOURCE MONITORING
# =============================================================================

@dataclass
class ResourceLimits:
    """Límites de recursos del sistema."""
    cpu_percent: float = 90.0
    memory_percent: float = 85.0
    gpu_memory_percent: float = 90.0
    disk_percent: float = 95.0
    max_file_size_mb: float = 1024.0  # 1GB
    max_batch_size: int = 32
    max_concurrent_operations: int = 10
    max_retry_attempts: int = 3
    timeout_seconds: float = 300.0  # 5 minutos

@dataclass
class ResourceUsage:
    """Uso actual de recursos."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_percent: float = 0.0
    available_memory_mb: float = 0.0
    active_threads: int = 0
    open_files: int = 0
    timestamp: float = field(default_factory=time.time)

class ResourceMonitor:
    """Monitor de recursos del sistema."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        
    """__init__ function."""
self.limits = limits or ResourceLimits()
        self.usage_history: deque = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Iniciar monitoreo de recursos."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("🔍 Monitoreo de recursos iniciado")
    
    def stop_monitoring(self) -> Any:
        """Detener monitoreo de recursos."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("🔍 Monitoreo de recursos detenido")
    
    def _monitor_loop(self, interval: float):
        """Loop de monitoreo."""
        while self._monitoring:
            try:
                usage = self.get_current_usage()
                self.usage_history.append(usage)
                
                # Verificar límites
                violations = self.check_resource_violations(usage)
                if violations:
                    self.logger.warning(f"⚠️ Violaciones de recursos: {violations}")
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error en monitoreo: {e}")
                time.sleep(interval)
    
    def get_current_usage(self) -> ResourceUsage:
        """Obtener uso actual de recursos."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU memory (si está disponible)
            gpu_memory_percent = 0.0
            try:
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_stats()
                    allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                    reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                    total = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_percent = (allocated / total) * 100
            except ImportError:
                pass
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_memory_percent=gpu_memory_percent,
                disk_percent=disk.percent,
                available_memory_mb=memory.available / (1024 * 1024),
                active_threads=threading.active_count(),
                open_files=len(psutil.Process().open_files())
            )
        except Exception as e:
            self.logger.error(f"Error obteniendo uso de recursos: {e}")
            return ResourceUsage()
    
    def check_resource_violations(self, usage: ResourceUsage) -> List[str]:
        """Verificar violaciones de límites de recursos."""
        violations = []
        
        if usage.cpu_percent > self.limits.cpu_percent:
            violations.append(f"CPU: {usage.cpu_percent:.1f}% > {self.limits.cpu_percent}%")
        
        if usage.memory_percent > self.limits.memory_percent:
            violations.append(f"Memoria: {usage.memory_percent:.1f}% > {self.limits.memory_percent}%")
        
        if usage.gpu_memory_percent > self.limits.gpu_memory_percent:
            violations.append(f"GPU: {usage.gpu_memory_percent:.1f}% > {self.limits.gpu_memory_percent}%")
        
        if usage.disk_percent > self.limits.disk_percent:
            violations.append(f"Disco: {usage.disk_percent:.1f}% > {self.limits.disk_percent}%")
        
        return violations
    
    def is_system_overloaded(self) -> bool:
        """Verificar si el sistema está sobrecargado."""
        if not self.usage_history:
            return False
        
        recent_usage = list(self.usage_history)[-10:]  # Últimos 10 registros
        avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage)
        avg_memory = sum(u.memory_percent for u in recent_usage) / len(recent_usage)
        
        return avg_cpu > self.limits.cpu_percent or avg_memory > self.limits.memory_percent

# =============================================================================
# BOUNDARY CONDITION HANDLING
# =============================================================================

class BoundaryConditionHandler:
    """Manejador de condiciones de borde."""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
    
    def validate_batch_size(self, batch_size: int, max_size: Optional[int] = None) -> int:
        """Validar y ajustar tamaño de batch."""
        if max_size is None:
            max_size = 32
        
        if batch_size <= 0:
            self.logger.warning("Tamaño de batch inválido, usando 1")
            return 1
        
        if batch_size > max_size:
            self.logger.warning(f"Tamaño de batch muy grande ({batch_size}), reduciendo a {max_size}")
            return max_size
        
        return batch_size
    
    def validate_sequence_length(self, length: int, max_length: int = 1024) -> int:
        """Validar longitud de secuencia."""
        if length <= 0:
            raise ValidationError("Longitud de secuencia debe ser positiva")
        
        if length > max_length:
            self.logger.warning(f"Longitud de secuencia muy grande ({length}), truncando a {max_length}")
            return max_length
        
        return length
    
    def validate_image_dimensions(self, width: int, height: int, max_dim: int = 2048) -> Tuple[int, int]:
        """Validar dimensiones de imagen."""
        if width <= 0 or height <= 0:
            raise ValidationError("Dimensiones de imagen deben ser positivas")
        
        if width > max_dim or height > max_dim:
            # Mantener aspect ratio
            if width > height:
                new_width = max_dim
                new_height = int((height * max_dim) / width)
            else:
                new_height = max_dim
                new_width = int((width * max_dim) / height)
            
            self.logger.warning(f"Dimensiones muy grandes ({width}x{height}), redimensionando a {new_width}x{new_height}")
            return new_width, new_height
        
        return width, height
    
    def validate_file_size(self, file_path: Path, max_size_mb: float = 1024.0) -> bool:
        """Validar tamaño de archivo."""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                raise ValidationError(f"Archivo muy grande: {file_size_mb:.1f}MB > {max_size_mb}MB")
            return True
        except Exception as e:
            raise ValidationError(f"Error validando archivo: {e}") from e
    
    def validate_memory_requirement(self, required_mb: float, available_mb: Optional[float] = None) -> bool:
        """Validar requerimiento de memoria."""
        if available_mb is None:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        if required_mb > available_mb:
            raise MemoryError(f"Memoria insuficiente: {required_mb:.1f}MB requerido, {available_mb:.1f}MB disponible")
        
        return True

# =============================================================================
# RACE CONDITION PREVENTION
# =============================================================================

class RaceConditionHandler:
    """Manejador de condiciones de carrera."""
    
    def __init__(self) -> Any:
        self.locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.async_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.logger = logging.getLogger(__name__)
    
    def get_lock(self, resource_name: str) -> threading.Lock:
        """Obtener lock para un recurso."""
        return self.locks[resource_name]
    
    def get_async_lock(self, resource_name: str) -> asyncio.Lock:
        """Obtener lock asíncrono para un recurso."""
        return self.async_locks[resource_name]
    
    @contextmanager
    def resource_lock(self, resource_name: str, timeout: float = 30.0):
        """Context manager para lock de recurso."""
        lock = self.get_lock(resource_name)
        acquired = lock.acquire(timeout=timeout)
        
        if not acquired:
            raise ConcurrencyError(f"Timeout adquiriendo lock para {resource_name}")
        
        try:
            yield
        finally:
            lock.release()
    
    @asynccontextmanager
    async def async_resource_lock(self, resource_name: str, timeout: float = 30.0):
        """Context manager asíncrono para lock de recurso."""
        lock = self.get_async_lock(resource_name)
        
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            yield
        finally:
            lock.release()
    
    def prevent_deadlock(self, resources: List[str]) -> List[str]:
        """Prevenir deadlock ordenando recursos."""
        # Ordenar recursos para prevenir deadlock
        return sorted(resources)
    
    def check_deadlock_risk(self, current_locks: Set[str], requested_locks: Set[str]) -> bool:
        """Verificar riesgo de deadlock."""
        # Implementar detección de deadlock
        return len(current_locks.intersection(requested_locks)) > 0

# =============================================================================
# MEMORY LEAK DETECTION
# =============================================================================

class MemoryLeakDetector:
    """Detector de memory leaks."""
    
    def __init__(self) -> Any:
        self.baseline_memory = 0
        self.memory_snapshots: List[Tuple[float, float]] = []
        self.logger = logging.getLogger(__name__)
        self._setup_baseline()
    
    def _setup_baseline(self) -> Any:
        """Configurar línea base de memoria."""
        self.baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.logger.info(f"Línea base de memoria: {self.baseline_memory:.1f}MB")
    
    def take_snapshot(self, label: str = ""):
        """Tomar snapshot de memoria."""
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        timestamp = time.time()
        
        self.memory_snapshots.append((timestamp, current_memory))
        
        if label:
            self.logger.info(f"Snapshot '{label}': {current_memory:.1f}MB")
        
        return current_memory
    
    def check_memory_growth(self, threshold_mb: float = 100.0) -> bool:
        """Verificar crecimiento de memoria."""
        if len(self.memory_snapshots) < 2:
            return False
        
        initial_memory = self.memory_snapshots[0][1]
        current_memory = self.memory_snapshots[-1][1]
        growth = current_memory - initial_memory
        
        if growth > threshold_mb:
            self.logger.warning(f"⚠️ Posible memory leak: crecimiento de {growth:.1f}MB")
            return True
        
        return False
    
    def force_garbage_collection(self) -> Any:
        """Forzar garbage collection."""
        collected = gc.collect()
        self.logger.info(f"Garbage collection: {collected} objetos recolectados")
        
        # Tomar snapshot después de GC
        self.take_snapshot("post_gc")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Obtener resumen de memoria."""
        if not self.memory_snapshots:
            return {"error": "No hay snapshots disponibles"}
        
        initial_memory = self.memory_snapshots[0][1]
        current_memory = self.memory_snapshots[-1][1]
        growth = current_memory - initial_memory
        
        return {
            "baseline_mb": self.baseline_memory,
            "current_mb": current_memory,
            "growth_mb": growth,
            "snapshots_count": len(self.memory_snapshots),
            "growth_percent": (growth / initial_memory) * 100 if initial_memory > 0 else 0
        }

# =============================================================================
# TIMEOUT HANDLING
# =============================================================================

class TimeoutHandler:
    """Manejador de timeouts."""
    
    def __init__(self, default_timeout: float = 300.0):
        
    """__init__ function."""
self.default_timeout = default_timeout
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def timeout_context(self, timeout: Optional[float] = None):
        """Context manager para timeout."""
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        def timeout_handler(signum, frame) -> Any:
            raise TimeoutError(f"Operación excedió timeout de {timeout}s")
        
        # Configurar signal handler (solo en Unix)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
        
        try:
            yield
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    async def async_timeout(self, coro, timeout: Optional[float] = None):
        """Timeout para operaciones asíncronas."""
        timeout = timeout or self.default_timeout
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operación asíncrona excedió timeout de {timeout}s")
    
    def check_operation_time(self, start_time: float, max_time: float) -> bool:
        """Verificar tiempo de operación."""
        elapsed = time.time() - start_time
        if elapsed > max_time:
            self.logger.warning(f"Operación lenta: {elapsed:.1f}s > {max_time}s")
            return False
        return True

# =============================================================================
# DATA VALIDATION & SANITIZATION
# =============================================================================

class DataValidator:
    """Validador de datos para casos edge."""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
    
    def validate_numpy_array(self, array: np.ndarray, expected_shape: Optional[Tuple] = None) -> np.ndarray:
        """Validar array de NumPy."""
        if not isinstance(array, np.ndarray):
            raise ValidationError("Datos deben ser array de NumPy")
        
        if array.size == 0:
            raise ValidationError("Array no puede estar vacío")
        
        if np.isnan(array).any():
            raise ValidationError("Array contiene valores NaN")
        
        if np.isinf(array).any():
            raise ValidationError("Array contiene valores infinitos")
        
        if expected_shape and array.shape != expected_shape:
            raise ValidationError(f"Shape esperado {expected_shape}, obtenido {array.shape}")
        
        return array
    
    def validate_video_data(self, video_data: np.ndarray) -> np.ndarray:
        """Validar datos de video."""
        if len(video_data.shape) != 4:  # (frames, height, width, channels)
            raise ValidationError(f"Video debe tener 4 dimensiones, obtenido {len(video_data.shape)}")
        
        frames, height, width, channels = video_data.shape
        
        if frames == 0:
            raise ValidationError("Video no puede tener 0 frames")
        
        if height <= 0 or width <= 0:
            raise ValidationError("Dimensiones de video deben ser positivas")
        
        if channels not in [1, 3, 4]:  # Grayscale, RGB, RGBA
            raise ValidationError(f"Número de canales inválido: {channels}")
        
        # Normalizar valores si es necesario
        if video_data.dtype != np.float32:
            if video_data.dtype in [np.uint8, np.uint16]:
                video_data = video_data.astype(np.float32) / 255.0
            else:
                video_data = video_data.astype(np.float32)
        
        return video_data
    
    def validate_model_input(self, input_data: Any, expected_type: type) -> bool:
        """Validar entrada de modelo."""
        if not isinstance(input_data, expected_type):
            raise ValidationError(f"Tipo esperado {expected_type}, obtenido {type(input_data)}")
        
        return input_data
    
    def sanitize_file_path(self, path: str) -> str:
        """Sanitizar ruta de archivo."""
        # Remover caracteres peligrosos
        dangerous_chars = ['..', '~', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            path = path.replace(char, '')
        
        # Normalizar ruta
        return str(Path(path).resolve())

# =============================================================================
# SYSTEM OVERLOAD PROTECTION
# =============================================================================

class SystemOverloadProtector:
    """Protector contra sobrecarga del sistema."""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        
    """__init__ function."""
self.resource_monitor = resource_monitor
        self.logger = logging.getLogger(__name__)
        self.operation_queue: deque = deque(maxlen=100)
        self.backpressure_threshold = 0.8  # 80% de uso
    
    def check_system_health(self) -> bool:
        """Verificar salud del sistema."""
        usage = self.resource_monitor.get_current_usage()
        
        # Verificar múltiples métricas
        cpu_ok = usage.cpu_percent < self.resource_monitor.limits.cpu_percent
        memory_ok = usage.memory_percent < self.resource_monitor.limits.memory_percent
        disk_ok = usage.disk_percent < self.resource_monitor.limits.disk_percent
        
        return cpu_ok and memory_ok and disk_ok
    
    def should_throttle(self) -> bool:
        """Determinar si se debe aplicar throttling."""
        usage = self.resource_monitor.get_current_usage()
        
        # Aplicar throttling si cualquier recurso está cerca del límite
        return (
            usage.cpu_percent > self.backpressure_threshold * self.resource_monitor.limits.cpu_percent or
            usage.memory_percent > self.backpressure_threshold * self.resource_monitor.limits.memory_percent or
            usage.disk_percent > self.backpressure_threshold * self.resource_monitor.limits.disk_percent
        )
    
    def apply_backpressure(self, operation: Callable, *args, **kwargs):
        """Aplicar backpressure a operaciones."""
        if self.should_throttle():
            self.logger.warning("⚠️ Aplicando backpressure - sistema sobrecargado")
            time.sleep(1.0)  # Pausa para reducir carga
        
        return operation(*args, **kwargs)
    
    async def apply_async_backpressure(self, operation: Callable, *args, **kwargs):
        """Aplicar backpressure a operaciones asíncronas."""
        if self.should_throttle():
            self.logger.warning("⚠️ Aplicando backpressure asíncrono - sistema sobrecargado")
            await asyncio.sleep(1.0)
        
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            return operation(*args, **kwargs)

# =============================================================================
# INTEGRATED EDGE CASE HANDLER
# =============================================================================

class EdgeCaseHandler:
    """Manejador integrado de casos edge."""
    
    def __init__(self) -> Any:
        self.resource_monitor = ResourceMonitor()
        self.boundary_handler = BoundaryConditionHandler()
        self.race_handler = RaceConditionHandler()
        self.memory_detector = MemoryLeakDetector()
        self.timeout_handler = TimeoutHandler()
        self.data_validator = DataValidator()
        self.overload_protector = SystemOverloadProtector(self.resource_monitor)
        self.logger = logging.getLogger(__name__)
        
        # Iniciar monitoreo
        self.resource_monitor.start_monitoring()
    
    def __enter__(self) -> Any:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        self.cleanup()
    
    def cleanup(self) -> Any:
        """Limpieza de recursos."""
        self.resource_monitor.stop_monitoring()
        self.memory_detector.force_garbage_collection()
        self.logger.info("🧹 Limpieza de casos edge completada")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado completo del sistema."""
        usage = self.resource_monitor.get_current_usage()
        memory_summary = self.memory_detector.get_memory_summary()
        
        return {
            "resource_usage": {
                "cpu_percent": usage.cpu_percent,
                "memory_percent": usage.memory_percent,
                "gpu_memory_percent": usage.gpu_memory_percent,
                "disk_percent": usage.disk_percent,
                "available_memory_mb": usage.available_memory_mb
            },
            "memory_leak": memory_summary,
            "system_healthy": self.overload_protector.check_system_health(),
            "should_throttle": self.overload_protector.should_throttle(),
            "active_threads": usage.active_threads,
            "open_files": usage.open_files
        }
    
    def safe_operation(self, operation: Callable, *args, **kwargs):
        """Ejecutar operación de forma segura."""
        # Verificar salud del sistema
        if not self.overload_protector.check_system_health():
            raise SystemError("Sistema no saludable para operación")
        
        # Aplicar backpressure si es necesario
        return self.overload_protector.apply_backpressure(operation, *args, **kwargs)
    
    async def safe_async_operation(self, operation: Callable, *args, **kwargs):
        """Ejecutar operación asíncrona de forma segura."""
        # Verificar salud del sistema
        if not self.overload_protector.check_system_health():
            raise SystemError("Sistema no saludable para operación asíncrona")
        
        # Aplicar backpressure si es necesario
        return await self.overload_protector.apply_async_backpressure(operation, *args, **kwargs)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_edge_case_handler() -> EdgeCaseHandler:
    """Crear instancia del manejador de casos edge."""
    return EdgeCaseHandler()

def with_edge_case_protection(operation: Callable):
    """Decorador para protección de casos edge."""
    def wrapper(*args, **kwargs) -> Any:
        with EdgeCaseHandler() as handler:
            return handler.safe_operation(operation, *args, **kwargs)
    
    async def async_wrapper(*args, **kwargs) -> Any:
        with EdgeCaseHandler() as handler:
            return await handler.safe_async_operation(operation, *args, **kwargs)
    
    return async_wrapper if asyncio.iscoroutinefunction(operation) else wrapper

def validate_system_requirements():
    """Validar requerimientos del sistema."""
    # Verificar memoria disponible
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
    if available_memory < 4.0:  # Mínimo 4GB
        raise SystemError(f"Memoria insuficiente: {available_memory:.1f}GB disponible, 4GB requerido")
    
    # Verificar espacio en disco
    disk_usage = psutil.disk_usage('/')
    free_space_gb = disk_usage.free / (1024 * 1024 * 1024)
    if free_space_gb < 10.0:  # Mínimo 10GB
        raise SystemError(f"Espacio en disco insuficiente: {free_space_gb:.1f}GB disponible, 10GB requerido")
    
    # Verificar CPU
    cpu_count = psutil.cpu_count()
    if cpu_count < 2:
        raise SystemError(f"CPU insuficiente: {cpu_count} cores, mínimo 2 requerido")
    
    return True

# =============================================================================
# INITIALIZATION
# =============================================================================

# Instancia global del manejador de casos edge
global_edge_handler = EdgeCaseHandler()

def get_edge_handler() -> EdgeCaseHandler:
    """Obtener instancia global del manejador de casos edge."""
    return global_edge_handler

def setup_edge_case_handling():
    """Configurar sistema de manejo de casos edge."""
    try:
        validate_system_requirements()
        logger = logging.getLogger(__name__)
        logger.info("🔍 Sistema de manejo de casos edge inicializado")
        return global_edge_handler
    except Exception as e:
        logging.error(f"Error configurando manejo de casos edge: {e}")
        raise

# Configuración automática al importar
setup_edge_case_handling() 