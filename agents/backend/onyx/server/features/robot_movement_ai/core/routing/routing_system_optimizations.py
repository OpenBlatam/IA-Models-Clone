"""
Routing System-Level Optimizations
===================================

Optimizaciones a nivel de sistema operativo y hardware.
Incluye: NUMA optimization, CPU affinity, I/O optimization, etc.
"""

import logging
import os
import sys
import threading
from typing import Dict, Any, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, some system optimizations disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CPUAffinityOptimizer:
    """Optimizador de afinidad de CPU para mejor rendimiento."""
    
    def __init__(self):
        """Inicializar optimizador de afinidad."""
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil not available")
        
        self.process = psutil.Process()
        self.original_cpus = self.process.cpu_affinity()
    
    def set_cpu_affinity(self, cpu_list: List[int]):
        """
        Establecer afinidad de CPU.
        
        Args:
            cpu_list: Lista de CPUs a usar
        """
        try:
            self.process.cpu_affinity(cpu_list)
            logger.info(f"CPU affinity set to CPUs: {cpu_list}")
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity: {e}")
    
    def set_high_performance_cpus(self):
        """Establecer CPUs de alto rendimiento (últimos cores, típicamente más rápidos)."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            logical_count = psutil.cpu_count(logical=True)
            
            # Usar los últimos cores físicos (típicamente más rápidos)
            high_perf_cpus = list(range(cpu_count, logical_count))
            if high_perf_cpus:
                self.set_cpu_affinity(high_perf_cpus)
            else:
                # Fallback: usar todos los cores
                self.set_cpu_affinity(list(range(logical_count)))
        except Exception as e:
            logger.warning(f"Failed to set high performance CPUs: {e}")
    
    def restore_affinity(self):
        """Restaurar afinidad original."""
        try:
            self.process.cpu_affinity(self.original_cpus)
            logger.info("CPU affinity restored")
        except Exception as e:
            logger.warning(f"Failed to restore CPU affinity: {e}")


class MemoryOptimizer:
    """Optimizador de memoria a nivel de sistema."""
    
    def __init__(self):
        """Inicializar optimizador de memoria."""
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil not available")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Obtener información de memoria del sistema."""
        if not PSUTIL_AVAILABLE:
            return {}
        
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent,
            'free_gb': mem.free / (1024**3)
        }
    
    def optimize_memory_settings(self):
        """Optimizar configuraciones de memoria."""
        if not PSUTIL_AVAILABLE:
            return
        
        # Sugerir configuración de PyTorch para mejor uso de memoria
        if TORCH_AVAILABLE:
            # Habilitar memory efficient attention si está disponible
            try:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                logger.info("PyTorch memory settings optimized")
            except Exception as e:
                logger.debug(f"Could not set PyTorch memory settings: {e}")


class IOOptimizer:
    """Optimizador de I/O para mejor rendimiento."""
    
    def __init__(self):
        """Inicializar optimizador de I/O."""
        self.original_buffering = None
    
    def set_buffering(self, buffering: int = 8192):
        """
        Configurar buffering para I/O.
        
        Args:
            buffering: Tamaño del buffer
        """
        try:
            # Configurar buffering para stdout/stderr
            sys.stdout.reconfigure(line_buffering=False)
            sys.stderr.reconfigure(line_buffering=False)
            logger.info(f"I/O buffering configured: {buffering}")
        except Exception as e:
            logger.debug(f"Could not configure I/O buffering: {e}")


class ThreadPoolOptimizer:
    """Optimizador de thread pools."""
    
    def __init__(self):
        """Inicializar optimizador de thread pools."""
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil not available")
    
    def get_optimal_thread_count(self, task_type: str = "cpu_bound") -> int:
        """
        Obtener número óptimo de threads.
        
        Args:
            task_type: Tipo de tarea ('cpu_bound', 'io_bound', 'mixed')
        
        Returns:
            Número óptimo de threads
        """
        cpu_count = psutil.cpu_count(logical=True)
        
        if task_type == "cpu_bound":
            # Para CPU-bound: usar todos los cores
            return cpu_count
        elif task_type == "io_bound":
            # Para I/O-bound: usar más threads
            return cpu_count * 2
        else:  # mixed
            # Para mixed: balanceado
            return int(cpu_count * 1.5)
    
    def optimize_torch_threads(self):
        """Optimizar número de threads de PyTorch."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(cpu_count)
            logger.info(f"PyTorch threads set to {cpu_count}")
        except Exception as e:
            logger.warning(f"Failed to optimize PyTorch threads: {e}")


class SystemPerformanceMonitor:
    """Monitor de rendimiento del sistema."""
    
    def __init__(self):
        """Inicializar monitor."""
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil not available")
        
        self.process = psutil.Process()
        self.start_time = time.time()
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.sample_interval = 1.0  # seconds
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Iniciar monitoreo en background.
        
        Args:
            interval: Intervalo de muestreo en segundos
        """
        if self.monitoring:
            return
        
        self.sample_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System performance monitoring started")
    
    def stop_monitoring(self):
        """Detener monitoreo."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("System performance monitoring stopped")
    
    def _monitor_loop(self):
        """Loop de monitoreo."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
                
                # Memory usage
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                self.memory_samples.append(mem_mb)
                
                # Mantener solo últimas 1000 muestras
                if len(self.cpu_samples) > 1000:
                    self.cpu_samples = self.cpu_samples[-1000:]
                if len(self.memory_samples) > 1000:
                    self.memory_samples = self.memory_samples[-1000:]
                
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento."""
        stats = {
            'cpu_usage_percent': self.process.cpu_percent(interval=0.1),
            'memory_mb': self.process.memory_info().rss / (1024 * 1024),
            'num_threads': self.process.num_threads(),
            'uptime_seconds': time.time() - self.start_time
        }
        
        if self.cpu_samples:
            stats['avg_cpu_usage'] = sum(self.cpu_samples) / len(self.cpu_samples)
            stats['max_cpu_usage'] = max(self.cpu_samples)
            stats['min_cpu_usage'] = min(self.cpu_samples)
        
        if self.memory_samples:
            stats['avg_memory_mb'] = sum(self.memory_samples) / len(self.memory_samples)
            stats['max_memory_mb'] = max(self.memory_samples)
            stats['min_memory_mb'] = min(self.memory_samples)
        
        # System-wide stats
        if PSUTIL_AVAILABLE:
            stats['system_cpu_percent'] = psutil.cpu_percent(interval=0.1)
            stats['system_memory_percent'] = psutil.virtual_memory().percent
        
        return stats


class SystemOptimizer:
    """Optimizador completo del sistema."""
    
    def __init__(self, enable_cpu_affinity: bool = False, enable_monitoring: bool = True):
        """
        Inicializar optimizador del sistema.
        
        Args:
            enable_cpu_affinity: Habilitar optimización de afinidad de CPU
            enable_monitoring: Habilitar monitoreo de rendimiento
        """
        self.cpu_optimizer = None
        self.memory_optimizer = None
        self.io_optimizer = None
        self.thread_optimizer = None
        self.monitor = None
        
        if PSUTIL_AVAILABLE:
            try:
                self.memory_optimizer = MemoryOptimizer()
                self.io_optimizer = IOOptimizer()
                self.thread_optimizer = ThreadPoolOptimizer()
                
                if enable_cpu_affinity:
                    self.cpu_optimizer = CPUAffinityOptimizer()
                
                if enable_monitoring:
                    self.monitor = SystemPerformanceMonitor()
                    self.monitor.start_monitoring()
                
                logger.info("System optimizer initialized")
            except Exception as e:
                logger.warning(f"System optimizer initialization failed: {e}")
        else:
            logger.warning("psutil not available, system optimizations disabled")
    
    def optimize_all(self):
        """Aplicar todas las optimizaciones."""
        # Optimizar threads de PyTorch
        if self.thread_optimizer:
            self.thread_optimizer.optimize_torch_threads()
        
        # Optimizar memoria
        if self.memory_optimizer:
            self.memory_optimizer.optimize_memory_settings()
        
        # Optimizar I/O
        if self.io_optimizer:
            self.io_optimizer.set_buffering()
        
        # Optimizar CPU affinity (opcional, puede afectar otros procesos)
        if self.cpu_optimizer:
            # No aplicar por defecto, solo si se solicita explícitamente
            pass
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtener información del sistema."""
        info = {}
        
        if PSUTIL_AVAILABLE:
            info['cpu_count_physical'] = psutil.cpu_count(logical=False)
            info['cpu_count_logical'] = psutil.cpu_count(logical=True)
            info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
            info['memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
        
        if self.monitor:
            info['performance_stats'] = self.monitor.get_stats()
        
        if self.memory_optimizer:
            info['memory_info'] = self.memory_optimizer.get_memory_info()
        
        return info
    
    def cleanup(self):
        """Limpiar recursos."""
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self.cpu_optimizer:
            self.cpu_optimizer.restore_affinity()

