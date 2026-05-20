"""
Diagnostics - Sistema de diagnóstico y depuración
==================================================

Utilidades para diagnóstico, depuración y análisis del sistema.
"""

import asyncio
import logging
import traceback
import sys
import gc
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import psutil
    _has_psutil = True
except ImportError:
    _has_psutil = False


@dataclass
class DiagnosticInfo:
    """Información de diagnóstico del sistema"""
    timestamp: datetime = field(default_factory=datetime.now)
    memory_usage: Optional[Dict[str, Any]] = None
    cpu_usage: Optional[float] = None
    active_tasks: int = 0
    thread_count: int = 0
    gc_stats: Optional[Dict[str, Any]] = None
    exception_count: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SystemDiagnostics:
    """
    Sistema de diagnóstico para monitorear el estado del sistema.
    
    Proporciona información detallada sobre:
    - Uso de memoria
    - Uso de CPU
    - Tareas activas
    - Estadísticas de garbage collection
    - Excepciones y errores
    """
    
    def __init__(self):
        self.exception_history: List[Dict[str, Any]] = []
        self.warning_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self._exception_count = defaultdict(int)
    
    def record_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar una excepción para diagnóstico.
        
        Args:
            exception: Excepción capturada
            context: Contexto adicional (opcional)
        """
        exc_info = {
            "type": type(exception).__name__,
            "message": str(exception),
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.exception_history.append(exc_info)
        self._exception_count[type(exception).__name__] += 1
        
        # Mantener historial limitado
        if len(self.exception_history) > self.max_history:
            self.exception_history = self.exception_history[-self.max_history:]
        
        logger.debug(f"📊 Exception recorded: {exc_info['type']}")
    
    def record_warning(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar una advertencia para diagnóstico.
        
        Args:
            message: Mensaje de advertencia
            context: Contexto adicional (opcional)
        """
        warning_info = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        self.warning_history.append(warning_info)
        
        # Mantener historial limitado
        if len(self.warning_history) > self.max_history:
            self.warning_history = self.warning_history[-self.max_history:]
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Obtener información de memoria"""
        if _has_psutil:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
                "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2),
                "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
                "total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 2)
            }
        else:
            return {"error": "psutil not available"}
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """Obtener información de CPU"""
        if _has_psutil:
            process = psutil.Process()
            return {
                "percent": round(process.cpu_percent(interval=0.1), 2),
                "count": psutil.cpu_count(),
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        else:
            return {"error": "psutil not available"}
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de garbage collection"""
        return {
            "collections": {
                str(gen): gc.get_stats()[gen] if gen < len(gc.get_stats()) else {}
                for gen in range(3)
            },
            "counts": gc.get_count(),
            "thresholds": gc.get_threshold()
        }
    
    def get_active_tasks_count(self) -> int:
        """Contar tareas asyncio activas"""
        try:
            return len([t for t in asyncio.all_tasks() if not t.done()])
        except RuntimeError:
            # No hay event loop activo
            return 0
    
    def get_diagnostic_info(self) -> DiagnosticInfo:
        """
        Obtener información completa de diagnóstico.
        
        Returns:
            DiagnosticInfo con toda la información del sistema
        """
        warnings = []
        errors = []
        
        # Analizar excepciones recientes
        recent_exceptions = self.exception_history[-10:]
        if recent_exceptions:
            error_types = defaultdict(int)
            for exc in recent_exceptions:
                error_types[exc["type"]] += 1
            
            for exc_type, count in error_types.items():
                if count > 5:
                    errors.append(f"High frequency of {exc_type} exceptions ({count} in recent history)")
        
        # Analizar memoria
        mem_info = self.get_memory_info()
        if isinstance(mem_info, dict) and "percent" in mem_info:
            if mem_info["percent"] > 90:
                warnings.append(f"High memory usage: {mem_info['percent']}%")
        
        # Analizar CPU
        cpu_info = self.get_cpu_info()
        if isinstance(cpu_info, dict) and "percent" in cpu_info:
            if cpu_info["percent"] > 90:
                warnings.append(f"High CPU usage: {cpu_info['percent']}%")
        
        return DiagnosticInfo(
            memory_usage=mem_info,
            cpu_usage=cpu_info.get("percent") if isinstance(cpu_info, dict) else None,
            active_tasks=self.get_active_tasks_count(),
            thread_count=len(psutil.Process().threads()) if _has_psutil else 0,
            gc_stats=self.get_gc_stats(),
            exception_count=len(self.exception_history),
            warnings=warnings,
            errors=errors
        )
    
    def get_exception_summary(self) -> Dict[str, Any]:
        """Obtener resumen de excepciones"""
        return {
            "total": len(self.exception_history),
            "by_type": dict(self._exception_count),
            "recent": [
                {
                    "type": exc["type"],
                    "message": exc["message"][:100],
                    "timestamp": exc["timestamp"]
                }
                for exc in self.exception_history[-10:]
            ]
        }
    
    def clear_history(self) -> None:
        """Limpiar historial de excepciones y advertencias"""
        self.exception_history.clear()
        self.warning_history.clear()
        self._exception_count.clear()
        logger.info("🧹 Diagnostic history cleared")


class TimeoutManager:
    """
    Gestor de timeouts con cancelación y notificaciones.
    
    Proporciona mejor control sobre timeouts con:
    - Cancelación graceful
    - Callbacks de timeout
    - Estadísticas de timeouts
    """
    
    def __init__(self):
        self.timeout_count = 0
        self.timeout_history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    async def execute_with_timeout(
        self,
        coro,
        timeout: float,
        timeout_callback: Optional[callable] = None,
        operation_name: Optional[str] = None
    ) -> Any:
        """
        Ejecutar coroutine con timeout y manejo mejorado.
        
        Args:
            coro: Coroutine a ejecutar
            timeout: Timeout en segundos
            timeout_callback: Callback opcional cuando ocurre timeout
            operation_name: Nombre de la operación para logging
            
        Returns:
            Resultado de la coroutine
            
        Raises:
            asyncio.TimeoutError: Si se excede el timeout
        """
        op_name = operation_name or "operation"
        
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.timeout_count += 1
            timeout_info = {
                "operation": op_name,
                "timeout": timeout,
                "timestamp": datetime.now().isoformat()
            }
            
            self.timeout_history.append(timeout_info)
            if len(self.timeout_history) > self.max_history:
                self.timeout_history = self.timeout_history[-self.max_history:]
            
            logger.warning(f"⏱️ Timeout in {op_name} after {timeout}s")
            
            if timeout_callback:
                try:
                    if asyncio.iscoroutinefunction(timeout_callback):
                        await timeout_callback()
                    else:
                        timeout_callback()
                except Exception as e:
                    logger.error(f"Error in timeout callback: {e}")
            
            raise
    
    def get_timeout_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de timeouts"""
        return {
            "total_timeouts": self.timeout_count,
            "recent_timeouts": len(self.timeout_history),
            "recent": self.timeout_history[-10:] if self.timeout_history else []
        }




