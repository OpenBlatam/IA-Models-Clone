from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import time
import psutil
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
        import os
from typing import Any, List, Dict, Optional
"""
🛠️ BLATAM AI UTILS MODULE v5.0.0
================================

Utilidades modulares y helpers:
- 🔧 Common helpers
- 📊 Data validators
- 🎯 Performance formatters
- ⚙️ Configuration utils
"""


logger = logging.getLogger(__name__)

# =============================================================================
# 🔧 COMMON HELPERS
# =============================================================================

def format_duration_ms(duration_ms: float) -> str:
    """Formatea duración en ms para display."""
    if duration_ms < 1:
        return f"{duration_ms:.3f}ms"
    elif duration_ms < 1000:
        return f"{duration_ms:.1f}ms"
    else:
        return f"{duration_ms/1000:.2f}s"

def format_size_bytes(size_bytes: int) -> str:
    """Formatea tamaño en bytes."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"

def calculate_improvement_percentage(old_value: float, new_value: float) -> float:
    """Calcula porcentaje de mejora."""
    if old_value == 0:
        return 0.0
    return ((old_value - new_value) / old_value) * 100

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """División segura."""
    return numerator / denominator if denominator != 0 else default

# =============================================================================
# 📊 DATA VALIDATORS
# =============================================================================

class DataValidator:
    """Validador de datos modular."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_fields: List[str]) -> bool:
        """Valida que config tenga campos requeridos."""
        return all(field in config for field in required_fields)
    
    @staticmethod
    def validate_processing_data(data: Any) -> bool:
        """Valida datos para procesamiento."""
        if data is None:
            return False
        
        if isinstance(data, str):
            return len(data.strip()) > 0
        elif isinstance(data, (dict, list)):
            return len(data) > 0
        else:
            return True
    
    @staticmethod
    def validate_metrics(metrics: Dict[str, Any]) -> bool:
        """Valida métricas."""
        numeric_fields = ['response_time_ms', 'throughput_rps', 'cpu_usage', 'memory_usage_mb']
        
        for field in numeric_fields:
            if field in metrics:
                try:
                    float(metrics[field])
                except (TypeError, ValueError):
                    return False
        
        return True

# =============================================================================
# 🎯 PERFORMANCE FORMATTERS
# =============================================================================

class PerformanceFormatter:
    """Formateador de métricas de rendimiento."""
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
        """Formatea métricas para display."""
        formatted = {}
        
        if 'response_time_ms' in metrics:
            formatted['response_time'] = format_duration_ms(metrics['response_time_ms'])
        
        if 'throughput_rps' in metrics:
            formatted['throughput'] = f"{metrics['throughput_rps']:.1f} req/s"
        
        if 'cpu_usage' in metrics:
            formatted['cpu_usage'] = f"{metrics['cpu_usage']:.1f}%"
        
        if 'memory_usage_mb' in metrics:
            formatted['memory_usage'] = format_size_bytes(metrics['memory_usage_mb'] * 1024 * 1024)
        
        if 'cache_hit_rate' in metrics:
            formatted['cache_hit_rate'] = f"{metrics['cache_hit_rate']:.1f}%"
        
        if 'accuracy' in metrics:
            formatted['accuracy'] = f"{metrics['accuracy']:.1f}%"
        
        return formatted
    
    @staticmethod
    def create_performance_summary(stats: Dict[str, Any]) -> str:
        """Crea resumen de rendimiento."""
        summary_parts = []
        
        if 'response_time_ms' in stats:
            summary_parts.append(f"⚡ {format_duration_ms(stats['response_time_ms'])}")
        
        if 'throughput_rps' in stats:
            summary_parts.append(f"🚀 {stats['throughput_rps']:.0f} req/s")
        
        if 'success_rate' in stats:
            summary_parts.append(f"✅ {stats['success_rate']:.1f}%")
        
        return " | ".join(summary_parts) if summary_parts else "📊 No metrics"

# =============================================================================
# ⚙️ CONFIGURATION UTILS
# =============================================================================

class ConfigMerger:
    """Utilidad para fusionar configuraciones."""
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fusiona configuraciones de forma profunda."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigMerger.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def apply_environment_overrides(config: Dict[str, Any], env_prefix: str = "BLATAM_") -> Dict[str, Any]:
        """Aplica overrides de variables de entorno."""
        
        updated_config = config.copy()
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                
                # Convert string values to appropriate types
                if value.lower() in ('true', 'false'):
                    updated_config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    updated_config[config_key] = int(value)
                else:
                    try:
                        updated_config[config_key] = float(value)
                    except ValueError:
                        updated_config[config_key] = value
        
        return updated_config

# =============================================================================
# 🕒 TIMING UTILS
# =============================================================================

class TimingUtils:
    """Utilidades de timing y medición."""
    
    @staticmethod
    async def time_async_function(func, *args, **kwargs) -> tuple[Any, float]:
        """Mide tiempo de ejecución de función async."""
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000
        return result, duration_ms
    
    @staticmethod
    def time_function(func, *args, **kwargs) -> tuple[Any, float]:
        """Mide tiempo de ejecución de función sync."""
        start_time = time.time()
        result = func(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000
        return result, duration_ms
    
    @staticmethod
    async def timeout_after(seconds: float, coro):
        """Aplica timeout a una corrutina."""
        try:
            return await asyncio.wait_for(coro, timeout=seconds)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {seconds}s")
            raise

# =============================================================================
# 💻 SYSTEM UTILS
# =============================================================================

class SystemUtils:
    """Utilidades del sistema."""
    
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Obtiene métricas del sistema."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage_mb': memory.used / (1024 * 1024),
                'memory_total_mb': memory.total / (1024 * 1024),
                'memory_percent': memory.percent,
                'disk_usage_gb': disk.used / (1024 * 1024 * 1024),
                'disk_total_gb': disk.total / (1024 * 1024 * 1024),
                'disk_percent': (disk.used / disk.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    @staticmethod
    def get_process_metrics() -> Dict[str, Any]:
        """Obtiene métricas del proceso actual."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'process_cpu_percent': process.cpu_percent(),
                'process_memory_mb': memory_info.rss / (1024 * 1024),
                'process_threads': process.num_threads(),
                'process_open_files': len(process.open_files())
            }
        except Exception as e:
            logger.error(f"Error getting process metrics: {e}")
            return {}

# =============================================================================
# 🎯 DATA PROCESSORS
# =============================================================================

class DataProcessor:
    """Procesador de datos genérico."""
    
    @staticmethod
    def detect_data_type(data: Any) -> str:
        """Detecta tipo de datos."""
        if isinstance(data, str):
            return "text"
        elif isinstance(data, dict):
            if 'product_name' in data or 'features' in data:
                return "product_data"
            else:
                return "structured_data"
        elif isinstance(data, (list, tuple)):
            return "list_data"
        elif isinstance(data, (int, float)):
            return "numeric_data"
        else:
            return "unknown"
    
    @staticmethod
    def sanitize_data(data: Any) -> Any:
        """Sanitiza datos para procesamiento."""
        if isinstance(data, str):
            return data.strip()
        elif isinstance(data, dict):
            return {k: DataProcessor.sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [DataProcessor.sanitize_data(item) for item in data]
        else:
            return data

# =============================================================================
# 🏆 STATS CALCULATOR
# =============================================================================

class StatsCalculator:
    """Calculadora de estadísticas."""
    
    @staticmethod
    def calculate_percentiles(values: List[float], percentiles: List[float] = [50, 90, 95, 99]) -> Dict[str, float]:
        """Calcula percentiles."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        results = {}
        
        for p in percentiles:
            index = int((p / 100) * len(sorted_values)) - 1
            index = max(0, min(index, len(sorted_values) - 1))
            results[f"p{p}"] = sorted_values[index]
        
        return results
    
    @staticmethod
    def calculate_moving_average(values: List[float], window_size: int = 10) -> List[float]:
        """Calcula promedio móvil."""
        if len(values) < window_size:
            return values
        
        moving_averages = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            moving_averages.append(sum(window) / window_size)
        
        return moving_averages

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    # Formatters
    "format_duration_ms", "format_size_bytes", "calculate_improvement_percentage", "safe_divide",
    
    # Validators
    "DataValidator",
    
    # Performance
    "PerformanceFormatter",
    
    # Config
    "ConfigMerger",
    
    # Timing
    "TimingUtils",
    
    # System
    "SystemUtils",
    
    # Data processing
    "DataProcessor",
    
    # Stats
    "StatsCalculator"
] 