"""
Sistema de Caché Real - Implementación práctica de caché distribuido
Sistema de caché funcional para AI History Comparison
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

# Simulación de Redis (reemplazar con redis real)
class MockRedis:
    """Mock de Redis para desarrollo local"""
    
    def __init__(self):
        self.data = {}
        self.expires = {}
    
    def get(self, key: str) -> Optional[str]:
        if key in self.expires and time.time() > self.expires[key]:
            del self.data[key]
            del self.expires[key]
            return None
        return self.data.get(key)
    
    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self.data[key] = value
        if ex:
            self.expires[key] = time.time() + ex
        return True
    
    def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
        if key in self.expires:
            del self.expires[key]
        return True
    
    def exists(self, key: str) -> bool:
        return key in self.data
    
    def keys(self, pattern: str = "*") -> list:
        if pattern == "*":
            return list(self.data.keys())
        return [k for k in self.data.keys() if pattern.replace("*", "") in k]

@dataclass
class CacheConfig:
    """Configuración del sistema de caché"""
    default_ttl: int = 3600  # 1 hora por defecto
    max_size: int = 10000   # Máximo 10k entradas
    cleanup_interval: int = 300  # Limpiar cada 5 minutos
    enable_compression: bool = True

class RealCacheSystem:
    """Sistema de caché real y funcional"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis = MockRedis()  # Cambiar por redis real en producción
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Iniciar tarea de limpieza automática"""
        async def cleanup():
            while True:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
        
        asyncio.create_task(cleanup())
    
    async def _cleanup_expired(self):
        """Limpiar entradas expiradas"""
        try:
            keys = self.redis.keys("*")
            current_time = time.time()
            
            for key in keys:
                if key in self.redis.expires and current_time > self.redis.expires[key]:
                    self.redis.delete(key)
        except Exception as e:
            self.stats["errors"] += 1
            print(f"Error en limpieza: {e}")
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generar clave única para el caché"""
        return f"{prefix}:{identifier}"
    
    def _serialize_data(self, data: Any) -> str:
        """Serializar datos para almacenamiento"""
        if self.config.enable_compression:
            # Aquí podrías añadir compresión real
            return json.dumps(data, default=str)
        return json.dumps(data, default=str)
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserializar datos del almacenamiento"""
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener datos del caché"""
        try:
            cached_data = self.redis.get(key)
            if cached_data is not None:
                self.stats["hits"] += 1
                return self._deserialize_data(cached_data)
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            self.stats["errors"] += 1
            print(f"Error obteniendo del caché: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Guardar datos en el caché"""
        try:
            ttl = ttl or self.config.default_ttl
            serialized_data = self._serialize_data(value)
            
            success = self.redis.set(key, serialized_data, ex=ttl)
            if success:
                self.stats["sets"] += 1
            return success
        except Exception as e:
            self.stats["errors"] += 1
            print(f"Error guardando en caché: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar datos del caché"""
        try:
            success = self.redis.delete(key)
            if success:
                self.stats["deletes"] += 1
            return success
        except Exception as e:
            self.stats["errors"] += 1
            print(f"Error eliminando del caché: {e}")
            return False
    
    async def get_or_set(self, key: str, factory_func, ttl: Optional[int] = None) -> Any:
        """Obtener del caché o generar y guardar"""
        # Intentar obtener del caché
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Generar nuevo valor
        try:
            if asyncio.iscoroutinefunction(factory_func):
                new_value = await factory_func()
            else:
                new_value = factory_func()
            
            # Guardar en caché
            await self.set(key, new_value, ttl)
            return new_value
        except Exception as e:
            self.stats["errors"] += 1
            print(f"Error en get_or_set: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.stats["hits"],
            "cache_misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.2f}%",
            "total_sets": self.stats["sets"],
            "total_deletes": self.stats["deletes"],
            "errors": self.stats["errors"],
            "total_requests": total_requests,
            "cache_size": len(self.redis.data),
            "config": {
                "default_ttl": self.config.default_ttl,
                "max_size": self.config.max_size,
                "compression_enabled": self.config.enable_compression
            }
        }

class ContentAnalysisCache:
    """Caché especializado para análisis de contenido"""
    
    def __init__(self, cache_system: RealCacheSystem):
        self.cache = cache_system
        self.prefix = "analysis"
    
    def _get_content_hash(self, content: str) -> str:
        """Generar hash único para el contenido"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_analysis(self, content: str) -> Optional[Dict[str, Any]]:
        """Obtener análisis desde caché"""
        content_hash = self._get_content_hash(content)
        key = self._generate_key(content_hash)
        return await self.cache.get(key)
    
    async def set_analysis(self, content: str, analysis: Dict[str, Any], ttl: int = 3600) -> bool:
        """Guardar análisis en caché"""
        content_hash = self._get_content_hash(content)
        key = self._generate_key(content_hash)
        return await self.cache.set(key, analysis, ttl)
    
    async def get_or_analyze(self, content: str, analyzer_func, ttl: int = 3600) -> Dict[str, Any]:
        """Obtener análisis del caché o generar nuevo"""
        content_hash = self._get_content_hash(content)
        key = self._generate_key(content_hash)
        
        async def analyze_content():
            """Función para generar análisis"""
            if asyncio.iscoroutinefunction(analyzer_func):
                return await analyzer_func(content)
            else:
                return analyzer_func(content)
        
        return await self.cache.get_or_set(key, analyze_content, ttl)
    
    def _generate_key(self, content_hash: str) -> str:
        """Generar clave para análisis"""
        return f"{self.prefix}:{content_hash}"

# Ejemplo de uso real
async def main():
    """Ejemplo de uso del sistema de caché"""
    
    # Configurar caché
    config = CacheConfig(
        default_ttl=1800,  # 30 minutos
        max_size=5000,
        enable_compression=True
    )
    
    cache_system = RealCacheSystem(config)
    analysis_cache = ContentAnalysisCache(cache_system)
    
    # Función de análisis simulada
    async def analyze_content(content: str) -> Dict[str, Any]:
        """Simular análisis de contenido"""
        await asyncio.sleep(0.5)  # Simular procesamiento
        
        return {
            "content_length": len(content),
            "word_count": len(content.split()),
            "sentiment": "positive" if "good" in content.lower() else "neutral",
            "timestamp": datetime.now().isoformat(),
            "analysis_id": f"analysis_{int(time.time())}"
        }
    
    # Probar caché
    test_content = "Este es un contenido de prueba para análisis"
    
    print("🚀 Probando sistema de caché...")
    
    # Primer análisis (sin caché)
    start_time = time.time()
    result1 = await analysis_cache.get_or_analyze(test_content, analyze_content)
    time1 = time.time() - start_time
    print(f"✅ Primer análisis: {time1:.3f}s")
    
    # Segundo análisis (con caché)
    start_time = time.time()
    result2 = await analysis_cache.get_or_analyze(test_content, analyze_content)
    time2 = time.time() - start_time
    print(f"⚡ Segundo análisis (caché): {time2:.3f}s")
    
    # Mostrar estadísticas
    stats = cache_system.get_stats()
    print(f"📊 Estadísticas del caché: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())





