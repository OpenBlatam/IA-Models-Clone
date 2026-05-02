"""
Sistema de cache warming para pre-cargar datos frecuentemente accedidos.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from config.logging_config import get_logger

logger = get_logger(__name__)


class CacheWarmingStrategy:
    """Estrategia de cache warming."""
    
    def __init__(
        self,
        name: str,
        warm_function: Callable,
        priority: int = 0,
        interval_seconds: int = 3600,
        enabled: bool = True
    ):
        """
        Inicializar estrategia.
        
        Args:
            name: Nombre de la estrategia
            warm_function: Función async que calienta el cache
            priority: Prioridad (mayor = más importante)
            interval_seconds: Intervalo entre ejecuciones
            enabled: Si está habilitada
        """
        self.name = name
        self.warm_function = warm_function
        self.priority = priority
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.last_run: Optional[datetime] = None
        self.run_count = 0
        self.success_count = 0
        self.error_count = 0


class CacheWarmingService:
    """Servicio para calentar el cache automáticamente."""
    
    def __init__(self):
        """Inicializar servicio."""
        self.strategies: List[CacheWarmingStrategy] = []
        self.running = False
        self.warming_task: Optional[asyncio.Task] = None
    
    def register_strategy(self, strategy: CacheWarmingStrategy) -> None:
        """
        Registrar estrategia de warming.
        
        Args:
            strategy: Estrategia a registrar
        """
        self.strategies.append(strategy)
        # Ordenar por prioridad
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
        logger.info(f"Estrategia de cache warming registrada: {strategy.name}")
    
    async def warm_cache(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Calentar cache.
        
        Args:
            strategy_name: Nombre de estrategia específica (opcional)
            
        Returns:
            Resultado del warming
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "strategies_run": 0,
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        strategies_to_run = (
            [s for s in self.strategies if s.name == strategy_name]
            if strategy_name
            else [s for s in self.strategies if s.enabled]
        )
        
        for strategy in strategies_to_run:
            try:
                logger.info(f"Ejecutando cache warming: {strategy.name}")
                await strategy.warm_function()
                strategy.last_run = datetime.now()
                strategy.run_count += 1
                strategy.success_count += 1
                results["strategies_run"] += 1
                results["successful"] += 1
                results["details"].append({
                    "strategy": strategy.name,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error en cache warming {strategy.name}: {e}", exc_info=True)
                strategy.run_count += 1
                strategy.error_count += 1
                results["strategies_run"] += 1
                results["failed"] += 1
                results["details"].append({
                    "strategy": strategy.name,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    async def start_auto_warming(self, interval_seconds: int = 3600) -> None:
        """
        Iniciar warming automático.
        
        Args:
            interval_seconds: Intervalo entre ejecuciones
        """
        if self.running:
            return
        
        self.running = True
        
        async def warming_loop():
            while self.running:
                try:
                    # Ejecutar estrategias según su intervalo
                    now = datetime.now()
                    for strategy in self.strategies:
                        if not strategy.enabled:
                            continue
                        
                        should_run = False
                        if strategy.last_run is None:
                            should_run = True
                        else:
                            elapsed = (now - strategy.last_run).total_seconds()
                            if elapsed >= strategy.interval_seconds:
                                should_run = True
                        
                        if should_run:
                            try:
                                await strategy.warm_function()
                                strategy.last_run = now
                                strategy.run_count += 1
                                strategy.success_count += 1
                            except Exception as e:
                                logger.error(
                                    f"Error en auto-warming {strategy.name}: {e}",
                                    exc_info=True
                                )
                                strategy.run_count += 1
                                strategy.error_count += 1
                    
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error en warming loop: {e}", exc_info=True)
                    await asyncio.sleep(interval_seconds)
        
        self.warming_task = asyncio.create_task(warming_loop())
        logger.info("Cache warming automático iniciado")
    
    async def stop_auto_warming(self) -> None:
        """Detener warming automático."""
        self.running = False
        if self.warming_task:
            await self.warming_task
        logger.info("Cache warming automático detenido")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_strategies": len(self.strategies),
            "enabled_strategies": len([s for s in self.strategies if s.enabled]),
            "running": self.running,
            "strategies": [
                {
                    "name": s.name,
                    "priority": s.priority,
                    "enabled": s.enabled,
                    "interval_seconds": s.interval_seconds,
                    "run_count": s.run_count,
                    "success_count": s.success_count,
                    "error_count": s.error_count,
                    "last_run": s.last_run.isoformat() if s.last_run else None
                }
                for s in self.strategies
            ]
        }



