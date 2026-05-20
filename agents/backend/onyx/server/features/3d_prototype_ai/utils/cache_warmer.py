"""
Cache Warmer - Sistema de cache warming
=========================================
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Sistema de cache warming (pre-carga de caché)"""
    
    def __init__(self):
        self.warming_tasks: List[Dict[str, Any]] = []
        self.warmed_keys: set = set()
    
    def register_warming_task(self, name: str, task_func: Callable, 
                            priority: int = 0, schedule: Optional[str] = None):
        """Registra una tarea de warming"""
        self.warming_tasks.append({
            "name": name,
            "func": task_func,
            "priority": priority,
            "schedule": schedule,
            "last_run": None,
            "success": False
        })
        
        # Ordenar por prioridad
        self.warming_tasks.sort(key=lambda x: x["priority"], reverse=True)
        
        logger.info(f"Tarea de warming registrada: {name}")
    
    async def warm_cache(self, task_name: Optional[str] = None):
        """Ejecuta warming del caché"""
        tasks_to_run = self.warming_tasks
        
        if task_name:
            tasks_to_run = [t for t in tasks_to_run if t["name"] == task_name]
        
        for task in tasks_to_run:
            try:
                logger.info(f"Ejecutando warming: {task['name']}")
                
                if asyncio.iscoroutinefunction(task["func"]):
                    result = await task["func"]()
                else:
                    result = task["func"]()
                
                task["last_run"] = datetime.now().isoformat()
                task["success"] = True
                
                # Registrar claves calentadas
                if isinstance(result, dict) and "keys" in result:
                    self.warmed_keys.update(result["keys"])
                
                logger.info(f"Warming completado: {task['name']}")
            
            except Exception as e:
                task["success"] = False
                logger.error(f"Error en warming {task['name']}: {e}")
    
    async def warm_common_prototypes(self, prototype_generator):
        """Pre-calienta prototipos comunes"""
        common_requests = [
            {"product_description": "licuadora básica", "product_type": "licuadora"},
            {"product_description": "estufa de gas", "product_type": "estufa"},
            {"product_description": "máquina de corte", "product_type": "maquina"}
        ]
        
        warmed_keys = []
        
        for req_data in common_requests:
            try:
                from models.schemas import PrototypeRequest, ProductType
                request = PrototypeRequest(**req_data)
                await prototype_generator.generate_prototype(request)
                
                cache_key = f"prototype:{req_data['product_type']}"
                warmed_keys.append(cache_key)
            
            except Exception as e:
                logger.warning(f"Error warming prototipo común: {e}")
        
        return {"keys": warmed_keys, "count": len(warmed_keys)}
    
    def get_warming_status(self) -> Dict[str, Any]:
        """Obtiene estado de warming"""
        return {
            "total_tasks": len(self.warming_tasks),
            "successful_tasks": sum(1 for t in self.warming_tasks if t["success"]),
            "warmed_keys_count": len(self.warmed_keys),
            "tasks": [
                {
                    "name": t["name"],
                    "priority": t["priority"],
                    "last_run": t["last_run"],
                    "success": t["success"]
                }
                for t in self.warming_tasks
            ]
        }




