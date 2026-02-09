"""
API Composition System
======================

Sistema de composición de APIs.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CompositionStrategy(Enum):
    """Estrategia de composición."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    AGGREGATE = "aggregate"


@dataclass
class APIEndpoint:
    """Endpoint de API."""
    endpoint_id: str
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionStep:
    """Paso de composición."""
    step_id: str
    name: str
    endpoint: APIEndpoint
    condition: Optional[Callable] = None
    transform: Optional[Callable] = None
    depends_on: List[str] = field(default_factory=list)


@dataclass
class APIComposition:
    """Composición de API."""
    composition_id: str
    name: str
    strategy: CompositionStrategy
    steps: List[CompositionStep]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIComposer:
    """
    Compositor de APIs.
    
    Compone múltiples llamadas a APIs en una sola respuesta.
    """
    
    def __init__(self):
        """Inicializar compositor."""
        self.compositions: Dict[str, APIComposition] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history = 10000
    
    def create_composition(
        self,
        name: str,
        strategy: CompositionStrategy,
        steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crear composición.
        
        Args:
            name: Nombre de la composición
            strategy: Estrategia de composición
            steps: Lista de pasos con endpoint, condition, transform, depends_on
            metadata: Metadata adicional
            
        Returns:
            ID de la composición
        """
        composition_id = str(uuid.uuid4())
        
        composition_steps = []
        for i, step_data in enumerate(steps):
            step_id = f"{composition_id}_step_{i}"
            
            endpoint_data = step_data["endpoint"]
            endpoint = APIEndpoint(
                endpoint_id=f"{step_id}_endpoint",
                name=endpoint_data.get("name", f"Endpoint {i}"),
                url=endpoint_data["url"],
                method=endpoint_data.get("method", "GET"),
                headers=endpoint_data.get("headers", {}),
                timeout=endpoint_data.get("timeout", 30.0),
                retry_count=endpoint_data.get("retry_count", 3)
            )
            
            step = CompositionStep(
                step_id=step_id,
                name=step_data.get("name", f"Step {i}"),
                endpoint=endpoint,
                condition=step_data.get("condition"),
                transform=step_data.get("transform"),
                depends_on=step_data.get("depends_on", [])
            )
            
            composition_steps.append(step)
        
        composition = APIComposition(
            composition_id=composition_id,
            name=name,
            strategy=strategy,
            steps=composition_steps,
            metadata=metadata or {}
        )
        
        self.compositions[composition_id] = composition
        logger.info(f"Created composition: {name} ({composition_id})")
        
        return composition_id
    
    async def execute_composition(
        self,
        composition_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar composición.
        
        Args:
            composition_id: ID de la composición
            context: Contexto adicional
            
        Returns:
            Resultado de la composición
        """
        if composition_id not in self.compositions:
            raise ValueError(f"Composition not found: {composition_id}")
        
        composition = self.compositions[composition_id]
        context = context or {}
        results = {}
        
        start_time = datetime.now()
        
        try:
            if composition.strategy == CompositionStrategy.SEQUENTIAL:
                results = await self._execute_sequential(composition, context)
            elif composition.strategy == CompositionStrategy.PARALLEL:
                results = await self._execute_parallel(composition, context)
            elif composition.strategy == CompositionStrategy.CONDITIONAL:
                results = await self._execute_conditional(composition, context)
            elif composition.strategy == CompositionStrategy.AGGREGATE:
                results = await self._execute_aggregate(composition, context)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            self.execution_history.append({
                "composition_id": composition_id,
                "name": composition.name,
                "strategy": composition.strategy.value,
                "duration_seconds": duration,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
            
            if len(self.execution_history) > self.max_history:
                self.execution_history = self.execution_history[-self.max_history:]
            
            return results
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            self.execution_history.append({
                "composition_id": composition_id,
                "name": composition.name,
                "strategy": composition.strategy.value,
                "duration_seconds": duration,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.error(f"Error executing composition {composition.name}: {e}", exc_info=True)
            raise
    
    async def _execute_sequential(
        self,
        composition: APIComposition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecutar composición secuencialmente."""
        results = {}
        
        for step in composition.steps:
            # Verificar dependencias
            if step.depends_on:
                for dep in step.depends_on:
                    if dep not in results:
                        raise ValueError(f"Dependency {dep} not found")
            
            # Verificar condición
            if step.condition:
                if not step.condition(context, results):
                    continue
            
            # Ejecutar endpoint
            response = await self._call_endpoint(step.endpoint, context, results)
            
            # Aplicar transformación
            if step.transform:
                response = step.transform(response, context, results)
            
            results[step.step_id] = response
        
        return results
    
    async def _execute_parallel(
        self,
        composition: APIComposition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecutar composición en paralelo."""
        import asyncio
        
        tasks = []
        for step in composition.steps:
            task = self._execute_step(step, context, {})
            tasks.append((step.step_id, task))
        
        results = {}
        for step_id, task in tasks:
            try:
                result = await task
                results[step_id] = result
            except Exception as e:
                logger.error(f"Error in parallel step {step_id}: {e}", exc_info=True)
                results[step_id] = {"error": str(e)}
        
        return results
    
    async def _execute_conditional(
        self,
        composition: APIComposition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecutar composición condicionalmente."""
        results = {}
        
        for step in composition.steps:
            if step.condition:
                if not step.condition(context, results):
                    continue
            
            response = await self._call_endpoint(step.endpoint, context, results)
            
            if step.transform:
                response = step.transform(response, context, results)
            
            results[step.step_id] = response
        
        return results
    
    async def _execute_aggregate(
        self,
        composition: APIComposition,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ejecutar composición con agregación."""
        results = {}
        
        for step in composition.steps:
            response = await self._call_endpoint(step.endpoint, context, results)
            results[step.step_id] = response
        
        # Agregar resultados
        aggregated = {}
        for step_id, result in results.items():
            aggregated.update(result if isinstance(result, dict) else {step_id: result})
        
        return aggregated
    
    async def _execute_step(
        self,
        step: CompositionStep,
        context: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Any:
        """Ejecutar paso individual."""
        if step.condition and not step.condition(context, results):
            return None
        
        response = await self._call_endpoint(step.endpoint, context, results)
        
        if step.transform:
            response = step.transform(response, context, results)
        
        return response
    
    async def _call_endpoint(
        self,
        endpoint: APIEndpoint,
        context: Dict[str, Any],
        results: Dict[str, Any]
    ) -> Any:
        """Llamar a endpoint."""
        import aiohttp
        
        for attempt in range(endpoint.retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=endpoint.method,
                        url=endpoint.url,
                        headers=endpoint.headers,
                        json=context.get("payload", {}),
                        timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                    ) as response:
                        return await response.json()
            except Exception as e:
                if attempt == endpoint.retry_count - 1:
                    raise
                logger.warning(f"Retry {attempt + 1} for {endpoint.url}: {e}")
                await asyncio.sleep(1.0)
        
        raise Exception("Max retries exceeded")
    
    def get_composition(self, composition_id: str) -> Optional[APIComposition]:
        """Obtener composición por ID."""
        return self.compositions.get(composition_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del compositor."""
        successful = sum(1 for h in self.execution_history if h.get("success"))
        failed = sum(1 for h in self.execution_history if not h.get("success"))
        
        avg_duration = 0.0
        if self.execution_history:
            durations = [h.get("duration_seconds", 0) for h in self.execution_history]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            "total_compositions": len(self.compositions),
            "total_executions": len(self.execution_history),
            "successful_executions": successful,
            "failed_executions": failed,
            "average_duration_seconds": avg_duration
        }


# Importar asyncio
import asyncio

# Instancia global
_api_composer: Optional[APIComposer] = None


def get_api_composer() -> APIComposer:
    """Obtener instancia global del compositor."""
    global _api_composer
    if _api_composer is None:
        _api_composer = APIComposer()
    return _api_composer


