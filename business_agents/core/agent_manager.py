"""
Agent Manager - Gestor central de agentes de negocio
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
import uuid

from .agent_base import BaseAgent, AgentType, AgentStatus, AgentTask
from .event_system import EventSystem

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Gestor central de agentes de negocio.
    """
    
    def __init__(self):
        """Inicializar el gestor de agentes."""
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[AgentType, List[str]] = {}
        self.event_system = EventSystem()
        
        # Configuración
        self.max_agents_per_type = 10
        self.agent_health_check_interval = 30  # segundos
        
        # Métricas globales
        self.global_metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0.0
        }
        
        # Inicializar tipos de agentes
        for agent_type in AgentType:
            self.agent_types[agent_type] = []
        
        logger.info("AgentManager inicializado")
    
    async def initialize(self):
        """Inicializar el gestor de agentes."""
        try:
            # Iniciar sistema de eventos
            await self.event_system.initialize()
            
            # Iniciar monitoreo de salud
            asyncio.create_task(self._health_monitoring_loop())
            
            logger.info("AgentManager inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar AgentManager: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el gestor de agentes."""
        try:
            # Detener todos los agentes
            for agent in self.agents.values():
                await agent.stop()
            
            # Cerrar sistema de eventos
            await self.event_system.shutdown()
            
            self.agents.clear()
            for agent_type in self.agent_types:
                self.agent_types[agent_type].clear()
            
            logger.info("AgentManager cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar AgentManager: {e}")
    
    async def register_agent(
        self,
        agent: BaseAgent,
        auto_start: bool = True
    ) -> bool:
        """
        Registrar un nuevo agente.
        
        Args:
            agent: Instancia del agente
            auto_start: Si iniciar el agente automáticamente
            
        Returns:
            True si el registro fue exitoso
        """
        try:
            # Verificar límites
            if len(self.agent_types[agent.agent_type]) >= self.max_agents_per_type:
                logger.warning(f"Límite de agentes alcanzado para tipo {agent.agent_type.value}")
                return False
            
            # Verificar ID único
            if agent.agent_id in self.agents:
                logger.warning(f"Agente con ID {agent.agent_id} ya existe")
                return False
            
            # Registrar agente
            self.agents[agent.agent_id] = agent
            self.agent_types[agent.agent_type].append(agent.agent_id)
            
            # Iniciar agente si se solicita
            if auto_start:
                success = await agent.start()
                if not success:
                    # Remover agente si no se pudo iniciar
                    del self.agents[agent.agent_id]
                    self.agent_types[agent.agent_type].remove(agent.agent_id)
                    return False
            
            # Actualizar métricas
            self.global_metrics["total_agents"] += 1
            if agent.status in [AgentStatus.RUNNING, AgentStatus.BUSY]:
                self.global_metrics["active_agents"] += 1
            
            # Emitir evento
            await self.event_system.emit_event("agent_registered", {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "name": agent.name
            })
            
            logger.info(f"Agente {agent.agent_id} registrado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al registrar agente {agent.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Desregistrar un agente.
        
        Args:
            agent_id: ID del agente
            
        Returns:
            True si el desregistro fue exitoso
        """
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agente {agent_id} no encontrado")
                return False
            
            agent = self.agents[agent_id]
            
            # Detener agente
            await agent.stop()
            
            # Remover de registros
            del self.agents[agent_id]
            self.agent_types[agent.agent_type].remove(agent_id)
            
            # Actualizar métricas
            self.global_metrics["total_agents"] -= 1
            if agent.status in [AgentStatus.RUNNING, AgentStatus.BUSY]:
                self.global_metrics["active_agents"] -= 1
            
            # Emitir evento
            await self.event_system.emit_event("agent_unregistered", {
                "agent_id": agent_id,
                "agent_type": agent.agent_type.value
            })
            
            logger.info(f"Agente {agent_id} desregistrado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al desregistrar agente {agent_id}: {e}")
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Obtener agente por ID."""
        return self.agents.get(agent_id)
    
    async def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Obtener agentes por tipo."""
        agent_ids = self.agent_types.get(agent_type, [])
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    async def get_available_agents(self, task_type: str) -> List[BaseAgent]:
        """Obtener agentes disponibles para un tipo de tarea."""
        available_agents = []
        
        for agent in self.agents.values():
            if (agent.status in [AgentStatus.IDLE, AgentStatus.RUNNING] and
                agent.can_handle_task(task_type)):
                available_agents.append(agent)
        
        return available_agents
    
    async def submit_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        agent_type: Optional[AgentType] = None,
        agent_id: Optional[str] = None,
        priority: int = 1
    ) -> Optional[str]:
        """
        Enviar tarea a un agente.
        
        Args:
            task_type: Tipo de tarea
            parameters: Parámetros de la tarea
            agent_type: Tipo de agente (opcional)
            agent_id: ID específico de agente (opcional)
            priority: Prioridad de la tarea
            
        Returns:
            ID de la tarea o None si no se pudo enviar
        """
        try:
            # Buscar agente específico
            if agent_id:
                agent = await self.get_agent(agent_id)
                if not agent or not agent.can_handle_task(task_type):
                    logger.warning(f"Agente {agent_id} no puede manejar tarea {task_type}")
                    return None
                
                task_id = await agent.submit_task(task_type, parameters, priority)
                self.global_metrics["total_tasks"] += 1
                return task_id
            
            # Buscar por tipo de agente
            if agent_type:
                agents = await self.get_agents_by_type(agent_type)
                available_agents = [a for a in agents if a.can_handle_task(task_type)]
            else:
                available_agents = await self.get_available_agents(task_type)
            
            if not available_agents:
                logger.warning(f"No hay agentes disponibles para tarea {task_type}")
                return None
            
            # Seleccionar agente con menor carga
            best_agent = min(available_agents, key=lambda a: len(a.task_queue))
            
            task_id = await best_agent.submit_task(task_type, parameters, priority)
            self.global_metrics["total_tasks"] += 1
            
            logger.info(f"Tarea {task_id} enviada al agente {best_agent.agent_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error al enviar tarea: {e}")
            return None
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de una tarea."""
        for agent in self.agents.values():
            status = await agent.get_task_status(task_id)
            if status:
                return status
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancelar una tarea."""
        for agent in self.agents.values():
            if await agent.cancel_task(task_id):
                return True
        return False
    
    async def get_all_agents_status(self) -> Dict[str, Any]:
        """Obtener estado de todos los agentes."""
        agents_status = {}
        
        for agent_id, agent in self.agents.items():
            agents_status[agent_id] = await agent.get_status()
        
        return {
            "total_agents": len(self.agents),
            "agents_by_type": {
                agent_type.value: len(agent_ids)
                for agent_type, agent_ids in self.agent_types.items()
                if agent_ids
            },
            "agents": agents_status,
            "global_metrics": self.global_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """Obtener métricas globales."""
        # Actualizar métricas de agentes activos
        active_count = sum(
            1 for agent in self.agents.values()
            if agent.status in [AgentStatus.RUNNING, AgentStatus.BUSY]
        )
        self.global_metrics["active_agents"] = active_count
        
        # Calcular métricas agregadas
        total_completed = sum(agent.metrics.completed_tasks for agent in self.agents.values())
        total_failed = sum(agent.metrics.failed_tasks for agent in self.agents.values())
        
        self.global_metrics["completed_tasks"] = total_completed
        self.global_metrics["failed_tasks"] = total_failed
        
        # Calcular tiempo promedio de respuesta
        if total_completed > 0:
            total_time = sum(agent.metrics.average_execution_time * agent.metrics.completed_tasks
                           for agent in self.agents.values())
            self.global_metrics["average_response_time"] = total_time / total_completed
        
        return {
            **self.global_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Verificar salud de todos los agentes."""
        health_results = {}
        healthy_count = 0
        
        for agent_id, agent in self.agents.items():
            health = await agent.health_check()
            health_results[agent_id] = health
            
            if health["status"] == "healthy":
                healthy_count += 1
        
        return {
            "total_agents": len(self.agents),
            "healthy_agents": healthy_count,
            "unhealthy_agents": len(self.agents) - healthy_count,
            "health_percentage": (healthy_count / len(self.agents) * 100) if self.agents else 0,
            "agents_health": health_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def restart_agent(self, agent_id: str) -> bool:
        """Reiniciar un agente."""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return False
            
            # Detener agente
            await agent.stop()
            
            # Esperar un momento
            await asyncio.sleep(1)
            
            # Iniciar agente
            success = await agent.start()
            
            if success:
                await self.event_system.emit_event("agent_restarted", {
                    "agent_id": agent_id
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Error al reiniciar agente {agent_id}: {e}")
            return False
    
    async def update_agent_configuration(
        self,
        agent_id: str,
        new_config: Dict[str, Any]
    ) -> bool:
        """Actualizar configuración de un agente."""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return False
            
            agent.update_configuration(new_config)
            
            await self.event_system.emit_event("agent_configuration_updated", {
                "agent_id": agent_id,
                "configuration": new_config
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar configuración del agente {agent_id}: {e}")
            return False
    
    async def _health_monitoring_loop(self):
        """Loop de monitoreo de salud de agentes."""
        while True:
            try:
                await asyncio.sleep(self.agent_health_check_interval)
                
                # Verificar salud de todos los agentes
                health_results = await self.health_check_all()
                
                # Emitir evento de salud
                await self.event_system.emit_event("health_check_completed", health_results)
                
                # Reiniciar agentes no saludables
                for agent_id, health in health_results["agents_health"].items():
                    if health["status"] == "unhealthy":
                        logger.warning(f"Agente {agent_id} no saludable, intentando reiniciar")
                        await self.restart_agent(agent_id)
                
            except Exception as e:
                logger.error(f"Error en loop de monitoreo de salud: {e}")
    
    async def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Obtener capacidades de todos los agentes."""
        capabilities = {}
        
        for agent_id, agent in self.agents.items():
            capabilities[agent_id] = agent.get_capabilities()
        
        return capabilities
    
    async def find_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Encontrar agentes por capacidad."""
        matching_agents = []
        
        for agent in self.agents.values():
            if capability in agent.get_capabilities():
                matching_agents.append(agent)
        
        return matching_agents




