"""
AI Agent Orchestration System
==============================
Sistema de orquestación de agentes de IA
"""

import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Tipos de agentes"""
    PROCESSOR = "processor"
    ANALYZER = "analyzer"
    OPTIMIZER = "optimizer"
    VALIDATOR = "validator"
    RECOMMENDER = "recommender"
    CUSTOM = "custom"


class AgentStatus(Enum):
    """Estados de agente"""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class Agent:
    """Agente de IA"""
    id: str
    name: str
    agent_type: AgentType
    capabilities: List[str]
    status: AgentStatus
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class AgentTask:
    """Tarea de agente"""
    id: str
    task_type: str
    input_data: Dict[str, Any]
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class AgentWorkflow:
    """Workflow de agentes"""
    id: str
    name: str
    steps: List[Dict[str, Any]]
    created_at: float
    executed_at: Optional[float] = None


class AgentOrchestration:
    """
    Sistema de orquestación de agentes de IA
    """
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.workflows: Dict[str, AgentWorkflow] = {}
        self.task_queue: List[str] = []
        self.agent_handlers: Dict[AgentType, Callable] = {}
    
    def register_agent(
        self,
        name: str,
        agent_type: AgentType,
        capabilities: List[str],
        handler: Optional[Callable] = None
    ) -> Agent:
        """
        Registrar agente
        
        Args:
            name: Nombre del agente
            agent_type: Tipo de agente
            capabilities: Capacidades del agente
            handler: Handler para ejecutar tareas
        """
        agent_id = f"agent_{int(time.time())}"
        
        agent = Agent(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            capabilities=capabilities,
            status=AgentStatus.IDLE
        )
        
        self.agents[agent_id] = agent
        
        if handler:
            self.agent_handlers[agent_type] = handler
        
        return agent
    
    def create_task(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None
    ) -> AgentTask:
        """
        Crear tarea
        
        Args:
            task_type: Tipo de tarea
            input_data: Datos de entrada
            required_capabilities: Capacidades requeridas
        """
        task_id = f"task_{int(time.time() * 1000)}"
        
        task = AgentTask(
            id=task_id,
            task_type=task_type,
            input_data=input_data
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # Asignar a agente disponible
        agent = self._find_available_agent(required_capabilities)
        if agent:
            self._assign_task(task_id, agent.id)
        
        return task
    
    def _find_available_agent(
        self,
        required_capabilities: Optional[List[str]]
    ) -> Optional[Agent]:
        """Encontrar agente disponible"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.IDLE
        ]
        
        if not available_agents:
            return None
        
        # Filtrar por capacidades
        if required_capabilities:
            capable_agents = [
                agent for agent in available_agents
                if all(cap in agent.capabilities for cap in required_capabilities)
            ]
            if capable_agents:
                available_agents = capable_agents
        
        # Seleccionar agente con menos tareas completadas (balanceo)
        return min(available_agents, key=lambda a: a.tasks_completed)
    
    def _assign_task(self, task_id: str, agent_id: str):
        """Asignar tarea a agente"""
        if task_id not in self.tasks or agent_id not in self.agents:
            return
        
        task = self.tasks[task_id]
        agent = self.agents[agent_id]
        
        task.assigned_agent = agent_id
        task.status = "assigned"
        agent.status = AgentStatus.WORKING
        agent.current_task = task_id
        
        # Ejecutar tarea
        task.started_at = time.time()
        task.status = "running"
        
        try:
            handler = self.agent_handlers.get(agent.agent_type)
            if handler:
                result = handler(task.input_data)
                task.result = result
                task.status = "completed"
                agent.tasks_completed += 1
            else:
                # Handler por defecto
                task.result = {"status": "completed", "agent": agent_id}
                task.status = "completed"
                agent.tasks_completed += 1
        except Exception as e:
            task.result = {"error": str(e)}
            task.status = "failed"
            agent.tasks_failed += 1
        
        task.completed_at = time.time()
        agent.status = AgentStatus.IDLE
        agent.current_task = None
    
    def create_workflow(
        self,
        name: str,
        steps: List[Dict[str, Any]]
    ) -> AgentWorkflow:
        """
        Crear workflow de agentes
        
        Args:
            name: Nombre del workflow
            steps: Pasos del workflow
        """
        workflow_id = f"workflow_{int(time.time())}"
        
        workflow = AgentWorkflow(
            id=workflow_id,
            name=name,
            steps=steps,
            created_at=time.time()
        )
        
        self.workflows[workflow_id] = workflow
        return workflow
    
    def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ejecutar workflow
        
        Args:
            workflow_id: ID del workflow
            input_data: Datos de entrada
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        workflow.executed_at = time.time()
        
        current_data = input_data
        results = {}
        
        # Ejecutar pasos en orden
        for step in workflow.steps:
            step_type = step.get('type')
            step_input = step.get('input', current_data)
            
            # Crear tarea para el paso
            task = self.create_task(
                task_type=step_type,
                input_data=step_input,
                required_capabilities=step.get('required_capabilities')
            )
            
            # Esperar completación (en implementación real, usar async)
            while task.status == "running" or task.status == "assigned":
                time.sleep(0.1)
            
            if task.status == "completed":
                current_data = task.result
                results[step.get('name', step_type)] = task.result
            else:
                raise Exception(f"Step {step_type} failed: {task.result}")
        
        return results
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de agente"""
        if agent_id not in self.agents:
            return {}
        
        agent = self.agents[agent_id]
        
        return {
            'agent_id': agent_id,
            'name': agent.name,
            'type': agent.agent_type.value,
            'status': agent.status.value,
            'tasks_completed': agent.tasks_completed,
            'tasks_failed': agent.tasks_failed,
            'success_rate': (
                agent.tasks_completed / (agent.tasks_completed + agent.tasks_failed)
                if (agent.tasks_completed + agent.tasks_failed) > 0 else 0
            ),
            'current_task': agent.current_task
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            'total_agents': len(self.agents),
            'idle_agents': len([a for a in self.agents.values() if a.status == AgentStatus.IDLE]),
            'working_agents': len([a for a in self.agents.values() if a.status == AgentStatus.WORKING]),
            'total_tasks': len(self.tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == 'pending']),
            'completed_tasks': len([t for t in self.tasks.values() if t.status == 'completed']),
            'total_workflows': len(self.workflows)
        }


# Instancia global
agent_orchestration = AgentOrchestration()

