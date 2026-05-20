"""
AI Agents Support for Autonomous Intelligent Systems
Sistema de Agentes AI para sistemas inteligentes autónomos ultra-optimizado
"""

import asyncio
import logging
import time
import json
import uuid
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import numpy as np
import random

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Tipos de agentes AI"""
    CONTENT_ANALYZER = "content_analyzer"
    SIMILARITY_DETECTOR = "similarity_detector"
    QUALITY_ASSESSOR = "quality_assessor"
    VARIATION_GENERATOR = "variation_generator"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    TOPIC_EXTRACTOR = "topic_extractor"
    LANGUAGE_DETECTOR = "language_detector"
    SUMMARIZER = "summarizer"
    TRANSLATOR = "translator"
    CLASSIFIER = "classifier"
    RECOMMENDER = "recommender"
    PREDICTOR = "predictor"


class AgentStatus(Enum):
    """Estados de agentes AI"""
    IDLE = "idle"
    THINKING = "thinking"
    PROCESSING = "processing"
    LEARNING = "learning"
    SLEEPING = "sleeping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Prioridades de tareas"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LearningMode(Enum):
    """Modos de aprendizaje"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    FEDERATED = "federated"


@dataclass
class AIAgent:
    """Agente AI"""
    id: str
    name: str
    type: AgentType
    status: AgentStatus
    capabilities: List[str]
    knowledge_base: Dict[str, Any]
    memory: Dict[str, Any]
    personality: Dict[str, Any]
    learning_mode: LearningMode
    performance_metrics: Dict[str, float]
    created_at: float
    last_active: float
    total_tasks: int
    successful_tasks: int
    metadata: Dict[str, Any]


@dataclass
class AgentTask:
    """Tarea de agente"""
    id: str
    agent_id: str
    task_type: str
    priority: TaskPriority
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]]
    status: str
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    learning_feedback: Optional[Dict[str, Any]]


@dataclass
class AgentMemory:
    """Memoria de agente"""
    id: str
    agent_id: str
    memory_type: str  # episodic, semantic, procedural
    content: Dict[str, Any]
    importance: float
    access_count: int
    last_accessed: float
    created_at: float


@dataclass
class AgentKnowledge:
    """Conocimiento de agente"""
    id: str
    agent_id: str
    domain: str
    knowledge_type: str  # fact, rule, pattern, model
    content: Dict[str, Any]
    confidence: float
    source: str
    created_at: float
    last_updated: float


@dataclass
class AgentCollaboration:
    """Colaboración entre agentes"""
    id: str
    agent_ids: List[str]
    collaboration_type: str  # parallel, sequential, hierarchical
    task_id: str
    status: str
    created_at: float
    completed_at: Optional[float]
    result: Optional[Dict[str, Any]]


class AgentMemoryManager:
    """Manager de memoria de agentes"""
    
    def __init__(self):
        self.memories: Dict[str, List[AgentMemory]] = defaultdict(list)
        self.knowledge: Dict[str, List[AgentKnowledge]] = defaultdict(list)
        self._lock = threading.Lock()
    
    async def store_memory(self, agent_id: str, memory_type: str, content: Dict[str, Any], 
                          importance: float = 0.5) -> str:
        """Almacenar memoria"""
        memory_id = f"memory_{uuid.uuid4().hex[:8]}"
        
        memory = AgentMemory(
            id=memory_id,
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            access_count=0,
            last_accessed=time.time(),
            created_at=time.time()
        )
        
        async with self._lock:
            self.memories[agent_id].append(memory)
            
            # Mantener solo las memorias más importantes
            if len(self.memories[agent_id]) > 1000:
                self.memories[agent_id].sort(key=lambda m: m.importance, reverse=True)
                self.memories[agent_id] = self.memories[agent_id][:1000]
        
        return memory_id
    
    async def retrieve_memory(self, agent_id: str, query: str, limit: int = 10) -> List[AgentMemory]:
        """Recuperar memoria"""
        async with self._lock:
            if agent_id not in self.memories:
                return []
            
            # Búsqueda simple por contenido
            relevant_memories = []
            for memory in self.memories[agent_id]:
                if query.lower() in str(memory.content).lower():
                    memory.access_count += 1
                    memory.last_accessed = time.time()
                    relevant_memories.append(memory)
            
            # Ordenar por importancia y acceso reciente
            relevant_memories.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
            return relevant_memories[:limit]
    
    async def store_knowledge(self, agent_id: str, domain: str, knowledge_type: str,
                            content: Dict[str, Any], confidence: float = 0.8, 
                            source: str = "agent") -> str:
        """Almacenar conocimiento"""
        knowledge_id = f"knowledge_{uuid.uuid4().hex[:8]}"
        
        knowledge = AgentKnowledge(
            id=knowledge_id,
            agent_id=agent_id,
            domain=domain,
            knowledge_type=knowledge_type,
            content=content,
            confidence=confidence,
            source=source,
            created_at=time.time(),
            last_updated=time.time()
        )
        
        async with self._lock:
            self.knowledge[agent_id].append(knowledge)
        
        return knowledge_id
    
    async def retrieve_knowledge(self, agent_id: str, domain: str, 
                               knowledge_type: Optional[str] = None) -> List[AgentKnowledge]:
        """Recuperar conocimiento"""
        async with self._lock:
            if agent_id not in self.knowledge:
                return []
            
            relevant_knowledge = []
            for knowledge in self.knowledge[agent_id]:
                if knowledge.domain == domain:
                    if knowledge_type is None or knowledge.knowledge_type == knowledge_type:
                        relevant_knowledge.append(knowledge)
            
            # Ordenar por confianza
            relevant_knowledge.sort(key=lambda k: k.confidence, reverse=True)
            return relevant_knowledge


class AgentLearningEngine:
    """Motor de aprendizaje de agentes"""
    
    def __init__(self):
        self.learning_models: Dict[str, Dict[str, Any]] = {}
        self.training_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
    
    async def train_agent(self, agent_id: str, training_data: List[Dict[str, Any]], 
                         learning_mode: LearningMode) -> Dict[str, Any]:
        """Entrenar agente"""
        try:
            # Simular entrenamiento
            start_time = time.time()
            
            # Generar modelo de aprendizaje simulado
            model = {
                "agent_id": agent_id,
                "learning_mode": learning_mode.value,
                "training_samples": len(training_data),
                "features": self._extract_features(training_data),
                "accuracy": np.random.uniform(0.7, 0.95),
                "precision": np.random.uniform(0.7, 0.95),
                "recall": np.random.uniform(0.7, 0.95),
                "f1_score": np.random.uniform(0.7, 0.95),
                "training_time": time.time() - start_time,
                "created_at": time.time()
            }
            
            self.learning_models[agent_id] = model
            self.training_data[agent_id].extend(training_data)
            
            # Actualizar historial de rendimiento
            self.performance_history[agent_id].append(model["accuracy"])
            
            return model
            
        except Exception as e:
            logger.error(f"Error training agent {agent_id}: {e}")
            raise
    
    def _extract_features(self, training_data: List[Dict[str, Any]]) -> List[str]:
        """Extraer características de los datos de entrenamiento"""
        features = set()
        for data in training_data:
            if isinstance(data, dict):
                features.update(data.keys())
        return list(features)
    
    async def predict(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Realizar predicción"""
        if agent_id not in self.learning_models:
            raise ValueError(f"Agent {agent_id} not trained")
        
        model = self.learning_models[agent_id]
        
        # Simular predicción
        prediction = {
            "agent_id": agent_id,
            "input": input_data,
            "prediction": np.random.uniform(0, 1),
            "confidence": np.random.uniform(0.7, 0.95),
            "model_accuracy": model["accuracy"],
            "timestamp": time.time()
        }
        
        return prediction
    
    async def evaluate_performance(self, agent_id: str) -> Dict[str, Any]:
        """Evaluar rendimiento del agente"""
        if agent_id not in self.performance_history:
            return {"error": "No performance history available"}
        
        history = self.performance_history[agent_id]
        
        return {
            "agent_id": agent_id,
            "current_accuracy": history[-1] if history else 0,
            "average_accuracy": np.mean(history),
            "best_accuracy": max(history),
            "worst_accuracy": min(history),
            "improvement_trend": self._calculate_trend(history),
            "training_epochs": len(history)
        }
    
    def _calculate_trend(self, history: List[float]) -> str:
        """Calcular tendencia de mejora"""
        if len(history) < 2:
            return "insufficient_data"
        
        recent = history[-5:] if len(history) >= 5 else history
        older = history[:len(history)-len(recent)] if len(history) > len(recent) else []
        
        if not older:
            return "insufficient_data"
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg + 0.05:
            return "improving"
        elif recent_avg < older_avg - 0.05:
            return "declining"
        else:
            return "stable"


class AgentCollaborationManager:
    """Manager de colaboración entre agentes"""
    
    def __init__(self):
        self.collaborations: Dict[str, AgentCollaboration] = {}
        self.agent_teams: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
    
    async def create_collaboration(self, agent_ids: List[str], collaboration_type: str, 
                                 task_id: str) -> str:
        """Crear colaboración"""
        collaboration_id = f"collab_{uuid.uuid4().hex[:8]}"
        
        collaboration = AgentCollaboration(
            id=collaboration_id,
            agent_ids=agent_ids,
            collaboration_type=collaboration_type,
            task_id=task_id,
            status="created",
            created_at=time.time(),
            completed_at=None,
            result=None
        )
        
        async with self._lock:
            self.collaborations[collaboration_id] = collaboration
            
            # Actualizar equipos de agentes
            for agent_id in agent_ids:
                self.agent_teams[agent_id].append(collaboration_id)
        
        return collaboration_id
    
    async def execute_collaboration(self, collaboration_id: str) -> Dict[str, Any]:
        """Ejecutar colaboración"""
        async with self._lock:
            if collaboration_id not in self.collaborations:
                raise ValueError(f"Collaboration {collaboration_id} not found")
            
            collaboration = self.collaborations[collaboration_id]
            collaboration.status = "executing"
        
        try:
            # Simular ejecución de colaboración
            start_time = time.time()
            
            if collaboration.collaboration_type == "parallel":
                result = await self._execute_parallel_collaboration(collaboration)
            elif collaboration.collaboration_type == "sequential":
                result = await self._execute_sequential_collaboration(collaboration)
            elif collaboration.collaboration_type == "hierarchical":
                result = await self._execute_hierarchical_collaboration(collaboration)
            else:
                raise ValueError(f"Unknown collaboration type: {collaboration.collaboration_type}")
            
            async with self._lock:
                collaboration.status = "completed"
                collaboration.completed_at = time.time()
                collaboration.result = result
            
            return result
            
        except Exception as e:
            async with self._lock:
                collaboration.status = "failed"
                collaboration.result = {"error": str(e)}
            raise
    
    async def _execute_parallel_collaboration(self, collaboration: AgentCollaboration) -> Dict[str, Any]:
        """Ejecutar colaboración paralela"""
        # Simular ejecución paralela
        await asyncio.sleep(0.5)
        
        return {
            "collaboration_type": "parallel",
            "agents": collaboration.agent_ids,
            "execution_time": 0.5,
            "results": {
                agent_id: f"Result from agent {agent_id}"
                for agent_id in collaboration.agent_ids
            },
            "combined_result": "Parallel collaboration completed successfully"
        }
    
    async def _execute_sequential_collaboration(self, collaboration: AgentCollaboration) -> Dict[str, Any]:
        """Ejecutar colaboración secuencial"""
        results = []
        
        for agent_id in collaboration.agent_ids:
            # Simular procesamiento secuencial
            await asyncio.sleep(0.2)
            results.append(f"Result from agent {agent_id}")
        
        return {
            "collaboration_type": "sequential",
            "agents": collaboration.agent_ids,
            "execution_time": 0.2 * len(collaboration.agent_ids),
            "results": results,
            "combined_result": "Sequential collaboration completed successfully"
        }
    
    async def _execute_hierarchical_collaboration(self, collaboration: AgentCollaboration) -> Dict[str, Any]:
        """Ejecutar colaboración jerárquica"""
        # Simular jerarquía (primer agente es coordinador)
        coordinator = collaboration.agent_ids[0]
        workers = collaboration.agent_ids[1:]
        
        # Coordinador planifica
        await asyncio.sleep(0.1)
        plan = f"Plan created by coordinator {coordinator}"
        
        # Trabajadores ejecutan
        worker_results = []
        for worker_id in workers:
            await asyncio.sleep(0.15)
            worker_results.append(f"Task completed by worker {worker_id}")
        
        # Coordinador integra resultados
        await asyncio.sleep(0.1)
        
        return {
            "collaboration_type": "hierarchical",
            "coordinator": coordinator,
            "workers": workers,
            "plan": plan,
            "worker_results": worker_results,
            "execution_time": 0.1 + 0.15 * len(workers) + 0.1,
            "combined_result": "Hierarchical collaboration completed successfully"
        }


class AIAgentManager:
    """Manager principal de agentes AI"""
    
    def __init__(self):
        self.agents: Dict[str, AIAgent] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.memory_manager = AgentMemoryManager()
        self.learning_engine = AgentLearningEngine()
        self.collaboration_manager = AgentCollaborationManager()
        self.is_running = False
        self._task_queue = deque()
        self._lock = threading.Lock()
    
    async def start(self):
        """Iniciar AI agent manager"""
        try:
            self.is_running = True
            
            # Inicializar agentes por defecto
            await self._initialize_default_agents()
            
            # Iniciar procesamiento de tareas
            asyncio.create_task(self._task_processing_loop())
            
            logger.info("AI agent manager started")
            
        except Exception as e:
            logger.error(f"Error starting AI agent manager: {e}")
            raise
    
    async def stop(self):
        """Detener AI agent manager"""
        try:
            self.is_running = False
            logger.info("AI agent manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping AI agent manager: {e}")
    
    async def _initialize_default_agents(self):
        """Inicializar agentes por defecto"""
        default_agents = [
            {
                "name": "Content Analyzer Agent",
                "type": AgentType.CONTENT_ANALYZER,
                "capabilities": ["text_analysis", "content_processing", "pattern_recognition"],
                "personality": {"analytical": 0.9, "precise": 0.8, "thorough": 0.9}
            },
            {
                "name": "Similarity Detector Agent",
                "type": AgentType.SIMILARITY_DETECTOR,
                "capabilities": ["similarity_analysis", "comparison", "matching"],
                "personality": {"comparative": 0.9, "accurate": 0.8, "efficient": 0.7}
            },
            {
                "name": "Quality Assessor Agent",
                "type": AgentType.QUALITY_ASSESSOR,
                "capabilities": ["quality_evaluation", "assessment", "grading"],
                "personality": {"critical": 0.8, "fair": 0.9, "detailed": 0.8}
            },
            {
                "name": "Variation Generator Agent",
                "type": AgentType.VARIATION_GENERATOR,
                "capabilities": ["content_generation", "variation_creation", "creativity"],
                "personality": {"creative": 0.9, "innovative": 0.8, "adaptive": 0.7}
            }
        ]
        
        for agent_info in default_agents:
            await self.create_agent(
                name=agent_info["name"],
                agent_type=agent_info["type"],
                capabilities=agent_info["capabilities"],
                personality=agent_info["personality"]
            )
    
    async def create_agent(self, name: str, agent_type: AgentType, 
                          capabilities: List[str], personality: Dict[str, float],
                          learning_mode: LearningMode = LearningMode.SUPERVISED) -> str:
        """Crear agente AI"""
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        agent = AIAgent(
            id=agent_id,
            name=name,
            type=agent_type,
            status=AgentStatus.IDLE,
            capabilities=capabilities,
            knowledge_base={},
            memory={},
            personality=personality,
            learning_mode=learning_mode,
            performance_metrics={
                "accuracy": 0.0,
                "efficiency": 0.0,
                "reliability": 0.0,
                "creativity": 0.0
            },
            created_at=time.time(),
            last_active=time.time(),
            total_tasks=0,
            successful_tasks=0,
            metadata={}
        )
        
        async with self._lock:
            self.agents[agent_id] = agent
        
        logger.info(f"AI agent created: {agent_id} ({name})")
        return agent_id
    
    async def assign_task(self, agent_id: str, task_type: str, input_data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         expected_output: Optional[Dict[str, Any]] = None) -> str:
        """Asignar tarea a agente"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = AgentTask(
            id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            priority=priority,
            input_data=input_data,
            expected_output=expected_output,
            status="pending",
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            result=None,
            error=None,
            learning_feedback=None
        )
        
        async with self._lock:
            self.tasks[task_id] = task
            self._task_queue.append(task_id)
        
        return task_id
    
    async def _task_processing_loop(self):
        """Loop de procesamiento de tareas"""
        while self.is_running:
            try:
                if self._task_queue:
                    task_id = self._task_queue.popleft()
                    await self._process_task(task_id)
                
                await asyncio.sleep(0.1)  # Procesar cada 100ms
                
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_task(self, task_id: str):
        """Procesar tarea"""
        try:
            task = self.tasks[task_id]
            agent = self.agents.get(task.agent_id)
            
            if not agent:
                task.status = "failed"
                task.error = f"Agent {task.agent_id} not found"
                return
            
            # Actualizar estado del agente
            agent.status = AgentStatus.PROCESSING
            agent.last_active = time.time()
            task.started_at = time.time()
            task.status = "processing"
            
            # Simular procesamiento de tarea
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Generar resultado simulado
            result = await self._generate_task_result(agent, task)
            
            # Actualizar tarea
            task.status = "completed"
            task.completed_at = time.time()
            task.result = result
            
            # Actualizar agente
            agent.status = AgentStatus.IDLE
            agent.total_tasks += 1
            agent.successful_tasks += 1
            
            # Actualizar métricas de rendimiento
            await self._update_agent_metrics(agent, task)
            
            # Almacenar en memoria
            await self.memory_manager.store_memory(
                agent.id, "episodic", {
                    "task_type": task.task_type,
                    "input": task.input_data,
                    "result": result,
                    "success": True
                }, importance=0.7
            )
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            task.status = "failed"
            task.error = str(e)
            
            if task.agent_id in self.agents:
                self.agents[task.agent_id].status = AgentStatus.ERROR
    
    async def _generate_task_result(self, agent: AIAgent, task: AgentTask) -> Dict[str, Any]:
        """Generar resultado de tarea"""
        # Simular resultado basado en el tipo de agente
        if agent.type == AgentType.CONTENT_ANALYZER:
            return {
                "analysis_type": "content_analysis",
                "word_count": len(str(task.input_data).split()),
                "complexity_score": np.random.uniform(0.3, 0.9),
                "readability_score": np.random.uniform(0.4, 0.8),
                "sentiment": random.choice(["positive", "negative", "neutral"]),
                "confidence": np.random.uniform(0.7, 0.95)
            }
        elif agent.type == AgentType.SIMILARITY_DETECTOR:
            return {
                "similarity_type": "text_similarity",
                "similarity_score": np.random.uniform(0.0, 1.0),
                "common_elements": random.randint(0, 10),
                "difference_score": np.random.uniform(0.0, 1.0),
                "confidence": np.random.uniform(0.7, 0.95)
            }
        elif agent.type == AgentType.QUALITY_ASSESSOR:
            return {
                "quality_type": "content_quality",
                "quality_score": np.random.uniform(0.0, 1.0),
                "grammar_score": np.random.uniform(0.6, 0.95),
                "clarity_score": np.random.uniform(0.5, 0.9),
                "coherence_score": np.random.uniform(0.4, 0.8),
                "confidence": np.random.uniform(0.7, 0.95)
            }
        elif agent.type == AgentType.VARIATION_GENERATOR:
            return {
                "variation_type": "content_variation",
                "variations": [
                    f"Variation {i+1} of the input content"
                    for i in range(random.randint(2, 5))
                ],
                "creativity_score": np.random.uniform(0.6, 0.9),
                "originality_score": np.random.uniform(0.5, 0.8),
                "confidence": np.random.uniform(0.7, 0.95)
            }
        else:
            return {
                "result_type": "generic_processing",
                "processed_data": task.input_data,
                "confidence": np.random.uniform(0.7, 0.95)
            }
    
    async def _update_agent_metrics(self, agent: AIAgent, task: AgentTask):
        """Actualizar métricas del agente"""
        # Simular actualización de métricas
        agent.performance_metrics["accuracy"] = min(1.0, agent.performance_metrics["accuracy"] + 0.01)
        agent.performance_metrics["efficiency"] = min(1.0, agent.performance_metrics["efficiency"] + 0.005)
        agent.performance_metrics["reliability"] = min(1.0, agent.performance_metrics["reliability"] + 0.008)
        
        if agent.type == AgentType.VARIATION_GENERATOR:
            agent.performance_metrics["creativity"] = min(1.0, agent.performance_metrics["creativity"] + 0.01)
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado del agente"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            "id": agent.id,
            "name": agent.name,
            "type": agent.type.value,
            "status": agent.status.value,
            "capabilities": agent.capabilities,
            "performance_metrics": agent.performance_metrics,
            "total_tasks": agent.total_tasks,
            "successful_tasks": agent.successful_tasks,
            "success_rate": agent.successful_tasks / agent.total_tasks if agent.total_tasks > 0 else 0,
            "last_active": agent.last_active,
            "created_at": agent.created_at
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "is_running": self.is_running,
            "agents": {
                "total": len(self.agents),
                "by_type": {
                    agent_type.value: sum(1 for a in self.agents.values() if a.type == agent_type)
                    for agent_type in AgentType
                },
                "by_status": {
                    status.value: sum(1 for a in self.agents.values() if a.status == status)
                    for status in AgentStatus
                }
            },
            "tasks": {
                "total": len(self.tasks),
                "pending": len(self._task_queue),
                "completed": sum(1 for t in self.tasks.values() if t.status == "completed"),
                "failed": sum(1 for t in self.tasks.values() if t.status == "failed")
            },
            "collaborations": len(self.collaboration_manager.collaborations),
            "memories": sum(len(memories) for memories in self.memory_manager.memories.values()),
            "knowledge": sum(len(knowledge) for knowledge in self.memory_manager.knowledge.values())
        }


# Instancia global del manager de agentes AI
ai_agent_manager = AIAgentManager()


# Router para endpoints de agentes AI
ai_agent_router = APIRouter()


@ai_agent_router.post("/ai-agents/create")
async def create_ai_agent_endpoint(agent_data: dict):
    """Crear agente AI"""
    try:
        name = agent_data["name"]
        agent_type = AgentType(agent_data["type"])
        capabilities = agent_data.get("capabilities", [])
        personality = agent_data.get("personality", {})
        learning_mode = LearningMode(agent_data.get("learning_mode", "supervised"))
        
        agent_id = await ai_agent_manager.create_agent(
            name, agent_type, capabilities, personality, learning_mode
        )
        
        return {
            "message": "AI agent created successfully",
            "agent_id": agent_id,
            "name": name,
            "type": agent_type.value
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid agent type or learning mode: {e}")
    except Exception as e:
        logger.error(f"Error creating AI agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create AI agent: {str(e)}")


@ai_agent_router.get("/ai-agents")
async def get_ai_agents_endpoint():
    """Obtener agentes AI"""
    try:
        agents = ai_agent_manager.agents
        return {
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.type.value,
                    "status": agent.status.value,
                    "capabilities": agent.capabilities,
                    "total_tasks": agent.total_tasks,
                    "successful_tasks": agent.successful_tasks,
                    "last_active": agent.last_active
                }
                for agent in agents.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting AI agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI agents: {str(e)}")


@ai_agent_router.get("/ai-agents/{agent_id}")
async def get_ai_agent_endpoint(agent_id: str):
    """Obtener agente AI específico"""
    try:
        status = await ai_agent_manager.get_agent_status(agent_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="AI agent not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI agent: {str(e)}")


@ai_agent_router.post("/ai-agents/{agent_id}/tasks")
async def assign_agent_task_endpoint(agent_id: str, task_data: dict):
    """Asignar tarea a agente AI"""
    try:
        task_type = task_data["task_type"]
        input_data = task_data["input_data"]
        priority = TaskPriority(task_data.get("priority", "medium"))
        expected_output = task_data.get("expected_output")
        
        task_id = await ai_agent_manager.assign_task(
            agent_id, task_type, input_data, priority, expected_output
        )
        
        return {
            "message": "Task assigned to AI agent successfully",
            "task_id": task_id,
            "agent_id": agent_id,
            "task_type": task_type
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {e}")
    except Exception as e:
        logger.error(f"Error assigning task to AI agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign task to AI agent: {str(e)}")


@ai_agent_router.get("/ai-agents/tasks/{task_id}")
async def get_agent_task_endpoint(task_id: str):
    """Obtener tarea de agente AI"""
    try:
        if task_id not in ai_agent_manager.tasks:
            raise HTTPException(status_code=404, detail="AI agent task not found")
        
        task = ai_agent_manager.tasks[task_id]
        
        return {
            "id": task.id,
            "agent_id": task.agent_id,
            "task_type": task.task_type,
            "priority": task.priority.value,
            "status": task.status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result,
            "error": task.error
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting AI agent task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI agent task: {str(e)}")


@ai_agent_router.post("/ai-agents/collaborations")
async def create_agent_collaboration_endpoint(collaboration_data: dict):
    """Crear colaboración entre agentes AI"""
    try:
        agent_ids = collaboration_data["agent_ids"]
        collaboration_type = collaboration_data["collaboration_type"]
        task_id = collaboration_data["task_id"]
        
        collaboration_id = await ai_agent_manager.collaboration_manager.create_collaboration(
            agent_ids, collaboration_type, task_id
        )
        
        return {
            "message": "AI agent collaboration created successfully",
            "collaboration_id": collaboration_id,
            "agent_ids": agent_ids,
            "collaboration_type": collaboration_type
        }
        
    except Exception as e:
        logger.error(f"Error creating AI agent collaboration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create AI agent collaboration: {str(e)}")


@ai_agent_router.post("/ai-agents/collaborations/{collaboration_id}/execute")
async def execute_agent_collaboration_endpoint(collaboration_id: str):
    """Ejecutar colaboración entre agentes AI"""
    try:
        result = await ai_agent_manager.collaboration_manager.execute_collaboration(collaboration_id)
        
        return {
            "message": "AI agent collaboration executed successfully",
            "collaboration_id": collaboration_id,
            "result": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing AI agent collaboration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute AI agent collaboration: {str(e)}")


@ai_agent_router.post("/ai-agents/{agent_id}/train")
async def train_ai_agent_endpoint(agent_id: str, training_data: dict):
    """Entrenar agente AI"""
    try:
        training_samples = training_data["training_samples"]
        learning_mode = LearningMode(training_data.get("learning_mode", "supervised"))
        
        result = await ai_agent_manager.learning_engine.train_agent(
            agent_id, training_samples, learning_mode
        )
        
        return {
            "message": "AI agent trained successfully",
            "agent_id": agent_id,
            "training_result": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid learning mode: {e}")
    except Exception as e:
        logger.error(f"Error training AI agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train AI agent: {str(e)}")


@ai_agent_router.get("/ai-agents/{agent_id}/performance")
async def get_agent_performance_endpoint(agent_id: str):
    """Obtener rendimiento del agente AI"""
    try:
        performance = await ai_agent_manager.learning_engine.evaluate_performance(agent_id)
        return performance
    except Exception as e:
        logger.error(f"Error getting AI agent performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI agent performance: {str(e)}")


@ai_agent_router.get("/ai-agents/stats")
async def get_ai_agent_stats_endpoint():
    """Obtener estadísticas de agentes AI"""
    try:
        stats = await ai_agent_manager.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting AI agent stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI agent stats: {str(e)}")


# Funciones de utilidad para integración
async def start_ai_agents():
    """Iniciar agentes AI"""
    await ai_agent_manager.start()


async def stop_ai_agents():
    """Detener agentes AI"""
    await ai_agent_manager.stop()


async def create_ai_agent(name: str, agent_type: AgentType, capabilities: List[str],
                         personality: Dict[str, float], learning_mode: LearningMode = LearningMode.SUPERVISED) -> str:
    """Crear agente AI"""
    return await ai_agent_manager.create_agent(name, agent_type, capabilities, personality, learning_mode)


async def assign_agent_task(agent_id: str, task_type: str, input_data: Dict[str, Any],
                           priority: TaskPriority = TaskPriority.MEDIUM) -> str:
    """Asignar tarea a agente AI"""
    return await ai_agent_manager.assign_task(agent_id, task_type, input_data, priority)


async def get_ai_agent_stats() -> Dict[str, Any]:
    """Obtener estadísticas de agentes AI"""
    return await ai_agent_manager.get_system_stats()


logger.info("AI agents support module loaded successfully")

