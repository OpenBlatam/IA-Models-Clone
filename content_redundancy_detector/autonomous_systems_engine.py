"""
Autonomous Systems Engine for Self-Governing Systems
Motor de Sistemas Autónomos para sistemas autogobernados ultra-optimizado
"""

import asyncio
import logging
import time
import json
import threading
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from datetime import datetime, timedelta
import statistics
import random

logger = logging.getLogger(__name__)


class AutonomousSystemType(Enum):
    """Tipos de sistemas autónomos"""
    SELF_HEALING = "self_healing"
    SELF_OPTIMIZING = "self_optimizing"
    SELF_CONFIGURING = "self_configuring"
    SELF_PROTECTING = "self_protecting"
    SELF_MONITORING = "self_monitoring"
    SELF_ADAPTING = "self_adapting"
    SELF_LEARNING = "self_learning"
    SELF_SCALING = "self_scaling"


class SystemState(Enum):
    """Estados del sistema"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    OPTIMIZING = "optimizing"
    ADAPTING = "adapting"
    LEARNING = "learning"


class ActionType(Enum):
    """Tipos de acciones"""
    HEAL = "heal"
    OPTIMIZE = "optimize"
    CONFIGURE = "configure"
    PROTECT = "protect"
    MONITOR = "monitor"
    ADAPT = "adapt"
    LEARN = "learn"
    SCALE = "scale"
    ALERT = "alert"
    RESTART = "restart"


class DecisionEngine(Enum):
    """Motores de decisión"""
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    EXPERT_SYSTEM = "expert_system"


@dataclass
class AutonomousSystem:
    """Sistema autónomo"""
    id: str
    name: str
    description: str
    system_type: AutonomousSystemType
    state: SystemState
    decision_engine: DecisionEngine
    health_score: float
    performance_metrics: Dict[str, float]
    configuration: Dict[str, Any]
    rules: List[Dict[str, Any]]
    learning_data: Dict[str, Any]
    created_at: float
    last_updated: float
    last_action: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class AutonomousAction:
    """Acción autónoma"""
    id: str
    system_id: str
    action_type: ActionType
    description: str
    parameters: Dict[str, Any]
    priority: int
    status: str
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    result: Optional[Dict[str, Any]]
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class SystemEvent:
    """Evento del sistema"""
    id: str
    system_id: str
    event_type: str
    severity: str
    message: str
    data: Dict[str, Any]
    timestamp: float
    is_resolved: bool
    resolution_action: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class LearningPattern:
    """Patrón de aprendizaje"""
    id: str
    system_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    actions: List[str]
    success_rate: float
    frequency: int
    last_used: float
    created_at: float
    metadata: Dict[str, Any]


class SelfHealingEngine:
    """Motor de auto-curación"""
    
    def __init__(self):
        self.healing_strategies: Dict[str, Callable] = {
            "restart_service": self._restart_service,
            "clear_cache": self._clear_cache,
            "reset_connections": self._reset_connections,
            "rollback_config": self._rollback_config,
            "scale_resources": self._scale_resources,
            "failover": self._failover
        }
    
    async def detect_issues(self, system: AutonomousSystem) -> List[Dict[str, Any]]:
        """Detectar problemas en el sistema"""
        issues = []
        
        # Verificar métricas de salud
        if system.health_score < 0.7:
            issues.append({
                "type": "health_degradation",
                "severity": "high",
                "description": f"Health score below threshold: {system.health_score}",
                "suggested_actions": ["restart_service", "clear_cache"]
            })
        
        # Verificar métricas de performance
        for metric, value in system.performance_metrics.items():
            if metric == "cpu_usage" and value > 80:
                issues.append({
                    "type": "high_cpu_usage",
                    "severity": "medium",
                    "description": f"CPU usage above threshold: {value}%",
                    "suggested_actions": ["scale_resources", "optimize_processes"]
                })
            elif metric == "memory_usage" and value > 85:
                issues.append({
                    "type": "high_memory_usage",
                    "severity": "medium",
                    "description": f"Memory usage above threshold: {value}%",
                    "suggested_actions": ["clear_cache", "scale_resources"]
                })
            elif metric == "response_time" and value > 1000:
                issues.append({
                    "type": "slow_response",
                    "severity": "high",
                    "description": f"Response time above threshold: {value}ms",
                    "suggested_actions": ["optimize_processes", "scale_resources"]
                })
        
        return issues
    
    async def execute_healing_action(self, system: AutonomousSystem, action: str, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar acción de curación"""
        try:
            strategy = self.healing_strategies.get(action)
            if not strategy:
                raise ValueError(f"Unknown healing action: {action}")
            
            result = await strategy(system, parameters)
            return {
                "action": action,
                "success": True,
                "result": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error executing healing action {action}: {e}")
            return {
                "action": action,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _restart_service(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Reiniciar servicio"""
        # Simular reinicio de servicio
        await asyncio.sleep(2)
        return {"message": "Service restarted successfully", "duration": 2}
    
    async def _clear_cache(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Limpiar caché"""
        # Simular limpieza de caché
        await asyncio.sleep(1)
        return {"message": "Cache cleared successfully", "duration": 1}
    
    async def _reset_connections(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Resetear conexiones"""
        # Simular reset de conexiones
        await asyncio.sleep(1.5)
        return {"message": "Connections reset successfully", "duration": 1.5}
    
    async def _rollback_config(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback de configuración"""
        # Simular rollback
        await asyncio.sleep(3)
        return {"message": "Configuration rolled back successfully", "duration": 3}
    
    async def _scale_resources(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Escalar recursos"""
        # Simular escalado
        await asyncio.sleep(5)
        return {"message": "Resources scaled successfully", "duration": 5}
    
    async def _failover(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Failover"""
        # Simular failover
        await asyncio.sleep(10)
        return {"message": "Failover completed successfully", "duration": 10}


class SelfOptimizingEngine:
    """Motor de auto-optimización"""
    
    def __init__(self):
        self.optimization_strategies: Dict[str, Callable] = {
            "optimize_cache": self._optimize_cache,
            "optimize_queries": self._optimize_queries,
            "optimize_resources": self._optimize_resources,
            "optimize_algorithms": self._optimize_algorithms,
            "optimize_network": self._optimize_network,
            "optimize_database": self._optimize_database
        }
    
    async def analyze_performance(self, system: AutonomousSystem) -> Dict[str, Any]:
        """Analizar performance del sistema"""
        analysis = {
            "bottlenecks": [],
            "optimization_opportunities": [],
            "recommendations": []
        }
        
        # Analizar métricas de performance
        for metric, value in system.performance_metrics.items():
            if metric == "cpu_usage" and value > 70:
                analysis["bottlenecks"].append({
                    "type": "cpu_bottleneck",
                    "severity": "medium",
                    "current_value": value,
                    "threshold": 70
                })
                analysis["optimization_opportunities"].append("optimize_resources")
            
            elif metric == "memory_usage" and value > 75:
                analysis["bottlenecks"].append({
                    "type": "memory_bottleneck",
                    "severity": "medium",
                    "current_value": value,
                    "threshold": 75
                })
                analysis["optimization_opportunities"].append("optimize_cache")
            
            elif metric == "response_time" and value > 500:
                analysis["bottlenecks"].append({
                    "type": "response_time_bottleneck",
                    "severity": "high",
                    "current_value": value,
                    "threshold": 500
                })
                analysis["optimization_opportunities"].append("optimize_queries")
        
        # Generar recomendaciones
        if analysis["optimization_opportunities"]:
            analysis["recommendations"] = [
                f"Consider {opp} to improve performance" 
                for opp in analysis["optimization_opportunities"]
            ]
        
        return analysis
    
    async def execute_optimization(self, system: AutonomousSystem, optimization: str, 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar optimización"""
        try:
            strategy = self.optimization_strategies.get(optimization)
            if not strategy:
                raise ValueError(f"Unknown optimization: {optimization}")
            
            result = await strategy(system, parameters)
            return {
                "optimization": optimization,
                "success": True,
                "result": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error executing optimization {optimization}: {e}")
            return {
                "optimization": optimization,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _optimize_cache(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar caché"""
        # Simular optimización de caché
        await asyncio.sleep(2)
        return {"message": "Cache optimized successfully", "improvement": "15%"}
    
    async def _optimize_queries(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar queries"""
        # Simular optimización de queries
        await asyncio.sleep(3)
        return {"message": "Queries optimized successfully", "improvement": "25%"}
    
    async def _optimize_resources(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar recursos"""
        # Simular optimización de recursos
        await asyncio.sleep(4)
        return {"message": "Resources optimized successfully", "improvement": "20%"}
    
    async def _optimize_algorithms(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar algoritmos"""
        # Simular optimización de algoritmos
        await asyncio.sleep(5)
        return {"message": "Algorithms optimized successfully", "improvement": "30%"}
    
    async def _optimize_network(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar red"""
        # Simular optimización de red
        await asyncio.sleep(2.5)
        return {"message": "Network optimized successfully", "improvement": "18%"}
    
    async def _optimize_database(self, system: AutonomousSystem, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar base de datos"""
        # Simular optimización de base de datos
        await asyncio.sleep(6)
        return {"message": "Database optimized successfully", "improvement": "22%"}


class SelfLearningEngine:
    """Motor de auto-aprendizaje"""
    
    def __init__(self):
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.experience_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def learn_from_experience(self, system: AutonomousSystem, 
                                  action_result: Dict[str, Any]) -> None:
        """Aprender de la experiencia"""
        system_id = system.id
        
        # Almacenar experiencia
        experience = {
            "timestamp": time.time(),
            "action": action_result.get("action"),
            "success": action_result.get("success", False),
            "context": {
                "health_score": system.health_score,
                "performance_metrics": system.performance_metrics,
                "state": system.state.value
            },
            "result": action_result.get("result")
        }
        
        self.experience_data[system_id].append(experience)
        
        # Mantener solo las últimas 1000 experiencias
        if len(self.experience_data[system_id]) > 1000:
            self.experience_data[system_id] = self.experience_data[system_id][-1000:]
        
        # Actualizar patrones de aprendizaje
        await self._update_learning_patterns(system_id, experience)
    
    async def _update_learning_patterns(self, system_id: str, experience: Dict[str, Any]) -> None:
        """Actualizar patrones de aprendizaje"""
        action = experience["action"]
        success = experience["success"]
        context = experience["context"]
        
        # Buscar patrón existente
        pattern_id = f"{system_id}_{action}_{context['state']}"
        
        if pattern_id in self.learning_patterns:
            pattern = self.learning_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_used = time.time()
            
            # Actualizar tasa de éxito
            if success:
                pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1) + 1) / pattern.frequency
            else:
                pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1)) / pattern.frequency
        else:
            # Crear nuevo patrón
            pattern = LearningPattern(
                id=pattern_id,
                system_id=system_id,
                pattern_type=action,
                conditions=context,
                actions=[action],
                success_rate=1.0 if success else 0.0,
                frequency=1,
                last_used=time.time(),
                created_at=time.time(),
                metadata={}
            )
            self.learning_patterns[pattern_id] = pattern
    
    async def get_recommendations(self, system: AutonomousSystem) -> List[Dict[str, Any]]:
        """Obtener recomendaciones basadas en aprendizaje"""
        recommendations = []
        system_id = system.id
        
        # Buscar patrones relevantes
        for pattern in self.learning_patterns.values():
            if pattern.system_id == system_id:
                # Verificar si las condiciones coinciden
                if self._conditions_match(pattern.conditions, system):
                    if pattern.success_rate > 0.7:  # Patrón exitoso
                        recommendations.append({
                            "action": pattern.pattern_type,
                            "confidence": pattern.success_rate,
                            "frequency": pattern.frequency,
                            "reason": f"Successful pattern with {pattern.success_rate:.2%} success rate"
                        })
        
        # Ordenar por confianza
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recommendations[:5]  # Top 5 recomendaciones
    
    def _conditions_match(self, pattern_conditions: Dict[str, Any], 
                         system: AutonomousSystem) -> bool:
        """Verificar si las condiciones coinciden"""
        # Verificar estado del sistema
        if pattern_conditions.get("state") != system.state.value:
            return False
        
        # Verificar health score (con tolerancia)
        pattern_health = pattern_conditions.get("health_score", 0)
        if abs(pattern_health - system.health_score) > 0.1:
            return False
        
        # Verificar métricas de performance (con tolerancia)
        pattern_metrics = pattern_conditions.get("performance_metrics", {})
        for metric, value in pattern_metrics.items():
            if metric in system.performance_metrics:
                if abs(value - system.performance_metrics[metric]) > value * 0.2:  # 20% tolerancia
                    return False
        
        return True


class DecisionEngine:
    """Motor de decisión autónoma"""
    
    def __init__(self):
        self.self_healing = SelfHealingEngine()
        self.self_optimizing = SelfOptimizingEngine()
        self.self_learning = SelfLearningEngine()
    
    async def make_decision(self, system: AutonomousSystem, 
                          event: Optional[SystemEvent] = None) -> Optional[AutonomousAction]:
        """Tomar decisión autónoma"""
        try:
            # Obtener recomendaciones de aprendizaje
            learning_recommendations = await self.self_learning.get_recommendations(system)
            
            # Detectar problemas
            issues = await self.self_healing.detect_issues(system)
            
            # Analizar performance
            performance_analysis = await self.self_optimizing.analyze_performance(system)
            
            # Decidir acción basada en el motor de decisión
            if system.decision_engine == DecisionEngine.RULE_BASED:
                action = await self._rule_based_decision(system, issues, performance_analysis)
            elif system.decision_engine == DecisionEngine.ML_BASED:
                action = await self._ml_based_decision(system, learning_recommendations)
            elif system.decision_engine == DecisionEngine.HYBRID:
                action = await self._hybrid_decision(system, issues, performance_analysis, learning_recommendations)
            else:
                action = None
            
            return action
            
        except Exception as e:
            logger.error(f"Error making autonomous decision: {e}")
            return None
    
    async def _rule_based_decision(self, system: AutonomousSystem, 
                                 issues: List[Dict[str, Any]], 
                                 performance_analysis: Dict[str, Any]) -> Optional[AutonomousAction]:
        """Decisión basada en reglas"""
        # Priorizar problemas críticos
        critical_issues = [issue for issue in issues if issue["severity"] == "high"]
        
        if critical_issues:
            issue = critical_issues[0]
            suggested_actions = issue.get("suggested_actions", [])
            
            if suggested_actions:
                return AutonomousAction(
                    id=f"action_{uuid.uuid4().hex[:8]}",
                    system_id=system.id,
                    action_type=ActionType.HEAL,
                    description=f"Addressing critical issue: {issue['type']}",
                    parameters={"issue": issue, "action": suggested_actions[0]},
                    priority=1,
                    status="pending",
                    created_at=time.time(),
                    started_at=None,
                    completed_at=None,
                    result=None,
                    execution_time=0.0,
                    metadata={}
                )
        
        # Considerar optimizaciones
        if performance_analysis["optimization_opportunities"]:
            optimization = performance_analysis["optimization_opportunities"][0]
            
            return AutonomousAction(
                id=f"action_{uuid.uuid4().hex[:8]}",
                system_id=system.id,
                action_type=ActionType.OPTIMIZE,
                description=f"Optimizing system performance: {optimization}",
                parameters={"optimization": optimization},
                priority=2,
                status="pending",
                created_at=time.time(),
                started_at=None,
                completed_at=None,
                result=None,
                execution_time=0.0,
                metadata={}
            )
        
        return None
    
    async def _ml_based_decision(self, system: AutonomousSystem, 
                               learning_recommendations: List[Dict[str, Any]]) -> Optional[AutonomousAction]:
        """Decisión basada en ML"""
        if learning_recommendations:
            recommendation = learning_recommendations[0]
            
            return AutonomousAction(
                id=f"action_{uuid.uuid4().hex[:8]}",
                system_id=system.id,
                action_type=ActionType.HEAL if "heal" in recommendation["action"] else ActionType.OPTIMIZE,
                description=f"ML-based action: {recommendation['action']}",
                parameters={"recommendation": recommendation},
                priority=1,
                status="pending",
                created_at=time.time(),
                started_at=None,
                completed_at=None,
                result=None,
                execution_time=0.0,
                metadata={}
            )
        
        return None
    
    async def _hybrid_decision(self, system: AutonomousSystem, 
                             issues: List[Dict[str, Any]], 
                             performance_analysis: Dict[str, Any], 
                             learning_recommendations: List[Dict[str, Any]]) -> Optional[AutonomousAction]:
        """Decisión híbrida"""
        # Combinar reglas y ML
        rule_action = await self._rule_based_decision(system, issues, performance_analysis)
        ml_action = await self._ml_based_decision(system, learning_recommendations)
        
        # Priorizar reglas para problemas críticos
        if rule_action and rule_action.priority == 1:
            return rule_action
        
        # Usar ML para optimizaciones
        if ml_action:
            return ml_action
        
        return rule_action


class AutonomousSystemsEngine:
    """Motor principal de sistemas autónomos"""
    
    def __init__(self):
        self.systems: Dict[str, AutonomousSystem] = {}
        self.actions: Dict[str, AutonomousAction] = {}
        self.events: Dict[str, SystemEvent] = {}
        self.decision_engine = DecisionEngine()
        self.is_running = False
        self._monitoring_task = None
        self._action_executor_task = None
        self._lock = threading.Lock()
    
    async def start(self):
        """Iniciar motor de sistemas autónomos"""
        try:
            self.is_running = True
            
            # Iniciar tareas de monitoreo y ejecución
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._action_executor_task = asyncio.create_task(self._action_executor_loop())
            
            logger.info("Autonomous systems engine started")
            
        except Exception as e:
            logger.error(f"Error starting autonomous systems engine: {e}")
            raise
    
    async def stop(self):
        """Detener motor de sistemas autónomos"""
        try:
            self.is_running = False
            
            # Detener tareas
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self._action_executor_task:
                self._action_executor_task.cancel()
                try:
                    await self._action_executor_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Autonomous systems engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping autonomous systems engine: {e}")
    
    async def _monitoring_loop(self):
        """Loop de monitoreo"""
        while self.is_running:
            try:
                # Monitorear todos los sistemas
                for system in self.systems.values():
                    await self._monitor_system(system)
                
                await asyncio.sleep(30)  # Monitorear cada 30 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _action_executor_loop(self):
        """Loop de ejecución de acciones"""
        while self.is_running:
            try:
                # Ejecutar acciones pendientes
                pending_actions = [action for action in self.actions.values() 
                                 if action.status == "pending"]
                
                for action in pending_actions:
                    await self._execute_action(action)
                
                await asyncio.sleep(10)  # Ejecutar cada 10 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in action executor loop: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_system(self, system: AutonomousSystem):
        """Monitorear sistema individual"""
        try:
            # Actualizar métricas de performance
            system.performance_metrics = {
                "cpu_usage": random.uniform(20, 90),
                "memory_usage": random.uniform(30, 85),
                "response_time": random.uniform(100, 2000),
                "throughput": random.uniform(100, 1000)
            }
            
            # Calcular health score
            system.health_score = self._calculate_health_score(system)
            
            # Actualizar estado
            if system.health_score > 0.8:
                system.state = SystemState.HEALTHY
            elif system.health_score > 0.6:
                system.state = SystemState.WARNING
            elif system.health_score > 0.4:
                system.state = SystemState.CRITICAL
            else:
                system.state = SystemState.FAILED
            
            # Tomar decisión autónoma
            action = await self.decision_engine.make_decision(system)
            
            if action:
                async with self._lock:
                    self.actions[action.id] = action
                    system.last_action = action.id
            
            system.last_updated = time.time()
            
        except Exception as e:
            logger.error(f"Error monitoring system {system.id}: {e}")
    
    async def _execute_action(self, action: AutonomousAction):
        """Ejecutar acción autónoma"""
        try:
            action.status = "executing"
            action.started_at = time.time()
            
            system = self.systems[action.system_id]
            
            # Ejecutar acción basada en el tipo
            if action.action_type == ActionType.HEAL:
                result = await self.decision_engine.self_healing.execute_healing_action(
                    system, action.parameters.get("action"), action.parameters
                )
            elif action.action_type == ActionType.OPTIMIZE:
                result = await self.decision_engine.self_optimizing.execute_optimization(
                    system, action.parameters.get("optimization"), action.parameters
                )
            else:
                result = {"message": "Action executed successfully"}
            
            # Actualizar acción
            action.status = "completed"
            action.completed_at = time.time()
            action.execution_time = action.completed_at - action.started_at
            action.result = result
            
            # Aprender de la experiencia
            await self.decision_engine.self_learning.learn_from_experience(system, result)
            
        except Exception as e:
            logger.error(f"Error executing action {action.id}: {e}")
            action.status = "failed"
            action.completed_at = time.time()
            action.execution_time = action.completed_at - action.started_at
            action.result = {"error": str(e)}
    
    def _calculate_health_score(self, system: AutonomousSystem) -> float:
        """Calcular health score del sistema"""
        metrics = system.performance_metrics
        
        # Normalizar métricas (0-1, donde 1 es mejor)
        cpu_score = max(0, 1 - metrics.get("cpu_usage", 0) / 100)
        memory_score = max(0, 1 - metrics.get("memory_usage", 0) / 100)
        response_score = max(0, 1 - min(1, metrics.get("response_time", 0) / 2000))
        throughput_score = min(1, metrics.get("throughput", 0) / 1000)
        
        # Calcular score promedio
        health_score = (cpu_score + memory_score + response_score + throughput_score) / 4
        
        return round(health_score, 3)
    
    async def create_autonomous_system(self, system_info: Dict[str, Any]) -> str:
        """Crear sistema autónomo"""
        system_id = f"system_{uuid.uuid4().hex[:8]}"
        
        system = AutonomousSystem(
            id=system_id,
            name=system_info["name"],
            description=system_info.get("description", ""),
            system_type=AutonomousSystemType(system_info["system_type"]),
            state=SystemState.HEALTHY,
            decision_engine=DecisionEngine(system_info.get("decision_engine", "rule_based")),
            health_score=1.0,
            performance_metrics={},
            configuration=system_info.get("configuration", {}),
            rules=system_info.get("rules", []),
            learning_data={},
            created_at=time.time(),
            last_updated=time.time(),
            last_action=None,
            metadata=system_info.get("metadata", {})
        )
        
        async with self._lock:
            self.systems[system_id] = system
        
        logger.info(f"Autonomous system created: {system_id} ({system.name})")
        return system_id
    
    async def get_system_status(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado del sistema"""
        if system_id not in self.systems:
            return None
        
        system = self.systems[system_id]
        return {
            "id": system.id,
            "name": system.name,
            "state": system.state.value,
            "health_score": system.health_score,
            "performance_metrics": system.performance_metrics,
            "last_updated": system.last_updated,
            "last_action": system.last_action
        }
    
    async def get_system_actions(self, system_id: str) -> List[Dict[str, Any]]:
        """Obtener acciones del sistema"""
        system_actions = [action for action in self.actions.values() 
                         if action.system_id == system_id]
        
        return [
            {
                "id": action.id,
                "action_type": action.action_type.value,
                "description": action.description,
                "status": action.status,
                "priority": action.priority,
                "created_at": action.created_at,
                "started_at": action.started_at,
                "completed_at": action.completed_at,
                "execution_time": action.execution_time,
                "result": action.result
            }
            for action in system_actions
        ]
    
    async def get_system_events(self, system_id: str) -> List[Dict[str, Any]]:
        """Obtener eventos del sistema"""
        system_events = [event for event in self.events.values() 
                        if event.system_id == system_id]
        
        return [
            {
                "id": event.id,
                "event_type": event.event_type,
                "severity": event.severity,
                "message": event.message,
                "timestamp": event.timestamp,
                "is_resolved": event.is_resolved,
                "resolution_action": event.resolution_action
            }
            for event in system_events
        ]
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "is_running": self.is_running,
            "systems": {
                "total": len(self.systems),
                "by_type": {
                    system_type.value: sum(1 for s in self.systems.values() if s.system_type == system_type)
                    for system_type in AutonomousSystemType
                },
                "by_state": {
                    state.value: sum(1 for s in self.systems.values() if s.state == state)
                    for state in SystemState
                }
            },
            "actions": {
                "total": len(self.actions),
                "by_status": {
                    "pending": sum(1 for a in self.actions.values() if a.status == "pending"),
                    "executing": sum(1 for a in self.actions.values() if a.status == "executing"),
                    "completed": sum(1 for a in self.actions.values() if a.status == "completed"),
                    "failed": sum(1 for a in self.actions.values() if a.status == "failed")
                }
            },
            "events": len(self.events),
            "learning_patterns": len(self.decision_engine.self_learning.learning_patterns)
        }


# Instancia global del motor de sistemas autónomos
autonomous_systems_engine = AutonomousSystemsEngine()


# Router para endpoints del motor de sistemas autónomos
autonomous_systems_router = APIRouter()


@autonomous_systems_router.post("/autonomous-systems")
async def create_autonomous_system_endpoint(system_data: dict):
    """Crear sistema autónomo"""
    try:
        system_id = await autonomous_systems_engine.create_autonomous_system(system_data)
        
        return {
            "message": "Autonomous system created successfully",
            "system_id": system_id,
            "name": system_data["name"],
            "system_type": system_data["system_type"],
            "decision_engine": system_data.get("decision_engine", "rule_based")
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid system type or decision engine: {e}")
    except Exception as e:
        logger.error(f"Error creating autonomous system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create autonomous system: {str(e)}")


@autonomous_systems_router.get("/autonomous-systems")
async def get_autonomous_systems_endpoint():
    """Obtener sistemas autónomos"""
    try:
        systems = autonomous_systems_engine.systems
        return {
            "systems": [
                {
                    "id": system.id,
                    "name": system.name,
                    "description": system.description,
                    "system_type": system.system_type.value,
                    "state": system.state.value,
                    "health_score": system.health_score,
                    "decision_engine": system.decision_engine.value,
                    "created_at": system.created_at,
                    "last_updated": system.last_updated,
                    "last_action": system.last_action
                }
                for system in systems.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting autonomous systems: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get autonomous systems: {str(e)}")


@autonomous_systems_router.get("/autonomous-systems/{system_id}")
async def get_autonomous_system_endpoint(system_id: str):
    """Obtener sistema autónomo específico"""
    try:
        status = await autonomous_systems_engine.get_system_status(system_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Autonomous system not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting autonomous system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get autonomous system: {str(e)}")


@autonomous_systems_router.get("/autonomous-systems/{system_id}/actions")
async def get_system_actions_endpoint(system_id: str):
    """Obtener acciones del sistema"""
    try:
        actions = await autonomous_systems_engine.get_system_actions(system_id)
        return {
            "system_id": system_id,
            "actions": actions,
            "count": len(actions)
        }
    except Exception as e:
        logger.error(f"Error getting system actions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system actions: {str(e)}")


@autonomous_systems_router.get("/autonomous-systems/{system_id}/events")
async def get_system_events_endpoint(system_id: str):
    """Obtener eventos del sistema"""
    try:
        events = await autonomous_systems_engine.get_system_events(system_id)
        return {
            "system_id": system_id,
            "events": events,
            "count": len(events)
        }
    except Exception as e:
        logger.error(f"Error getting system events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system events: {str(e)}")


@autonomous_systems_router.get("/autonomous-systems/stats")
async def get_autonomous_systems_stats_endpoint():
    """Obtener estadísticas del motor de sistemas autónomos"""
    try:
        stats = await autonomous_systems_engine.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting autonomous systems stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get autonomous systems stats: {str(e)}")


# Funciones de utilidad para integración
async def start_autonomous_systems_engine():
    """Iniciar motor de sistemas autónomos"""
    await autonomous_systems_engine.start()


async def stop_autonomous_systems_engine():
    """Detener motor de sistemas autónomos"""
    await autonomous_systems_engine.stop()


async def create_autonomous_system(system_info: Dict[str, Any]) -> str:
    """Crear sistema autónomo"""
    return await autonomous_systems_engine.create_autonomous_system(system_info)


async def get_autonomous_systems_engine_stats() -> Dict[str, Any]:
    """Obtener estadísticas del motor de sistemas autónomos"""
    return await autonomous_systems_engine.get_system_stats()


logger.info("Autonomous systems engine module loaded successfully")

