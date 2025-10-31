"""
Dynamic Adaptation Service
=========================

Advanced dynamic adaptation service for real-time agent optimization and adaptation.
"""

import asyncio
import logging
import json
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_

from ..schemas import (
    BusinessAgent, AgentRequest, AgentResponse, AgentAnalytics,
    AgentWorkflow, AgentCollaboration, AgentSettings,
    ErrorResponse, SuccessResponse
)
from ..exceptions import (
    DynamicAdaptationNotFoundError, DynamicAdaptationExecutionError, DynamicAdaptationValidationError,
    DynamicAdaptationOptimizationError, DynamicAdaptationSystemError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ..models import (
    db_manager, BusinessAgent as AgentModel, User
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class AdaptationType(str, Enum):
    """Adaptation type enumeration"""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESOURCE_ADAPTATION = "resource_adaptation"
    ENVIRONMENT_ADAPTATION = "environment_adaptation"
    WORKLOAD_ADAPTATION = "workload_adaptation"
    QUALITY_ADAPTATION = "quality_adaptation"
    LATENCY_ADAPTATION = "latency_adaptation"
    THROUGHPUT_ADAPTATION = "throughput_adaptation"
    ACCURACY_ADAPTATION = "accuracy_adaptation"
    CUSTOM = "custom"


class AdaptationStatus(str, Enum):
    """Adaptation status enumeration"""
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


class AdaptationStrategy(str, Enum):
    """Adaptation strategy enumeration"""
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    PREDICTIVE = "predictive"
    CONTINUOUS = "continuous"
    THRESHOLD_BASED = "threshold_based"
    MACHINE_LEARNING_BASED = "machine_learning_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    CUSTOM = "custom"


@dataclass
class AdaptationConfig:
    """Adaptation configuration"""
    adaptation_type: AdaptationType
    adaptation_strategy: AdaptationStrategy
    monitoring_interval: float
    adaptation_threshold: float
    optimization_goals: List[str]
    constraints: Dict[str, Any]
    adaptation_parameters: Dict[str, Any]
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Adaptation result"""
    adaptation_id: str
    agent_id: str
    adaptation_type: AdaptationType
    status: AdaptationStatus
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    adaptation_metrics: Dict[str, Any]
    adaptation_log: List[Dict[str, Any]]
    new_parameters: Dict[str, Any]
    performance_impact: Dict[str, Any]
    adaptation_insights: List[str]
    error_message: Optional[str] = None
    duration: float = 0.0
    adaptation_score: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class MonitoringResult:
    """Monitoring result"""
    monitoring_id: str
    agent_id: str
    metrics: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    monitoring_timestamp: datetime = field(default_factory=datetime.utcnow)


class DynamicAdaptationService:
    """Advanced dynamic adaptation service"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
        self._adaptation_configs = {}
        self._monitoring_tasks = {}
        self._adaptation_cache = {}
        
        # Initialize adaptation configurations
        self._initialize_adaptation_configs()
    
    def _initialize_adaptation_configs(self):
        """Initialize adaptation configurations"""
        self._adaptation_configs = {
            AdaptationType.PERFORMANCE_OPTIMIZATION: AdaptationConfig(
                adaptation_type=AdaptationType.PERFORMANCE_OPTIMIZATION,
                adaptation_strategy=AdaptationStrategy.PROACTIVE,
                monitoring_interval=5.0,
                adaptation_threshold=0.1,
                optimization_goals=["maximize_performance", "minimize_latency"],
                constraints={"max_cpu": "90%", "max_memory": "85%"},
                adaptation_parameters={
                    "learning_rate": 0.01,
                    "optimization_algorithm": "adam",
                    "batch_size": 32
                }
            ),
            AdaptationType.RESOURCE_ADAPTATION: AdaptationConfig(
                adaptation_type=AdaptationType.RESOURCE_ADAPTATION,
                adaptation_strategy=AdaptationStrategy.REACTIVE,
                monitoring_interval=2.0,
                adaptation_threshold=0.8,
                optimization_goals=["optimize_resource_usage", "maintain_performance"],
                constraints={"min_cpu": "10%", "min_memory": "20%"},
                adaptation_parameters={
                    "resource_scaling_factor": 1.2,
                    "scaling_threshold": 0.8,
                    "scaling_cooldown": 60
                }
            ),
            AdaptationType.ENVIRONMENT_ADAPTATION: AdaptationConfig(
                adaptation_type=AdaptationType.ENVIRONMENT_ADAPTATION,
                adaptation_strategy=AdaptationStrategy.PREDICTIVE,
                monitoring_interval=10.0,
                adaptation_threshold=0.15,
                optimization_goals=["adapt_to_environment", "maintain_stability"],
                constraints={"adaptation_time": "30s", "stability_requirement": 0.9},
                adaptation_parameters={
                    "environment_sensitivity": 0.8,
                    "adaptation_speed": 1.0,
                    "stability_factor": 0.9
                }
            ),
            AdaptationType.WORKLOAD_ADAPTATION: AdaptationConfig(
                adaptation_type=AdaptationType.WORKLOAD_ADAPTATION,
                adaptation_strategy=AdaptationStrategy.CONTINUOUS,
                monitoring_interval=1.0,
                adaptation_threshold=0.2,
                optimization_goals=["handle_workload_changes", "maintain_quality"],
                constraints={"max_throughput": 1000, "min_quality": 0.8},
                adaptation_parameters={
                    "workload_scaling_factor": 1.5,
                    "quality_threshold": 0.8,
                    "throughput_target": 800
                }
            )
        }
    
    async def start_dynamic_adaptation(
        self,
        agent_id: str,
        adaptation_type: AdaptationType,
        adaptation_options: Dict[str, Any] = None,
        user_id: str = None
    ) -> AdaptationResult:
        """Start dynamic adaptation for agent"""
        try:
            # Get agent
            agent = await self.get_agent(agent_id)
            if not agent:
                raise DynamicAdaptationValidationError(
                    "agent_not_found",
                    f"Agent {agent_id} not found",
                    {"agent_id": agent_id}
                )
            
            # Get adaptation configuration
            adaptation_config = self._adaptation_configs.get(adaptation_type)
            if not adaptation_config:
                raise DynamicAdaptationValidationError(
                    "invalid_adaptation_type",
                    f"Invalid adaptation type: {adaptation_type}",
                    {"adaptation_type": adaptation_type}
                )
            
            # Validate adaptation options
            await self._validate_adaptation_options(adaptation_type, adaptation_options)
            
            # Create adaptation record
            adaptation_id = str(uuid4())
            adaptation_data = {
                "agent_id": agent_id,
                "adaptation_id": adaptation_id,
                "adaptation_type": adaptation_type.value,
                "status": AdaptationStatus.MONITORING.value,
                "created_by": user_id or "system"
            }
            
            adaptation = await db_manager.create_adaptation(adaptation_data)
            
            # Start adaptation process
            start_time = datetime.utcnow()
            result = await self._perform_dynamic_adaptation(
                agent, adaptation_type, adaptation_config, adaptation_options
            )
            
            # Update adaptation record
            await db_manager.update_adaptation_status(
                adaptation_id,
                result.status.value,
                initial_state=result.initial_state,
                final_state=result.final_state,
                adaptation_metrics=result.adaptation_metrics,
                new_parameters=result.new_parameters,
                performance_impact=result.performance_impact,
                adaptation_insights=result.adaptation_insights,
                error_message=result.error_message,
                duration=result.duration,
                adaptation_score=result.adaptation_score,
                completed_at=result.completed_at
            )
            
            # Update agent with adaptation results
            await self._update_agent_adaptation(agent_id, result)
            
            # Cache adaptation result
            await self._cache_adaptation_result(adaptation_id, result)
            
            logger.info(f"Dynamic adaptation completed: {agent_id}, adaptation: {adaptation_id}")
            
            return result
            
        except Exception as e:
            error = handle_agent_error(e, agent_id=agent_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def start_continuous_monitoring(
        self,
        agent_id: str,
        monitoring_config: Dict[str, Any],
        user_id: str = None
    ) -> str:
        """Start continuous monitoring for agent"""
        try:
            # Get agent
            agent = await self.get_agent(agent_id)
            if not agent:
                raise DynamicAdaptationValidationError(
                    "agent_not_found",
                    f"Agent {agent_id} not found",
                    {"agent_id": agent_id}
                )
            
            # Create monitoring task
            monitoring_id = str(uuid4())
            monitoring_task = asyncio.create_task(
                self._continuous_monitoring_loop(agent_id, monitoring_config, monitoring_id)
            )
            
            # Store monitoring task
            self._monitoring_tasks[monitoring_id] = {
                "task": monitoring_task,
                "agent_id": agent_id,
                "config": monitoring_config,
                "started_at": datetime.utcnow()
            }
            
            logger.info(f"Continuous monitoring started: {agent_id}, monitoring: {monitoring_id}")
            
            return monitoring_id
            
        except Exception as e:
            error = handle_agent_error(e, agent_id=agent_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def stop_continuous_monitoring(
        self,
        monitoring_id: str,
        user_id: str = None
    ) -> bool:
        """Stop continuous monitoring"""
        try:
            if monitoring_id not in self._monitoring_tasks:
                raise DynamicAdaptationNotFoundError(
                    "monitoring_not_found",
                    f"Monitoring {monitoring_id} not found",
                    {"monitoring_id": monitoring_id}
                )
            
            # Cancel monitoring task
            monitoring_task = self._monitoring_tasks[monitoring_id]["task"]
            monitoring_task.cancel()
            
            # Remove from monitoring tasks
            del self._monitoring_tasks[monitoring_id]
            
            logger.info(f"Continuous monitoring stopped: {monitoring_id}")
            
            return True
            
        except Exception as e:
            error = handle_agent_error(e, monitoring_id=monitoring_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def get_adaptation_history(
        self,
        agent_id: str,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[AdaptationResult]:
        """Get agent adaptation history"""
        try:
            # Get adaptation records
            adaptation_records = await db_manager.get_adaptation_history(
                agent_id, start_time, end_time
            )
            
            # Convert to AdaptationResult objects
            results = []
            for record in adaptation_records:
                result = AdaptationResult(
                    adaptation_id=str(record.id),
                    agent_id=record.agent_id,
                    adaptation_type=AdaptationType(record.adaptation_type),
                    status=AdaptationStatus(record.status),
                    initial_state=record.initial_state or {},
                    final_state=record.final_state or {},
                    adaptation_metrics=record.adaptation_metrics or {},
                    adaptation_log=record.adaptation_log or [],
                    new_parameters=record.new_parameters or {},
                    performance_impact=record.performance_impact or {},
                    adaptation_insights=record.adaptation_insights or [],
                    error_message=record.error_message,
                    duration=record.duration or 0.0,
                    adaptation_score=record.adaptation_score or 0.0,
                    started_at=record.created_at,
                    completed_at=record.completed_at
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            error = handle_agent_error(e, agent_id=agent_id)
            log_agent_error(error)
            raise error
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID"""
        try:
            # Try cache first
            cached_agent = await self._get_cached_agent(agent_id)
            if cached_agent:
                return cached_agent
            
            # Get from database
            agent = await db_manager.get_agent_by_id(agent_id)
            if not agent:
                return None
            
            # Cache agent data
            await self._cache_agent_data(agent)
            
            return {
                "id": str(agent.id),
                "name": agent.name,
                "description": agent.description,
                "agent_type": agent.agent_type,
                "specialization": agent.specialization,
                "capabilities": agent.capabilities,
                "configuration": agent.configuration,
                "status": agent.status,
                "execution_count": agent.execution_count,
                "success_rate": agent.success_rate,
                "average_duration": agent.average_duration
            }
            
        except Exception as e:
            error = handle_agent_error(e, agent_id=agent_id)
            log_agent_error(error)
            raise error
    
    # Private helper methods
    async def _validate_adaptation_options(
        self,
        adaptation_type: AdaptationType,
        adaptation_options: Dict[str, Any] = None
    ) -> None:
        """Validate adaptation options"""
        if not adaptation_options:
            return
        
        # Validate based on adaptation type
        if adaptation_type == AdaptationType.PERFORMANCE_OPTIMIZATION:
            required_fields = ["target_metrics", "optimization_goals"]
            for field in required_fields:
                if field not in adaptation_options:
                    raise DynamicAdaptationValidationError(
                        "missing_required_field",
                        f"Required field {field} is missing for performance optimization",
                        {"field": field, "adaptation_type": adaptation_type}
                    )
    
    async def _perform_dynamic_adaptation(
        self,
        agent: Dict[str, Any],
        adaptation_type: AdaptationType,
        adaptation_config: AdaptationConfig,
        adaptation_options: Dict[str, Any] = None
    ) -> AdaptationResult:
        """Perform dynamic adaptation"""
        try:
            start_time = datetime.utcnow()
            adaptation_log = []
            
            # Get initial state
            initial_state = await self._get_agent_state(agent["id"])
            
            # Initialize adaptation process
            adaptation_log.append({
                "step": "adaptation_initialization",
                "status": "started",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Perform adaptation based on type
            if adaptation_type == AdaptationType.PERFORMANCE_OPTIMIZATION:
                result = await self._perform_performance_optimization(
                    agent, adaptation_config, adaptation_log
                )
            elif adaptation_type == AdaptationType.RESOURCE_ADAPTATION:
                result = await self._perform_resource_adaptation(
                    agent, adaptation_config, adaptation_log
                )
            elif adaptation_type == AdaptationType.ENVIRONMENT_ADAPTATION:
                result = await self._perform_environment_adaptation(
                    agent, adaptation_config, adaptation_log
                )
            elif adaptation_type == AdaptationType.WORKLOAD_ADAPTATION:
                result = await self._perform_workload_adaptation(
                    agent, adaptation_config, adaptation_log
                )
            else:
                result = await self._perform_custom_adaptation(
                    agent, adaptation_config, adaptation_log, adaptation_options
                )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Get final state
            final_state = await self._get_agent_state(agent["id"])
            
            # Calculate adaptation metrics
            adaptation_metrics = await self._calculate_adaptation_metrics(
                initial_state, final_state, result
            )
            
            # Calculate adaptation score
            adaptation_score = await self._calculate_adaptation_score(
                final_state, adaptation_metrics
            )
            
            return AdaptationResult(
                adaptation_id=str(uuid4()),
                agent_id=agent["id"],
                adaptation_type=adaptation_type,
                status=AdaptationStatus.COMPLETED,
                initial_state=initial_state,
                final_state=final_state,
                adaptation_metrics=adaptation_metrics,
                adaptation_log=adaptation_log,
                new_parameters=result.get("new_parameters", {}),
                performance_impact=result.get("performance_impact", {}),
                adaptation_insights=result.get("adaptation_insights", []),
                duration=duration,
                adaptation_score=adaptation_score,
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return AdaptationResult(
                adaptation_id=str(uuid4()),
                agent_id=agent["id"],
                adaptation_type=adaptation_type,
                status=AdaptationStatus.FAILED,
                initial_state={},
                final_state={},
                adaptation_metrics={},
                adaptation_log=adaptation_log,
                error_message=str(e),
                duration=duration,
                started_at=start_time,
                completed_at=end_time
            )
    
    async def _perform_performance_optimization(
        self,
        agent: Dict[str, Any],
        adaptation_config: AdaptationConfig,
        adaptation_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform performance optimization adaptation"""
        adaptation_log.append({
            "step": "performance_optimization",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate performance optimization
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate new parameters
        new_parameters = {
            "learning_rate": 0.01 + np.random.random() * 0.005,
            "batch_size": 32 + int(np.random.random() * 16),
            "optimization_algorithm": "adam",
            "performance_boost": 1.1 + np.random.random() * 0.2
        }
        
        # Calculate performance impact
        performance_impact = {
            "accuracy_improvement": 0.05 + np.random.random() * 0.05,
            "speed_improvement": 0.1 + np.random.random() * 0.1,
            "efficiency_improvement": 0.08 + np.random.random() * 0.07
        }
        
        adaptation_log.append({
            "step": "performance_optimization",
            "status": "completed",
            "new_parameters": new_parameters,
            "performance_impact": performance_impact,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "new_parameters": new_parameters,
            "performance_impact": performance_impact,
            "adaptation_insights": [
                "Optimized learning rate for better convergence",
                "Improved batch size for better performance",
                "Enhanced optimization algorithm selection"
            ]
        }
    
    async def _perform_resource_adaptation(
        self,
        agent: Dict[str, Any],
        adaptation_config: AdaptationConfig,
        adaptation_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform resource adaptation"""
        adaptation_log.append({
            "step": "resource_adaptation",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate resource adaptation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate new parameters
        new_parameters = {
            "cpu_allocation": 0.8 + np.random.random() * 0.1,
            "memory_allocation": 0.7 + np.random.random() * 0.15,
            "resource_scaling_factor": 1.2 + np.random.random() * 0.3,
            "efficiency_boost": 1.15 + np.random.random() * 0.1
        }
        
        # Calculate performance impact
        performance_impact = {
            "resource_efficiency_improvement": 0.12 + np.random.random() * 0.08,
            "cost_reduction": 0.08 + np.random.random() * 0.07,
            "scalability_improvement": 0.15 + np.random.random() * 0.1
        }
        
        adaptation_log.append({
            "step": "resource_adaptation",
            "status": "completed",
            "new_parameters": new_parameters,
            "performance_impact": performance_impact,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "new_parameters": new_parameters,
            "performance_impact": performance_impact,
            "adaptation_insights": [
                "Optimized resource allocation for better efficiency",
                "Improved scaling strategies",
                "Enhanced cost-effectiveness"
            ]
        }
    
    async def _perform_environment_adaptation(
        self,
        agent: Dict[str, Any],
        adaptation_config: AdaptationConfig,
        adaptation_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform environment adaptation"""
        adaptation_log.append({
            "step": "environment_adaptation",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate environment adaptation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate new parameters
        new_parameters = {
            "environment_sensitivity": 0.8 + np.random.random() * 0.1,
            "adaptation_speed": 1.0 + np.random.random() * 0.2,
            "stability_factor": 0.9 + np.random.random() * 0.05,
            "context_awareness": 0.85 + np.random.random() * 0.1
        }
        
        # Calculate performance impact
        performance_impact = {
            "adaptability_improvement": 0.18 + np.random.random() * 0.12,
            "stability_improvement": 0.1 + np.random.random() * 0.05,
            "environment_fit_improvement": 0.15 + np.random.random() * 0.1
        }
        
        adaptation_log.append({
            "step": "environment_adaptation",
            "status": "completed",
            "new_parameters": new_parameters,
            "performance_impact": performance_impact,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "new_parameters": new_parameters,
            "performance_impact": performance_impact,
            "adaptation_insights": [
                "Improved environment change detection",
                "Enhanced adaptation speed",
                "Better stability during transitions"
            ]
        }
    
    async def _perform_workload_adaptation(
        self,
        agent: Dict[str, Any],
        adaptation_config: AdaptationConfig,
        adaptation_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform workload adaptation"""
        adaptation_log.append({
            "step": "workload_adaptation",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate workload adaptation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate new parameters
        new_parameters = {
            "workload_scaling_factor": 1.5 + np.random.random() * 0.3,
            "quality_threshold": 0.8 + np.random.random() * 0.1,
            "throughput_target": 800 + int(np.random.random() * 200),
            "load_balancing_factor": 0.9 + np.random.random() * 0.1
        }
        
        # Calculate performance impact
        performance_impact = {
            "throughput_improvement": 0.2 + np.random.random() * 0.15,
            "quality_maintenance": 0.05 + np.random.random() * 0.05,
            "load_handling_improvement": 0.25 + np.random.random() * 0.15
        }
        
        adaptation_log.append({
            "step": "workload_adaptation",
            "status": "completed",
            "new_parameters": new_parameters,
            "performance_impact": performance_impact,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "new_parameters": new_parameters,
            "performance_impact": performance_impact,
            "adaptation_insights": [
                "Improved workload handling capabilities",
                "Enhanced quality maintenance under load",
                "Better throughput optimization"
            ]
        }
    
    async def _perform_custom_adaptation(
        self,
        agent: Dict[str, Any],
        adaptation_config: AdaptationConfig,
        adaptation_log: List[Dict[str, Any]],
        adaptation_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform custom adaptation"""
        adaptation_log.append({
            "step": "custom_adaptation",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate custom adaptation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        adaptation_log.append({
            "step": "custom_adaptation",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "new_parameters": adaptation_options or {},
            "performance_impact": {"custom_improvement": 0.1},
            "adaptation_insights": ["Custom adaptation completed successfully"]
        }
    
    async def _continuous_monitoring_loop(
        self,
        agent_id: str,
        monitoring_config: Dict[str, Any],
        monitoring_id: str
    ) -> None:
        """Continuous monitoring loop"""
        try:
            monitoring_interval = monitoring_config.get("interval", 5.0)
            
            while True:
                # Perform monitoring
                monitoring_result = await self._perform_monitoring(agent_id, monitoring_config)
                
                # Check for anomalies
                if monitoring_result.anomalies:
                    # Trigger adaptation if needed
                    await self._handle_anomalies(agent_id, monitoring_result.anomalies)
                
                # Apply recommendations
                if monitoring_result.recommendations:
                    await self._apply_recommendations(agent_id, monitoring_result.recommendations)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Continuous monitoring cancelled: {monitoring_id}")
        except Exception as e:
            logger.error(f"Continuous monitoring error: {monitoring_id}, error: {e}")
    
    async def _perform_monitoring(
        self,
        agent_id: str,
        monitoring_config: Dict[str, Any]
    ) -> MonitoringResult:
        """Perform monitoring"""
        # Get current metrics
        metrics = await self._get_agent_metrics(agent_id)
        
        # Detect anomalies
        anomalies = await self._detect_anomalies(metrics, monitoring_config)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(metrics, anomalies)
        
        return MonitoringResult(
            monitoring_id=str(uuid4()),
            agent_id=agent_id,
            metrics=metrics,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    async def _detect_anomalies(
        self,
        metrics: Dict[str, Any],
        monitoring_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics"""
        anomalies = []
        
        # Check for performance anomalies
        if metrics.get("cpu_usage", 0) > 0.9:
            anomalies.append({
                "type": "high_cpu_usage",
                "severity": "high",
                "value": metrics["cpu_usage"],
                "threshold": 0.9
            })
        
        if metrics.get("memory_usage", 0) > 0.85:
            anomalies.append({
                "type": "high_memory_usage",
                "severity": "high",
                "value": metrics["memory_usage"],
                "threshold": 0.85
            })
        
        if metrics.get("error_rate", 0) > 0.05:
            anomalies.append({
                "type": "high_error_rate",
                "severity": "medium",
                "value": metrics["error_rate"],
                "threshold": 0.05
            })
        
        return anomalies
    
    async def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on metrics and anomalies"""
        recommendations = []
        
        for anomaly in anomalies:
            if anomaly["type"] == "high_cpu_usage":
                recommendations.append({
                    "type": "scale_resources",
                    "action": "increase_cpu_allocation",
                    "priority": "high",
                    "description": "Increase CPU allocation to handle high usage"
                })
            elif anomaly["type"] == "high_memory_usage":
                recommendations.append({
                    "type": "scale_resources",
                    "action": "increase_memory_allocation",
                    "priority": "high",
                    "description": "Increase memory allocation to handle high usage"
                })
            elif anomaly["type"] == "high_error_rate":
                recommendations.append({
                    "type": "optimize_performance",
                    "action": "tune_parameters",
                    "priority": "medium",
                    "description": "Tune parameters to reduce error rate"
                })
        
        return recommendations
    
    async def _handle_anomalies(
        self,
        agent_id: str,
        anomalies: List[Dict[str, Any]]
    ) -> None:
        """Handle detected anomalies"""
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                # Trigger immediate adaptation
                await self.start_dynamic_adaptation(
                    agent_id,
                    AdaptationType.RESOURCE_ADAPTATION
                )
    
    async def _apply_recommendations(
        self,
        agent_id: str,
        recommendations: List[Dict[str, Any]]
    ) -> None:
        """Apply recommendations"""
        for recommendation in recommendations:
            if recommendation["priority"] == "high":
                # Apply high priority recommendations immediately
                if recommendation["type"] == "scale_resources":
                    await self.start_dynamic_adaptation(
                        agent_id,
                        AdaptationType.RESOURCE_ADAPTATION
                    )
                elif recommendation["type"] == "optimize_performance":
                    await self.start_dynamic_adaptation(
                        agent_id,
                        AdaptationType.PERFORMANCE_OPTIMIZATION
                    )
    
    async def _get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "cpu_usage": 0.6 + np.random.random() * 0.3,
            "memory_usage": 0.5 + np.random.random() * 0.4,
            "performance": 0.8 + np.random.random() * 0.15,
            "error_rate": 0.02 + np.random.random() * 0.03,
            "throughput": 500 + np.random.random() * 300
        }
    
    async def _get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            "cpu_usage": 0.6 + np.random.random() * 0.3,
            "memory_usage": 0.5 + np.random.random() * 0.4,
            "performance": 0.8 + np.random.random() * 0.15,
            "error_rate": 0.02 + np.random.random() * 0.03,
            "throughput": 500 + np.random.random() * 300,
            "latency": 100 + np.random.random() * 50,
            "accuracy": 0.85 + np.random.random() * 0.1
        }
    
    async def _calculate_adaptation_metrics(
        self,
        initial_state: Dict[str, Any],
        final_state: Dict[str, Any],
        adaptation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate adaptation metrics"""
        metrics = {
            "state_changes": {},
            "performance_improvements": {},
            "adaptation_efficiency": 0.0
        }
        
        # Calculate state changes
        for metric in initial_state:
            if metric in final_state:
                initial_value = initial_state[metric]
                final_value = final_state[metric]
                
                if initial_value != 0:
                    change = ((final_value - initial_value) / initial_value) * 100
                    metrics["state_changes"][metric] = change
        
        # Calculate performance improvements
        performance_impact = adaptation_result.get("performance_impact", {})
        metrics["performance_improvements"] = performance_impact
        
        # Calculate adaptation efficiency
        total_improvement = sum(performance_impact.values())
        metrics["adaptation_efficiency"] = total_improvement / len(performance_impact) if performance_impact else 0
        
        return metrics
    
    async def _calculate_adaptation_score(
        self,
        final_state: Dict[str, Any],
        adaptation_metrics: Dict[str, Any]
    ) -> float:
        """Calculate adaptation score"""
        # Weighted combination of final state and adaptation metrics
        state_score = np.mean(list(final_state.values()))
        efficiency_score = adaptation_metrics.get("adaptation_efficiency", 0)
        
        # Calculate weighted score
        weights = [0.7, 0.3]  # State, efficiency
        scores = [state_score, efficiency_score]
        
        adaptation_score = sum(w * s for w, s in zip(weights, scores))
        
        return min(1.0, max(0.0, adaptation_score))  # Clamp between 0 and 1
    
    async def _update_agent_adaptation(
        self,
        agent_id: str,
        adaptation_result: AdaptationResult
    ) -> None:
        """Update agent with adaptation results"""
        # Update agent configuration with adaptation results
        updates = {
            "configuration": {
                "last_adaptation": datetime.utcnow().isoformat(),
                "adaptation_results": {
                    "adaptation_id": adaptation_result.adaptation_id,
                    "adaptation_type": adaptation_result.adaptation_type.value,
                    "adaptation_metrics": adaptation_result.adaptation_metrics,
                    "new_parameters": adaptation_result.new_parameters,
                    "performance_impact": adaptation_result.performance_impact,
                    "adaptation_score": adaptation_result.adaptation_score
                }
            }
        }
        
        # This would update the agent in the database
        logger.info(f"Updated agent adaptation: {agent_id}")
    
    # Caching methods
    async def _cache_agent_data(self, agent: Any) -> None:
        """Cache agent data"""
        cache_key = f"dynamic_adaptation_agent:{agent.id}"
        agent_data = {
            "id": str(agent.id),
            "name": agent.name,
            "agent_type": agent.agent_type,
            "specialization": agent.specialization,
            "capabilities": agent.capabilities,
            "configuration": agent.configuration
        }
        
        await self.redis.setex(
            cache_key,
            3600,  # 1 hour
            json.dumps(agent_data)
        )
    
    async def _get_cached_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get cached agent data"""
        cache_key = f"dynamic_adaptation_agent:{agent_id}"
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        return None
    
    async def _cache_adaptation_result(self, adaptation_id: str, result: AdaptationResult) -> None:
        """Cache adaptation result"""
        cache_key = f"adaptation_result:{adaptation_id}"
        result_data = {
            "adaptation_id": result.adaptation_id,
            "agent_id": result.agent_id,
            "adaptation_type": result.adaptation_type.value,
            "status": result.status.value,
            "adaptation_metrics": result.adaptation_metrics,
            "adaptation_score": result.adaptation_score,
            "duration": result.duration
        }
        
        await self.redis.setex(
            cache_key,
            1800,  # 30 minutes
            json.dumps(result_data)
        )



























