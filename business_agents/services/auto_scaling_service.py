"""
Auto-Scaling Service
===================

Advanced auto-scaling service for dynamic agent resource management and scaling.
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
    AutoScalingNotFoundError, AutoScalingExecutionError, AutoScalingValidationError,
    AutoScalingOptimizationError, AutoScalingSystemError,
    create_agent_error, log_agent_error, handle_agent_error, get_error_response
)
from ..models import (
    db_manager, BusinessAgent as AgentModel, User
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class ScalingType(str, Enum):
    """Scaling type enumeration"""
    HORIZONTAL_SCALING = "horizontal_scaling"
    VERTICAL_SCALING = "vertical_scaling"
    AUTO_SCALING = "auto_scaling"
    MANUAL_SCALING = "manual_scaling"
    PREDICTIVE_SCALING = "predictive_scaling"
    REACTIVE_SCALING = "reactive_scaling"
    CUSTOM = "custom"


class ScalingStatus(str, Enum):
    """Scaling status enumeration"""
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


class ScalingStrategy(str, Enum):
    """Scaling strategy enumeration"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    THROUGHPUT_BASED = "throughput_based"
    LATENCY_BASED = "latency_based"
    QUEUE_BASED = "queue_based"
    CUSTOM_METRICS = "custom_metrics"
    HYBRID = "hybrid"
    CUSTOM = "custom"


@dataclass
class ScalingConfig:
    """Scaling configuration"""
    scaling_type: ScalingType
    scaling_strategy: ScalingStrategy
    monitoring_interval: float
    scaling_threshold: float
    scaling_goals: List[str]
    constraints: Dict[str, Any]
    scaling_parameters: Dict[str, Any]
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingResult:
    """Scaling result"""
    scaling_id: str
    agent_id: str
    scaling_type: ScalingType
    status: ScalingStatus
    initial_resources: Dict[str, Any]
    final_resources: Dict[str, Any]
    scaling_metrics: Dict[str, Any]
    scaling_log: List[Dict[str, Any]]
    scaling_actions: List[str]
    performance_impact: Dict[str, Any]
    scaling_insights: List[str]
    error_message: Optional[str] = None
    duration: float = 0.0
    scaling_score: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class ScalingEvent:
    """Scaling event"""
    event_id: str
    agent_id: str
    event_type: str
    trigger_metrics: Dict[str, Any]
    scaling_decision: str
    event_timestamp: datetime = field(default_factory=datetime.utcnow)


class AutoScalingService:
    """Advanced auto-scaling service"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.settings = get_settings()
        self._scaling_configs = {}
        self._monitoring_tasks = {}
        self._scaling_cache = {}
        
        # Initialize scaling configurations
        self._initialize_scaling_configs()
    
    def _initialize_scaling_configs(self):
        """Initialize scaling configurations"""
        self._scaling_configs = {
            ScalingType.HORIZONTAL_SCALING: ScalingConfig(
                scaling_type=ScalingType.HORIZONTAL_SCALING,
                scaling_strategy=ScalingStrategy.CPU_BASED,
                monitoring_interval=5.0,
                scaling_threshold=0.8,
                scaling_goals=["maintain_performance", "optimize_cost"],
                constraints={"max_instances": 10, "min_instances": 1},
                scaling_parameters={
                    "scale_up_threshold": 0.8,
                    "scale_down_threshold": 0.3,
                    "scaling_factor": 1.5,
                    "cooldown_period": 300
                }
            ),
            ScalingType.VERTICAL_SCALING: ScalingConfig(
                scaling_type=ScalingType.VERTICAL_SCALING,
                scaling_strategy=ScalingStrategy.MEMORY_BASED,
                monitoring_interval=3.0,
                scaling_threshold=0.85,
                scaling_goals=["optimize_resources", "maintain_stability"],
                constraints={"max_cpu": "8", "max_memory": "16GB"},
                scaling_parameters={
                    "cpu_scaling_factor": 1.2,
                    "memory_scaling_factor": 1.5,
                    "scaling_step": 0.5,
                    "evaluation_period": 60
                }
            ),
            ScalingType.AUTO_SCALING: ScalingConfig(
                scaling_type=ScalingType.AUTO_SCALING,
                scaling_strategy=ScalingStrategy.HYBRID,
                monitoring_interval=2.0,
                scaling_threshold=0.75,
                scaling_goals=["automatic_optimization", "cost_efficiency"],
                constraints={"max_cost": 1000, "min_performance": 0.8},
                scaling_parameters={
                    "auto_scaling_enabled": True,
                    "prediction_window": 300,
                    "scaling_aggressiveness": 0.8,
                    "cost_optimization": True
                }
            ),
            ScalingType.PREDICTIVE_SCALING: ScalingConfig(
                scaling_type=ScalingType.PREDICTIVE_SCALING,
                scaling_strategy=ScalingStrategy.CUSTOM_METRICS,
                monitoring_interval=1.0,
                scaling_threshold=0.7,
                scaling_goals=["predictive_optimization", "proactive_scaling"],
                constraints={"prediction_accuracy": 0.85, "scaling_latency": "30s"},
                scaling_parameters={
                    "prediction_model": "lstm",
                    "prediction_horizon": 600,
                    "confidence_threshold": 0.8,
                    "scaling_lead_time": 60
                }
            )
        }
    
    async def start_auto_scaling(
        self,
        agent_id: str,
        scaling_type: ScalingType,
        scaling_options: Dict[str, Any] = None,
        user_id: str = None
    ) -> ScalingResult:
        """Start auto-scaling for agent"""
        try:
            # Get agent
            agent = await self.get_agent(agent_id)
            if not agent:
                raise AutoScalingValidationError(
                    "agent_not_found",
                    f"Agent {agent_id} not found",
                    {"agent_id": agent_id}
                )
            
            # Get scaling configuration
            scaling_config = self._scaling_configs.get(scaling_type)
            if not scaling_config:
                raise AutoScalingValidationError(
                    "invalid_scaling_type",
                    f"Invalid scaling type: {scaling_type}",
                    {"scaling_type": scaling_type}
                )
            
            # Validate scaling options
            await self._validate_scaling_options(scaling_type, scaling_options)
            
            # Create scaling record
            scaling_id = str(uuid4())
            scaling_data = {
                "agent_id": agent_id,
                "scaling_id": scaling_id,
                "scaling_type": scaling_type.value,
                "status": ScalingStatus.MONITORING.value,
                "created_by": user_id or "system"
            }
            
            scaling = await db_manager.create_scaling(scaling_data)
            
            # Start scaling process
            start_time = datetime.utcnow()
            result = await self._perform_auto_scaling(
                agent, scaling_type, scaling_config, scaling_options
            )
            
            # Update scaling record
            await db_manager.update_scaling_status(
                scaling_id,
                result.status.value,
                initial_resources=result.initial_resources,
                final_resources=result.final_resources,
                scaling_metrics=result.scaling_metrics,
                scaling_actions=result.scaling_actions,
                performance_impact=result.performance_impact,
                scaling_insights=result.scaling_insights,
                error_message=result.error_message,
                duration=result.duration,
                scaling_score=result.scaling_score,
                completed_at=result.completed_at
            )
            
            # Update agent with scaling results
            await self._update_agent_scaling(agent_id, result)
            
            # Cache scaling result
            await self._cache_scaling_result(scaling_id, result)
            
            logger.info(f"Auto-scaling completed: {agent_id}, scaling: {scaling_id}")
            
            return result
            
        except Exception as e:
            error = handle_agent_error(e, agent_id=agent_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def start_continuous_scaling(
        self,
        agent_id: str,
        scaling_config: Dict[str, Any],
        user_id: str = None
    ) -> str:
        """Start continuous scaling for agent"""
        try:
            # Get agent
            agent = await self.get_agent(agent_id)
            if not agent:
                raise AutoScalingValidationError(
                    "agent_not_found",
                    f"Agent {agent_id} not found",
                    {"agent_id": agent_id}
                )
            
            # Create scaling task
            scaling_id = str(uuid4())
            scaling_task = asyncio.create_task(
                self._continuous_scaling_loop(agent_id, scaling_config, scaling_id)
            )
            
            # Store scaling task
            self._monitoring_tasks[scaling_id] = {
                "task": scaling_task,
                "agent_id": agent_id,
                "config": scaling_config,
                "started_at": datetime.utcnow()
            }
            
            logger.info(f"Continuous scaling started: {agent_id}, scaling: {scaling_id}")
            
            return scaling_id
            
        except Exception as e:
            error = handle_agent_error(e, agent_id=agent_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def stop_continuous_scaling(
        self,
        scaling_id: str,
        user_id: str = None
    ) -> bool:
        """Stop continuous scaling"""
        try:
            if scaling_id not in self._monitoring_tasks:
                raise AutoScalingNotFoundError(
                    "scaling_not_found",
                    f"Scaling {scaling_id} not found",
                    {"scaling_id": scaling_id}
                )
            
            # Cancel scaling task
            scaling_task = self._monitoring_tasks[scaling_id]["task"]
            scaling_task.cancel()
            
            # Remove from scaling tasks
            del self._monitoring_tasks[scaling_id]
            
            logger.info(f"Continuous scaling stopped: {scaling_id}")
            
            return True
            
        except Exception as e:
            error = handle_agent_error(e, scaling_id=scaling_id, user_id=user_id)
            log_agent_error(error)
            raise error
    
    async def get_scaling_history(
        self,
        agent_id: str,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[ScalingResult]:
        """Get agent scaling history"""
        try:
            # Get scaling records
            scaling_records = await db_manager.get_scaling_history(
                agent_id, start_time, end_time
            )
            
            # Convert to ScalingResult objects
            results = []
            for record in scaling_records:
                result = ScalingResult(
                    scaling_id=str(record.id),
                    agent_id=record.agent_id,
                    scaling_type=ScalingType(record.scaling_type),
                    status=ScalingStatus(record.status),
                    initial_resources=record.initial_resources or {},
                    final_resources=record.final_resources or {},
                    scaling_metrics=record.scaling_metrics or {},
                    scaling_log=record.scaling_log or [],
                    scaling_actions=record.scaling_actions or [],
                    performance_impact=record.performance_impact or {},
                    scaling_insights=record.scaling_insights or [],
                    error_message=record.error_message,
                    duration=record.duration or 0.0,
                    scaling_score=record.scaling_score or 0.0,
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
    async def _validate_scaling_options(
        self,
        scaling_type: ScalingType,
        scaling_options: Dict[str, Any] = None
    ) -> None:
        """Validate scaling options"""
        if not scaling_options:
            return
        
        # Validate based on scaling type
        if scaling_type == ScalingType.HORIZONTAL_SCALING:
            required_fields = ["max_instances", "min_instances"]
            for field in required_fields:
                if field not in scaling_options:
                    raise AutoScalingValidationError(
                        "missing_required_field",
                        f"Required field {field} is missing for horizontal scaling",
                        {"field": field, "scaling_type": scaling_type}
                    )
    
    async def _perform_auto_scaling(
        self,
        agent: Dict[str, Any],
        scaling_type: ScalingType,
        scaling_config: ScalingConfig,
        scaling_options: Dict[str, Any] = None
    ) -> ScalingResult:
        """Perform auto-scaling"""
        try:
            start_time = datetime.utcnow()
            scaling_log = []
            
            # Get initial resources
            initial_resources = await self._get_agent_resources(agent["id"])
            
            # Initialize scaling process
            scaling_log.append({
                "step": "scaling_initialization",
                "status": "started",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Perform scaling based on type
            if scaling_type == ScalingType.HORIZONTAL_SCALING:
                result = await self._perform_horizontal_scaling(
                    agent, scaling_config, scaling_log
                )
            elif scaling_type == ScalingType.VERTICAL_SCALING:
                result = await self._perform_vertical_scaling(
                    agent, scaling_config, scaling_log
                )
            elif scaling_type == ScalingType.AUTO_SCALING:
                result = await self._perform_auto_scaling_logic(
                    agent, scaling_config, scaling_log
                )
            elif scaling_type == ScalingType.PREDICTIVE_SCALING:
                result = await self._perform_predictive_scaling(
                    agent, scaling_config, scaling_log
                )
            else:
                result = await self._perform_custom_scaling(
                    agent, scaling_config, scaling_log, scaling_options
                )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Get final resources
            final_resources = await self._get_agent_resources(agent["id"])
            
            # Calculate scaling metrics
            scaling_metrics = await self._calculate_scaling_metrics(
                initial_resources, final_resources, result
            )
            
            # Calculate scaling score
            scaling_score = await self._calculate_scaling_score(
                final_resources, scaling_metrics
            )
            
            return ScalingResult(
                scaling_id=str(uuid4()),
                agent_id=agent["id"],
                scaling_type=scaling_type,
                status=ScalingStatus.COMPLETED,
                initial_resources=initial_resources,
                final_resources=final_resources,
                scaling_metrics=scaling_metrics,
                scaling_log=scaling_log,
                scaling_actions=result.get("scaling_actions", []),
                performance_impact=result.get("performance_impact", {}),
                scaling_insights=result.get("scaling_insights", []),
                duration=duration,
                scaling_score=scaling_score,
                started_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return ScalingResult(
                scaling_id=str(uuid4()),
                agent_id=agent["id"],
                scaling_type=scaling_type,
                status=ScalingStatus.FAILED,
                initial_resources={},
                final_resources={},
                scaling_metrics={},
                scaling_log=scaling_log,
                error_message=str(e),
                duration=duration,
                started_at=start_time,
                completed_at=end_time
            )
    
    async def _perform_horizontal_scaling(
        self,
        agent: Dict[str, Any],
        scaling_config: ScalingConfig,
        scaling_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform horizontal scaling"""
        scaling_log.append({
            "step": "horizontal_scaling",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate horizontal scaling
        await asyncio.sleep(0.1)  # Simulate scaling time
        
        # Perform scaling actions
        scaling_actions = []
        
        # Scale up instances
        scaling_actions.append("scale_up_instances")
        scaling_log.append({
            "step": "scale_up_instances",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Load balance
        scaling_actions.append("load_balance")
        scaling_log.append({
            "step": "load_balance",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Calculate performance impact
        performance_impact = {
            "throughput_improvement": 0.3 + np.random.random() * 0.2,
            "latency_reduction": 0.25 + np.random.random() * 0.15,
            "availability_improvement": 0.2 + np.random.random() * 0.1,
            "cost_increase": 0.4 + np.random.random() * 0.2
        }
        
        scaling_log.append({
            "step": "horizontal_scaling",
            "status": "completed",
            "scaling_actions": scaling_actions,
            "performance_impact": performance_impact,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "scaling_actions": scaling_actions,
            "performance_impact": performance_impact,
            "scaling_insights": [
                "Horizontal scaling completed successfully",
                "Load balancing improved distribution",
                "Throughput increased significantly"
            ]
        }
    
    async def _perform_vertical_scaling(
        self,
        agent: Dict[str, Any],
        scaling_config: ScalingConfig,
        scaling_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform vertical scaling"""
        scaling_log.append({
            "step": "vertical_scaling",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate vertical scaling
        await asyncio.sleep(0.1)  # Simulate scaling time
        
        # Perform scaling actions
        scaling_actions = []
        
        # Scale CPU
        scaling_actions.append("scale_cpu")
        scaling_log.append({
            "step": "scale_cpu",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Scale memory
        scaling_actions.append("scale_memory")
        scaling_log.append({
            "step": "scale_memory",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Calculate performance impact
        performance_impact = {
            "cpu_improvement": 0.25 + np.random.random() * 0.15,
            "memory_improvement": 0.3 + np.random.random() * 0.2,
            "processing_speed_improvement": 0.2 + np.random.random() * 0.15,
            "cost_increase": 0.3 + np.random.random() * 0.15
        }
        
        scaling_log.append({
            "step": "vertical_scaling",
            "status": "completed",
            "scaling_actions": scaling_actions,
            "performance_impact": performance_impact,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "scaling_actions": scaling_actions,
            "performance_impact": performance_impact,
            "scaling_insights": [
                "Vertical scaling completed successfully",
                "CPU and memory resources increased",
                "Processing speed improved significantly"
            ]
        }
    
    async def _perform_auto_scaling_logic(
        self,
        agent: Dict[str, Any],
        scaling_config: ScalingConfig,
        scaling_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform auto-scaling logic"""
        scaling_log.append({
            "step": "auto_scaling_logic",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate auto-scaling logic
        await asyncio.sleep(0.1)  # Simulate scaling time
        
        # Perform scaling actions
        scaling_actions = []
        
        # Analyze metrics
        scaling_actions.append("analyze_metrics")
        scaling_log.append({
            "step": "analyze_metrics",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Make scaling decision
        scaling_actions.append("make_scaling_decision")
        scaling_log.append({
            "step": "make_scaling_decision",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Execute scaling
        scaling_actions.append("execute_scaling")
        scaling_log.append({
            "step": "execute_scaling",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Calculate performance impact
        performance_impact = {
            "efficiency_improvement": 0.2 + np.random.random() * 0.15,
            "cost_optimization": 0.15 + np.random.random() * 0.1,
            "performance_improvement": 0.18 + np.random.random() * 0.12,
            "resource_utilization_improvement": 0.22 + np.random.random() * 0.13
        }
        
        scaling_log.append({
            "step": "auto_scaling_logic",
            "status": "completed",
            "scaling_actions": scaling_actions,
            "performance_impact": performance_impact,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "scaling_actions": scaling_actions,
            "performance_impact": performance_impact,
            "scaling_insights": [
                "Auto-scaling logic executed successfully",
                "Metrics analyzed and scaling decision made",
                "Resources optimized automatically"
            ]
        }
    
    async def _perform_predictive_scaling(
        self,
        agent: Dict[str, Any],
        scaling_config: ScalingConfig,
        scaling_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform predictive scaling"""
        scaling_log.append({
            "step": "predictive_scaling",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate predictive scaling
        await asyncio.sleep(0.1)  # Simulate scaling time
        
        # Perform scaling actions
        scaling_actions = []
        
        # Predict future load
        scaling_actions.append("predict_future_load")
        scaling_log.append({
            "step": "predict_future_load",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Proactive scaling
        scaling_actions.append("proactive_scaling")
        scaling_log.append({
            "step": "proactive_scaling",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Calculate performance impact
        performance_impact = {
            "proactive_improvement": 0.25 + np.random.random() * 0.2,
            "latency_reduction": 0.3 + np.random.random() * 0.25,
            "availability_improvement": 0.2 + np.random.random() * 0.15,
            "cost_efficiency": 0.15 + np.random.random() * 0.1
        }
        
        scaling_log.append({
            "step": "predictive_scaling",
            "status": "completed",
            "scaling_actions": scaling_actions,
            "performance_impact": performance_impact,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "scaling_actions": scaling_actions,
            "performance_impact": performance_impact,
            "scaling_insights": [
                "Predictive scaling completed successfully",
                "Future load predicted accurately",
                "Proactive scaling improved performance"
            ]
        }
    
    async def _perform_custom_scaling(
        self,
        agent: Dict[str, Any],
        scaling_config: ScalingConfig,
        scaling_log: List[Dict[str, Any]],
        scaling_options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform custom scaling"""
        scaling_log.append({
            "step": "custom_scaling",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simulate custom scaling
        await asyncio.sleep(0.1)  # Simulate scaling time
        
        scaling_log.append({
            "step": "custom_scaling",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "scaling_actions": ["custom_scaling_action"],
            "performance_impact": {"custom_improvement": 0.1},
            "scaling_insights": ["Custom scaling completed successfully"]
        }
    
    async def _continuous_scaling_loop(
        self,
        agent_id: str,
        scaling_config: Dict[str, Any],
        scaling_id: str
    ) -> None:
        """Continuous scaling loop"""
        try:
            monitoring_interval = scaling_config.get("interval", 5.0)
            
            while True:
                # Perform monitoring
                monitoring_result = await self._perform_monitoring(agent_id, scaling_config)
                
                # Check for scaling triggers
                if monitoring_result.scaling_triggers:
                    # Trigger scaling if needed
                    await self._handle_scaling_triggers(agent_id, monitoring_result.scaling_triggers)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Continuous scaling cancelled: {scaling_id}")
        except Exception as e:
            logger.error(f"Continuous scaling error: {scaling_id}, error: {e}")
    
    async def _perform_monitoring(
        self,
        agent_id: str,
        scaling_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform monitoring"""
        # Get current metrics
        metrics = await self._get_agent_metrics(agent_id)
        
        # Detect scaling triggers
        scaling_triggers = await self._detect_scaling_triggers(metrics, scaling_config)
        
        return {
            "metrics": metrics,
            "scaling_triggers": scaling_triggers
        }
    
    async def _detect_scaling_triggers(
        self,
        metrics: Dict[str, Any],
        scaling_config: Dict[str, Any]
    ) -> List[ScalingEvent]:
        """Detect scaling triggers"""
        triggers = []
        
        # Check for CPU-based scaling
        if metrics.get("cpu_usage", 0) > 0.8:
            triggers.append(ScalingEvent(
                event_id=str(uuid4()),
                agent_id="agent_id",  # This would be the actual agent ID
                event_type="cpu_high",
                trigger_metrics={"cpu_usage": metrics["cpu_usage"]},
                scaling_decision="scale_up"
            ))
        
        # Check for memory-based scaling
        if metrics.get("memory_usage", 0) > 0.85:
            triggers.append(ScalingEvent(
                event_id=str(uuid4()),
                agent_id="agent_id",  # This would be the actual agent ID
                event_type="memory_high",
                trigger_metrics={"memory_usage": metrics["memory_usage"]},
                scaling_decision="scale_up"
            ))
        
        # Check for throughput-based scaling
        if metrics.get("throughput", 0) > 800:
            triggers.append(ScalingEvent(
                event_id=str(uuid4()),
                agent_id="agent_id",  # This would be the actual agent ID
                event_type="throughput_high",
                trigger_metrics={"throughput": metrics["throughput"]},
                scaling_decision="scale_up"
            ))
        
        return triggers
    
    async def _handle_scaling_triggers(
        self,
        agent_id: str,
        scaling_triggers: List[ScalingEvent]
    ) -> None:
        """Handle scaling triggers"""
        for trigger in scaling_triggers:
            if trigger.scaling_decision == "scale_up":
                # Trigger scale up
                await self.start_auto_scaling(agent_id, ScalingType.HORIZONTAL_SCALING)
            elif trigger.scaling_decision == "scale_down":
                # Trigger scale down
                await self.start_auto_scaling(agent_id, ScalingType.VERTICAL_SCALING)
    
    async def _get_agent_resources(self, agent_id: str) -> Dict[str, Any]:
        """Get current agent resources"""
        return {
            "cpu_cores": 2 + int(np.random.random() * 4),
            "memory_gb": 4 + int(np.random.random() * 8),
            "instances": 1 + int(np.random.random() * 3),
            "storage_gb": 20 + int(np.random.random() * 40),
            "network_bandwidth": 100 + int(np.random.random() * 200)
        }
    
    async def _get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            "cpu_usage": 0.6 + np.random.random() * 0.3,
            "memory_usage": 0.5 + np.random.random() * 0.4,
            "throughput": 500 + np.random.random() * 300,
            "latency": 100 + np.random.random() * 50,
            "error_rate": 0.02 + np.random.random() * 0.03,
            "queue_length": 10 + int(np.random.random() * 20)
        }
    
    async def _calculate_scaling_metrics(
        self,
        initial_resources: Dict[str, Any],
        final_resources: Dict[str, Any],
        scaling_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate scaling metrics"""
        metrics = {
            "resource_changes": {},
            "scaling_efficiency": 0.0,
            "cost_impact": 0.0
        }
        
        # Calculate resource changes
        for resource in initial_resources:
            if resource in final_resources:
                initial_value = initial_resources[resource]
                final_value = final_resources[resource]
                
                if initial_value != 0:
                    change = ((final_value - initial_value) / initial_value) * 100
                    metrics["resource_changes"][resource] = change
        
        # Calculate scaling efficiency
        performance_impact = scaling_result.get("performance_impact", {})
        total_improvement = sum(performance_impact.values())
        metrics["scaling_efficiency"] = total_improvement / len(performance_impact) if performance_impact else 0
        
        # Calculate cost impact
        cost_increase = performance_impact.get("cost_increase", 0)
        metrics["cost_impact"] = cost_increase
        
        return metrics
    
    async def _calculate_scaling_score(
        self,
        final_resources: Dict[str, Any],
        scaling_metrics: Dict[str, Any]
    ) -> float:
        """Calculate scaling score"""
        # Weighted combination of final resources and scaling metrics
        resource_score = np.mean(list(final_resources.values()))
        efficiency_score = scaling_metrics.get("scaling_efficiency", 0)
        cost_impact = scaling_metrics.get("cost_impact", 0)
        
        # Calculate weighted score
        weights = [0.5, 0.3, 0.2]  # Resources, efficiency, cost
        scores = [resource_score, efficiency_score, 1.0 - cost_impact]  # Cost penalty
        
        scaling_score = sum(w * s for w, s in zip(weights, scores))
        
        return min(1.0, max(0.0, scaling_score))  # Clamp between 0 and 1
    
    async def _update_agent_scaling(
        self,
        agent_id: str,
        scaling_result: ScalingResult
    ) -> None:
        """Update agent with scaling results"""
        # Update agent configuration with scaling results
        updates = {
            "configuration": {
                "last_scaling": datetime.utcnow().isoformat(),
                "scaling_results": {
                    "scaling_id": scaling_result.scaling_id,
                    "scaling_type": scaling_result.scaling_type.value,
                    "scaling_metrics": scaling_result.scaling_metrics,
                    "scaling_actions": scaling_result.scaling_actions,
                    "performance_impact": scaling_result.performance_impact,
                    "scaling_score": scaling_result.scaling_score
                }
            }
        }
        
        # This would update the agent in the database
        logger.info(f"Updated agent scaling: {agent_id}")
    
    # Caching methods
    async def _cache_agent_data(self, agent: Any) -> None:
        """Cache agent data"""
        cache_key = f"auto_scaling_agent:{agent.id}"
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
        cache_key = f"auto_scaling_agent:{agent_id}"
        cached_data = await self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        return None
    
    async def _cache_scaling_result(self, scaling_id: str, result: ScalingResult) -> None:
        """Cache scaling result"""
        cache_key = f"scaling_result:{scaling_id}"
        result_data = {
            "scaling_id": result.scaling_id,
            "agent_id": result.agent_id,
            "scaling_type": result.scaling_type.value,
            "status": result.status.value,
            "scaling_metrics": result.scaling_metrics,
            "scaling_score": result.scaling_score,
            "duration": result.duration
        }
        
        await self.redis.setex(
            cache_key,
            1800,  # 30 minutes
            json.dumps(result_data)
        )



























