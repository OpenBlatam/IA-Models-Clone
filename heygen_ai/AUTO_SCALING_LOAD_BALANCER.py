#!/usr/bin/env python3
"""
⚖️ HeyGen AI - Auto-Scaling Load Balancer System
================================================

This module implements an intelligent auto-scaling and load balancing system
that automatically adjusts resources based on demand, optimizes performance,
and ensures high availability of the HeyGen AI system.
"""

import asyncio
import logging
import time
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import hashlib
import secrets
import base64
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import aiohttp
import asyncio
from aiohttp import web, WSMsgType
import ssl
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalingPolicy(str, Enum):
    """Scaling policies"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"

class LoadBalancingAlgorithm(str, Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_LOAD = "least_load"
    ADAPTIVE = "adaptive"

class InstanceStatus(str, Enum):
    """Instance status"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ScalingAction(str, Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    EMERGENCY_SCALE_UP = "emergency_scale_up"
    MAINTENANCE_MODE = "maintenance_mode"

@dataclass
class Instance:
    """Service instance"""
    instance_id: str
    instance_type: str
    status: InstanceStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    response_time: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingRule:
    """Scaling rule definition"""
    rule_id: str
    metric_name: str
    threshold: float
    comparison_operator: str  # ">", "<", ">=", "<=", "=="
    scaling_action: ScalingAction
    cooldown_period: int = 300  # seconds
    min_instances: int = 1
    max_instances: int = 10
    scale_up_step: int = 1
    scale_down_step: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""
    algorithm: LoadBalancingAlgorithm
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5  # seconds
    max_retries: int = 3
    retry_delay: int = 1  # seconds
    sticky_sessions: bool = False
    session_timeout: int = 3600  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingDecision:
    """Scaling decision"""
    decision_id: str
    action: ScalingAction
    reason: str
    current_instances: int
    target_instances: int
    confidence: float
    estimated_cost: float = 0.0
    estimated_performance: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class HealthChecker:
    """Health check service"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize health checker"""
        self.initialized = True
        logger.info("✅ Health Checker initialized")
    
    async def check_instance_health(self, instance: Instance) -> bool:
        """Check instance health"""
        if not self.initialized:
            return False
        
        try:
            # Simulate health check
            # In real implementation, this would make HTTP requests to the instance
            
            # Check if instance is responsive
            if instance.status != InstanceStatus.RUNNING:
                return False
            
            # Check resource usage
            if instance.cpu_usage > 95 or instance.memory_usage > 95:
                return False
            
            # Check response time
            if instance.response_time > 5000:  # 5 seconds
                return False
            
            # Update last health check
            instance.last_health_check = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check failed for instance {instance.instance_id}: {e}")
            return False
    
    async def check_all_instances(self, instances: List[Instance]) -> Dict[str, bool]:
        """Check health of all instances"""
        health_status = {}
        
        for instance in instances:
            health_status[instance.instance_id] = await self.check_instance_health(instance)
        
        return health_status

class MetricsCollector:
    """Metrics collection service"""
    
    def __init__(self):
        self.metrics_buffer: Dict[str, List[float]] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize metrics collector"""
        self.initialized = True
        logger.info("✅ Metrics Collector initialized")
    
    async def collect_instance_metrics(self, instance: Instance) -> Dict[str, float]:
        """Collect metrics for an instance"""
        if not self.initialized:
            return {}
        
        try:
            # Simulate metrics collection
            # In real implementation, this would collect real metrics
            
            metrics = {
                'cpu_usage': instance.cpu_usage,
                'memory_usage': instance.memory_usage,
                'active_connections': instance.active_connections,
                'response_time': instance.response_time,
                'requests_per_second': np.random.poisson(10),
                'error_rate': np.random.uniform(0, 0.05)
            }
            
            # Store metrics in buffer
            for metric_name, value in metrics.items():
                if metric_name not in self.metrics_buffer:
                    self.metrics_buffer[metric_name] = []
                self.metrics_buffer[metric_name].append(value)
                
                # Keep only last 1000 values
                if len(self.metrics_buffer[metric_name]) > 1000:
                    self.metrics_buffer[metric_name] = self.metrics_buffer[metric_name][-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to collect metrics for instance {instance.instance_id}: {e}")
            return {}
    
    async def get_aggregated_metrics(self, metric_name: str, time_window: int = 300) -> Dict[str, float]:
        """Get aggregated metrics for a time window"""
        if metric_name not in self.metrics_buffer:
            return {}
        
        values = self.metrics_buffer[metric_name]
        
        if not values:
            return {}
        
        # Get recent values (simplified)
        recent_values = values[-min(100, len(values)):]
        
        return {
            'current': recent_values[-1] if recent_values else 0,
            'average': np.mean(recent_values),
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'p95': np.percentile(recent_values, 95),
            'p99': np.percentile(recent_values, 99)
        }

class ScalingEngine:
    """Auto-scaling engine"""
    
    def __init__(self):
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: List[ScalingDecision] = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize scaling engine"""
        self.initialized = True
        logger.info("✅ Scaling Engine initialized")
    
    async def add_scaling_rule(self, rule: ScalingRule) -> bool:
        """Add scaling rule"""
        try:
            self.scaling_rules[rule.rule_id] = rule
            logger.info(f"✅ Scaling rule added: {rule.rule_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add scaling rule: {e}")
            return False
    
    async def evaluate_scaling(self, instances: List[Instance], 
                             metrics: Dict[str, float]) -> Optional[ScalingDecision]:
        """Evaluate scaling needs"""
        if not self.initialized or not instances:
            return None
        
        try:
            # Check each scaling rule
            for rule in self.scaling_rules.values():
                if not rule.enabled:
                    continue
                
                # Get metric value
                metric_value = metrics.get(rule.metric_name, 0)
                
                # Check if rule is triggered
                if self._evaluate_condition(metric_value, rule.comparison_operator, rule.threshold):
                    # Check cooldown
                    if self._is_in_cooldown(rule):
                        continue
                    
                    # Determine scaling action
                    current_instances = len([i for i in instances if i.status == InstanceStatus.RUNNING])
                    target_instances = self._calculate_target_instances(
                        current_instances, rule, metric_value
                    )
                    
                    if target_instances != current_instances:
                        decision = ScalingDecision(
                            decision_id=str(uuid.uuid4()),
                            action=rule.scaling_action,
                            reason=f"Rule {rule.rule_id} triggered: {rule.metric_name} {rule.comparison_operator} {rule.threshold}",
                            current_instances=current_instances,
                            target_instances=target_instances,
                            confidence=0.9,
                            estimated_cost=self._estimate_cost(target_instances),
                            estimated_performance=self._estimate_performance(target_instances)
                        )
                        
                        self.scaling_history.append(decision)
                        logger.info(f"✅ Scaling decision made: {decision.action} to {target_instances} instances")
                        return decision
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Scaling evaluation failed: {e}")
            return None
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate scaling condition"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        else:
            return False
    
    def _is_in_cooldown(self, rule: ScalingRule) -> bool:
        """Check if rule is in cooldown period"""
        if not rule.metadata.get('last_triggered'):
            return False
        
        last_triggered = rule.metadata['last_triggered']
        cooldown_end = last_triggered + timedelta(seconds=rule.cooldown_period)
        
        return datetime.now() < cooldown_end
    
    def _calculate_target_instances(self, current_instances: int, rule: ScalingRule, 
                                  metric_value: float) -> int:
        """Calculate target number of instances"""
        if rule.scaling_action == ScalingAction.SCALE_UP:
            target = current_instances + rule.scale_up_step
        elif rule.scaling_action == ScalingAction.SCALE_DOWN:
            target = current_instances - rule.scale_down_step
        elif rule.scaling_action == ScalingAction.EMERGENCY_SCALE_UP:
            target = current_instances + rule.scale_up_step * 2
        else:
            target = current_instances
        
        # Apply min/max constraints
        target = max(rule.min_instances, min(target, rule.max_instances))
        
        return target
    
    def _estimate_cost(self, instances: int) -> float:
        """Estimate cost for number of instances"""
        # Simplified cost estimation
        base_cost = 0.1  # per instance per hour
        return instances * base_cost
    
    def _estimate_performance(self, instances: int) -> float:
        """Estimate performance for number of instances"""
        # Simplified performance estimation
        if instances == 0:
            return 0.0
        return min(1.0, instances / 5.0)  # Normalize to 0-1

class LoadBalancer:
    """Load balancer service"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.instances: List[Instance] = []
        self.current_index = 0
        self.initialized = False
    
    async def initialize(self):
        """Initialize load balancer"""
        self.initialized = True
        logger.info(f"✅ Load Balancer initialized with {self.config.algorithm.value} algorithm")
    
    async def add_instance(self, instance: Instance) -> bool:
        """Add instance to load balancer"""
        try:
            self.instances.append(instance)
            logger.info(f"✅ Instance added to load balancer: {instance.instance_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add instance: {e}")
            return False
    
    async def remove_instance(self, instance_id: str) -> bool:
        """Remove instance from load balancer"""
        try:
            self.instances = [i for i in self.instances if i.instance_id != instance_id]
            logger.info(f"✅ Instance removed from load balancer: {instance_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove instance: {e}")
            return False
    
    async def select_instance(self, request_metadata: Dict[str, Any] = None) -> Optional[Instance]:
        """Select instance for request"""
        if not self.initialized or not self.instances:
            return None
        
        try:
            # Filter healthy instances
            healthy_instances = [i for i in self.instances if i.status == InstanceStatus.RUNNING]
            
            if not healthy_instances:
                return None
            
            # Select instance based on algorithm
            if self.config.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                return self._round_robin_selection(healthy_instances)
            elif self.config.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                return self._least_connections_selection(healthy_instances)
            elif self.config.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(healthy_instances)
            elif self.config.algorithm == LoadBalancingAlgorithm.LEAST_LOAD:
                return self._least_load_selection(healthy_instances)
            else:
                return self._round_robin_selection(healthy_instances)
                
        except Exception as e:
            logger.error(f"❌ Instance selection failed: {e}")
            return None
    
    def _round_robin_selection(self, instances: List[Instance]) -> Instance:
        """Round robin selection"""
        instance = instances[self.current_index % len(instances)]
        self.current_index = (self.current_index + 1) % len(instances)
        return instance
    
    def _least_connections_selection(self, instances: List[Instance]) -> Instance:
        """Least connections selection"""
        return min(instances, key=lambda x: x.active_connections)
    
    def _least_response_time_selection(self, instances: List[Instance]) -> Instance:
        """Least response time selection"""
        return min(instances, key=lambda x: x.response_time)
    
    def _least_load_selection(self, instances: List[Instance]) -> Instance:
        """Least load selection (CPU + Memory)"""
        return min(instances, key=lambda x: x.cpu_usage + x.memory_usage)
    
    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        healthy_instances = [i for i in self.instances if i.status == InstanceStatus.RUNNING]
        
        if not healthy_instances:
            return {
                'total_instances': len(self.instances),
                'healthy_instances': 0,
                'average_cpu': 0,
                'average_memory': 0,
                'average_connections': 0,
                'average_response_time': 0
            }
        
        return {
            'total_instances': len(self.instances),
            'healthy_instances': len(healthy_instances),
            'average_cpu': np.mean([i.cpu_usage for i in healthy_instances]),
            'average_memory': np.mean([i.memory_usage for i in healthy_instances]),
            'average_connections': np.mean([i.active_connections for i in healthy_instances]),
            'average_response_time': np.mean([i.response_time for i in healthy_instances])
        }

class AutoScalingLoadBalancer:
    """Main auto-scaling load balancer system"""
    
    def __init__(self):
        self.instances: Dict[str, Instance] = {}
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector()
        self.scaling_engine = ScalingEngine()
        self.load_balancer: Optional[LoadBalancer] = None
        self.scaling_enabled = True
        self.initialized = False
    
    async def initialize(self, load_balancer_config: LoadBalancerConfig = None):
        """Initialize auto-scaling load balancer"""
        try:
            logger.info("⚖️ Initializing Auto-Scaling Load Balancer...")
            
            # Initialize components
            await self.health_checker.initialize()
            await self.metrics_collector.initialize()
            await self.scaling_engine.initialize()
            
            # Initialize load balancer
            config = load_balancer_config or LoadBalancerConfig(
                algorithm=LoadBalancingAlgorithm.LEAST_LOAD
            )
            self.load_balancer = LoadBalancer(config)
            await self.load_balancer.initialize()
            
            # Add default scaling rules
            await self._add_default_scaling_rules()
            
            self.initialized = True
            logger.info("✅ Auto-Scaling Load Balancer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Auto-Scaling Load Balancer: {e}")
            raise
    
    async def _add_default_scaling_rules(self):
        """Add default scaling rules"""
        default_rules = [
            ScalingRule(
                rule_id="cpu_scale_up",
                metric_name="cpu_usage",
                threshold=80.0,
                comparison_operator=">",
                scaling_action=ScalingAction.SCALE_UP,
                cooldown_period=300,
                min_instances=1,
                max_instances=10
            ),
            ScalingRule(
                rule_id="cpu_scale_down",
                metric_name="cpu_usage",
                threshold=30.0,
                comparison_operator="<",
                scaling_action=ScalingAction.SCALE_DOWN,
                cooldown_period=600,
                min_instances=1,
                max_instances=10
            ),
            ScalingRule(
                rule_id="memory_scale_up",
                metric_name="memory_usage",
                threshold=85.0,
                comparison_operator=">",
                scaling_action=ScalingAction.SCALE_UP,
                cooldown_period=300,
                min_instances=1,
                max_instances=10
            ),
            ScalingRule(
                rule_id="response_time_scale_up",
                metric_name="response_time",
                threshold=1000.0,  # 1 second
                comparison_operator=">",
                scaling_action=ScalingAction.SCALE_UP,
                cooldown_period=180,
                min_instances=1,
                max_instances=10
            )
        ]
        
        for rule in default_rules:
            await self.scaling_engine.add_scaling_rule(rule)
    
    async def create_instance(self, instance_type: str = "standard") -> str:
        """Create new instance"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        try:
            instance_id = str(uuid.uuid4())
            
            instance = Instance(
                instance_id=instance_id,
                instance_type=instance_type,
                status=InstanceStatus.STARTING
            )
            
            # Add to instances
            self.instances[instance_id] = instance
            
            # Add to load balancer
            if self.load_balancer:
                await self.load_balancer.add_instance(instance)
            
            # Simulate instance startup
            await asyncio.sleep(1)
            instance.status = InstanceStatus.RUNNING
            
            logger.info(f"✅ Instance created: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create instance: {e}")
            raise
    
    async def destroy_instance(self, instance_id: str) -> bool:
        """Destroy instance"""
        try:
            if instance_id not in self.instances:
                return False
            
            instance = self.instances[instance_id]
            instance.status = InstanceStatus.STOPPING
            
            # Remove from load balancer
            if self.load_balancer:
                await self.load_balancer.remove_instance(instance_id)
            
            # Simulate instance shutdown
            await asyncio.sleep(1)
            instance.status = InstanceStatus.STOPPED
            
            # Remove from instances
            del self.instances[instance_id]
            
            logger.info(f"✅ Instance destroyed: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to destroy instance {instance_id}: {e}")
            return False
    
    async def process_request(self, request_metadata: Dict[str, Any] = None) -> Optional[str]:
        """Process request through load balancer"""
        if not self.initialized or not self.load_balancer:
            return None
        
        try:
            # Select instance
            instance = await self.load_balancer.select_instance(request_metadata)
            
            if not instance:
                return None
            
            # Simulate request processing
            instance.active_connections += 1
            
            # Simulate processing time
            processing_time = np.random.exponential(0.1)  # 100ms average
            await asyncio.sleep(processing_time)
            
            # Update instance metrics
            instance.response_time = processing_time * 1000  # Convert to ms
            instance.cpu_usage = min(100, instance.cpu_usage + np.random.uniform(0, 5))
            instance.memory_usage = min(100, instance.memory_usage + np.random.uniform(0, 2))
            
            # Decrement connections
            instance.active_connections = max(0, instance.active_connections - 1)
            
            return instance.instance_id
            
        except Exception as e:
            logger.error(f"❌ Request processing failed: {e}")
            return None
    
    async def run_scaling_cycle(self):
        """Run scaling evaluation cycle"""
        if not self.initialized or not self.scaling_enabled:
            return
        
        try:
            # Collect metrics for all instances
            all_metrics = {}
            for instance in self.instances.values():
                if instance.status == InstanceStatus.RUNNING:
                    metrics = await self.metrics_collector.collect_instance_metrics(instance)
                    for metric_name, value in metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)
            
            # Calculate aggregated metrics
            aggregated_metrics = {}
            for metric_name, values in all_metrics.items():
                if values:
                    aggregated_metrics[metric_name] = np.mean(values)
            
            # Evaluate scaling
            scaling_decision = await self.scaling_engine.evaluate_scaling(
                list(self.instances.values()), aggregated_metrics
            )
            
            if scaling_decision:
                await self._execute_scaling_decision(scaling_decision)
            
        except Exception as e:
            logger.error(f"❌ Scaling cycle failed: {e}")
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision"""
        try:
            current_instances = len([i for i in self.instances.values() if i.status == InstanceStatus.RUNNING])
            target_instances = decision.target_instances
            
            if target_instances > current_instances:
                # Scale up
                instances_to_add = target_instances - current_instances
                for _ in range(instances_to_add):
                    await self.create_instance()
            elif target_instances < current_instances:
                # Scale down
                instances_to_remove = current_instances - target_instances
                running_instances = [i for i in self.instances.values() if i.status == InstanceStatus.RUNNING]
                
                for i in range(min(instances_to_remove, len(running_instances))):
                    instance = running_instances[i]
                    await self.destroy_instance(instance.instance_id)
            
            logger.info(f"✅ Scaling decision executed: {decision.action} to {target_instances} instances")
            
        except Exception as e:
            logger.error(f"❌ Failed to execute scaling decision: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.initialized:
            return {}
        
        # Get load balancer stats
        lb_stats = {}
        if self.load_balancer:
            lb_stats = await self.load_balancer.get_load_balancer_stats()
        
        # Get scaling stats
        scaling_stats = {
            'scaling_rules': len(self.scaling_engine.scaling_rules),
            'scaling_decisions': len(self.scaling_engine.scaling_history),
            'scaling_enabled': self.scaling_enabled
        }
        
        return {
            'initialized': self.initialized,
            'total_instances': len(self.instances),
            'running_instances': len([i for i in self.instances.values() if i.status == InstanceStatus.RUNNING]),
            'load_balancer_stats': lb_stats,
            'scaling_stats': scaling_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_auto_scaling(self, interval: int = 30):
        """Start auto-scaling loop"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        self.scaling_enabled = True
        
        async def scaling_loop():
            while self.scaling_enabled:
                await self.run_scaling_cycle()
                await asyncio.sleep(interval)
        
        asyncio.create_task(scaling_loop())
        logger.info(f"✅ Auto-scaling started with {interval}s interval")
    
    async def stop_auto_scaling(self):
        """Stop auto-scaling loop"""
        self.scaling_enabled = False
        logger.info("✅ Auto-scaling stopped")
    
    async def shutdown(self):
        """Shutdown auto-scaling load balancer"""
        await self.stop_auto_scaling()
        self.initialized = False
        logger.info("✅ Auto-Scaling Load Balancer shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the auto-scaling load balancer"""
    print("⚖️ HeyGen AI - Auto-Scaling Load Balancer Demo")
    print("=" * 60)
    
    # Initialize system
    system = AutoScalingLoadBalancer()
    
    try:
        # Initialize the system
        print("\n🚀 Initializing Auto-Scaling Load Balancer...")
        await system.initialize()
        print("✅ Auto-Scaling Load Balancer initialized successfully")
        
        # Get system status
        print("\n📊 System Status:")
        status = await system.get_system_status()
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Create initial instances
        print("\n🏗️ Creating Initial Instances...")
        
        for i in range(3):
            instance_id = await system.create_instance(f"instance_{i+1}")
            print(f"  ✅ Created instance: {instance_id}")
        
        # Start auto-scaling
        print("\n⚖️ Starting Auto-Scaling...")
        await system.start_auto_scaling(interval=5)  # 5 second interval for demo
        
        # Simulate load
        print("\n📈 Simulating Load...")
        
        for round_num in range(10):
            print(f"\n  Round {round_num + 1}:")
            
            # Process some requests
            for _ in range(20):
                instance_id = await system.process_request()
                if instance_id:
                    print(f"    Request processed by {instance_id[:8]}...")
            
            # Update instance metrics (simulate load)
            for instance in system.instances.values():
                if instance.status == InstanceStatus.RUNNING:
                    # Simulate varying load
                    load_factor = 1.0 + np.sin(round_num * 0.5) * 0.5
                    instance.cpu_usage = min(100, instance.cpu_usage * load_factor + np.random.uniform(0, 10))
                    instance.memory_usage = min(100, instance.memory_usage * load_factor + np.random.uniform(0, 5))
            
            # Run scaling cycle
            await system.run_scaling_cycle()
            
            # Show current status
            current_status = await system.get_system_status()
            print(f"    Running instances: {current_status['running_instances']}")
            
            await asyncio.sleep(2)  # Wait between rounds
        
        # Stop auto-scaling
        print("\n🛑 Stopping Auto-Scaling...")
        await system.stop_auto_scaling()
        
        # Final status
        print("\n📊 Final Status:")
        final_status = await system.get_system_status()
        for key, value in final_status.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Shutdown
        await system.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


