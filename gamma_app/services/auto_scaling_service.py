"""
Gamma App - Advanced Auto-Scaling Service
Ultra-advanced auto-scaling with ML-powered predictions and intelligent resource management
"""

import asyncio
import logging
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import redis
import kubernetes
import docker
import boto3
import google.cloud
import azure.identity
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import structlog
import prometheus_client
from prometheus_client import Gauge, Counter, Histogram
import requests
import yaml
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import schedule
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)

class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"

class ScalingTrigger(Enum):
    """Scaling triggers"""
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_SIZE = "queue_size"
    ERROR_RATE = "error_rate"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"

class ScalingPolicy(Enum):
    """Scaling policies"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CUSTOM = "custom"

@dataclass
class ScalingMetrics:
    """Scaling metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    queue_size: int
    error_rate: float
    active_connections: int
    throughput: float
    latency_p95: float
    latency_p99: float

@dataclass
class ScalingDecision:
    """Scaling decision"""
    action: ScalingAction
    trigger: ScalingTrigger
    current_replicas: int
    target_replicas: int
    confidence: float
    reasoning: str
    estimated_impact: Dict[str, float]
    timestamp: datetime
    policy: ScalingPolicy

@dataclass
class ScalingConfig:
    """Scaling configuration"""
    min_replicas: int = 1
    max_replicas: int = 100
    target_cpu: float = 70.0
    target_memory: float = 80.0
    target_response_time: float = 500.0  # ms
    target_error_rate: float = 1.0  # %
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    predictive_scaling: bool = True
    scheduled_scaling: bool = True
    policy: ScalingPolicy = ScalingPolicy.BALANCED

class AdvancedAutoScalingService:
    """
    Ultra-advanced auto-scaling service with ML-powered predictions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize advanced auto-scaling service"""
        self.config = config or {}
        
        # Core components
        self.redis_client = None
        self.k8s_client = None
        self.docker_client = None
        self.cloud_clients = {}
        
        # Scaling configuration
        self.scaling_config = ScalingConfig()
        self.scaling_policies = {
            ScalingPolicy.CONSERVATIVE: {
                "scale_up_threshold": 0.9,
                "scale_down_threshold": 0.2,
                "cooldown_multiplier": 1.5
            },
            ScalingPolicy.AGGRESSIVE: {
                "scale_up_threshold": 0.6,
                "scale_down_threshold": 0.4,
                "cooldown_multiplier": 0.5
            },
            ScalingPolicy.BALANCED: {
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.3,
                "cooldown_multiplier": 1.0
            }
        }
        
        # Metrics collection
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_decisions: List[ScalingDecision] = []
        self.last_scaling_action: Optional[datetime] = None
        
        # ML models
        self.demand_predictor = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Prometheus metrics
        self.prometheus_metrics = {
            'scaling_actions_total': Counter('scaling_actions_total', 'Total scaling actions', ['action', 'trigger']),
            'current_replicas': Gauge('current_replicas', 'Current number of replicas'),
            'target_replicas': Gauge('target_replicas', 'Target number of replicas'),
            'scaling_confidence': Gauge('scaling_confidence', 'Confidence in scaling decision'),
            'prediction_accuracy': Gauge('prediction_accuracy', 'ML model prediction accuracy'),
            'scaling_duration': Histogram('scaling_duration_seconds', 'Time taken to complete scaling action')
        }
        
        # Scheduled scaling
        self.scheduled_scaling_rules = []
        self.schedule_thread = None
        
        # Auto-scaling state
        self.auto_scaling_enabled = True
        self.scaling_active = False
        self.emergency_mode = False
        
        # Performance tracking
        self.performance_history = []
        self.cost_tracking = {}
        
        logger.info("Advanced Auto-Scaling Service initialized")
    
    async def initialize(self):
        """Initialize auto-scaling service"""
        try:
            # Initialize Redis
            await self._initialize_redis()
            
            # Initialize Kubernetes
            await self._initialize_kubernetes()
            
            # Initialize Docker
            await self._initialize_docker()
            
            # Initialize cloud clients
            await self._initialize_cloud_clients()
            
            # Load scaling configuration
            await self._load_scaling_config()
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Setup scheduled scaling
            await self._setup_scheduled_scaling()
            
            # Start monitoring
            await self._start_monitoring()
            
            logger.info("Auto-scaling service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize auto-scaling service: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established for auto-scaling")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    async def _initialize_kubernetes(self):
        """Initialize Kubernetes client"""
        try:
            kubernetes.config.load_kube_config()
            self.k8s_client = kubernetes.client.AppsV1Api()
            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.warning(f"Kubernetes client initialization failed: {e}")
    
    async def _initialize_docker(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning(f"Docker client initialization failed: {e}")
    
    async def _initialize_cloud_clients(self):
        """Initialize cloud service clients"""
        try:
            # AWS
            if self.config.get('aws_enabled'):
                self.cloud_clients['aws'] = {
                    'ec2': boto3.client('ec2'),
                    'ecs': boto3.client('ecs'),
                    'eks': boto3.client('eks'),
                    'cloudwatch': boto3.client('cloudwatch')
                }
            
            # Google Cloud
            if self.config.get('gcp_enabled'):
                self.cloud_clients['gcp'] = {
                    'compute': google.cloud.compute_v1.InstancesClient(),
                    'gke': google.cloud.container_v1.ClusterManagerClient()
                }
            
            # Azure
            if self.config.get('azure_enabled'):
                self.cloud_clients['azure'] = {
                    'compute': azure.identity.DefaultAzureCredential(),
                    'aks': azure.identity.DefaultAzureCredential()
                }
            
            logger.info("Cloud clients initialized")
            
        except Exception as e:
            logger.warning(f"Cloud clients initialization failed: {e}")
    
    async def _load_scaling_config(self):
        """Load scaling configuration"""
        try:
            config_file = self.config.get('scaling_config', 'scaling.yaml')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self.scaling_config = ScalingConfig(**config_data)
                    logger.info("Scaling configuration loaded")
            else:
                logger.warning("Scaling configuration file not found, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load scaling configuration: {e}")
    
    async def _initialize_ml_models(self):
        """Initialize ML models for predictive scaling"""
        try:
            # Demand prediction model
            self.demand_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Load pre-trained models if available
            model_path = Path("models/scaling_models")
            if model_path.exists():
                try:
                    self.demand_predictor = joblib.load(model_path / "demand_predictor.pkl")
                    self.anomaly_detector = joblib.load(model_path / "anomaly_detector.pkl")
                    self.scaler = joblib.load(model_path / "scaler.pkl")
                    self.model_trained = True
                    logger.info("Pre-trained ML models loaded")
                except Exception as e:
                    logger.warning(f"Failed to load pre-trained models: {e}")
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    async def _setup_scheduled_scaling(self):
        """Setup scheduled scaling rules"""
        try:
            # Load scheduled scaling rules
            schedule_config = self.config.get('scheduled_scaling', {})
            
            # Example: Scale up during business hours
            if schedule_config.get('business_hours_scaling'):
                schedule.every().monday.at("08:00").do(self._scheduled_scale_up)
                schedule.every().tuesday.at("08:00").do(self._scheduled_scale_up)
                schedule.every().wednesday.at("08:00").do(self._scheduled_scale_up)
                schedule.every().thursday.at("08:00").do(self._scheduled_scale_up)
                schedule.every().friday.at("08:00").do(self._scheduled_scale_up)
                
                schedule.every().monday.at("18:00").do(self._scheduled_scale_down)
                schedule.every().tuesday.at("18:00").do(self._scheduled_scale_down)
                schedule.every().wednesday.at("18:00").do(self._scheduled_scale_down)
                schedule.every().thursday.at("18:00").do(self._scheduled_scale_down)
                schedule.every().friday.at("18:00").do(self._scheduled_scale_down)
            
            # Start schedule thread
            self.schedule_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.schedule_thread.start()
            
            logger.info("Scheduled scaling setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup scheduled scaling: {e}")
    
    def _run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    async def _start_monitoring(self):
        """Start monitoring for auto-scaling"""
        try:
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            # Start ML model training loop
            asyncio.create_task(self._ml_training_loop())
            
            # Start performance tracking
            asyncio.create_task(self._performance_tracking_loop())
            
            logger.info("Auto-scaling monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for auto-scaling"""
        while self.auto_scaling_enabled:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Make scaling decision
                if not self.scaling_active and not self.emergency_mode:
                    decision = await self._make_scaling_decision(metrics)
                    
                    if decision.action != ScalingAction.NO_ACTION:
                        await self._execute_scaling_decision(decision)
                
                # Update Prometheus metrics
                await self._update_prometheus_metrics(metrics)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Application metrics (would integrate with monitoring system)
            request_rate = await self._get_request_rate()
            response_time = await self._get_response_time()
            queue_size = await self._get_queue_size()
            error_rate = await self._get_error_rate()
            active_connections = await self._get_active_connections()
            throughput = await self._get_throughput()
            latency_p95 = await self._get_latency_p95()
            latency_p99 = await self._get_latency_p99()
            
            metrics = ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_rate=request_rate,
                response_time=response_time,
                queue_size=queue_size,
                error_rate=error_rate,
                active_connections=active_connections,
                throughput=throughput,
                latency_p95=latency_p95,
                latency_p99=latency_p99
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return default metrics
            return ScalingMetrics(
                timestamp=datetime.now(),
                cpu_usage=0,
                memory_usage=0,
                request_rate=0,
                response_time=0,
                queue_size=0,
                error_rate=0,
                active_connections=0,
                throughput=0,
                latency_p95=0,
                latency_p99=0
            )
    
    async def _get_request_rate(self) -> float:
        """Get current request rate"""
        try:
            if self.redis_client:
                # Get request rate from Redis
                rate = self.redis_client.get("metrics:request_rate")
                return float(rate) if rate else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    async def _get_response_time(self) -> float:
        """Get current response time"""
        try:
            if self.redis_client:
                response_time = self.redis_client.get("metrics:response_time")
                return float(response_time) if response_time else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    async def _get_queue_size(self) -> int:
        """Get current queue size"""
        try:
            if self.redis_client:
                queue_size = self.redis_client.get("metrics:queue_size")
                return int(queue_size) if queue_size else 0
            return 0
        except Exception:
            return 0
    
    async def _get_error_rate(self) -> float:
        """Get current error rate"""
        try:
            if self.redis_client:
                error_rate = self.redis_client.get("metrics:error_rate")
                return float(error_rate) if error_rate else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    async def _get_active_connections(self) -> int:
        """Get current active connections"""
        try:
            if self.redis_client:
                connections = self.redis_client.get("metrics:active_connections")
                return int(connections) if connections else 0
            return 0
        except Exception:
            return 0
    
    async def _get_throughput(self) -> float:
        """Get current throughput"""
        try:
            if self.redis_client:
                throughput = self.redis_client.get("metrics:throughput")
                return float(throughput) if throughput else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    async def _get_latency_p95(self) -> float:
        """Get 95th percentile latency"""
        try:
            if self.redis_client:
                latency = self.redis_client.get("metrics:latency_p95")
                return float(latency) if latency else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    async def _get_latency_p99(self) -> float:
        """Get 99th percentile latency"""
        try:
            if self.redis_client:
                latency = self.redis_client.get("metrics:latency_p99")
                return float(latency) if latency else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    async def _make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Make intelligent scaling decision"""
        try:
            current_replicas = await self._get_current_replicas()
            
            # Check cooldown period
            if self.last_scaling_action:
                cooldown = self.scaling_config.scale_up_cooldown
                if (datetime.now() - self.last_scaling_action).seconds < cooldown:
                    return ScalingDecision(
                        action=ScalingAction.NO_ACTION,
                        trigger=ScalingTrigger.CPU_THRESHOLD,
                        current_replicas=current_replicas,
                        target_replicas=current_replicas,
                        confidence=1.0,
                        reasoning="Cooldown period active",
                        estimated_impact={},
                        timestamp=datetime.now(),
                        policy=self.scaling_config.policy
                    )
            
            # Check if we're at limits
            if current_replicas >= self.scaling_config.max_replicas:
                return ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    trigger=ScalingTrigger.CPU_THRESHOLD,
                    current_replicas=current_replicas,
                    target_replicas=current_replicas,
                    confidence=1.0,
                    reasoning="Maximum replicas reached",
                    estimated_impact={},
                    timestamp=datetime.now(),
                    policy=self.scaling_config.policy
                )
            
            if current_replicas <= self.scaling_config.min_replicas:
                return ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    trigger=ScalingTrigger.CPU_THRESHOLD,
                    current_replicas=current_replicas,
                    target_replicas=current_replicas,
                    confidence=1.0,
                    reasoning="Minimum replicas reached",
                    estimated_impact={},
                    timestamp=datetime.now(),
                    policy=self.scaling_config.policy
                )
            
            # Analyze metrics and make decision
            decision = await self._analyze_metrics_for_scaling(metrics, current_replicas)
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to make scaling decision: {e}")
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                trigger=ScalingTrigger.CPU_THRESHOLD,
                current_replicas=current_replicas,
                target_replicas=current_replicas,
                confidence=0.0,
                reasoning=f"Error in decision making: {e}",
                estimated_impact={},
                timestamp=datetime.now(),
                policy=self.scaling_config.policy
            )
    
    async def _analyze_metrics_for_scaling(self, metrics: ScalingMetrics, 
                                         current_replicas: int) -> ScalingDecision:
        """Analyze metrics to determine scaling action"""
        try:
            # Get policy thresholds
            policy_config = self.scaling_policies[self.scaling_config.policy]
            scale_up_threshold = policy_config["scale_up_threshold"]
            scale_down_threshold = policy_config["scale_down_threshold"]
            
            # Calculate scaling scores
            cpu_score = metrics.cpu_usage / 100.0
            memory_score = metrics.memory_usage / 100.0
            response_time_score = min(metrics.response_time / self.scaling_config.target_response_time, 2.0)
            error_rate_score = min(metrics.error_rate / self.scaling_config.target_error_rate, 2.0)
            
            # Weighted score
            weighted_score = (
                cpu_score * 0.3 +
                memory_score * 0.3 +
                response_time_score * 0.2 +
                error_rate_score * 0.2
            )
            
            # Determine action
            if weighted_score > scale_up_threshold:
                # Scale up
                target_replicas = min(
                    current_replicas * 2,
                    self.scaling_config.max_replicas
                )
                
                confidence = min(weighted_score, 1.0)
                reasoning = f"High load detected (score: {weighted_score:.2f})"
                
                return ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    trigger=ScalingTrigger.CPU_THRESHOLD,
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    confidence=confidence,
                    reasoning=reasoning,
                    estimated_impact={
                        "cpu_reduction": (target_replicas - current_replicas) * 0.3,
                        "response_time_reduction": (target_replicas - current_replicas) * 0.2,
                        "cost_increase": (target_replicas - current_replicas) * 0.1
                    },
                    timestamp=datetime.now(),
                    policy=self.scaling_config.policy
                )
            
            elif weighted_score < scale_down_threshold:
                # Scale down
                target_replicas = max(
                    current_replicas // 2,
                    self.scaling_config.min_replicas
                )
                
                confidence = 1.0 - weighted_score
                reasoning = f"Low load detected (score: {weighted_score:.2f})"
                
                return ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    trigger=ScalingTrigger.CPU_THRESHOLD,
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    confidence=confidence,
                    reasoning=reasoning,
                    estimated_impact={
                        "cpu_increase": (current_replicas - target_replicas) * 0.3,
                        "response_time_increase": (current_replicas - target_replicas) * 0.2,
                        "cost_reduction": (current_replicas - target_replicas) * 0.1
                    },
                    timestamp=datetime.now(),
                    policy=self.scaling_config.policy
                )
            
            else:
                # No action needed
                return ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    trigger=ScalingTrigger.CPU_THRESHOLD,
                    current_replicas=current_replicas,
                    target_replicas=current_replicas,
                    confidence=1.0,
                    reasoning=f"Load within acceptable range (score: {weighted_score:.2f})",
                    estimated_impact={},
                    timestamp=datetime.now(),
                    policy=self.scaling_config.policy
                )
            
        except Exception as e:
            logger.error(f"Failed to analyze metrics: {e}")
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                trigger=ScalingTrigger.CPU_THRESHOLD,
                current_replicas=current_replicas,
                target_replicas=current_replicas,
                confidence=0.0,
                reasoning=f"Error in analysis: {e}",
                estimated_impact={},
                timestamp=datetime.now(),
                policy=self.scaling_config.policy
            )
    
    async def _get_current_replicas(self) -> int:
        """Get current number of replicas"""
        try:
            if self.k8s_client:
                # Get from Kubernetes
                deployment = self.k8s_client.read_namespaced_deployment(
                    name="gamma-app",
                    namespace="default"
                )
                return deployment.spec.replicas
            elif self.docker_client:
                # Get from Docker
                containers = self.docker_client.containers.list(
                    filters={"label": "app=gamma-app"}
                )
                return len(containers)
            else:
                return 1
        except Exception as e:
            logger.error(f"Failed to get current replicas: {e}")
            return 1
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision"""
        try:
            self.scaling_active = True
            start_time = time.time()
            
            logger.info(f"Executing scaling decision: {decision.action.value}")
            
            # Execute scaling action
            if decision.action == ScalingAction.SCALE_UP:
                await self._scale_up(decision.target_replicas)
            elif decision.action == ScalingAction.SCALE_DOWN:
                await self._scale_down(decision.target_replicas)
            elif decision.action == ScalingAction.SCALE_OUT:
                await self._scale_out(decision.target_replicas)
            elif decision.action == ScalingAction.SCALE_IN:
                await self._scale_in(decision.target_replicas)
            
            # Record scaling action
            self.scaling_decisions.append(decision)
            self.last_scaling_action = datetime.now()
            
            # Update Prometheus metrics
            self.prometheus_metrics['scaling_actions_total'].labels(
                action=decision.action.value,
                trigger=decision.trigger.value
            ).inc()
            
            duration = time.time() - start_time
            self.prometheus_metrics['scaling_duration'].observe(duration)
            
            logger.info(f"Scaling action completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
        finally:
            self.scaling_active = False
    
    async def _scale_up(self, target_replicas: int):
        """Scale up deployment"""
        try:
            if self.k8s_client:
                # Kubernetes scaling
                deployment = self.k8s_client.read_namespaced_deployment(
                    name="gamma-app",
                    namespace="default"
                )
                deployment.spec.replicas = target_replicas
                self.k8s_client.patch_namespaced_deployment(
                    name="gamma-app",
                    namespace="default",
                    body=deployment
                )
                logger.info(f"Scaled up to {target_replicas} replicas (Kubernetes)")
            
            elif self.docker_client:
                # Docker scaling
                for i in range(target_replicas):
                    container = self.docker_client.containers.run(
                        "gamma-app:latest",
                        name=f"gamma-app-{i}",
                        detach=True,
                        labels={"app": "gamma-app"}
                    )
                    logger.info(f"Started container: {container.id[:12]}")
            
            # Update current replicas metric
            self.prometheus_metrics['current_replicas'].set(target_replicas)
            
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
            raise
    
    async def _scale_down(self, target_replicas: int):
        """Scale down deployment"""
        try:
            if self.k8s_client:
                # Kubernetes scaling
                deployment = self.k8s_client.read_namespaced_deployment(
                    name="gamma-app",
                    namespace="default"
                )
                deployment.spec.replicas = target_replicas
                self.k8s_client.patch_namespaced_deployment(
                    name="gamma-app",
                    namespace="default",
                    body=deployment
                )
                logger.info(f"Scaled down to {target_replicas} replicas (Kubernetes)")
            
            elif self.docker_client:
                # Docker scaling
                containers = self.docker_client.containers.list(
                    filters={"label": "app=gamma-app"}
                )
                
                # Remove excess containers
                for container in containers[target_replicas:]:
                    container.stop()
                    container.remove()
                    logger.info(f"Stopped container: {container.id[:12]}")
            
            # Update current replicas metric
            self.prometheus_metrics['current_replicas'].set(target_replicas)
            
        except Exception as e:
            logger.error(f"Failed to scale down: {e}")
            raise
    
    async def _scale_out(self, target_replicas: int):
        """Scale out (horizontal scaling)"""
        try:
            # Similar to scale_up but with different logic
            await self._scale_up(target_replicas)
            logger.info(f"Scaled out to {target_replicas} replicas")
            
        except Exception as e:
            logger.error(f"Failed to scale out: {e}")
            raise
    
    async def _scale_in(self, target_replicas: int):
        """Scale in (horizontal scaling)"""
        try:
            # Similar to scale_down but with different logic
            await self._scale_down(target_replicas)
            logger.info(f"Scaled in to {target_replicas} replicas")
            
        except Exception as e:
            logger.error(f"Failed to scale in: {e}")
            raise
    
    async def _scheduled_scale_up(self):
        """Scheduled scale up"""
        try:
            current_replicas = await self._get_current_replicas()
            target_replicas = min(current_replicas * 2, self.scaling_config.max_replicas)
            
            decision = ScalingDecision(
                action=ScalingAction.SCALE_UP,
                trigger=ScalingTrigger.SCHEDULED,
                current_replicas=current_replicas,
                target_replicas=target_replicas,
                confidence=1.0,
                reasoning="Scheduled scale up for business hours",
                estimated_impact={},
                timestamp=datetime.now(),
                policy=self.scaling_config.policy
            )
            
            await self._execute_scaling_decision(decision)
            
        except Exception as e:
            logger.error(f"Scheduled scale up failed: {e}")
    
    async def _scheduled_scale_down(self):
        """Scheduled scale down"""
        try:
            current_replicas = await self._get_current_replicas()
            target_replicas = max(current_replicas // 2, self.scaling_config.min_replicas)
            
            decision = ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                trigger=ScalingTrigger.SCHEDULED,
                current_replicas=current_replicas,
                target_replicas=target_replicas,
                confidence=1.0,
                reasoning="Scheduled scale down for off-hours",
                estimated_impact={},
                timestamp=datetime.now(),
                policy=self.scaling_config.policy
            )
            
            await self._execute_scaling_decision(decision)
            
        except Exception as e:
            logger.error(f"Scheduled scale down failed: {e}")
    
    async def _ml_training_loop(self):
        """ML model training loop"""
        while self.auto_scaling_enabled:
            try:
                # Train models every hour if we have enough data
                if len(self.metrics_history) > 100:
                    await self._train_ml_models()
                
                await asyncio.sleep(3600)  # Train every hour
                
            except Exception as e:
                logger.error(f"ML training loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _train_ml_models(self):
        """Train ML models for predictive scaling"""
        try:
            if len(self.metrics_history) < 100:
                return
            
            # Prepare training data
            df = pd.DataFrame([asdict(metrics) for metrics in self.metrics_history[-500:]])
            
            # Features for demand prediction
            feature_columns = [
                'cpu_usage', 'memory_usage', 'request_rate', 
                'response_time', 'queue_size', 'error_rate',
                'active_connections', 'throughput'
            ]
            
            X = df[feature_columns].values
            y = df['request_rate'].shift(-1).dropna().values  # Predict next request rate
            
            if len(X) != len(y):
                X = X[:len(y)]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train demand predictor
            self.demand_predictor.fit(X_scaled, y)
            
            # Train anomaly detector
            self.anomaly_detector.fit(X_scaled)
            
            # Save models
            model_path = Path("models/scaling_models")
            model_path.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.demand_predictor, model_path / "demand_predictor.pkl")
            joblib.dump(self.anomaly_detector, model_path / "anomaly_detector.pkl")
            joblib.dump(self.scaler, model_path / "scaler.pkl")
            
            self.model_trained = True
            
            # Calculate prediction accuracy
            if len(self.scaling_decisions) > 10:
                accuracy = await self._calculate_prediction_accuracy()
                self.prometheus_metrics['prediction_accuracy'].set(accuracy)
            
            logger.info("ML models trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train ML models: {e}")
    
    async def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy"""
        try:
            # Simple accuracy calculation based on scaling decisions
            correct_predictions = 0
            total_predictions = len(self.scaling_decisions)
            
            for decision in self.scaling_decisions[-10:]:  # Last 10 decisions
                # Check if the scaling decision was correct
                # This is a simplified calculation
                if decision.confidence > 0.7:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            return accuracy
            
        except Exception as e:
            logger.error(f"Failed to calculate prediction accuracy: {e}")
            return 0.0
    
    async def _performance_tracking_loop(self):
        """Performance tracking loop"""
        while self.auto_scaling_enabled:
            try:
                # Track performance metrics
                current_replicas = await self._get_current_replicas()
                current_metrics = self.metrics_history[-1] if self.metrics_history else None
                
                if current_metrics:
                    performance_data = {
                        "timestamp": datetime.now(),
                        "replicas": current_replicas,
                        "cpu_usage": current_metrics.cpu_usage,
                        "memory_usage": current_metrics.memory_usage,
                        "response_time": current_metrics.response_time,
                        "throughput": current_metrics.throughput
                    }
                    
                    self.performance_history.append(performance_data)
                    
                    # Keep only last 1000 performance records
                    if len(self.performance_history) > 1000:
                        self.performance_history = self.performance_history[-1000:]
                
                await asyncio.sleep(300)  # Track every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance tracking loop error: {e}")
                await asyncio.sleep(300)
    
    async def _update_prometheus_metrics(self, metrics: ScalingMetrics):
        """Update Prometheus metrics"""
        try:
            current_replicas = await self._get_current_replicas()
            
            self.prometheus_metrics['current_replicas'].set(current_replicas)
            self.prometheus_metrics['target_replicas'].set(current_replicas)
            
            if self.scaling_decisions:
                last_decision = self.scaling_decisions[-1]
                self.prometheus_metrics['scaling_confidence'].set(last_decision.confidence)
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    async def get_scaling_dashboard(self) -> Dict[str, Any]:
        """Get auto-scaling dashboard data"""
        try:
            current_replicas = await self._get_current_replicas()
            current_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            dashboard = {
                "timestamp": datetime.now().isoformat(),
                "auto_scaling_enabled": self.auto_scaling_enabled,
                "scaling_active": self.scaling_active,
                "emergency_mode": self.emergency_mode,
                "current_replicas": current_replicas,
                "scaling_config": asdict(self.scaling_config),
                "current_metrics": asdict(current_metrics) if current_metrics else None,
                "recent_decisions": [asdict(d) for d in self.scaling_decisions[-10:]],
                "performance_summary": {
                    "total_scaling_actions": len(self.scaling_decisions),
                    "successful_scaling_actions": len([d for d in self.scaling_decisions if d.action != ScalingAction.NO_ACTION]),
                    "average_confidence": sum(d.confidence for d in self.scaling_decisions) / len(self.scaling_decisions) if self.scaling_decisions else 0,
                    "model_trained": self.model_trained
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to get scaling dashboard: {e}")
            return {}
    
    async def set_scaling_policy(self, policy: ScalingPolicy):
        """Set scaling policy"""
        try:
            self.scaling_config.policy = policy
            logger.info(f"Scaling policy set to: {policy.value}")
        except Exception as e:
            logger.error(f"Failed to set scaling policy: {e}")
    
    async def enable_emergency_mode(self):
        """Enable emergency mode (disable auto-scaling)"""
        try:
            self.emergency_mode = True
            logger.warning("Emergency mode enabled - auto-scaling disabled")
        except Exception as e:
            logger.error(f"Failed to enable emergency mode: {e}")
    
    async def disable_emergency_mode(self):
        """Disable emergency mode (enable auto-scaling)"""
        try:
            self.emergency_mode = False
            logger.info("Emergency mode disabled - auto-scaling enabled")
        except Exception as e:
            logger.error(f"Failed to disable emergency mode: {e}")
    
    async def close(self):
        """Close auto-scaling service"""
        try:
            self.auto_scaling_enabled = False
            
            # Stop schedule thread
            if self.schedule_thread:
                self.schedule_thread.join(timeout=5)
            
            # Close clients
            if self.redis_client:
                self.redis_client.close()
            
            if self.docker_client:
                self.docker_client.close()
            
            logger.info("Auto-scaling service closed")
            
        except Exception as e:
            logger.error(f"Error closing auto-scaling service: {e}")

# Global auto-scaling service instance
auto_scaling_service = None

async def initialize_auto_scaling_service(config: Optional[Dict] = None):
    """Initialize global auto-scaling service"""
    global auto_scaling_service
    auto_scaling_service = AdvancedAutoScalingService(config)
    await auto_scaling_service.initialize()
    return auto_scaling_service

async def get_auto_scaling_service() -> AdvancedAutoScalingService:
    """Get auto-scaling service instance"""
    if not auto_scaling_service:
        raise RuntimeError("Auto-scaling service not initialized")
    return auto_scaling_service














