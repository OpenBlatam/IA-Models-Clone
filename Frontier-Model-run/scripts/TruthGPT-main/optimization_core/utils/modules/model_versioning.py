"""
TruthGPT Model Versioning & A/B Testing Features
Advanced model versioning, A/B testing, and canary deployments for TruthGPT
"""

import asyncio
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import pickle
import threading
from datetime import datetime, timedelta
import uuid
import math
import random
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import heapq
import queue
import sqlite3
import hashlib
import shutil
import tempfile
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import TruthGPT modules
from .models import TruthGPTModel, TruthGPTModelConfig
from .advanced_security import TruthGPTSecurityManager, SecurityConfig
from .distributed_computing import DistributedCoordinator, DistributedWorker
from .real_time_computing import RealTimeManager, StreamProcessor


class ModelStatus(Enum):
    """Model status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class ExperimentStatus(Enum):
    """Experiment status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TrafficAllocation(Enum):
    """Traffic allocation strategies"""
    EQUAL = "equal"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"
    BANDIT = "bandit"
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"


class MetricType(Enum):
    """Metric types"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"
    BUSINESS_METRIC = "business_metric"
    CUSTOM = "custom"


class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"


@dataclass
class ModelVersion:
    """Model version"""
    version_id: str
    model_name: str
    version_number: str
    model_path: str
    config_path: str
    status: ModelStatus = ModelStatus.DEVELOPMENT
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    size_bytes: int = 0


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_id: str
    name: str
    description: str
    model_versions: List[str] = field(default_factory=list)
    traffic_allocation: TrafficAllocation = TrafficAllocation.EQUAL
    traffic_weights: Dict[str, float] = field(default_factory=dict)
    metrics: List[MetricType] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    duration_hours: float = 24.0
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    enable_early_stopping: bool = True
    enable_adaptive_allocation: bool = True
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    target_audience: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Experiment result"""
    experiment_id: str
    model_version: str
    user_id: str
    timestamp: float = field(default_factory=time.time)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class ModelRegistryConfig:
    """Model registry configuration"""
    registry_path: str = "./model_registry"
    enable_versioning: bool = True
    enable_experiments: bool = True
    enable_canary_deployments: bool = True
    max_versions_per_model: int = 10
    enable_automatic_cleanup: bool = True
    cleanup_threshold_days: int = 30
    enable_checksums: bool = True
    enable_metadata_tracking: bool = True
    enable_dependency_tracking: bool = True


class ModelRegistry:
    """Model registry for TruthGPT"""
    
    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self.logger = logging.getLogger(f"ModelRegistry_{id(self)}")
        
        # Registry data
        self.models: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        
        # Database
        self.db_path = os.path.join(config.registry_path, "registry.db")
        self._init_database()
        
        # Load existing data
        self._load_registry_data()
    
    def _init_database(self):
        """Initialize registry database"""
        os.makedirs(self.config.registry_path, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                    version_number TEXT NOT NULL,
                model_path TEXT NOT NULL,
                config_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                created_by TEXT,
                description TEXT,
                tags TEXT,
                metrics TEXT,
                metadata TEXT,
                dependencies TEXT,
                checksum TEXT,
                size_bytes INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    model_versions TEXT NOT NULL,
                    traffic_allocation TEXT NOT NULL,
                traffic_weights TEXT,
                metrics TEXT,
                success_criteria TEXT,
                duration_hours REAL,
                min_sample_size INTEGER,
                confidence_level REAL,
                statistical_power REAL,
                enable_early_stopping BOOLEAN,
                enable_adaptive_allocation BOOLEAN,
                    created_at REAL NOT NULL,
                created_by TEXT,
                target_audience TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiment_results (
                    result_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                user_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                input_data TEXT,
                output_data TEXT,
                metrics TEXT,
                processing_time REAL,
                error TEXT
                )
            ''')
            
        conn.commit()
        conn.close()
    
    def _load_registry_data(self):
        """Load registry data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load model versions
        cursor.execute('SELECT * FROM model_versions')
        for row in cursor.fetchall():
            version = ModelVersion(
                version_id=row[0],
                model_name=row[1],
                version_number=row[2],
                model_path=row[3],
                config_path=row[4],
                status=ModelStatus(row[5]),
                created_at=row[6],
                created_by=row[7] or "",
                description=row[8] or "",
                tags=json.loads(row[9]) if row[9] else [],
                metrics=json.loads(row[10]) if row[10] else {},
                metadata=json.loads(row[11]) if row[11] else {},
                dependencies=json.loads(row[12]) if row[12] else [],
                checksum=row[13] or "",
                size_bytes=row[14] or 0
            )
            self.models[version.model_name][version.version_number] = version
        
        # Load experiments
        cursor.execute('SELECT * FROM experiments')
        for row in cursor.fetchall():
            experiment = ExperimentConfig(
                experiment_id=row[0],
                name=row[1],
                description=row[2] or "",
                model_versions=json.loads(row[3]) if row[3] else [],
                traffic_allocation=TrafficAllocation(row[4]),
                traffic_weights=json.loads(row[5]) if row[5] else {},
                metrics=[MetricType(m) for m in json.loads(row[6])] if row[6] else [],
                success_criteria=json.loads(row[7]) if row[7] else {},
                duration_hours=row[8] or 24.0,
                min_sample_size=row[9] or 1000,
                confidence_level=row[10] or 0.95,
                statistical_power=row[11] or 0.8,
                enable_early_stopping=bool(row[12]),
                enable_adaptive_allocation=bool(row[13]),
                created_at=row[14],
                created_by=row[15] or "",
                target_audience=json.loads(row[16]) if row[16] else {}
            )
            self.experiments[experiment.experiment_id] = experiment
        
        conn.close()
    
    def register_model(self, model: TruthGPTModel, model_config: TruthGPTModelConfig,
                      model_name: str, version_number: str, created_by: str = "",
                      description: str = "", tags: List[str] = None) -> ModelVersion:
        """Register model version"""
            version_id = str(uuid.uuid4())
            
        # Create version directory
        version_dir = os.path.join(self.config.registry_path, model_name, version_number)
        os.makedirs(version_dir, exist_ok=True)
            
        # Save model
        model_path = os.path.join(version_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            
        # Save config
        config_path = os.path.join(version_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config.__dict__, f, indent=2)
        
        # Calculate checksum
        checksum = self._calculate_checksum(model_path)
        
        # Get file size
        size_bytes = os.path.getsize(model_path)
        
        # Create model version
        version = ModelVersion(
                version_id=version_id,
            model_name=model_name,
                version_number=version_number,
            model_path=model_path,
            config_path=config_path,
            status=ModelStatus.DEVELOPMENT,
                created_by=created_by,
            description=description,
            tags=tags or [],
            checksum=checksum,
            size_bytes=size_bytes
        )
        
        # Store in registry
        self.models[model_name][version_number] = version
        self._save_model_version(version)
        
        self.logger.info(f"Registered model {model_name} version {version_number}")
        return version
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _save_model_version(self, version: ModelVersion):
        """Save model version to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO model_versions 
            (version_id, model_name, version_number, model_path, config_path, status,
             created_at, created_by, description, tags, metrics, metadata, dependencies,
             checksum, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            version.version_id, version.model_name, version.version_number,
            version.model_path, version.config_path, version.status.value,
            version.created_at, version.created_by, version.description,
            json.dumps(version.tags), json.dumps(version.metrics),
            json.dumps(version.metadata), json.dumps(version.dependencies),
            version.checksum, version.size_bytes
        ))
        
        conn.commit()
        conn.close()
    
    def get_model_version(self, model_name: str, version_number: str) -> Optional[ModelVersion]:
        """Get model version"""
        return self.models.get(model_name, {}).get(version_number)
    
    def list_model_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model"""
        return list(self.models.get(model_name, {}).values())
    
    def update_model_status(self, model_name: str, version_number: str, status: ModelStatus):
        """Update model status"""
        version = self.get_model_version(model_name, version_number)
        if version:
            version.status = status
            self._save_model_version(version)
    
    def delete_model_version(self, model_name: str, version_number: str) -> bool:
        """Delete model version"""
        version = self.get_model_version(model_name, version_number)
        if not version:
            return False
        
        # Delete files
        try:
            shutil.rmtree(os.path.dirname(version.model_path))
        except Exception as e:
            self.logger.error(f"Failed to delete model files: {e}")
        
        # Remove from registry
        del self.models[model_name][version_number]
        
        # Remove from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM model_versions WHERE version_id = ?', (version.version_id,))
        conn.commit()
        conn.close()
        
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_models = len(self.models)
        total_versions = sum(len(versions) for versions in self.models.values())
        total_experiments = len(self.experiments)
        
        return {
            "config": self.config.__dict__,
            "total_models": total_models,
            "total_versions": total_versions,
            "total_experiments": total_experiments,
            "models": list(self.models.keys())
        }


class ExperimentManager:
    """Experiment manager for TruthGPT"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.logger = logging.getLogger(f"ExperimentManager_{id(self)}")
        
        # Experiment state
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        
        # Statistical analysis
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Traffic allocation
        self.traffic_allocator = TrafficAllocator()
    
    def create_experiment(self, name: str, description: str, model_versions: List[str],
                         traffic_allocation: TrafficAllocation = TrafficAllocation.EQUAL,
                         traffic_weights: Dict[str, float] = None,
                         metrics: List[MetricType] = None,
                         success_criteria: Dict[str, float] = None,
                         duration_hours: float = 24.0,
                         created_by: str = "") -> ExperimentConfig:
        """Create new experiment"""
        experiment_id = str(uuid.uuid4())
        
            # Validate model versions
        for version_id in model_versions:
            if not self._validate_model_version(version_id):
                raise ValueError(f"Invalid model version: {version_id}")
        
        # Create experiment config
        experiment = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            model_versions=model_versions,
            traffic_allocation=traffic_allocation,
            traffic_weights=traffic_weights or {},
            metrics=metrics or [MetricType.ACCURACY, MetricType.LATENCY],
            success_criteria=success_criteria or {},
            duration_hours=duration_hours,
            created_by=created_by
        )
            
            # Store experiment
        self.model_registry.experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        self.logger.info(f"Created experiment {name} with ID {experiment_id}")
        return experiment
    
    def _validate_model_version(self, version_id: str) -> bool:
        """Validate model version exists"""
        # Simplified validation - in real implementation, check registry
        return True
    
    def _save_experiment(self, experiment: ExperimentConfig):
        """Save experiment to database"""
        conn = sqlite3.connect(self.model_registry.db_path)
        cursor = conn.cursor()
        
            cursor.execute('''
            INSERT OR REPLACE INTO experiments 
                (experiment_id, name, description, model_versions, traffic_allocation,
             traffic_weights, metrics, success_criteria, duration_hours, min_sample_size,
             confidence_level, statistical_power, enable_early_stopping, enable_adaptive_allocation,
             created_at, created_by, target_audience)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
            experiment.experiment_id, experiment.name, experiment.description,
            json.dumps(experiment.model_versions), experiment.traffic_allocation.value,
            json.dumps(experiment.traffic_weights), json.dumps([m.value for m in experiment.metrics]),
            json.dumps(experiment.success_criteria), experiment.duration_hours,
            experiment.min_sample_size, experiment.confidence_level, experiment.statistical_power,
            experiment.enable_early_stopping, experiment.enable_adaptive_allocation,
            experiment.created_at, experiment.created_by, json.dumps(experiment.target_audience)
        ))
        
        conn.commit()
        conn.close()
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start experiment"""
        experiment = self.model_registry.experiments.get(experiment_id)
        if not experiment:
            return False
        
        # Initialize traffic allocation
        self.traffic_allocator.initialize_experiment(experiment)
        
        # Mark as running
        experiment.status = ExperimentStatus.RUNNING
        self.active_experiments[experiment_id] = experiment
        
        self.logger.info(f"Started experiment {experiment.name}")
        return True
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop experiment"""
            if experiment_id not in self.active_experiments:
            return False
            
            experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        
        # Remove from active experiments
        del self.active_experiments[experiment_id]
        
        # Analyze results
        await self._analyze_experiment_results(experiment_id)
        
        self.logger.info(f"Stopped experiment {experiment.name}")
            return True
            
    def assign_user_to_experiment(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Assign user to experiment variant"""
            if experiment_id not in self.active_experiments:
            return None
            
            experiment = self.active_experiments[experiment_id]
            
        # Get traffic allocation
        variant = self.traffic_allocator.assign_user(experiment, user_id)
        
        return variant
    
    def record_experiment_result(self, experiment_id: str, model_version: str,
                                user_id: str, input_data: Dict[str, Any],
                                output_data: Dict[str, Any], metrics: Dict[str, float],
                                processing_time: float = 0.0, error: str = None):
        """Record experiment result"""
            result = ExperimentResult(
                experiment_id=experiment_id,
            model_version=model_version,
            user_id=user_id,
            input_data=input_data,
            output_data=output_data,
                metrics=metrics,
            processing_time=processing_time,
            error=error
        )
        
        self.experiment_results[experiment_id].append(result)
        self._save_experiment_result(result)
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to database"""
        conn = sqlite3.connect(self.model_registry.db_path)
        cursor = conn.cursor()
        
        result_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO experiment_results 
            (result_id, experiment_id, model_version, user_id, timestamp,
             input_data, output_data, metrics, processing_time, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
            result_id, result.experiment_id, result.model_version, result.user_id,
            result.timestamp, json.dumps(result.input_data), json.dumps(result.output_data),
            json.dumps(result.metrics), result.processing_time, result.error
        ))
        
        conn.commit()
        conn.close()
    
    async def _analyze_experiment_results(self, experiment_id: str):
        """Analyze experiment results"""
        results = self.experiment_results.get(experiment_id, [])
        if not results:
            return
        
        # Group results by model version
        results_by_version = defaultdict(list)
        for result in results:
            results_by_version[result.model_version].append(result)
        
        # Analyze each variant
        analysis_results = {}
        for version, version_results in results_by_version.items():
            analysis = self.statistical_analyzer.analyze_results(version_results)
            analysis_results[version] = analysis
        
        self.logger.info(f"Experiment {experiment_id} analysis completed")
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get experiment statistics"""
        return {
            "active_experiments": len(self.active_experiments),
            "total_experiments": len(self.model_registry.experiments),
            "total_results": sum(len(results) for results in self.experiment_results.values()),
            "active_experiment_ids": list(self.active_experiments.keys())
        }


class StatisticalAnalyzer:
    """Statistical analyzer for experiments"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"StatisticalAnalyzer_{id(self)}")
    
    def analyze_results(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze experiment results"""
        if not results:
            return {}
        
        # Extract metrics
        metrics_data = defaultdict(list)
        processing_times = []
        error_count = 0
        
        for result in results:
            for metric, value in result.metrics.items():
                metrics_data[metric].append(value)
            
            processing_times.append(result.processing_time)
            
            if result.error:
                error_count += 1
        
        # Calculate statistics
        analysis = {
            "total_samples": len(results),
            "error_count": error_count,
            "error_rate": error_count / len(results),
            "processing_time": {
                "mean": np.mean(processing_times),
                "std": np.std(processing_times),
                "min": np.min(processing_times),
                "max": np.max(processing_times)
            }
        }
        
        # Calculate metric statistics
        for metric, values in metrics_data.items():
            analysis[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        return analysis
    
    def calculate_statistical_significance(self, control_results: List[ExperimentResult],
                                         treatment_results: List[ExperimentResult],
                                         metric: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Calculate statistical significance"""
        # Extract metric values
        control_values = [r.metrics.get(metric, 0) for r in control_results]
        treatment_values = [r.metrics.get(metric, 0) for r in treatment_results]
        
        if not control_values or not treatment_values:
            return {"significant": False, "p_value": 1.0}
        
        # Perform t-test (simplified)
        from scipy import stats
        
        try:
            t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
            
            return {
                "significant": p_value < alpha,
                "p_value": p_value,
                "t_statistic": t_stat,
                "control_mean": np.mean(control_values),
                "treatment_mean": np.mean(treatment_values),
                "effect_size": (np.mean(treatment_values) - np.mean(control_values)) / np.std(control_values)
            }
        except Exception as e:
            self.logger.error(f"Statistical analysis error: {e}")
            return {"significant": False, "p_value": 1.0}


class TrafficAllocator:
    """Traffic allocator for experiments"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"TrafficAllocator_{id(self)}")
        self.experiment_allocations: Dict[str, Dict[str, float]] = {}
    
    def initialize_experiment(self, experiment: ExperimentConfig):
        """Initialize traffic allocation for experiment"""
        if experiment.traffic_allocation == TrafficAllocation.EQUAL:
            # Equal allocation
            weight_per_version = 1.0 / len(experiment.model_versions)
            allocation = {version: weight_per_version for version in experiment.model_versions}
        elif experiment.traffic_allocation == TrafficAllocation.WEIGHTED:
            # Weighted allocation
            total_weight = sum(experiment.traffic_weights.values())
            allocation = {version: weight / total_weight 
                         for version, weight in experiment.traffic_weights.items()}
            else:
            # Default to equal allocation
            weight_per_version = 1.0 / len(experiment.model_versions)
            allocation = {version: weight_per_version for version in experiment.model_versions}
        
        self.experiment_allocations[experiment.experiment_id] = allocation
    
    def assign_user(self, experiment: ExperimentConfig, user_id: str) -> str:
        """Assign user to experiment variant"""
        allocation = self.experiment_allocations.get(experiment.experiment_id, {})
        
        if not allocation:
            # Default to first version
            return experiment.model_versions[0]
        
        # Use user ID hash for consistent assignment
        user_hash = hash(user_id) % 10000
        cumulative_prob = 0
        
        for version, weight in allocation.items():
            cumulative_prob += weight * 10000
            if user_hash < cumulative_prob:
                return version
        
        # Fallback to first version
        return experiment.model_versions[0]


class CanaryDeploymentManager:
    """Canary deployment manager for TruthGPT"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.logger = logging.getLogger(f"CanaryDeploymentManager_{id(self)}")
        
        # Deployment state
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.deployment_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def create_canary_deployment(self, model_name: str, new_version: str,
                               current_version: str, traffic_percentage: float = 10.0,
                               success_criteria: Dict[str, float] = None,
                               rollback_threshold: float = 0.05) -> str:
        """Create canary deployment"""
        deployment_id = str(uuid.uuid4())
        
        deployment = {
            "deployment_id": deployment_id,
            "model_name": model_name,
            "new_version": new_version,
            "current_version": current_version,
            "traffic_percentage": traffic_percentage,
            "success_criteria": success_criteria or {},
            "rollback_threshold": rollback_threshold,
            "status": "active",
            "created_at": time.time(),
            "metrics": defaultdict(list)
        }
        
        self.active_deployments[deployment_id] = deployment
        
        self.logger.info(f"Created canary deployment {deployment_id} for {model_name}")
        return deployment_id
    
    def update_deployment_metrics(self, deployment_id: str, metrics: Dict[str, float]):
        """Update deployment metrics"""
        if deployment_id not in self.active_deployments:
            return
        
        deployment = self.active_deployments[deployment_id]
        
        # Add timestamp to metrics
        metrics_with_time = {
            **metrics,
            "timestamp": time.time()
        }
        
        deployment["metrics"].append(metrics_with_time)
        self.deployment_metrics[deployment_id].append(metrics_with_time)
        
        # Check rollback conditions
        if self._should_rollback(deployment):
            self._rollback_deployment(deployment_id)
    
    def _should_rollback(self, deployment: Dict[str, Any]) -> bool:
        """Check if deployment should be rolled back"""
        metrics = deployment["metrics"]
        if len(metrics) < 10:  # Need minimum samples
            return False
    
        # Check error rate
        error_rates = [m.get("error_rate", 0) for m in metrics[-10:]]
        avg_error_rate = np.mean(error_rates)
        
        if avg_error_rate > deployment["rollback_threshold"]:
            return True
        
        # Check latency
        latencies = [m.get("latency", 0) for m in metrics[-10:]]
        if latencies:
            avg_latency = np.mean(latencies)
            if avg_latency > 1000:  # 1 second threshold
                return True
        
        return False
    
    def _rollback_deployment(self, deployment_id: str):
        """Rollback deployment"""
        deployment = self.active_deployments[deployment_id]
        deployment["status"] = "rolled_back"
        
        self.logger.warning(f"Rolled back deployment {deployment_id}")
    
    def promote_deployment(self, deployment_id: str) -> bool:
        """Promote deployment to full traffic"""
        if deployment_id not in self.active_deployments:
            return False
        
        deployment = self.active_deployments[deployment_id]
        deployment["status"] = "promoted"
        deployment["traffic_percentage"] = 100.0
        
        self.logger.info(f"Promoted deployment {deployment_id}")
        return True
    
    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        return {
            "active_deployments": len(self.active_deployments),
            "total_deployments": len(self.deployment_metrics),
            "deployment_ids": list(self.active_deployments.keys())
        }


class TruthGPTVersioningManager:
    """Unified versioning manager for TruthGPT"""
    
    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self.logger = logging.getLogger(f"TruthGPTVersioningManager_{id(self)}")
        
        # Core components
        self.model_registry = ModelRegistry(config)
        self.experiment_manager = ExperimentManager(self.model_registry)
        self.canary_manager = CanaryDeploymentManager(self.model_registry)
        
        # Integration components
        self.security_manager: Optional[TruthGPTSecurityManager] = None
        self.distributed_coordinator: Optional[DistributedCoordinator] = None
        self.real_time_manager: Optional[RealTimeManager] = None
    
    def set_security_manager(self, security_manager: TruthGPTSecurityManager):
        """Set security manager"""
        self.security_manager = security_manager
    
    def set_distributed_coordinator(self, coordinator: DistributedCoordinator):
        """Set distributed coordinator"""
        self.distributed_coordinator = coordinator
    
    def set_real_time_manager(self, manager: RealTimeManager):
        """Set real-time manager"""
        self.real_time_manager = manager
    
    def register_model(self, model: TruthGPTModel, model_config: TruthGPTModelConfig,
                      model_name: str, version_number: str, created_by: str = "",
                      description: str = "", tags: List[str] = None) -> ModelVersion:
        """Register model with security"""
        # Encrypt model if security manager is available
        if self.security_manager:
            encrypted_model = self.security_manager.encrypt_model(model)
            # Save encrypted model (simplified)
            model_path = f"./encrypted_models/{model_name}_{version_number}.enc"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                f.write(encrypted_model)
        
        return self.model_registry.register_model(
            model, model_config, model_name, version_number,
            created_by, description, tags
        )
    
    def create_experiment(self, name: str, description: str, model_versions: List[str],
                         **kwargs) -> ExperimentConfig:
        """Create experiment"""
        return self.experiment_manager.create_experiment(
            name, description, model_versions, **kwargs
        )
    
    def create_canary_deployment(self, model_name: str, new_version: str,
                               current_version: str, **kwargs) -> str:
        """Create canary deployment"""
        return self.canary_manager.create_canary_deployment(
            model_name, new_version, current_version, **kwargs
        )
    
    def get_versioning_stats(self) -> Dict[str, Any]:
        """Get versioning statistics"""
        return {
            "config": self.config.__dict__,
            "registry_stats": self.model_registry.get_registry_stats(),
            "experiment_stats": self.experiment_manager.get_experiment_stats(),
            "deployment_stats": self.canary_manager.get_deployment_stats()
        }


def create_model_registry_config(registry_path: str = "./model_registry") -> ModelRegistryConfig:
    """Create model registry configuration"""
    return ModelRegistryConfig(registry_path=registry_path)


def create_model_version(model_name: str, version_number: str, model_path: str,
                        config_path: str) -> ModelVersion:
    """Create model version"""
    return ModelVersion(
        version_id=str(uuid.uuid4()),
        model_name=model_name,
        version_number=version_number,
        model_path=model_path,
        config_path=config_path
    )


def create_experiment_config(name: str, description: str, model_versions: List[str]) -> ExperimentConfig:
    """Create experiment configuration"""
    return ExperimentConfig(
        experiment_id=str(uuid.uuid4()),
        name=name,
        description=description,
        model_versions=model_versions
    )


def create_model_registry(config: ModelRegistryConfig) -> ModelRegistry:
    """Create model registry"""
    return ModelRegistry(config)


def create_experiment_manager(model_registry: ModelRegistry) -> ExperimentManager:
    """Create experiment manager"""
    return ExperimentManager(model_registry)


def create_canary_deployment_manager(model_registry: ModelRegistry) -> CanaryDeploymentManager:
    """Create canary deployment manager"""
    return CanaryDeploymentManager(model_registry)


def create_versioning_manager(config: ModelRegistryConfig) -> TruthGPTVersioningManager:
    """Create versioning manager"""
    return TruthGPTVersioningManager(config)


def quick_versioning_setup(model_name: str = "truthgpt_model") -> TruthGPTVersioningManager:
    """Quick setup for versioning"""
    config = ModelRegistryConfig(registry_path=f"./registry_{model_name}")
    return TruthGPTVersioningManager(config)


# Example usage
if __name__ == "__main__":
    async def main():
    # Create versioning manager
        versioning_manager = quick_versioning_setup("example_model")
        
        # Register a model
        model_config = TruthGPTModelConfig()
        model = TruthGPTModel(model_config)
        
        version = versioning_manager.register_model(
            model, model_config, "example_model", "v1.0.0",
            created_by="developer", description="Initial version"
        )
        
        # Create experiment
        experiment = versioning_manager.create_experiment(
            "A/B Test", "Testing new model version",
            [version.version_id]
        )
        
        # Start experiment
        await versioning_manager.experiment_manager.start_experiment(experiment.experiment_id)
    
    # Create canary deployment
        deployment_id = versioning_manager.create_canary_deployment(
            "example_model", "v1.1.0", "v1.0.0", traffic_percentage=5.0
        )
        
        # Get stats
        stats = versioning_manager.get_versioning_stats()
        print(f"Versioning stats: {stats}")
    
    # Run example
    asyncio.run(main())