"""
TruthGPT Model Versioning and A/B Testing
Advanced model versioning, A/B testing, and experiment management for enterprise deployments
"""

import time
import json
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import sqlite3
import threading
from datetime import datetime, timedelta
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiofiles
import yaml
import pickle
from contextlib import contextmanager

# Import TruthGPT modules
from .models import TruthGPTModel, TruthGPTModelConfig
from .inference import TruthGPTInference, TruthGPTInferenceConfig
from .evaluation import TruthGPTEvaluator, TruthGPTEvaluationConfig
from .analytics import TruthGPTAnalyticsManager
from .config import TruthGPTConfigManager


class ModelStatus(Enum):
    """Model deployment status"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ExperimentStatus(Enum):
    """A/B test experiment status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficAllocation(Enum):
    """Traffic allocation strategies"""
    RANDOM = "random"
    USER_ID_HASH = "user_id_hash"
    SESSION_ID_HASH = "session_id_hash"
    CUSTOM = "custom"


class MetricType(Enum):
    """Metric types for evaluation"""
    ACCURACY = "accuracy"
    PERPLEXITY = "perplexity"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    USER_SATISFACTION = "user_satisfaction"
    CUSTOM = "custom"


@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_path: str
    config: TruthGPTModelConfig
    created_at: datetime
    created_by: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.DEVELOPMENT
    metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration"""
    experiment_id: str
    name: str
    description: str
    variants: List[Dict[str, Any]]  # List of model versions and their configs
    traffic_allocation: Dict[str, float]  # Traffic percentage per variant
    allocation_strategy: TrafficAllocation = TrafficAllocation.RANDOM
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    success_metrics: List[MetricType] = field(default_factory=lambda: [MetricType.ACCURACY])
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """A/B test experiment results"""
    experiment_id: str
    variant_results: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, float]
    winner: Optional[str] = None
    confidence_interval: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelRegistryConfig:
    """Configuration for model registry"""
    registry_path: str = "./model_registry"
    max_versions_per_model: int = 50
    enable_automatic_cleanup: bool = True
    cleanup_retention_days: int = 30
    enable_metrics_tracking: bool = True
    enable_experiment_tracking: bool = True
    database_path: str = "./model_registry.db"
    backup_interval_hours: int = 24


class TruthGPTModelRegistry:
    """Advanced model versioning and registry system"""
    
    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self.logger = logging.getLogger(f"TruthGPTModelRegistry_{id(self)}")
        self.registry_path = Path(config.registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load existing models
        self._models: Dict[str, List[ModelVersion]] = {}
        self._load_registry()
        
        # Start background tasks
        if config.enable_automatic_cleanup:
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
        
        if config.enable_metrics_tracking:
            self._metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
            self._metrics_thread.start()
    
    def _init_database(self):
        """Initialize SQLite database for model registry"""
        self.db_path = Path(self.config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT,
                    created_by TEXT,
                    current_version TEXT,
                    status TEXT
                )
            """)
            
            # Versions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    model_path TEXT,
                    config TEXT,
                    created_at TEXT,
                    created_by TEXT,
                    description TEXT,
                    tags TEXT,
                    status TEXT,
                    metrics TEXT,
                    dependencies TEXT,
                    metadata TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)
            
            # Experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    config TEXT,
                    status TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    results TEXT,
                    created_at TEXT,
                    created_by TEXT
                )
            """)
            
            # Experiment metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    variant TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            conn.commit()
    
    def _load_registry(self):
        """Load existing models from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT v.*, m.name as model_name 
                    FROM versions v 
                    JOIN models m ON v.model_id = m.model_id
                    ORDER BY v.created_at DESC
                """)
                
                for row in cursor.fetchall():
                    version_id, model_id, model_path, config_json, created_at, created_by, description, tags_json, status, metrics_json, dependencies_json, metadata_json, model_name = row
                    
                    version = ModelVersion(
                        version_id=version_id,
                        model_path=model_path,
                        config=TruthGPTModelConfig.from_dict(json.loads(config_json)),
                        created_at=datetime.fromisoformat(created_at),
                        created_by=created_by,
                        description=description,
                        tags=json.loads(tags_json) if tags_json else [],
                        status=ModelStatus(status),
                        metrics=json.loads(metrics_json) if metrics_json else {},
                        dependencies=json.loads(dependencies_json) if dependencies_json else [],
                        metadata=json.loads(metadata_json) if metadata_json else {}
                    )
                    
                    if model_id not in self._models:
                        self._models[model_id] = []
                    self._models[model_id].append(version)
                    
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
    
    def register_model(self, model_id: str, name: str, description: str = "",
                      created_by: str = "system") -> str:
        """Register a new model"""
        if model_id in self._models:
            raise ValueError(f"Model {model_id} already exists")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO models (model_id, name, description, created_at, created_by, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_id, name, description, datetime.now().isoformat(), created_by, ModelStatus.DEVELOPMENT.value))
            conn.commit()
        
        self._models[model_id] = []
        self.logger.info(f"Registered model {model_id}: {name}")
        return model_id
    
    def create_version(self, model_id: str, model: TruthGPTModel, 
                      description: str = "", tags: List[str] = None,
                      created_by: str = "system") -> str:
        """Create a new model version"""
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")
        
        version_id = str(uuid.uuid4())
        model_dir = self.registry_path / model_id / version_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save config
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(model.config.to_dict(), f, indent=2)
        
        # Create version metadata
        version = ModelVersion(
            version_id=version_id,
            model_path=str(model_path),
            config=model.config,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            tags=tags or [],
            status=ModelStatus.DEVELOPMENT
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO versions 
                (version_id, model_id, model_path, config, created_at, created_by, 
                 description, tags, status, metrics, dependencies, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version_id, model_id, str(model_path), json.dumps(model.config.to_dict()),
                version.created_at.isoformat(), created_by, description,
                json.dumps(tags or []), ModelStatus.DEVELOPMENT.value,
                json.dumps({}), json.dumps([]), json.dumps({})
            ))
            conn.commit()
        
        self._models[model_id].append(version)
        self.logger.info(f"Created version {version_id} for model {model_id}")
        return version_id
    
    def load_version(self, model_id: str, version_id: str) -> TruthGPTModel:
        """Load a specific model version"""
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")
        
        version = None
        for v in self._models[model_id]:
            if v.version_id == version_id:
                version = v
                break
        
        if not version:
            raise ValueError(f"Version {version_id} not found for model {model_id}")
        
        # Load model
        model = TruthGPTModel(version.config)
        model.load_state_dict(torch.load(version.model_path))
        
        return model
    
    def promote_version(self, model_id: str, version_id: str, 
                      target_status: ModelStatus, created_by: str = "system"):
        """Promote a model version to a new status"""
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")
        
        version = None
        for v in self._models[model_id]:
            if v.version_id == version_id:
                version = v
                break
        
        if not version:
            raise ValueError(f"Version {version_id} not found for model {model_id}")
        
        # Update status
        version.status = target_status
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE versions SET status = ? WHERE version_id = ?
            """, (target_status.value, version_id))
            
            if target_status == ModelStatus.PRODUCTION:
                conn.execute("""
                    UPDATE models SET current_version = ? WHERE model_id = ?
                """, (version_id, model_id))
            
            conn.commit()
        
        self.logger.info(f"Promoted version {version_id} to {target_status.value}")
    
    def get_model_versions(self, model_id: str, status: Optional[ModelStatus] = None) -> List[ModelVersion]:
        """Get all versions for a model, optionally filtered by status"""
        if model_id not in self._models:
            return []
        
        versions = self._models[model_id]
        if status:
            versions = [v for v in versions if v.status == status]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def compare_versions(self, model_id: str, version_ids: List[str], 
                       test_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compare multiple model versions"""
        results = {}
        
        for version_id in version_ids:
            try:
                model = self.load_version(model_id, version_id)
                
                # Create evaluator
                eval_config = TruthGPTEvaluationConfig(
                    batch_size=32,
                    device="cpu",
                    metrics=["perplexity", "accuracy"]
                )
                evaluator = TruthGPTEvaluator(model, eval_config)
                
                # Evaluate model
                metrics = evaluator.evaluate(test_data)
                results[version_id] = metrics
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate version {version_id}: {e}")
                results[version_id] = {"error": str(e)}
        
        return results
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self._cleanup_old_versions()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def _cleanup_old_versions(self):
        """Clean up old model versions"""
        cutoff_date = datetime.now() - timedelta(days=self.config.cleanup_retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT version_id, model_id, model_path 
                FROM versions 
                WHERE created_at < ? AND status = ?
            """, (cutoff_date.isoformat(), ModelStatus.ARCHIVED.value))
            
            for version_id, model_id, model_path in cursor.fetchall():
                try:
                    # Remove files
                    if Path(model_path).exists():
                        shutil.rmtree(Path(model_path).parent)
                    
                    # Remove from database
                    conn.execute("DELETE FROM versions WHERE version_id = ?", (version_id,))
                    
                    # Remove from memory
                    if model_id in self._models:
                        self._models[model_id] = [
                            v for v in self._models[model_id] if v.version_id != version_id
                        ]
                    
                    self.logger.info(f"Cleaned up old version {version_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to cleanup version {version_id}: {e}")
            
            conn.commit()
    
    def _metrics_loop(self):
        """Background metrics collection loop"""
        while True:
            try:
                time.sleep(1800)  # Run every 30 minutes
                self._collect_model_metrics()
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    def _collect_model_metrics(self):
        """Collect metrics for all production models"""
        # This would integrate with monitoring systems
        # For now, just log the activity
        self.logger.debug("Collecting model metrics")


class TruthGPTExperimentManager:
    """A/B testing and experiment management for TruthGPT"""
    
    def __init__(self, registry: TruthGPTModelRegistry):
        self.registry = registry
        self.logger = logging.getLogger(f"TruthGPTExperimentManager_{id(self)}")
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._results: Dict[str, ExperimentResult] = {}
        self._active_experiments: Dict[str, ExperimentConfig] = {}
        
        # Load existing experiments
        self._load_experiments()
    
    def _load_experiments(self):
        """Load existing experiments from database"""
        try:
            with sqlite3.connect(self.registry.db_path) as conn:
                cursor = conn.execute("SELECT experiment_id, config FROM experiments")
                
                for experiment_id, config_json in cursor.fetchall():
                    config_dict = json.loads(config_json)
                    config = ExperimentConfig(**config_dict)
                    self._experiments[experiment_id] = config
                    
                    if config.start_date and config.end_date and config.start_date <= datetime.now() <= config.end_date:
                        self._active_experiments[experiment_id] = config
                        
        except Exception as e:
            self.logger.error(f"Failed to load experiments: {e}")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment"""
        experiment_id = str(uuid.uuid4())
        config.experiment_id = experiment_id
        
        # Validate variants
        total_allocation = sum(config.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError("Traffic allocation must sum to 1.0")
        
        # Save to database
        with sqlite3.connect(self.registry.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments 
                (experiment_id, name, description, config, status, created_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, config.name, config.description,
                json.dumps(config.__dict__), ExperimentStatus.DRAFT.value,
                datetime.now().isoformat(), "system"
            ))
            conn.commit()
        
        self._experiments[experiment_id] = config
        self.logger.info(f"Created experiment {experiment_id}: {config.name}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str):
        """Start an A/B test experiment"""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self._experiments[experiment_id]
        config.start_date = datetime.now()
        config.status = ExperimentStatus.RUNNING
        
        # Update database
        with sqlite3.connect(self.registry.db_path) as conn:
            conn.execute("""
                UPDATE experiments 
                SET status = ?, start_date = ?, config = ?
                WHERE experiment_id = ?
            """, (
                ExperimentStatus.RUNNING.value, config.start_date.isoformat(),
                json.dumps(config.__dict__), experiment_id
            ))
            conn.commit()
        
        self._active_experiments[experiment_id] = config
        self.logger.info(f"Started experiment {experiment_id}")
    
    def get_variant_for_user(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Get variant assignment for a user"""
        if experiment_id not in self._active_experiments:
            return None
        
        config = self._active_experiments[experiment_id]
        
        if config.allocation_strategy == TrafficAllocation.USER_ID_HASH:
            # Use user ID hash for consistent assignment
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            random_value = (hash_value % 10000) / 10000.0
        else:
            # Random assignment
            random_value = np.random.random()
        
        # Assign variant based on traffic allocation
        cumulative = 0.0
        for variant, allocation in config.traffic_allocation.items():
            cumulative += allocation
            if random_value <= cumulative:
                return variant
        
        return None
    
    def record_metric(self, experiment_id: str, variant: str, metric_name: str,
                     metric_value: float, user_id: str = None, session_id: str = None):
        """Record a metric for an experiment variant"""
        with sqlite3.connect(self.registry.db_path) as conn:
            conn.execute("""
                INSERT INTO experiment_metrics 
                (experiment_id, variant, metric_name, metric_value, timestamp, user_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, variant, metric_name, metric_value,
                datetime.now().isoformat(), user_id, session_id
            ))
            conn.commit()
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Analyze experiment results and determine statistical significance"""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self._experiments[experiment_id]
        
        # Get metrics from database
        with sqlite3.connect(self.registry.db_path) as conn:
            cursor = conn.execute("""
                SELECT variant, metric_name, metric_value 
                FROM experiment_metrics 
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            metrics_data = cursor.fetchall()
        
        # Organize metrics by variant
        variant_metrics = {}
        for variant, metric_name, metric_value in metrics_data:
            if variant not in variant_metrics:
                variant_metrics[variant] = {}
            if metric_name not in variant_metrics[variant]:
                variant_metrics[variant][metric_name] = []
            variant_metrics[variant][metric_name].append(metric_value)
        
        # Calculate statistics
        variant_results = {}
        statistical_significance = {}
        confidence_intervals = {}
        p_values = {}
        effect_sizes = {}
        sample_sizes = {}
        
        for variant, metrics in variant_metrics.items():
            variant_results[variant] = {}
            sample_sizes[variant] = {}
            
            for metric_name, values in metrics.items():
                if not values:
                    continue
                
                values = np.array(values)
                mean_val = np.mean(values)
                std_val = np.std(values)
                n = len(values)
                
                variant_results[variant][metric_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "count": n,
                    "min": np.min(values),
                    "max": np.max(values)
                }
                sample_sizes[variant][metric_name] = n
        
        # Calculate statistical significance (simplified t-test)
        variants = list(variant_metrics.keys())
        if len(variants) >= 2:
            control_variant = variants[0]
            
            for metric_name in config.success_metrics:
                if metric_name.value in variant_results[control_variant]:
                    control_values = np.array(variant_metrics[control_variant][metric_name.value])
                    
                    for variant in variants[1:]:
                        if metric_name.value in variant_metrics[variant]:
                            variant_values = np.array(variant_metrics[variant][metric_name.value])
                            
                            # Simplified t-test
                            from scipy import stats
                            t_stat, p_value = stats.ttest_ind(control_values, variant_values)
                            
                            key = f"{control_variant}_vs_{variant}_{metric_name.value}"
                            statistical_significance[key] = p_value
                            p_values[key] = p_value
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values) + 
                                                 (len(variant_values) - 1) * np.var(variant_values)) / 
                                                (len(control_values) + len(variant_values) - 2))
                            effect_size = (np.mean(variant_values) - np.mean(control_values)) / pooled_std
                            effect_sizes[key] = effect_size
        
        # Determine winner
        winner = None
        if statistical_significance:
            # Find variant with best performance and significant improvement
            best_metric = config.success_metrics[0].value
            best_performance = float('-inf')
            
            for variant in variants:
                if best_metric in variant_results[variant]:
                    performance = variant_results[variant][best_metric]["mean"]
                    if performance > best_performance:
                        best_performance = performance
                        winner = variant
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            winner=winner,
            confidence_interval=confidence_intervals,
            sample_sizes=sample_sizes,
            p_values=p_values,
            effect_sizes=effect_sizes
        )
        
        self._results[experiment_id] = result
        return result
    
    def stop_experiment(self, experiment_id: str):
        """Stop an A/B test experiment"""
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self._active_experiments[experiment_id]
        config.end_date = datetime.now()
        config.status = ExperimentStatus.COMPLETED
        
        # Update database
        with sqlite3.connect(self.registry.db_path) as conn:
            conn.execute("""
                UPDATE experiments 
                SET status = ?, end_date = ?, config = ?
                WHERE experiment_id = ?
            """, (
                ExperimentStatus.COMPLETED.value, config.end_date.isoformat(),
                json.dumps(config.__dict__), experiment_id
            ))
            conn.commit()
        
        del self._active_experiments[experiment_id]
        self.logger.info(f"Stopped experiment {experiment_id}")
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment results"""
        return self._results.get(experiment_id)
    
    def list_active_experiments(self) -> List[ExperimentConfig]:
        """List all active experiments"""
        return list(self._active_experiments.values())


class TruthGPTVersioningManager:
    """Unified model versioning and A/B testing manager"""
    
    def __init__(self, config: ModelRegistryConfig):
        self.registry = TruthGPTModelRegistry(config)
        self.experiment_manager = TruthGPTExperimentManager(self.registry)
        self.logger = logging.getLogger(f"TruthGPTVersioningManager_{id(self)}")
    
    def create_model_with_versioning(self, model_id: str, name: str, 
                                   model: TruthGPTModel, description: str = "") -> Tuple[str, str]:
        """Create a model with automatic versioning"""
        # Register model if not exists
        if model_id not in self.registry._models:
            self.registry.register_model(model_id, name, description)
        
        # Create initial version
        version_id = self.registry.create_version(model_id, model, description)
        
        return model_id, version_id
    
    def create_ab_test(self, name: str, description: str, 
                      model_versions: List[Tuple[str, str]],  # (model_id, version_id)
                      traffic_allocation: Dict[str, float]) -> str:
        """Create an A/B test experiment"""
        # Create experiment config
        variants = []
        for i, (model_id, version_id) in enumerate(model_versions):
            variants.append({
                "model_id": model_id,
                "version_id": version_id,
                "name": f"variant_{i+1}"
            })
        
        config = ExperimentConfig(
            experiment_id="",  # Will be set by create_experiment
            name=name,
            description=description,
            variants=variants,
            traffic_allocation=traffic_allocation,
            allocation_strategy=TrafficAllocation.RANDOM
        )
        
        return self.experiment_manager.create_experiment(config)
    
    def get_model_for_inference(self, model_id: str, user_id: str = None) -> Tuple[TruthGPTModel, str]:
        """Get model for inference, considering active A/B tests"""
        # Check for active experiments involving this model
        for experiment in self.experiment_manager.list_active_experiments():
            for variant in experiment.variants:
                if variant["model_id"] == model_id:
                    # User is in this experiment
                    assigned_variant = self.experiment_manager.get_variant_for_user(experiment.experiment_id, user_id or "anonymous")
                    if assigned_variant == variant["name"]:
                        model = self.registry.load_version(model_id, variant["version_id"])
                        return model, f"{experiment.experiment_id}:{variant['name']}"
        
        # No active experiment, use production version
        versions = self.registry.get_model_versions(model_id, ModelStatus.PRODUCTION)
        if versions:
            model = self.registry.load_version(model_id, versions[0].version_id)
            return model, "production"
        
        raise ValueError(f"No production version found for model {model_id}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive versioning and experiment statistics"""
        return {
            "models": {
                "total_models": len(self.registry._models),
                "total_versions": sum(len(versions) for versions in self.registry._models.values()),
                "production_versions": sum(
                    len([v for v in versions if v.status == ModelStatus.PRODUCTION])
                    for versions in self.registry._models.values()
                )
            },
            "experiments": {
                "total_experiments": len(self.experiment_manager._experiments),
                "active_experiments": len(self.experiment_manager._active_experiments),
                "completed_experiments": len([
                    exp for exp in self.experiment_manager._experiments.values()
                    if exp.status == ExperimentStatus.COMPLETED
                ])
            }
        }


def create_truthgpt_versioning_manager(
    config: Optional[ModelRegistryConfig] = None
) -> TruthGPTVersioningManager:
    """Create TruthGPT versioning manager with default configuration"""
    if config is None:
        config = ModelRegistryConfig()
    
    return TruthGPTVersioningManager(config)


def quick_truthgpt_versioning_setup(
    registry_path: str = "./model_registry",
    max_versions: int = 50
) -> TruthGPTVersioningManager:
    """Quick setup for TruthGPT model versioning and A/B testing"""
    config = ModelRegistryConfig(
        registry_path=registry_path,
        max_versions_per_model=max_versions,
        enable_automatic_cleanup=True,
        enable_metrics_tracking=True,
        enable_experiment_tracking=True
    )
    
    return TruthGPTVersioningManager(config)


# Example usage
if __name__ == "__main__":
    # Create versioning manager
    versioning_manager = quick_truthgpt_versioning_setup()
    
    # Create a model
    model_config = TruthGPTModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        num_heads=4
    )
    model = TruthGPTModel(model_config)
    
    # Register model and create version
    model_id, version_id = versioning_manager.create_model_with_versioning(
        "test_model", "Test Model", model, "Initial version"
    )
    
    # Promote to production
    versioning_manager.registry.promote_version(model_id, version_id, ModelStatus.PRODUCTION)
    
    # Create A/B test
    experiment_id = versioning_manager.create_ab_test(
        "Test Experiment",
        "Testing different model configurations",
        [(model_id, version_id)],
        {"variant_1": 1.0}
    )
    
    # Get stats
    stats = versioning_manager.get_comprehensive_stats()
    print(f"Versioning stats: {stats}")
