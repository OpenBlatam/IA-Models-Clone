#!/usr/bin/env python3
"""
Advanced Model Versioning and A/B Testing Framework
Enterprise-grade model versioning, A/B testing, and deployment management
"""

import logging
import time
import json
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import uuid
import sqlite3
from contextlib import contextmanager

# ===== ENHANCED ENUMS =====

class VersionStatus(Enum):
    """Version status enumeration."""
    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class TestStatus(Enum):
    """A/B test status enumeration."""
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TestResult(Enum):
    """A/B test result enumeration."""
    WINNER_A = "winner_a"
    WINNER_B = "winner_b"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    INCONCLUSIVE = "inconclusive"

class TrafficSplit(Enum):
    """Traffic split enumeration."""
    EQUAL = "equal"
    WEIGHTED = "weighted"
    CUSTOM = "custom"

# ===== ENHANCED CONFIGURATION =====

@dataclass
class VersioningConfig:
    """Configuration for model versioning."""
    storage_path: str = "./model_versions"
    database_path: str = "./versioning.db"
    auto_versioning: bool = True
    semantic_versioning: bool = True
    metadata_tracking: bool = True
    performance_tracking: bool = True
    rollback_enabled: bool = True
    backup_enabled: bool = True
    encryption_enabled: bool = False

@dataclass
class ABTestingConfig:
    """Configuration for A/B testing."""
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    max_duration_days: int = 30
    auto_stop_enabled: bool = True
    early_stopping_enabled: bool = True
    multiple_testing_correction: bool = True
    bayesian_testing: bool = False

# ===== ENHANCED DATA MODELS =====

@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_id: str
    version_number: str
    status: VersionStatus
    created_at: datetime
    created_by: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    deployment_info: Optional[Dict[str, Any]] = None
    parent_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class ABTest:
    """A/B test configuration and results."""
    test_id: str
    name: str
    description: str
    model_a_version: str
    model_b_version: str
    traffic_split: TrafficSplit
    split_ratio: float = 0.5
    status: TestStatus = TestStatus.PLANNED
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    metrics: List[str] = field(default_factory=list)
    results: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"

@dataclass
class TestResult:
    """A/B test result data."""
    test_id: str
    timestamp: datetime
    model_a_metrics: Dict[str, float]
    model_b_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    sample_sizes: Dict[str, int]
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    recommendation: str

# ===== ADVANCED VERSIONING SYSTEM =====

class AdvancedModelVersioningSystem:
    """Advanced model versioning system with enterprise features."""
    
    def __init__(self, config: VersioningConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.VersioningSystem")
        
        # Initialize storage
        self._initialize_storage()
        
        # Version tracking
        self.versions = {}
        self.version_history = defaultdict(list)
        self.current_versions = {}
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        
        # Threading
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Start monitoring
        self._start_monitoring()
    
    def _initialize_storage(self) -> None:
        """Initialize storage systems."""
        try:
            # Create storage directory
            Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            self._initialize_database()
            
            self.logger.info("Storage systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
            raise
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for versioning."""
        try:
            self.db_path = Path(self.config.database_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create versions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        version_id TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        version_number TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        created_by TEXT NOT NULL,
                        description TEXT,
                        metadata TEXT,
                        performance_metrics TEXT,
                        deployment_info TEXT,
                        parent_version TEXT,
                        tags TEXT
                    )
                """)
                
                # Create performance history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version_id TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metrics TEXT NOT NULL,
                        FOREIGN KEY (version_id) REFERENCES model_versions (version_id)
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_id ON model_versions (model_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON model_versions (status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_history (timestamp)")
                
                conn.commit()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_version(self, 
                      model_id: str, 
                      model: Any, 
                      metadata: Dict[str, Any],
                      created_by: str = "system",
                      description: str = "") -> str:
        """Create a new model version."""
        try:
            # Generate version ID
            version_id = str(uuid.uuid4())
            
            # Determine version number
            version_number = self._generate_version_number(model_id)
            
            # Create version object
            version = ModelVersion(
                version_id=version_id,
                model_id=model_id,
                version_number=version_number,
                status=VersionStatus.DRAFT,
                created_at=datetime.now(),
                created_by=created_by,
                description=description,
                metadata=metadata
            )
            
            # Store model
            self._store_model_version(version, model)
            
            # Store in database
            self._store_version_in_db(version)
            
            # Update tracking
            self.versions[version_id] = version
            self.version_history[model_id].append(version)
            
            self.logger.info(f"Version created: {version_id} ({version_number})")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to create version: {e}")
            raise
    
    def promote_version(self, version_id: str, target_status: VersionStatus) -> bool:
        """Promote version to target status."""
        try:
            if version_id not in self.versions:
                raise ValueError(f"Version not found: {version_id}")
            
            version = self.versions[version_id]
            old_status = version.status
            
            # Validate promotion
            if not self._validate_promotion(version, target_status):
                self.logger.warning(f"Invalid promotion: {old_status} -> {target_status}")
                return False
            
            # Update status
            version.status = target_status
            
            # Update current version if promoting to production
            if target_status == VersionStatus.PRODUCTION:
                self.current_versions[version.model_id] = version_id
            
            # Update database
            self._update_version_status(version_id, target_status)
            
            self.logger.info(f"Version promoted: {version_id} ({old_status} -> {target_status})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to promote version: {e}")
            return False
    
    def rollback_version(self, model_id: str, target_version_id: str) -> bool:
        """Rollback to a previous version."""
        try:
            if not self.config.rollback_enabled:
                self.logger.warning("Rollback is disabled")
                return False
            
            if model_id not in self.current_versions:
                raise ValueError(f"No current version for model: {model_id}")
            
            current_version_id = self.current_versions[model_id]
            
            # Validate rollback
            if not self._validate_rollback(model_id, target_version_id):
                self.logger.warning(f"Invalid rollback: {current_version_id} -> {target_version_id}")
                return False
            
            # Perform rollback
            self.current_versions[model_id] = target_version_id
            
            # Update statuses
            self._update_version_status(current_version_id, VersionStatus.DEPRECATED)
            self._update_version_status(target_version_id, VersionStatus.PRODUCTION)
            
            self.logger.info(f"Rollback completed: {model_id} -> {target_version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback version: {e}")
            return False
    
    def get_version_history(self, model_id: str) -> List[ModelVersion]:
        """Get version history for a model."""
        try:
            return self.version_history.get(model_id, [])
        except Exception as e:
            self.logger.error(f"Failed to get version history: {e}")
            return []
    
    def get_current_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get current production version for a model."""
        try:
            if model_id not in self.current_versions:
                return None
            
            version_id = self.current_versions[model_id]
            return self.versions.get(version_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get current version: {e}")
            return None
    
    def _generate_version_number(self, model_id: str) -> str:
        """Generate semantic version number."""
        try:
            if not self.config.semantic_versioning:
                return str(int(time.time()))
            
            # Get existing versions for this model
            existing_versions = self.version_history.get(model_id, [])
            
            if not existing_versions:
                return "1.0.0"
            
            # Parse existing version numbers
            version_numbers = []
            for version in existing_versions:
                try:
                    parts = version.version_number.split('.')
                    if len(parts) == 3:
                        version_numbers.append(tuple(map(int, parts)))
                except:
                    continue
            
            if not version_numbers:
                return "1.0.0"
            
            # Get latest version
            latest_version = max(version_numbers)
            
            # Increment patch version
            new_version = (latest_version[0], latest_version[1], latest_version[2] + 1)
            
            return f"{new_version[0]}.{new_version[1]}.{new_version[2]}"
            
        except Exception as e:
            self.logger.error(f"Failed to generate version number: {e}")
            return "1.0.0"
    
    def _validate_promotion(self, version: ModelVersion, target_status: VersionStatus) -> bool:
        """Validate version promotion."""
        try:
            current_status = version.status
            
            # Define valid promotion paths
            valid_promotions = {
                VersionStatus.DRAFT: [VersionStatus.STAGING],
                VersionStatus.STAGING: [VersionStatus.PRODUCTION, VersionStatus.DRAFT],
                VersionStatus.PRODUCTION: [VersionStatus.DEPRECATED],
                VersionStatus.DEPRECATED: [VersionStatus.ARCHIVED],
                VersionStatus.ARCHIVED: []
            }
            
            return target_status in valid_promotions.get(current_status, [])
            
        except Exception as e:
            self.logger.error(f"Failed to validate promotion: {e}")
            return False
    
    def _validate_rollback(self, model_id: str, target_version_id: str) -> bool:
        """Validate version rollback."""
        try:
            if target_version_id not in self.versions:
                return False
            
            target_version = self.versions[target_version_id]
            
            # Can only rollback to production-ready versions
            return target_version.status in [VersionStatus.PRODUCTION, VersionStatus.STAGING]
            
        except Exception as e:
            self.logger.error(f"Failed to validate rollback: {e}")
            return False
    
    def _store_model_version(self, version: ModelVersion, model: Any) -> None:
        """Store model version."""
        try:
            # Create version directory
            version_dir = Path(self.config.storage_path) / version.model_id / version.version_number
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Store model
            import joblib
            model_path = version_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Store metadata
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(version.__dict__, f, default=str, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to store model version: {e}")
            raise
    
    def _store_version_in_db(self, version: ModelVersion) -> None:
        """Store version in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO model_versions 
                    (version_id, model_id, version_number, status, created_at, created_by, 
                     description, metadata, performance_metrics, deployment_info, parent_version, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    version.version_id,
                    version.model_id,
                    version.version_number,
                    version.status.value,
                    version.created_at,
                    version.created_by,
                    version.description,
                    json.dumps(version.metadata),
                    json.dumps(version.performance_metrics),
                    json.dumps(version.deployment_info) if version.deployment_info else None,
                    version.parent_version,
                    json.dumps(version.tags)
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store version in database: {e}")
            raise
    
    def _update_version_status(self, version_id: str, status: VersionStatus) -> None:
        """Update version status in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE model_versions 
                    SET status = ? 
                    WHERE version_id = ?
                """, (status.value, version_id))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update version status: {e}")
            raise
    
    def _start_monitoring(self) -> None:
        """Start version monitoring."""
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Version monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def _monitoring_loop(self) -> None:
        """Version monitoring loop."""
        while self.monitoring_active:
            try:
                # Monitor version performance
                self._monitor_version_performance()
                
                # Clean up old versions
                self._cleanup_old_versions()
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _monitor_version_performance(self) -> None:
        """Monitor version performance."""
        try:
            # Implement performance monitoring logic
            pass
            
        except Exception as e:
            self.logger.error(f"Failed to monitor version performance: {e}")
    
    def _cleanup_old_versions(self) -> None:
        """Clean up old versions."""
        try:
            # Implement cleanup logic
            pass
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old versions: {e}")
    
    def stop(self) -> None:
        """Stop the versioning system."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("Versioning system stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop versioning system: {e}")

# ===== ADVANCED A/B TESTING SYSTEM =====

class AdvancedABTestingSystem:
    """Advanced A/B testing system for model comparison."""
    
    def __init__(self, config: ABTestingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ABTestingSystem")
        
        # Test tracking
        self.tests = {}
        self.test_results = defaultdict(list)
        self.active_tests = {}
        
        # Statistical analysis
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Threading
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Start monitoring
        self._start_monitoring()
    
    def create_test(self, 
                   name: str,
                   model_a_version: str,
                   model_b_version: str,
                   traffic_split: TrafficSplit = TrafficSplit.EQUAL,
                   split_ratio: float = 0.5,
                   metrics: List[str] = None,
                   created_by: str = "system") -> str:
        """Create a new A/B test."""
        try:
            # Generate test ID
            test_id = str(uuid.uuid4())
            
            # Default metrics
            if metrics is None:
                metrics = ["accuracy", "precision", "recall", "f1_score"]
            
            # Create test
            test = ABTest(
                test_id=test_id,
                name=name,
                description=f"A/B test comparing {model_a_version} vs {model_b_version}",
                model_a_version=model_a_version,
                model_b_version=model_b_version,
                traffic_split=traffic_split,
                split_ratio=split_ratio,
                metrics=metrics,
                created_by=created_by
            )
            
            # Store test
            self.tests[test_id] = test
            
            self.logger.info(f"A/B test created: {test_id} ({name})")
            return test_id
            
        except Exception as e:
            self.logger.error(f"Failed to create A/B test: {e}")
            raise
    
    def start_test(self, test_id: str) -> bool:
        """Start an A/B test."""
        try:
            if test_id not in self.tests:
                raise ValueError(f"Test not found: {test_id}")
            
            test = self.tests[test_id]
            
            if test.status != TestStatus.PLANNED:
                self.logger.warning(f"Test cannot be started: {test.status}")
                return False
            
            # Update test status
            test.status = TestStatus.RUNNING
            test.start_date = datetime.now()
            
            # Add to active tests
            self.active_tests[test_id] = test
            
            self.logger.info(f"A/B test started: {test_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start A/B test: {e}")
            return False
    
    def stop_test(self, test_id: str) -> bool:
        """Stop an A/B test."""
        try:
            if test_id not in self.tests:
                raise ValueError(f"Test not found: {test_id}")
            
            test = self.tests[test_id]
            
            if test.status != TestStatus.RUNNING:
                self.logger.warning(f"Test cannot be stopped: {test.status}")
                return False
            
            # Update test status
            test.status = TestStatus.COMPLETED
            test.end_date = datetime.now()
            
            # Remove from active tests
            if test_id in self.active_tests:
                del self.active_tests[test_id]
            
            # Analyze results
            self._analyze_test_results(test_id)
            
            self.logger.info(f"A/B test stopped: {test_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop A/B test: {e}")
            return False
    
    def add_test_data(self, test_id: str, model_version: str, metrics: Dict[str, float]) -> None:
        """Add data to an A/B test."""
        try:
            if test_id not in self.tests:
                raise ValueError(f"Test not found: {test_id}")
            
            test = self.tests[test_id]
            
            if test.status != TestStatus.RUNNING:
                self.logger.warning(f"Cannot add data to test: {test.status}")
                return
            
            # Store test data
            test_data = {
                "timestamp": datetime.now(),
                "model_version": model_version,
                "metrics": metrics
            }
            
            self.test_results[test_id].append(test_data)
            
            # Check for early stopping
            if self.config.early_stopping_enabled:
                self._check_early_stopping(test_id)
            
        except Exception as e:
            self.logger.error(f"Failed to add test data: {e}")
    
    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results."""
        try:
            if test_id not in self.tests:
                return None
            
            test = self.tests[test_id]
            return test.results
            
        except Exception as e:
            self.logger.error(f"Failed to get test results: {e}")
            return None
    
    def _analyze_test_results(self, test_id: str) -> None:
        """Analyze A/B test results."""
        try:
            if test_id not in self.tests:
                return
            
            test = self.tests[test_id]
            test_data = self.test_results[test_id]
            
            if not test_data:
                self.logger.warning(f"No data available for test: {test_id}")
                return
            
            # Separate data by model version
            model_a_data = [d for d in test_data if d["model_version"] == test.model_a_version]
            model_b_data = [d for d in test_data if d["model_version"] == test.model_b_version]
            
            if not model_a_data or not model_b_data:
                self.logger.warning(f"Insufficient data for analysis: {test_id}")
                return
            
            # Perform statistical analysis
            results = self.statistical_analyzer.analyze_test(
                model_a_data, model_b_data, test.metrics, test.confidence_level
            )
            
            # Store results
            test.results = results
            
            self.logger.info(f"Test analysis completed: {test_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze test results: {e}")
    
    def _check_early_stopping(self, test_id: str) -> None:
        """Check for early stopping conditions."""
        try:
            if test_id not in self.tests:
                return
            
            test = self.tests[test_id]
            test_data = self.test_results[test_id]
            
            # Check sample size
            if len(test_data) >= test.min_sample_size:
                # Check for statistical significance
                if self._is_statistically_significant(test_id):
                    self.logger.info(f"Early stopping triggered for test: {test_id}")
                    self.stop_test(test_id)
            
        except Exception as e:
            self.logger.error(f"Failed to check early stopping: {e}")
    
    def _is_statistically_significant(self, test_id: str) -> bool:
        """Check if test results are statistically significant."""
        try:
            # Implement statistical significance check
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check statistical significance: {e}")
            return False
    
    def _start_monitoring(self) -> None:
        """Start test monitoring."""
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("A/B test monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def _monitoring_loop(self) -> None:
        """Test monitoring loop."""
        while self.monitoring_active:
            try:
                # Monitor active tests
                for test_id, test in self.active_tests.items():
                    self._monitor_test(test_id, test)
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _monitor_test(self, test_id: str, test: ABTest) -> None:
        """Monitor individual test."""
        try:
            # Check test duration
            if test.start_date:
                duration = datetime.now() - test.start_date
                if duration.days >= self.config.max_duration_days:
                    self.logger.info(f"Test duration exceeded for: {test_id}")
                    self.stop_test(test_id)
            
        except Exception as e:
            self.logger.error(f"Failed to monitor test: {e}")
    
    def stop(self) -> None:
        """Stop the A/B testing system."""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("A/B testing system stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop A/B testing system: {e}")

# ===== STATISTICAL ANALYZER =====

class StatisticalAnalyzer:
    """Statistical analysis for A/B testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StatisticalAnalyzer")
    
    def analyze_test(self, 
                    model_a_data: List[Dict[str, Any]], 
                    model_b_data: List[Dict[str, Any]], 
                    metrics: List[str],
                    confidence_level: float = 0.95) -> Dict[str, Any]:
        """Analyze A/B test results."""
        try:
            results = {
                "analysis_timestamp": datetime.now(),
                "model_a_sample_size": len(model_a_data),
                "model_b_sample_size": len(model_b_data),
                "metrics_analysis": {},
                "overall_recommendation": "inconclusive"
            }
            
            # Analyze each metric
            for metric in metrics:
                metric_analysis = self._analyze_metric(
                    model_a_data, model_b_data, metric, confidence_level
                )
                results["metrics_analysis"][metric] = metric_analysis
            
            # Determine overall recommendation
            results["overall_recommendation"] = self._determine_overall_recommendation(
                results["metrics_analysis"]
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze test: {e}")
            return {"error": str(e)}
    
    def _analyze_metric(self, 
                       model_a_data: List[Dict[str, Any]], 
                       model_b_data: List[Dict[str, Any]], 
                       metric: str,
                       confidence_level: float) -> Dict[str, Any]:
        """Analyze a specific metric."""
        try:
            # Extract metric values
            a_values = [d["metrics"].get(metric, 0.0) for d in model_a_data]
            b_values = [d["metrics"].get(metric, 0.0) for d in model_b_data]
            
            if not a_values or not b_values:
                return {"error": "No data available"}
            
            # Calculate statistics
            a_mean = np.mean(a_values)
            b_mean = np.mean(b_values)
            a_std = np.std(a_values)
            b_std = np.std(b_values)
            
            # Perform t-test
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(a_values, b_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(a_values) - 1) * a_std**2 + (len(b_values) - 1) * b_std**2) / 
                                (len(a_values) + len(b_values) - 2))
            effect_size = (a_mean - b_mean) / pooled_std if pooled_std > 0 else 0
            
            # Calculate confidence intervals
            a_ci = self._calculate_confidence_interval(a_values, confidence_level)
            b_ci = self._calculate_confidence_interval(b_values, confidence_level)
            
            return {
                "model_a_mean": a_mean,
                "model_b_mean": b_mean,
                "model_a_std": a_std,
                "model_b_std": b_std,
                "difference": a_mean - b_mean,
                "t_statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "model_a_confidence_interval": a_ci,
                "model_b_confidence_interval": b_ci,
                "statistically_significant": p_value < (1 - confidence_level),
                "recommendation": self._get_metric_recommendation(a_mean, b_mean, p_value, confidence_level)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze metric {metric}: {e}")
            return {"error": str(e)}
    
    def _calculate_confidence_interval(self, values: List[float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval."""
        try:
            from scipy import stats
            
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)
            
            # Calculate t-value
            alpha = 1 - confidence_level
            t_value = stats.t.ppf(1 - alpha/2, n - 1)
            
            # Calculate margin of error
            margin_error = t_value * (std / np.sqrt(n))
            
            return (mean - margin_error, mean + margin_error)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate confidence interval: {e}")
            return (0.0, 0.0)
    
    def _get_metric_recommendation(self, a_mean: float, b_mean: float, p_value: float, confidence_level: float) -> str:
        """Get recommendation for a metric."""
        try:
            if p_value >= (1 - confidence_level):
                return "no_significant_difference"
            elif a_mean > b_mean:
                return "model_a_better"
            else:
                return "model_b_better"
                
        except Exception as e:
            self.logger.error(f"Failed to get metric recommendation: {e}")
            return "inconclusive"
    
    def _determine_overall_recommendation(self, metrics_analysis: Dict[str, Any]) -> str:
        """Determine overall recommendation based on all metrics."""
        try:
            recommendations = []
            
            for metric, analysis in metrics_analysis.items():
                if "recommendation" in analysis:
                    recommendations.append(analysis["recommendation"])
            
            if not recommendations:
                return "inconclusive"
            
            # Count recommendations
            a_better = recommendations.count("model_a_better")
            b_better = recommendations.count("model_b_better")
            no_diff = recommendations.count("no_significant_difference")
            
            if a_better > b_better and a_better > no_diff:
                return "model_a_winner"
            elif b_better > a_better and b_better > no_diff:
                return "model_b_winner"
            else:
                return "no_significant_difference"
                
        except Exception as e:
            self.logger.error(f"Failed to determine overall recommendation: {e}")
            return "inconclusive"

# ===== MAIN EXECUTION =====

def main():
    """Main execution function."""
    print("🚀 Advanced Model Versioning and A/B Testing Framework")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configurations
    versioning_config = VersioningConfig()
    ab_testing_config = ABTestingConfig()
    
    # Create systems
    versioning_system = AdvancedModelVersioningSystem(versioning_config)
    ab_testing_system = AdvancedABTestingSystem(ab_testing_config)
    
    try:
        print("✅ Systems initialized successfully")
        
        # Example usage
        print("📊 System Status:")
        print(f"   Versioning System: Active")
        print(f"   A/B Testing System: Active")
        
        # Keep systems running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Systems interrupted by user")
    except Exception as e:
        print(f"❌ Systems failed: {e}")
        raise
    finally:
        versioning_system.stop()
        ab_testing_system.stop()

if __name__ == "__main__":
    main()
