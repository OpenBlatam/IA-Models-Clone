"""
Experiment tracker for the ads training system.

This module consolidates all experiment tracking functionality into a unified,
clean architecture for monitoring training experiments.
"""

import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3
import hashlib
import uuid

from .base_trainer import TrainingMetrics, TrainingResult

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class ExperimentRun:
    """A single experiment run."""
    run_id: str
    experiment_name: str
    status: str = "running"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metrics: List[TrainingMetrics] = field(default_factory=list)
    final_result: Optional[TrainingResult] = None
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        data['metrics'] = [m.to_dict() for m in self.metrics]
        if self.final_result:
            data['final_result'] = self.final_result.to_dict()
        return data

class ExperimentTracker:
    """
    Unified experiment tracker for the ads training system.
    
    This tracker consolidates all experiment tracking functionality including:
    - Experiment configuration management
    - Run tracking and metrics collection
    - Artifact storage and retrieval
    - Performance analysis and comparison
    - Export and reporting capabilities
    """
    
    def __init__(self, storage_path: str = "./experiments"):
        """Initialize the experiment tracker."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Database setup
        self.db_path = self.storage_path / "experiments.db"
        self._init_database()
        
        # Current experiment tracking
        self.current_experiment: Optional[ExperimentConfig] = None
        self.current_run: Optional[ExperimentRun] = None
        
        logger.info(f"Experiment tracker initialized at: {self.storage_path}")
    
    def _init_database(self):
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create experiments table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS experiments (
                        name TEXT PRIMARY KEY,
                        description TEXT,
                        tags TEXT,
                        hyperparameters TEXT,
                        dataset_info TEXT,
                        model_info TEXT,
                        created_at TEXT
                    )
                ''')
                
                # Create runs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        experiment_name TEXT,
                        status TEXT,
                        start_time TEXT,
                        end_time TEXT,
                        final_result TEXT,
                        artifacts TEXT,
                        notes TEXT,
                        FOREIGN KEY (experiment_name) REFERENCES experiments (name)
                    )
                ''')
                
                # Create metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT,
                        epoch INTEGER,
                        step INTEGER,
                        loss REAL,
                        accuracy REAL,
                        learning_rate REAL,
                        validation_loss REAL,
                        validation_accuracy REAL,
                        timestamp TEXT,
                        FOREIGN KEY (run_id) REFERENCES runs (run_id)
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO experiments 
                    (name, description, tags, hyperparameters, dataset_info, model_info, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    config.name,
                    config.description,
                    json.dumps(config.tags),
                    json.dumps(config.hyperparameters),
                    json.dumps(config.dataset_info),
                    json.dumps(config.model_info),
                    config.created_at.isoformat()
                ))
                
                conn.commit()
                
                # Store current experiment
                self.current_experiment = config
                
                logger.info(f"Experiment created: {config.name}")
                return config.name
                
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    def start_run(self, experiment_name: str, run_id: Optional[str] = None) -> str:
        """Start a new experiment run."""
        if not run_id:
            run_id = str(uuid.uuid4())
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO runs 
                    (run_id, experiment_name, status, start_time)
                    VALUES (?, ?, ?, ?)
                ''', (
                    run.run_id,
                    run.experiment_name,
                    run.status,
                    run.start_time.isoformat()
                ))
                
                conn.commit()
                
                # Store current run
                self.current_run = run
                
                logger.info(f"Run started: {run_id} for experiment: {experiment_name}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            raise
    
    def log_metrics(self, metrics: TrainingMetrics, run_id: Optional[str] = None):
        """Log training metrics for a run."""
        if not run_id and self.current_run:
            run_id = self.current_run.run_id
        elif not run_id:
            raise ValueError("No run_id provided and no current run")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO metrics 
                    (run_id, epoch, step, loss, accuracy, learning_rate, 
                     validation_loss, validation_accuracy, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    metrics.epoch,
                    metrics.step,
                    metrics.loss,
                    metrics.accuracy,
                    metrics.learning_rate,
                    metrics.validation_loss,
                    metrics.validation_accuracy,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                
                # Update current run if applicable
                if self.current_run and self.current_run.run_id == run_id:
                    self.current_run.metrics.append(metrics)
                
                logger.debug(f"Metrics logged for run {run_id}: epoch {metrics.epoch}")
                
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    def complete_run(self, run_id: str, result: TrainingResult, 
                    artifacts: Optional[List[str]] = None, notes: str = ""):
        """Complete an experiment run."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE runs 
                    SET status = ?, end_time = ?, final_result = ?, artifacts = ?, notes = ?
                    WHERE run_id = ?
                ''', (
                    result.status.value,
                    datetime.now().isoformat(),
                    json.dumps(result.to_dict()),
                    json.dumps(artifacts or []),
                    notes,
                    run_id
                ))
                
                conn.commit()
                
                # Update current run if applicable
                if self.current_run and self.current_run.run_id == run_id:
                    self.current_run.status = result.status.value
                    self.current_run.end_time = datetime.now()
                    self.current_run.final_result = result
                    self.current_run.artifacts = artifacts or []
                    self.current_run.notes = notes
                
                logger.info(f"Run completed: {run_id}")
                
        except Exception as e:
            logger.error(f"Failed to complete run: {e}")
            raise
    
    def get_experiment(self, name: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration by name."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT name, description, tags, hyperparameters, dataset_info, model_info, created_at
                    FROM experiments WHERE name = ?
                ''', (name,))
                
                row = cursor.fetchone()
                if row:
                    return ExperimentConfig(
                        name=row[0],
                        description=row[1],
                        tags=json.loads(row[2]),
                        hyperparameters=json.loads(row[3]),
                        dataset_info=json.loads(row[4]),
                        model_info=json.loads(row[5]),
                        created_at=datetime.fromisoformat(row[6])
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get experiment: {e}")
            return None
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get experiment run by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT run_id, experiment_name, status, start_time, end_time, 
                           final_result, artifacts, notes
                    FROM runs WHERE run_id = ?
                ''', (run_id,))
                
                row = cursor.fetchone()
                if row:
                    # Get metrics for this run
                    cursor.execute('''
                        SELECT epoch, step, loss, accuracy, learning_rate, 
                               validation_loss, validation_accuracy, timestamp
                        FROM metrics WHERE run_id = ? ORDER BY epoch, step
                    ''', (run_id,))
                    
                    metrics_rows = cursor.fetchall()
                    metrics = []
                    
                    for m_row in metrics_rows:
                        metrics.append(TrainingMetrics(
                            epoch=m_row[0],
                            step=m_row[1],
                            loss=m_row[2],
                            accuracy=m_row[3],
                            learning_rate=m_row[4],
                            validation_loss=m_row[5],
                            validation_accuracy=m_row[6]
                        ))
                    
                    return ExperimentRun(
                        run_id=row[0],
                        experiment_name=row[1],
                        status=row[2],
                        start_time=datetime.fromisoformat(row[3]),
                        end_time=datetime.fromisoformat(row[4]) if row[4] else None,
                        final_result=TrainingResult(**json.loads(row[5])) if row[5] else None,
                        artifacts=json.loads(row[6]) if row[6] else [],
                        notes=row[7] or "",
                        metrics=metrics
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get run: {e}")
            return None
    
    def list_experiments(self) -> List[str]:
        """List all experiment names."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT name FROM experiments ORDER BY created_at DESC')
                rows = cursor.fetchall()
                
                return [row[0] for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def list_runs(self, experiment_name: Optional[str] = None) -> List[str]:
        """List all run IDs, optionally filtered by experiment."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if experiment_name:
                    cursor.execute('SELECT run_id FROM runs WHERE experiment_name = ? ORDER BY start_time DESC', 
                                 (experiment_name,))
                else:
                    cursor.execute('SELECT run_id FROM runs ORDER BY start_time DESC')
                
                rows = cursor.fetchall()
                return [row[0] for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        runs = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                runs.append(run)
        
        if not runs:
            return {"error": "No valid runs found"}
        
        comparison = {
            "run_ids": run_ids,
            "comparison_time": datetime.now().isoformat(),
            "runs": []
        }
        
        for run in runs:
            run_data = {
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "status": run.status,
                "duration": None,
                "final_metrics": None,
                "best_metrics": None
            }
            
            # Calculate duration
            if run.end_time and run.start_time:
                duration = (run.end_time - run.start_time).total_seconds()
                run_data["duration"] = duration
            
            # Get final metrics
            if run.final_result and run.final_result.final_metrics:
                run_data["final_metrics"] = run.final_result.final_metrics.to_dict()
            
            # Get best metrics
            if run.metrics:
                best_loss = min(run.metrics, key=lambda x: x.loss or float('inf'))
                run_data["best_metrics"] = best_loss.to_dict()
            
            comparison["runs"].append(run_data)
        
        return comparison
    
    def export_experiment(self, experiment_name: str, format: str = "json") -> str:
        """Export experiment data."""
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        runs = self.list_runs(experiment_name)
        experiment_data = {
            "experiment": experiment.to_dict(),
            "runs": []
        }
        
        for run_id in runs:
            run = self.get_run(run_id)
            if run:
                experiment_data["runs"].append(run.to_dict())
        
        if format.lower() == "json":
            export_path = self.storage_path / f"{experiment_name}_export.json"
            with open(export_path, 'w') as f:
                json.dump(experiment_data, f, indent=2)
            
            return str(export_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_experiment(self, experiment_name: str) -> bool:
        """Clean up an experiment and all its runs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete metrics first (foreign key constraint)
                cursor.execute('''
                    DELETE FROM metrics 
                    WHERE run_id IN (SELECT run_id FROM runs WHERE experiment_name = ?)
                ''', (experiment_name,))
                
                # Delete runs
                cursor.execute('DELETE FROM runs WHERE experiment_name = ?', (experiment_name,))
                
                # Delete experiment
                cursor.execute('DELETE FROM experiments WHERE name = ?', (experiment_name,))
                
                conn.commit()
                
                logger.info(f"Experiment cleaned up: {experiment_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cleanup experiment: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about experiments and runs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count experiments
                cursor.execute('SELECT COUNT(*) FROM experiments')
                experiment_count = cursor.fetchone()[0]
                
                # Count runs
                cursor.execute('SELECT COUNT(*) FROM runs')
                run_count = cursor.fetchone()[0]
                
                # Count completed runs
                cursor.execute('SELECT COUNT(*) FROM runs WHERE status = "completed"')
                completed_runs = cursor.fetchone()[0]
                
                # Count failed runs
                cursor.execute('SELECT COUNT(*) FROM runs WHERE status = "failed"')
                failed_runs = cursor.fetchone()[0]
                
                # Get recent activity
                cursor.execute('''
                    SELECT experiment_name, start_time 
                    FROM runs 
                    ORDER BY start_time DESC 
                    LIMIT 5
                ''')
                recent_runs = cursor.fetchall()
                
                return {
                    "total_experiments": experiment_count,
                    "total_runs": run_count,
                    "completed_runs": completed_runs,
                    "failed_runs": failed_runs,
                    "success_rate": completed_runs / run_count if run_count > 0 else 0,
                    "recent_runs": [
                        {"experiment": row[0], "start_time": row[1]} 
                        for row in recent_runs
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def add_artifact(self, run_id: str, artifact_path: str, artifact_type: str = "model"):
        """Add an artifact to a run."""
        if not self.current_run or self.current_run.run_id != run_id:
            raise ValueError(f"No active run with ID: {run_id}")
        
        # Copy artifact to experiment storage
        artifact_name = f"{run_id}_{artifact_type}_{os.path.basename(artifact_path)}"
        artifact_dest = self.storage_path / "artifacts" / artifact_name
        
        artifact_dest.parent.mkdir(exist_ok=True)
        
        try:
            import shutil
            shutil.copy2(artifact_path, artifact_dest)
            
            # Update run artifacts
            if artifact_dest not in self.current_run.artifacts:
                self.current_run.artifacts.append(str(artifact_dest))
            
            logger.info(f"Artifact added: {artifact_name}")
            
        except Exception as e:
            logger.error(f"Failed to add artifact: {e}")
            raise
