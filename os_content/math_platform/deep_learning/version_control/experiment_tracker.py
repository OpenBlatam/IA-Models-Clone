from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import json
import yaml
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import sqlite3
import uuid
import pickle
import zipfile
from git_manager import GitManager, GitConfig
        import argparse
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Experiment Tracker
Comprehensive experiment tracking system with Git version control integration.
"""


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    # Experiment settings
    experiment_name: str
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Tracking settings
    track_code: bool = True
    track_configs: bool = True
    track_models: bool = True
    track_metrics: bool = True
    track_artifacts: bool = True
    track_logs: bool = True
    
    # Storage settings
    experiment_dir: str = "./experiments"
    artifacts_dir: str = "./artifacts"
    logs_dir: str = "./logs"
    
    # Database settings
    db_path: str = "./experiments.db"
    
    # Git integration
    git_enabled: bool = True
    git_auto_commit: bool = True
    git_commit_configs: bool = True
    git_commit_models: bool = False
    
    # Metadata
    created_by: str = "Unknown"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    framework: str = "pytorch"
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class ExperimentRun:
    """Information about an experiment run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    run_name: str = ""
    status: str = "running"  # running, completed, failed, cancelled
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    duration: Optional[float] = None
    
    # Configuration
    config_hash: str = ""
    config_file: str = ""
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Artifacts
    artifacts: List[str] = field(default_factory=list)
    model_files: List[str] = field(default_factory=list)
    log_files: List[str] = field(default_factory=list)
    
    # Git information
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    
    # System information
    system_info: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    
    # Notes
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save run to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentRun':
        """Load run from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            run_dict = json.load(f)
        return cls(**run_dict)


class ExperimentTracker:
    """Comprehensive experiment tracking system."""
    
    def __init__(self, config: ExperimentConfig):
        
    """__init__ function."""
self.config = config
        self.experiment_dir = Path(config.experiment_dir)
        self.artifacts_dir = Path(config.artifacts_dir)
        self.logs_dir = Path(config.logs_dir)
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = Path(config.db_path)
        self._init_database()
        
        # Initialize Git manager if enabled
        self.git_manager = None
        if config.git_enabled:
            git_config = GitConfig(
                repo_path=str(self.experiment_dir),
                author_name=config.created_by,
                author_email="experiment@example.com",
                track_configs=config.git_commit_configs,
                track_models=config.git_commit_models
            )
            self.git_manager = GitManager(git_config)
        
        # Current run
        self.current_run = None
        
        logger.info(f"Initialized experiment tracker: {config.experiment_name}")
    
    def _init_database(self) -> Any:
        """Initialize SQLite database for experiment tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                created_by TEXT,
                created_date TEXT,
                framework TEXT,
                version TEXT
            )
        ''')
        
        # Create runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                run_name TEXT,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                duration REAL,
                config_hash TEXT,
                config_file TEXT,
                metrics TEXT,
                final_metrics TEXT,
                artifacts TEXT,
                model_files TEXT,
                log_files TEXT,
                git_commit TEXT,
                git_branch TEXT,
                git_dirty BOOLEAN,
                system_info TEXT,
                environment TEXT,
                notes TEXT,
                tags TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                step INTEGER,
                timestamp TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_run(self, run_name: str, config_file: str = None) -> str:
        """Start a new experiment run."""
        run_id = str(uuid.uuid4())
        
        # Create run directory
        run_dir = self.experiment_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run object
        self.current_run = ExperimentRun(
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            run_name=run_name,
            status="running",
            start_time=datetime.now().isoformat()
        )
        
        # Save configuration
        if config_file:
            self.current_run.config_file = config_file
            config_hash = self._calculate_file_hash(config_file)
            self.current_run.config_hash = config_hash
            
            # Copy config file
            config_dest = run_dir / "config.yaml"
            shutil.copy2(config_file, config_dest)
        
        # Save run info
        run_info_file = run_dir / "run_info.json"
        self.current_run.save(str(run_info_file))
        
        # Save experiment config
        exp_config_file = run_dir / "experiment_config.json"
        self.config.save(str(exp_config_file))
        
        # Git commit if enabled
        if self.git_manager and self.config.git_auto_commit:
            self.git_manager.add_file(str(run_info_file), f"Start run: {run_name}")
            self.git_manager.add_file(str(exp_config_file), f"Experiment config for run: {run_name}")
        
        # Insert into database
        self._insert_run_to_db(self.current_run)
        
        logger.info(f"Started experiment run: {run_id} - {run_name}")
        return run_id
    
    def end_run(self, status: str = "completed", final_metrics: Dict[str, Any] = None):
        """End the current experiment run."""
        if not self.current_run:
            logger.warning("No active run to end")
            return
        
        # Update run status
        self.current_run.status = status
        self.current_run.end_time = datetime.now().isoformat()
        
        # Calculate duration
        start_time = datetime.fromisoformat(self.current_run.start_time)
        end_time = datetime.fromisoformat(self.current_run.end_time)
        self.current_run.duration = (end_time - start_time).total_seconds()
        
        # Update final metrics
        if final_metrics:
            self.current_run.final_metrics = final_metrics
        
        # Update Git information
        if self.git_manager:
            git_status = self.git_manager.get_status()
            self.current_run.git_commit = git_status.get('last_commit', {}).get('hash', '')
            self.current_run.git_branch = git_status.get('active_branch', '')
            self.current_run.git_dirty = git_status.get('is_dirty', False)
        
        # Save updated run info
        run_dir = self.experiment_dir / self.current_run.run_id
        run_info_file = run_dir / "run_info.json"
        self.current_run.save(str(run_info_file))
        
        # Update database
        self._update_run_in_db(self.current_run)
        
        # Git commit if enabled
        if self.git_manager and self.config.git_auto_commit:
            self.git_manager.add_file(str(run_info_file), f"End run: {self.current_run.run_name} - {status}")
        
        logger.info(f"Ended experiment run: {self.current_run.run_id} - {status}")
        self.current_run = None
    
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric for the current run."""
        if not self.current_run:
            logger.warning("No active run to log metric")
            return
        
        # Add to current run metrics
        if step is not None:
            if 'step_metrics' not in self.current_run.metrics:
                self.current_run.metrics['step_metrics'] = {}
            if step not in self.current_run.metrics['step_metrics']:
                self.current_run.metrics['step_metrics'][step] = {}
            self.current_run.metrics['step_metrics'][step][name] = value
        else:
            self.current_run.metrics[name] = value
        
        # Save to database
        self._insert_metric_to_db(self.current_run.run_id, name, value, step)
        
        # Update run info file
        run_dir = self.experiment_dir / self.current_run.run_id
        run_info_file = run_dir / "run_info.json"
        self.current_run.save(str(run_info_file))
        
        logger.debug(f"Logged metric: {name} = {value} (step: {step})")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def save_artifact(self, file_path: str, artifact_name: str = None) -> str:
        """Save an artifact for the current run."""
        if not self.current_run:
            logger.warning("No active run to save artifact")
            return ""
        
        source_path = Path(file_path)
        if not source_path.exists():
            logger.error(f"Artifact file does not exist: {file_path}")
            return ""
        
        # Create artifact name
        if artifact_name is None:
            artifact_name = source_path.name
        
        # Copy to artifacts directory
        run_artifacts_dir = self.artifacts_dir / self.current_run.run_id
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = run_artifacts_dir / artifact_name
        shutil.copy2(source_path, dest_path)
        
        # Add to run artifacts
        self.current_run.artifacts.append(str(dest_path))
        
        # Update run info
        run_dir = self.experiment_dir / self.current_run.run_id
        run_info_file = run_dir / "run_info.json"
        self.current_run.save(str(run_info_file))
        
        logger.info(f"Saved artifact: {artifact_name}")
        return str(dest_path)
    
    def save_model(self, model_path: str, model_name: str = None) -> str:
        """Save a model file for the current run."""
        if not self.current_run:
            logger.warning("No active run to save model")
            return ""
        
        source_path = Path(model_path)
        if not source_path.exists():
            logger.error(f"Model file does not exist: {model_path}")
            return ""
        
        # Create model name
        if model_name is None:
            model_name = source_path.name
        
        # Copy to run directory
        run_dir = self.experiment_dir / self.current_run.run_id
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = models_dir / model_name
        shutil.copy2(source_path, dest_path)
        
        # Add to run model files
        self.current_run.model_files.append(str(dest_path))
        
        # Update run info
        run_info_file = run_dir / "run_info.json"
        self.current_run.save(str(run_info_file))
        
        # Git commit if enabled
        if self.git_manager and self.config.git_commit_models:
            self.git_manager.add_file(str(dest_path), f"Save model: {model_name}")
        
        logger.info(f"Saved model: {model_name}")
        return str(dest_path)
    
    def save_log(self, log_path: str, log_name: str = None) -> str:
        """Save a log file for the current run."""
        if not self.current_run:
            logger.warning("No active run to save log")
            return ""
        
        source_path = Path(log_path)
        if not source_path.exists():
            logger.error(f"Log file does not exist: {log_path}")
            return ""
        
        # Create log name
        if log_name is None:
            log_name = source_path.name
        
        # Copy to run directory
        run_dir = self.experiment_dir / self.current_run.run_id
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = logs_dir / log_name
        shutil.copy2(source_path, dest_path)
        
        # Add to run log files
        self.current_run.log_files.append(str(dest_path))
        
        # Update run info
        run_info_file = run_dir / "run_info.json"
        self.current_run.save(str(run_info_file))
        
        logger.info(f"Saved log: {log_name}")
        return str(dest_path)
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get experiment run by ID."""
        run_dir = self.experiment_dir / run_id
        run_info_file = run_dir / "run_info.json"
        
        if run_info_file.exists():
            return ExperimentRun.load(str(run_info_file))
        return None
    
    def list_runs(self, experiment_id: str = None) -> List[ExperimentRun]:
        """List all runs for an experiment."""
        if experiment_id is None:
            experiment_id = self.config.experiment_id
        
        runs = []
        for run_dir in self.experiment_dir.iterdir():
            if run_dir.is_dir():
                run_info_file = run_dir / "run_info.json"
                if run_info_file.exists():
                    run = ExperimentRun.load(str(run_info_file))
                    if run.experiment_id == experiment_id:
                        runs.append(run)
        
        # Sort by start time (newest first)
        runs.sort(key=lambda x: x.start_time, reverse=True)
        return runs
    
    def get_best_run(self, metric: str, experiment_id: str = None, maximize: bool = True) -> Optional[ExperimentRun]:
        """Get the best run based on a metric."""
        runs = self.list_runs(experiment_id)
        if not runs:
            return None
        
        # Filter completed runs
        completed_runs = [run for run in runs if run.status == "completed"]
        if not completed_runs:
            return None
        
        # Find best run
        best_run = None
        best_value = float('-inf') if maximize else float('inf')
        
        for run in completed_runs:
            if metric in run.final_metrics:
                value = run.final_metrics[metric]
                if maximize and value > best_value:
                    best_value = value
                    best_run = run
                elif not maximize and value < best_value:
                    best_value = value
                    best_run = run
        
        return best_run
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        runs = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                runs.append(run)
        
        if len(runs) < 2:
            return {}
        
        comparison = {
            'runs': [run.to_dict() for run in runs],
            'metrics_comparison': {},
            'config_comparison': {},
            'duration_comparison': {
                'durations': [run.duration for run in runs if run.duration],
                'avg_duration': sum(run.duration for run in runs if run.duration) / len(runs)
            }
        }
        
        # Compare metrics
        all_metrics = set()
        for run in runs:
            all_metrics.update(run.final_metrics.keys())
        
        for metric in all_metrics:
            values = []
            for run in runs:
                if metric in run.final_metrics:
                    values.append(run.final_metrics[metric])
                else:
                    values.append(None)
            
            comparison['metrics_comparison'][metric] = {
                'values': values,
                'min': min(v for v in values if v is not None) if any(v is not None for v in values) else None,
                'max': max(v for v in values if v is not None) if any(v is not None for v in values) else None,
                'avg': sum(v for v in values if v is not None) / len([v for v in values if v is not None]) if any(v is not None for v in values) else None
            }
        
        return comparison
    
    def export_experiment(self, experiment_id: str = None, export_path: str = None) -> str:
        """Export an experiment with all its runs."""
        if experiment_id is None:
            experiment_id = self.config.experiment_id
        
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"experiment_{experiment_id}_{timestamp}.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add experiment config
            exp_config_file = self.experiment_dir / "experiment_config.json"
            if exp_config_file.exists():
                zipf.write(exp_config_file, "experiment_config.json")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Add all runs
            runs = self.list_runs(experiment_id)
            for run in runs:
                run_dir = self.experiment_dir / run.run_id
                if run_dir.exists():
                    for file_path in run_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = f"runs/{run.run_id}/{file_path.relative_to(run_dir)}"
                            zipf.write(file_path, arcname)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Exported experiment to: {export_path}")
        return export_path
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for chunk in iter(lambda: f.read(4096), b""):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _insert_run_to_db(self, run: ExperimentRun):
        """Insert run into database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO runs (
                run_id, experiment_id, run_name, status, start_time, end_time,
                duration, config_hash, config_file, metrics, final_metrics,
                artifacts, model_files, log_files, git_commit, git_branch,
                git_dirty, system_info, environment, notes, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run.run_id, run.experiment_id, run.run_name, run.status,
            run.start_time, run.end_time, run.duration, run.config_hash,
            run.config_file, json.dumps(run.metrics), json.dumps(run.final_metrics),
            json.dumps(run.artifacts), json.dumps(run.model_files),
            json.dumps(run.log_files), run.git_commit, run.git_branch,
            run.git_dirty, json.dumps(run.system_info), json.dumps(run.environment),
            run.notes, json.dumps(run.tags)
        ))
        
        conn.commit()
        conn.close()
    
    def _update_run_in_db(self, run: ExperimentRun):
        """Update run in database."""
        self._insert_run_to_db(run)  # SQLite INSERT OR REPLACE handles updates
    
    def _insert_metric_to_db(self, run_id: str, metric_name: str, metric_value: float, step: int = None):
        """Insert metric into database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (run_id, metric_name, metric_value, step, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (run_id, metric_name, metric_value, step, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()


class ExperimentTrackerCLI:
    """Command-line interface for experiment tracking."""
    
    def __init__(self, tracker: ExperimentTracker):
        
    """__init__ function."""
self.tracker = tracker
    
    def create_parser(self) -> Any:
        """Create command-line argument parser."""
        
        parser = argparse.ArgumentParser(description="Experiment Tracking CLI")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Start run
        start_parser = subparsers.add_parser('start', help='Start a new experiment run')
        start_parser.add_argument('run_name', help='Name of the run')
        start_parser.add_argument('--config', help='Configuration file path')
        
        # End run
        end_parser = subparsers.add_parser('end', help='End the current run')
        end_parser.add_argument('--status', default='completed', help='Run status')
        
        # Log metric
        metric_parser = subparsers.add_parser('log-metric', help='Log a metric')
        metric_parser.add_argument('name', help='Metric name')
        metric_parser.add_argument('value', type=float, help='Metric value')
        metric_parser.add_argument('--step', type=int, help='Step number')
        
        # List runs
        list_parser = subparsers.add_parser('list', help='List experiment runs')
        
        # Get run
        get_parser = subparsers.add_parser('get', help='Get run information')
        get_parser.add_argument('run_id', help='Run ID')
        
        # Compare runs
        compare_parser = subparsers.add_parser('compare', help='Compare runs')
        compare_parser.add_argument('run_ids', nargs='+', help='Run IDs to compare')
        
        # Export experiment
        export_parser = subparsers.add_parser('export', help='Export experiment')
        export_parser.add_argument('--output', help='Output file path')
        
        return parser
    
    def run(self, args=None) -> Any:
        """Run the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if parsed_args.command == 'start':
            run_id = self.tracker.start_run(parsed_args.run_name, parsed_args.config)
            print(f"Started run: {run_id}")
        elif parsed_args.command == 'end':
            self.tracker.end_run(parsed_args.status)
            print("Ended run")
        elif parsed_args.command == 'log-metric':
            self.tracker.log_metric(parsed_args.name, parsed_args.value, parsed_args.step)
            print(f"Logged metric: {parsed_args.name} = {parsed_args.value}")
        elif parsed_args.command == 'list':
            runs = self.tracker.list_runs()
            for run in runs:
                print(f"{run.run_id}: {run.run_name} ({run.status}) - {run.start_time}")
        elif parsed_args.command == 'get':
            run = self.tracker.get_run(parsed_args.run_id)
            if run:
                print(json.dumps(run.to_dict(), indent=2))
            else:
                print("Run not found")
        elif parsed_args.command == 'compare':
            comparison = self.tracker.compare_runs(parsed_args.run_ids)
            print(json.dumps(comparison, indent=2))
        elif parsed_args.command == 'export':
            export_path = self.tracker.export_experiment(export_path=parsed_args.output)
            print(f"Exported to: {export_path}")
        else:
            parser.print_help()


# Example usage
if __name__ == "__main__":
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="transformer_classification",
        description="Training transformer model for image classification",
        tags=["transformer", "classification", "vision"],
        created_by="Deep Learning Team",
        track_code=True,
        track_configs=True,
        track_models=True,
        git_enabled=True
    )
    
    # Create experiment tracker
    tracker = ExperimentTracker(config)
    
    # Create CLI
    cli = ExperimentTrackerCLI(tracker)
    
    # Run CLI
    cli.run() 