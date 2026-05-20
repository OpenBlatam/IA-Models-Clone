from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import json
import yaml
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import structlog
import git
from git import Repo, GitCommandError
import difflib
import tempfile
import zipfile
import pickle
from contextlib import contextmanager
from experiment_tracking import ExperimentTracker
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Version Control System for Deep Learning Projects
================================================

This module provides comprehensive version control capabilities for tracking
changes in code, configurations, models, and experiments. It integrates with
Git and provides additional versioning features specific to deep learning
workflows.

Key Features:
1. Git integration for code versioning
2. Configuration versioning and diff tracking
3. Model versioning with metadata
4. Experiment versioning and reproducibility
5. Automated commit and tagging
6. Change tracking and rollback capabilities
7. Integration with experiment tracking
"""


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlog.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# =============================================================================
# VERSION METADATA
# =============================================================================

@dataclass
class VersionMetadata:
    """Metadata for versioned items."""
    
    version_id: str
    item_type: str  # code, config, model, experiment
    item_name: str
    timestamp: datetime
    author: str
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    tags: List[str] = None
    description: Optional[str] = None
    
    # Change tracking
    changes: List[str] = None
    files_modified: List[str] = None
    files_added: List[str] = None
    files_deleted: List[str] = None
    
    # Dependencies
    dependencies: Dict[str, str] = None
    requirements_hash: Optional[str] = None
    
    # Performance metrics (for models)
    performance_metrics: Dict[str, float] = None
    
    # Configuration diff
    config_diff: Optional[str] = None
    
    def __post_init__(self) -> Any:
        """Initialize default values."""
        if self.tags is None:
            self.tags = []
        if self.changes is None:
            self.changes = []
        if self.files_modified is None:
            self.files_modified = []
        if self.files_added is None:
            self.files_added = []
        if self.files_deleted is None:
            self.files_deleted = []
        if self.dependencies is None:
            self.dependencies = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def generate_id(self) -> str:
        """Generate unique version ID."""
        timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(self.item_name.encode()).hexdigest()[:8]
        return f"{self.item_type}_{name_hash}_{timestamp}"


# =============================================================================
# GIT INTEGRATION
# =============================================================================

class GitManager:
    """Git repository management and operations."""
    
    def __init__(self, repo_path: str = "."):
        
    """__init__ function."""
self.repo_path = Path(repo_path)
        self.logger = structlog.get_logger(__name__)
        
        # Initialize or open repository
        if not self._is_git_repo():
            self._init_repo()
        
        self.repo = Repo(self.repo_path)
    
    def _is_git_repo(self) -> bool:
        """Check if directory is a Git repository."""
        return (self.repo_path / ".git").exists()
    
    def _init_repo(self) -> Any:
        """Initialize Git repository."""
        try:
            Repo.init(self.repo_path)
            self.logger.info("Git repository initialized", path=str(self.repo_path))
        except Exception as e:
            self.logger.error("Failed to initialize Git repository", error=str(e))
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get Git repository status."""
        try:
            status = {
                "branch": self.repo.active_branch.name,
                "commit_hash": self.repo.head.commit.hexsha,
                "commit_message": self.repo.head.commit.message,
                "is_dirty": self.repo.is_dirty(),
                "untracked_files": self.repo.untracked_files,
                "modified_files": [item.a_path for item in self.repo.index.diff(None)],
                "staged_files": [item.a_path for item in self.repo.index.diff('HEAD')]
            }
            return status
        except Exception as e:
            self.logger.error("Failed to get Git status", error=str(e))
            return {}
    
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> str:
        """Commit changes to Git repository."""
        try:
            # Add files to staging
            if files:
                for file_path in files:
                    self.repo.index.add([file_path])
            else:
                self.repo.index.add('*')
            
            # Commit changes
            commit = self.repo.index.commit(message)
            self.logger.info("Changes committed", commit_hash=commit.hexsha, message=message)
            return commit.hexsha
        except Exception as e:
            self.logger.error("Failed to commit changes", error=str(e))
            raise
    
    def create_tag(self, tag_name: str, message: str = "") -> str:
        """Create Git tag."""
        try:
            tag = self.repo.create_tag(tag_name, message=message)
            self.logger.info("Tag created", tag_name=tag_name, commit_hash=tag.commit.hexsha)
            return tag.name
        except Exception as e:
            self.logger.error("Failed to create tag", error=str(e))
            raise
    
    def get_diff(self, commit1: str = "HEAD", commit2: str = "HEAD~1") -> str:
        """Get diff between two commits."""
        try:
            diff = self.repo.git.diff(commit1, commit2)
            return diff
        except Exception as e:
            self.logger.error("Failed to get diff", error=str(e))
            return ""
    
    def checkout_branch(self, branch_name: str, create: bool = False):
        """Checkout Git branch."""
        try:
            if create and branch_name not in [b.name for b in self.repo.branches]:
                new_branch = self.repo.create_head(branch_name)
                new_branch.checkout()
            else:
                self.repo.heads[branch_name].checkout()
            
            self.logger.info("Branch checked out", branch=branch_name)
        except Exception as e:
            self.logger.error("Failed to checkout branch", error=str(e))
            raise
    
    def get_commit_history(self, max_count: int = 10) -> List[Dict[str, Any]]:
        """Get commit history."""
        try:
            commits = []
            for commit in self.repo.iter_commits('HEAD', max_count=max_count):
                commits.append({
                    "hash": commit.hexsha,
                    "author": commit.author.name,
                    "date": datetime.fromtimestamp(commit.committed_date),
                    "message": commit.message.strip(),
                    "files": list(commit.stats.files.keys())
                })
            return commits
        except Exception as e:
            self.logger.error("Failed to get commit history", error=str(e))
            return []


# =============================================================================
# CONFIGURATION VERSIONING
# =============================================================================

class ConfigurationVersioner:
    """Version control for configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger(__name__)
        
        # Version history file
        self.history_file = self.config_dir / "version_history.json"
        self.version_history = self._load_history()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load version history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.load(f)
        return {"configs": {}, "versions": []}
    
    def _save_history(self) -> Any:
        """Save version history to file."""
        with open(self.history_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.version_history, f, indent=2, default=str)
    
    def version_config(self, config_name: str, config_data: Dict[str, Any], 
                      description: str = "", tags: List[str] = None) -> str:
        """
        Version a configuration file.
        
        Args:
            config_name: Name of the configuration
            config_data: Configuration data
            description: Description of changes
            tags: Tags for the version
            
        Returns:
            Version ID
        """
        # Create version metadata
        metadata = VersionMetadata(
            version_id="",
            item_type="config",
            item_name=config_name,
            timestamp=datetime.now(),
            author=os.getenv("USER", "unknown"),
            description=description,
            tags=tags or []
        )
        metadata.version_id = metadata.generate_id()
        
        # Calculate configuration hash
        config_str = json.dumps(config_data, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Create version directory
        version_dir = self.config_dir / "versions" / metadata.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration file
        config_file = version_dir / f"{config_name}.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f, default_flow_style=False)
        
        # Generate diff if previous version exists
        if config_name in self.version_history["configs"]:
            prev_version = self.version_history["configs"][config_name]["latest"]
            prev_file = self.config_dir / "versions" / prev_version / f"{config_name}.yaml"
            
            if prev_file.exists():
                with open(prev_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    prev_config = yaml.safe_load(f)
                
                metadata.config_diff = self._generate_config_diff(prev_config, config_data)
        
        # Update version history
        self.version_history["configs"][config_name] = {
            "latest": metadata.version_id,
            "config_hash": config_hash,
            "last_updated": metadata.timestamp.isoformat()
        }
        
        self.version_history["versions"].append(metadata.to_dict())
        self._save_history()
        
        self.logger.info("Configuration versioned", 
                        config_name=config_name,
                        version_id=metadata.version_id)
        
        return metadata.version_id
    
    def _generate_config_diff(self, old_config: Dict, new_config: Dict) -> str:
        """Generate diff between two configurations."""
        old_str = yaml.dump(old_config, default_flow_style=False)
        new_str = yaml.dump(new_config, default_flow_style=False)
        
        diff = difflib.unified_diff(
            old_str.splitlines(keepends=True),
            new_str.splitlines(keepends=True),
            fromfile="old_config",
            tofile="new_config"
        )
        
        return ''.join(diff)
    
    def get_config_version(self, config_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Get specific version of configuration."""
        config_file = self.config_dir / "versions" / version_id / f"{config_name}.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return yaml.safe_load(f)
        return None
    
    def get_latest_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Get latest version of configuration."""
        if config_name in self.version_history["configs"]:
            latest_version = self.version_history["configs"][config_name]["latest"]
            return self.get_config_version(config_name, latest_version)
        return None
    
    def list_config_versions(self, config_name: str) -> List[Dict[str, Any]]:
        """List all versions of a configuration."""
        versions = []
        for version_data in self.version_history["versions"]:
            if version_data["item_name"] == config_name:
                versions.append(version_data)
        
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
    
    def rollback_config(self, config_name: str, version_id: str) -> bool:
        """Rollback configuration to specific version."""
        config_data = self.get_config_version(config_name, version_id)
        if config_data:
            # Save as new version with rollback description
            self.version_config(
                config_name=config_name,
                config_data=config_data,
                description=f"Rollback to version {version_id}",
                tags=["rollback"]
            )
            return True
        return False


# =============================================================================
# MODEL VERSIONING
# =============================================================================

class ModelVersioner:
    """Version control for model files and metadata."""
    
    def __init__(self, model_dir: str = "models"):
        
    """__init__ function."""
self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger(__name__)
        
        # Version history file
        self.history_file = self.model_dir / "version_history.json"
        self.version_history = self._load_history()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load version history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.load(f)
        return {"models": {}, "versions": []}
    
    def _save_history(self) -> Any:
        """Save version history to file."""
        with open(self.history_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.version_history, f, indent=2, default=str)
    
    def version_model(self, model_name: str, model_path: str, 
                     metadata: Dict[str, Any], description: str = "",
                     tags: List[str] = None) -> str:
        """
        Version a model file.
        
        Args:
            model_name: Name of the model
            model_path: Path to model file
            metadata: Model metadata (architecture, hyperparameters, etc.)
            description: Description of changes
            tags: Tags for the version
            
        Returns:
            Version ID
        """
        # Create version metadata
        version_metadata = VersionMetadata(
            version_id="",
            item_type="model",
            item_name=model_name,
            timestamp=datetime.now(),
            author=os.getenv("USER", "unknown"),
            description=description,
            tags=tags or []
        )
        version_metadata.version_id = version_metadata.generate_id()
        
        # Create version directory
        version_dir = self.model_dir / "versions" / version_metadata.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        model_file = Path(model_path)
        if model_file.exists():
            shutil.copy2(model_file, version_dir / model_file.name)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(model_file)
            version_metadata.dependencies["model_file_hash"] = file_hash
        else:
            self.logger.warning("Model file not found", path=model_path)
        
        # Save metadata
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metadata, f, indent=2, default=str)
        
        # Update version history
        self.version_history["models"][model_name] = {
            "latest": version_metadata.version_id,
            "last_updated": version_metadata.timestamp.isoformat()
        }
        
        self.version_history["versions"].append(version_metadata.to_dict())
        self._save_history()
        
        self.logger.info("Model versioned", 
                        model_name=model_name,
                        version_id=version_metadata.version_id)
        
        return version_metadata.version_id
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
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
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def get_model_version(self, model_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Get specific version of model."""
        version_dir = self.model_dir / "versions" / version_id
        
        if version_dir.exists():
            # Find model file
            model_files = list(version_dir.glob("*.pt")) + list(version_dir.glob("*.pth"))
            if model_files:
                model_file = model_files[0]
                
                # Load metadata
                metadata_file = version_dir / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        metadata = json.load(f)
                
                return {
                    "model_path": str(model_file),
                    "metadata": metadata
                }
        return None
    
    def get_latest_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get latest version of model."""
        if model_name in self.version_history["models"]:
            latest_version = self.version_history["models"][model_name]["latest"]
            return self.get_model_version(model_name, latest_version)
        return None
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        versions = []
        for version_data in self.version_history["versions"]:
            if version_data["item_name"] == model_name:
                versions.append(version_data)
        
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)


# =============================================================================
# EXPERIMENT VERSIONING
# =============================================================================

class ExperimentVersioner:
    """Version control for experiments and their results."""
    
    def __init__(self, experiment_dir: str = "experiments"):
        
    """__init__ function."""
self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger(__name__)
        
        # Version history file
        self.history_file = self.experiment_dir / "version_history.json"
        self.version_history = self._load_history()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load version history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.load(f)
        return {"experiments": {}, "versions": []}
    
    def _save_history(self) -> Any:
        """Save version history to file."""
        with open(self.history_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.version_history, f, indent=2, default=str)
    
    def version_experiment(self, experiment_id: str, experiment_data: Dict[str, Any],
                          results: Dict[str, Any], description: str = "",
                          tags: List[str] = None) -> str:
        """
        Version an experiment and its results.
        
        Args:
            experiment_id: Experiment identifier
            experiment_data: Experiment configuration and metadata
            results: Experiment results and metrics
            description: Description of changes
            tags: Tags for the version
            
        Returns:
            Version ID
        """
        # Create version metadata
        version_metadata = VersionMetadata(
            version_id="",
            item_type="experiment",
            item_name=experiment_id,
            timestamp=datetime.now(),
            author=os.getenv("USER", "unknown"),
            description=description,
            tags=tags or []
        )
        version_metadata.version_id = version_metadata.generate_id()
        
        # Add performance metrics
        if "metrics" in results:
            version_metadata.performance_metrics = results["metrics"]
        
        # Create version directory
        version_dir = self.experiment_dir / "versions" / version_metadata.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment data
        experiment_file = version_dir / "experiment.json"
        with open(experiment_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(experiment_data, f, indent=2, default=str)
        
        # Save results
        results_file = version_dir / "results.json"
        with open(results_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        # Save plots and artifacts if they exist
        if "plots" in results:
            plots_dir = version_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            for plot_name, plot_path in results["plots"].items():
                if Path(plot_path).exists():
                    shutil.copy2(plot_path, plots_dir / f"{plot_name}.png")
        
        # Update version history
        self.version_history["experiments"][experiment_id] = {
            "latest": version_metadata.version_id,
            "last_updated": version_metadata.timestamp.isoformat()
        }
        
        self.version_history["versions"].append(version_metadata.to_dict())
        self._save_history()
        
        self.logger.info("Experiment versioned", 
                        experiment_id=experiment_id,
                        version_id=version_metadata.version_id)
        
        return version_metadata.version_id
    
    def get_experiment_version(self, experiment_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Get specific version of experiment."""
        version_dir = self.experiment_dir / "versions" / version_id
        
        if version_dir.exists():
            experiment_file = version_dir / "experiment.json"
            results_file = version_dir / "results.json"
            
            experiment_data = {}
            results_data = {}
            
            if experiment_file.exists():
                with open(experiment_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    experiment_data = json.load(f)
            
            if results_file.exists():
                with open(results_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    results_data = json.load(f)
            
            return {
                "experiment": experiment_data,
                "results": results_data,
                "plots_dir": str(version_dir / "plots") if (version_dir / "plots").exists() else None
            }
        return None
    
    def get_latest_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get latest version of experiment."""
        if experiment_id in self.version_history["experiments"]:
            latest_version = self.version_history["experiments"][experiment_id]["latest"]
            return self.get_experiment_version(experiment_id, latest_version)
        return None
    
    def list_experiment_versions(self, experiment_id: str) -> List[Dict[str, Any]]:
        """List all versions of an experiment."""
        versions = []
        for version_data in self.version_history["versions"]:
            if version_data["item_name"] == experiment_id:
                versions.append(version_data)
        
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)


# =============================================================================
# MAIN VERSION CONTROL SYSTEM
# =============================================================================

class VersionControlSystem:
    """Main version control system integrating all components."""
    
    def __init__(self, project_root: str = ".", auto_commit: bool = True):
        
    """__init__ function."""
self.project_root = Path(project_root)
        self.auto_commit = auto_commit
        self.logger = structlog.get_logger(__name__)
        
        # Initialize components
        self.git_manager = GitManager(self.project_root)
        self.config_versioner = ConfigurationVersioner(self.project_root / "configs")
        self.model_versioner = ModelVersioner(self.project_root / "models")
        self.experiment_versioner = ExperimentVersioner(self.project_root / "experiments")
        
        # Version control metadata
        self.vc_metadata_file = self.project_root / ".version_control.json"
        self.vc_metadata = self._load_vc_metadata()
    
    def _load_vc_metadata(self) -> Dict[str, Any]:
        """Load version control metadata."""
        if self.vc_metadata_file.exists():
            with open(self.vc_metadata_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.load(f)
        return {
            "project_info": {
                "name": self.project_root.name,
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "versioning_enabled": True,
            "auto_commit": self.auto_commit
        }
    
    def _save_vc_metadata(self) -> Any:
        """Save version control metadata."""
        self.vc_metadata["project_info"]["last_updated"] = datetime.now().isoformat()
        with open(self.vc_metadata_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.vc_metadata, f, indent=2, default=str)
    
    def version_configuration(self, config_name: str, config_data: Dict[str, Any],
                             description: str = "", tags: List[str] = None) -> str:
        """Version a configuration with Git integration."""
        # Version the configuration
        version_id = self.config_versioner.version_config(
            config_name, config_data, description, tags
        )
        
        # Auto-commit if enabled
        if self.auto_commit:
            try:
                commit_message = f"Version configuration {config_name}: {description}"
                self.git_manager.commit_changes(commit_message)
                self.git_manager.create_tag(f"config-{config_name}-{version_id}")
            except Exception as e:
                self.logger.warning("Failed to auto-commit configuration version", error=str(e))
        
        return version_id
    
    def version_model(self, model_name: str, model_path: str, metadata: Dict[str, Any],
                     description: str = "", tags: List[str] = None) -> str:
        """Version a model with Git integration."""
        # Version the model
        version_id = self.model_versioner.version_model(
            model_name, model_path, metadata, description, tags
        )
        
        # Auto-commit if enabled
        if self.auto_commit:
            try:
                commit_message = f"Version model {model_name}: {description}"
                self.git_manager.commit_changes(commit_message)
                self.git_manager.create_tag(f"model-{model_name}-{version_id}")
            except Exception as e:
                self.logger.warning("Failed to auto-commit model version", error=str(e))
        
        return version_id
    
    def version_experiment(self, experiment_id: str, experiment_data: Dict[str, Any],
                          results: Dict[str, Any], description: str = "",
                          tags: List[str] = None) -> str:
        """Version an experiment with Git integration."""
        # Version the experiment
        version_id = self.experiment_versioner.version_experiment(
            experiment_id, experiment_data, results, description, tags
        )
        
        # Auto-commit if enabled
        if self.auto_commit:
            try:
                commit_message = f"Version experiment {experiment_id}: {description}"
                self.git_manager.commit_changes(commit_message)
                self.git_manager.create_tag(f"experiment-{experiment_id}-{version_id}")
            except Exception as e:
                self.logger.warning("Failed to auto-commit experiment version", error=str(e))
        
        return version_id
    
    def get_version_info(self, item_type: str, item_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version."""
        if item_type == "config":
            config_data = self.config_versioner.get_config_version(item_name, version_id)
            if config_data:
                return {"type": "config", "data": config_data}
        
        elif item_type == "model":
            model_data = self.model_versioner.get_model_version(item_name, version_id)
            if model_data:
                return {"type": "model", "data": model_data}
        
        elif item_type == "experiment":
            experiment_data = self.experiment_versioner.get_experiment_version(item_name, version_id)
            if experiment_data:
                return {"type": "experiment", "data": experiment_data}
        
        return None
    
    def list_versions(self, item_type: str = None, item_name: str = None) -> List[Dict[str, Any]]:
        """List all versions, optionally filtered by type and name."""
        all_versions = []
        
        # Collect versions from all versioners
        if item_type is None or item_type == "config":
            config_versions = self.config_versioner.version_history["versions"]
            all_versions.extend(config_versions)
        
        if item_type is None or item_type == "model":
            model_versions = self.model_versioner.version_history["versions"]
            all_versions.extend(model_versions)
        
        if item_type is None or item_type == "experiment":
            experiment_versions = self.experiment_versioner.version_history["versions"]
            all_versions.extend(experiment_versions)
        
        # Filter by item name if specified
        if item_name:
            all_versions = [v for v in all_versions if v["item_name"] == item_name]
        
        # Sort by timestamp
        return sorted(all_versions, key=lambda x: x["timestamp"], reverse=True)
    
    def create_snapshot(self, snapshot_name: str, description: str = "") -> str:
        """Create a complete project snapshot."""
        try:
            # Get current Git status
            git_status = self.git_manager.get_status()
            
            # Create snapshot metadata
            snapshot_metadata = {
                "snapshot_name": snapshot_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "git_status": git_status,
                "configs": self.config_versioner.version_history["configs"],
                "models": self.model_versioner.version_history["models"],
                "experiments": self.experiment_versioner.version_history["experiments"]
            }
            
            # Save snapshot
            snapshot_dir = self.project_root / "snapshots" / snapshot_name
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_file = snapshot_dir / "snapshot.json"
            with open(snapshot_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(snapshot_metadata, f, indent=2, default=str)
            
            # Create Git tag for snapshot
            self.git_manager.create_tag(f"snapshot-{snapshot_name}", description)
            
            self.logger.info("Project snapshot created", snapshot_name=snapshot_name)
            return snapshot_name
            
        except Exception as e:
            self.logger.error("Failed to create snapshot", error=str(e))
            raise
    
    def rollback_to_snapshot(self, snapshot_name: str) -> bool:
        """Rollback project to a specific snapshot."""
        try:
            snapshot_dir = self.project_root / "snapshots" / snapshot_name
            snapshot_file = snapshot_dir / "snapshot.json"
            
            if not snapshot_file.exists():
                self.logger.error("Snapshot not found", snapshot_name=snapshot_name)
                return False
            
            # Load snapshot metadata
            with open(snapshot_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                snapshot_metadata = json.load(f)
            
            # Checkout Git tag
            self.git_manager.repo.git.checkout(f"snapshot-{snapshot_name}")
            
            self.logger.info("Project rolled back to snapshot", snapshot_name=snapshot_name)
            return True
            
        except Exception as e:
            self.logger.error("Failed to rollback to snapshot", error=str(e))
            return False
    
    def get_project_status(self) -> Dict[str, Any]:
        """Get comprehensive project status."""
        git_status = self.git_manager.get_status()
        
        return {
            "project_info": self.vc_metadata["project_info"],
            "git_status": git_status,
            "version_counts": {
                "configs": len(self.config_versioner.version_history["configs"]),
                "models": len(self.model_versioner.version_history["models"]),
                "experiments": len(self.experiment_versioner.version_history["experiments"])
            },
            "recent_versions": self.list_versions()[:5]  # Last 5 versions
        }


# =============================================================================
# CONTEXT MANAGER FOR VERSION CONTROL
# =============================================================================

@contextmanager
def version_control(project_root: str = ".", auto_commit: bool = True):
    """
    Context manager for version control operations.
    
    Usage:
        with version_control("my_project") as vc:
            # Make changes
            vc.version_configuration("model_config", config_data)
            vc.version_model("transformer", "model.pt", metadata)
    """
    vc_system = VersionControlSystem(project_root, auto_commit)
    
    try:
        yield vc_system
    finally:
        # Save metadata
        vc_system._save_vc_metadata()


# =============================================================================
# INTEGRATION WITH EXPERIMENT TRACKING
# =============================================================================

class VersionedExperimentTracker:
    """Experiment tracker with integrated version control."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any], 
                 project_root: str = ".", auto_commit: bool = True):
        
    """__init__ function."""
        
        self.experiment_tracker = ExperimentTracker(experiment_name, config)
        self.vc_system = VersionControlSystem(project_root, auto_commit)
        self.logger = structlog.get_logger(__name__)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics with version control."""
        self.experiment_tracker.log_metrics(metrics, step)
    
    def save_checkpoint(self, model, epoch: int, step: int, **kwargs) -> str:
        """Save checkpoint with version control."""
        # Save checkpoint through experiment tracker
        checkpoint_path = self.experiment_tracker.save_checkpoint(
            model, epoch, step, **kwargs
        )
        
        # Version the model
        model_metadata = {
            "experiment_id": self.experiment_tracker.metadata.experiment_id,
            "epoch": epoch,
            "step": step,
            "checkpoint_path": checkpoint_path,
            **kwargs
        }
        
        self.vc_system.version_model(
            model_name=f"{self.experiment_tracker.metadata.experiment_id}_checkpoint",
            model_path=checkpoint_path,
            metadata=model_metadata,
            description=f"Checkpoint at epoch {epoch}, step {step}",
            tags=["checkpoint", f"epoch_{epoch}"]
        )
        
        return checkpoint_path
    
    def finish(self, status: str = "completed"):
        """Finish experiment with version control."""
        # Finish experiment tracking
        self.experiment_tracker.finish(status)
        
        # Version the experiment
        experiment_data = {
            "metadata": self.experiment_tracker.metadata.to_dict(),
            "config": self.experiment_tracker.config,
            "metrics_history": self.experiment_tracker.metrics_history
        }
        
        results = {
            "status": status,
            "final_metrics": {
                "total_steps": len(self.experiment_tracker.metrics_history["train_loss"]),
                "final_train_loss": self.experiment_tracker.metrics_history["train_loss"][-1] if self.experiment_tracker.metrics_history["train_loss"] else None,
                "final_val_loss": self.experiment_tracker.metrics_history["val_loss"][-1] if self.experiment_tracker.metrics_history["val_loss"] else None
            }
        }
        
        self.vc_system.version_experiment(
            experiment_id=self.experiment_tracker.metadata.experiment_id,
            experiment_data=experiment_data,
            results=results,
            description=f"Experiment completed with status: {status}",
            tags=["experiment", status]
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of the version control system."""
    
    # Example configuration
    config = {
        "model": {
            "type": "transformer",
            "name": "bert-base-uncased"
        },
        "training": {
            "epochs": 10,
            "learning_rate": 2e-5
        }
    }
    
    # Example usage with context manager
    with version_control("example_project", auto_commit=True) as vc:
        # Version configuration
        config_version = vc.version_configuration(
            config_name="transformer_config",
            config_data=config,
            description="Initial transformer configuration",
            tags=["transformer", "initial"]
        )
        
        print(f"Configuration versioned: {config_version}")
        
        # Version model (simulated)
        model_metadata = {
            "architecture": "transformer",
            "parameters": 110000000,
            "training_epochs": 5
        }
        
        model_version = vc.version_model(
            model_name="transformer_model",
            model_path="dummy_model.pt",  # Would be real model path
            metadata=model_metadata,
            description="Trained transformer model",
            tags=["transformer", "trained"]
        )
        
        print(f"Model versioned: {model_version}")
        
        # Create project snapshot
        snapshot_name = vc.create_snapshot(
            snapshot_name="v1.0.0",
            description="First stable version"
        )
        
        print(f"Snapshot created: {snapshot_name}")
        
        # Get project status
        status = vc.get_project_status()
        print(f"Project status: {status}")
    
    print("Version control example completed!")


match __name__:
    case "__main__":
    main() 