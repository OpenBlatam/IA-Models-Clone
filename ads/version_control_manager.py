from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import logging
from enum import Enum
import shutil
import tempfile
from contextlib import contextmanager
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.config_manager import ConfigManager, ConfigType
from onyx.server.features.ads.experiment_tracker import ExperimentTracker, ExperimentMetadata
from typing import Any, List, Dict, Optional
import asyncio
"""
Version Control Manager for ML Projects

This module provides comprehensive version control management for ML projects including:
- Git integration for code and configuration tracking
- Automated commit management for experiments
- Configuration versioning and rollback
- Experiment reproducibility through git commits
- Branch management for different experiment versions
- Integration with experiment tracking system
"""


logger = setup_logger()

class GitStatus(Enum):
    """Git status enumeration."""
    CLEAN = "clean"
    MODIFIED = "modified"
    UNTRACKED = "untracked"
    CONFLICT = "conflict"
    AHEAD = "ahead"
    BEHIND = "behind"

@dataclass
class GitCommit:
    """Git commit information."""
    hash: str
    author: str
    date: datetime
    message: str
    files_changed: List[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0

@dataclass
class VersionInfo:
    """Version information for files and configurations."""
    file_path: str
    git_hash: str
    commit_date: datetime
    commit_message: str
    branch: str
    is_dirty: bool = False
    untracked_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)

@dataclass
class ExperimentVersion:
    """Experiment version information."""
    experiment_id: str
    git_hash: str
    branch: str
    config_hash: str
    code_hash: str
    commit_date: datetime
    commit_message: str
    is_reproducible: bool = True
    dependencies: Dict[str, str] = field(default_factory=dict)

class VersionControlManager:
    """Comprehensive version control manager for ML projects."""
    
    def __init__(self, project_root: str = ".", auto_commit: bool = True):
        
    """__init__ function."""
self.project_root = Path(project_root).resolve()
        self.auto_commit = auto_commit
        self.logger = logger
        
        # Verify git repository
        if not self._is_git_repository():
            self._initialize_git_repository()
        
        # Initialize config manager
        self.config_manager = ConfigManager(str(self.project_root / "configs"))
        
    def _is_git_repository(self) -> bool:
        """Check if the directory is a git repository."""
        git_dir = self.project_root / ".git"
        return git_dir.exists() and git_dir.is_dir()
    
    def _initialize_git_repository(self) -> Any:
        """Initialize a new git repository."""
        try:
            subprocess.run(
                ["git", "init"],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            # Create initial .gitignore
            self._create_gitignore()
            
            # Make initial commit
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            subprocess.run(
                ["git", "commit", "-m", "Initial commit: ML project setup"],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            self.logger.info(f"Initialized git repository at {self.project_root}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to initialize git repository: {e}")
            raise
    
    def _create_gitignore(self) -> Any:
        """Create a comprehensive .gitignore file for ML projects."""
        gitignore_content = """# ML Project .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/
.env/

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt
*.ckpt

# TensorFlow
*.h5
*.hdf5
*.pb
*.pkl

# Checkpoints and models
checkpoints/
models/
saved_models/
*.model

# Data
data/
datasets/
*.csv
*.json
*.parquet
*.h5
*.hdf5

# Logs
logs/
*.log
tensorboard_logs/
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
*.tmp

# Environment variables
.env
.env.local
.env.production

# Config files with secrets
config/secrets.yaml
config/local.yaml

# Large files
*.zip
*.tar.gz
*.rar

# Profiling outputs
profiles/
*.prof
"""
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(gitignore_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def get_git_status(self) -> GitStatus:
        """Get the current git status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if not result.stdout.strip():
                return GitStatus.CLEAN
            
            # Check for conflicts
            if "UU" in result.stdout or "AA" in result.stdout:
                return GitStatus.CONFLICT
            
            # Check for untracked files
            if any(line.startswith("??") for line in result.stdout.splitlines()):
                return GitStatus.UNTRACKED
            
            return GitStatus.MODIFIED
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get git status: {e}")
            return GitStatus.CONFLICT
    
    def get_current_branch(self) -> str:
        """Get the current git branch."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get current branch: {e}")
            return "main"
    
    def get_current_commit(self) -> GitCommit:
        """Get information about the current commit."""
        try:
            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            commit_hash = hash_result.stdout.strip()
            
            # Get commit details
            log_result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%H|%an|%ad|%s", "--date=iso"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if log_result.stdout.strip():
                parts = log_result.stdout.strip().split("|")
                if len(parts) >= 4:
                    hash_val, author, date_str, message = parts[:4]
                    commit_date = datetime.fromisoformat(date_str.replace(" ", "T"))
                    
                    return GitCommit(
                        hash=hash_val,
                        author=author,
                        date=commit_date,
                        message=message
                    )
            
            # Fallback
            return GitCommit(
                hash=commit_hash,
                author="Unknown",
                date=datetime.now(),
                message="Unknown commit"
            )
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get current commit: {e}")
            return GitCommit(
                hash="unknown",
                author="Unknown",
                date=datetime.now(),
                message="Failed to get commit info"
            )
    
    def get_file_version_info(self, file_path: str) -> VersionInfo:
        """Get version information for a specific file."""
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.project_root / file_path
        
        try:
            # Get git log for the file
            log_result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%H|%an|%ad|%s", "--date=iso", str(file_path)],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if log_result.stdout.strip():
                parts = log_result.stdout.strip().split("|")
                if len(parts) >= 4:
                    hash_val, author, date_str, message = parts[:4]
                    commit_date = datetime.fromisoformat(date_str.replace(" ", "T"))
                else:
                    hash_val, commit_date, message = "unknown", datetime.now(), "Unknown"
            else:
                hash_val, commit_date, message = "unknown", datetime.now(), "File not tracked"
            
            # Get current branch
            branch = self.get_current_branch()
            
            # Check if file is modified
            status_result = subprocess.run(
                ["git", "status", "--porcelain", str(file_path)],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            is_dirty = bool(status_result.stdout.strip())
            
            # Get untracked and modified files
            untracked_files = []
            modified_files = []
            
            if is_dirty:
                status_lines = status_result.stdout.strip().splitlines()
                for line in status_lines:
                    if line.startswith("??"):
                        untracked_files.append(line[3:])
                    elif line.startswith(" M") or line.startswith("M "):
                        modified_files.append(line[3:])
            
            return VersionInfo(
                file_path=str(file_path),
                git_hash=hash_val,
                commit_date=commit_date,
                commit_message=message,
                branch=branch,
                is_dirty=is_dirty,
                untracked_files=untracked_files,
                modified_files=modified_files
            )
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get file version info: {e}")
            return VersionInfo(
                file_path=str(file_path),
                git_hash="unknown",
                commit_date=datetime.now(),
                commit_message="Failed to get version info",
                branch="unknown",
                is_dirty=True
            )
    
    def commit_experiment(self, 
                         experiment_id: str,
                         experiment_name: str,
                         configs: Dict[str, Any],
                         message: Optional[str] = None) -> str:
        """Commit experiment changes with proper versioning."""
        try:
            # Create commit message
            if message is None:
                message = f"Experiment: {experiment_name} ({experiment_id})"
            
            # Add all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            # Check if there are changes to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if not status_result.stdout.strip():
                self.logger.info("No changes to commit")
                return self.get_current_commit().hash
            
            # Create commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            commit_hash = self.get_current_commit().hash
            
            # Save experiment version info
            self._save_experiment_version(experiment_id, commit_hash, configs)
            
            self.logger.info(f"Committed experiment {experiment_id} with hash {commit_hash}")
            return commit_hash
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to commit experiment: {e}")
            raise
    
    def create_experiment_branch(self, 
                                experiment_id: str,
                                base_branch: str = "main") -> str:
        """Create a new branch for an experiment."""
        branch_name = f"experiment/{experiment_id}"
        
        try:
            # Checkout base branch
            subprocess.run(
                ["git", "checkout", base_branch],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            # Create and checkout new branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            self.logger.info(f"Created experiment branch: {branch_name}")
            return branch_name
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create experiment branch: {e}")
            raise
    
    def switch_to_branch(self, branch_name: str) -> bool:
        """Switch to a specific branch."""
        try:
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            self.logger.info(f"Switched to branch: {branch_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return False
    
    def get_experiment_version(self, experiment_id: str) -> Optional[ExperimentVersion]:
        """Get version information for an experiment."""
        version_file = self.project_root / "experiment_versions" / f"{experiment_id}.json"
        
        if not version_file.exists():
            return None
        
        try:
            with open(version_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                data = json.load(f)
            
            return ExperimentVersion(
                experiment_id=data['experiment_id'],
                git_hash=data['git_hash'],
                branch=data['branch'],
                config_hash=data['config_hash'],
                code_hash=data['code_hash'],
                commit_date=datetime.fromisoformat(data['commit_date']),
                commit_message=data['commit_message'],
                is_reproducible=data.get('is_reproducible', True),
                dependencies=data.get('dependencies', {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment version: {e}")
            return None
    
    def reproduce_experiment(self, experiment_id: str) -> bool:
        """Reproduce an experiment by checking out its exact version."""
        version_info = self.get_experiment_version(experiment_id)
        
        if not version_info:
            self.logger.error(f"Version info not found for experiment {experiment_id}")
            return False
        
        try:
            # Checkout the specific commit
            subprocess.run(
                ["git", "checkout", version_info.git_hash],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            self.logger.info(f"Reproduced experiment {experiment_id} at commit {version_info.git_hash}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to reproduce experiment: {e}")
            return False
    
    def get_config_changes(self, config_path: str, since_commit: str = "HEAD~1") -> List[Dict[str, Any]]:
        """Get changes made to a configuration file."""
        try:
            result = subprocess.run(
                ["git", "diff", since_commit, "HEAD", "--", config_path],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if not result.stdout.strip():
                return []
            
            # Parse diff output
            changes = []
            current_change = {}
            
            for line in result.stdout.splitlines():
                if line.startswith("diff --git"):
                    if current_change:
                        changes.append(current_change)
                    current_change = {"type": "config_change", "file": config_path}
                elif line.startswith("+") and not line.startswith("+++"):
                    current_change.setdefault("additions", []).append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    current_change.setdefault("deletions", []).append(line[1:])
            
            if current_change:
                changes.append(current_change)
            
            return changes
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get config changes: {e}")
            return []
    
    def create_config_snapshot(self, configs: Dict[str, Any]) -> str:
        """Create a snapshot of current configurations."""
        snapshot_dir = self.project_root / "config_snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = snapshot_dir / f"config_snapshot_{timestamp}.yaml"
        
        # Add metadata
        snapshot_data = {
            "timestamp": timestamp,
            "git_hash": self.get_current_commit().hash,
            "branch": self.get_current_branch(),
            "configs": configs
        }
        
        with open(snapshot_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(snapshot_data, f, default_flow_style=False, indent=2)
        
        return str(snapshot_file)
    
    def restore_config_snapshot(self, snapshot_file: str) -> bool:
        """Restore configurations from a snapshot."""
        try:
            with open(snapshot_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                snapshot_data = yaml.safe_load(f)
            
            configs = snapshot_data.get('configs', {})
            
            # Restore each configuration
            for config_type, config in configs.items():
                if config_type in ['model', 'training', 'data', 'experiment', 'optimization', 'deployment']:
                    config_file = self.project_root / "configs" / f"{config_type}_config.yaml"
                    with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Restored configurations from snapshot: {snapshot_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore config snapshot: {e}")
            return False
    
    def get_experiment_history(self, experiment_id: str) -> List[GitCommit]:
        """Get commit history for an experiment."""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--grep", experiment_id],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            commits = []
            for line in result.stdout.splitlines():
                if line.strip():
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        commit_hash, message = parts
                        commits.append(GitCommit(
                            hash=commit_hash,
                            author="Unknown",
                            date=datetime.now(),
                            message=message
                        ))
            
            return commits
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get experiment history: {e}")
            return []
    
    def _save_experiment_version(self, experiment_id: str, commit_hash: str, configs: Dict[str, Any]):
        """Save experiment version information."""
        version_dir = self.project_root / "experiment_versions"
        version_dir.mkdir(exist_ok=True)
        
        # Calculate config hash
        config_str = json.dumps(configs, sort_keys=True)
        config_hash = str(hash(config_str))
        
        # Calculate code hash (hash of tracked files)
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            code_files = result.stdout.splitlines()
            code_str = "\n".join(sorted(code_files))
            code_hash = str(hash(code_str))
            
        except subprocess.CalledProcessError:
            code_hash = "unknown"
        
        version_info = ExperimentVersion(
            experiment_id=experiment_id,
            git_hash=commit_hash,
            branch=self.get_current_branch(),
            config_hash=config_hash,
            code_hash=code_hash,
            commit_date=datetime.now(),
            commit_message=self.get_current_commit().message,
            is_reproducible=True,
            dependencies=self._get_dependencies()
        )
        
        version_file = version_dir / f"{experiment_id}.json"
        with open(version_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(asdict(version_info), f, indent=2, default=str)
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current project dependencies."""
        dependencies = {}
        
        # Check for requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "==" in line:
                            package, version = line.split("==", 1)
                            dependencies[package.strip()] = version.strip()
        
        return dependencies
    
    def tag_experiment(self, experiment_id: str, tag_name: str, message: str = "") -> bool:
        """Create a git tag for an experiment."""
        try:
            tag_message = f"Experiment {experiment_id}: {message}" if message else f"Experiment {experiment_id}"
            
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", tag_message],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            
            self.logger.info(f"Created tag {tag_name} for experiment {experiment_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create tag: {e}")
            return False
    
    def get_experiment_tags(self, experiment_id: str) -> List[str]:
        """Get all tags for an experiment."""
        try:
            result = subprocess.run(
                ["git", "tag", "--list", f"*{experiment_id}*"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            return [tag.strip() for tag in result.stdout.splitlines() if tag.strip()]
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get experiment tags: {e}")
            return []

class ExperimentVersionControl:
    """High-level interface for experiment version control."""
    
    def __init__(self, project_root: str = ".", auto_commit: bool = True):
        
    """__init__ function."""
self.vc_manager = VersionControlManager(project_root, auto_commit)
        self.logger = logger
    
    def start_experiment(self, 
                        experiment_id: str,
                        experiment_name: str,
                        configs: Dict[str, Any],
                        create_branch: bool = True) -> str:
        """Start a new experiment with version control."""
        try:
            # Create experiment branch if requested
            if create_branch:
                self.vc_manager.create_experiment_branch(experiment_id)
            
            # Create config snapshot
            snapshot_file = self.vc_manager.create_config_snapshot(configs)
            
            # Commit initial state
            commit_hash = self.vc_manager.commit_experiment(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                configs=configs,
                message=f"Start experiment: {experiment_name}"
            )
            
            self.logger.info(f"Started experiment {experiment_id} with version control")
            return commit_hash
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment with version control: {e}")
            raise
    
    def commit_experiment_changes(self,
                                 experiment_id: str,
                                 experiment_name: str,
                                 configs: Dict[str, Any],
                                 message: str = "") -> str:
        """Commit changes during an experiment."""
        commit_message = f"Experiment {experiment_id} update: {message}" if message else f"Experiment {experiment_id} update"
        
        return self.vc_manager.commit_experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            configs=configs,
            message=commit_message
        )
    
    def end_experiment(self,
                      experiment_id: str,
                      experiment_name: str,
                      configs: Dict[str, Any],
                      final_metrics: Dict[str, float],
                      tag_name: Optional[str] = None) -> str:
        """End an experiment with version control."""
        # Commit final state
        commit_hash = self.vc_manager.commit_experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            configs=configs,
            message=f"End experiment: {experiment_name} - Final metrics: {final_metrics}"
        )
        
        # Create tag if specified
        if tag_name:
            self.vc_manager.tag_experiment(experiment_id, tag_name, f"Final version with metrics: {final_metrics}")
        
        self.logger.info(f"Ended experiment {experiment_id} with version control")
        return commit_hash
    
    def reproduce_experiment(self, experiment_id: str) -> bool:
        """Reproduce an experiment from its exact version."""
        return self.vc_manager.reproduce_experiment(experiment_id)
    
    def get_experiment_info(self, experiment_id: str) -> Optional[ExperimentVersion]:
        """Get detailed information about an experiment version."""
        return self.vc_manager.get_experiment_version(experiment_id)

# Utility functions
def create_version_control_manager(project_root: str = ".", auto_commit: bool = True) -> VersionControlManager:
    """Create a version control manager."""
    return VersionControlManager(project_root, auto_commit)

def create_experiment_vc(project_root: str = ".", auto_commit: bool = True) -> ExperimentVersionControl:
    """Create an experiment version control interface."""
    return ExperimentVersionControl(project_root, auto_commit)

@contextmanager
def experiment_version_context(experiment_id: str, experiment_name: str, configs: Dict[str, Any]):
    """Context manager for experiment version control."""
    vc = create_experiment_vc()
    try:
        commit_hash = vc.start_experiment(experiment_id, experiment_name, configs)
        yield vc, commit_hash
    finally:
        pass

# Example usage
if __name__ == "__main__":
    # Create version control manager
    vc_manager = create_version_control_manager()
    
    # Get current status
    status = vc_manager.get_git_status()
    print(f"Git status: {status.value}")
    
    # Get current commit
    commit = vc_manager.get_current_commit()
    print(f"Current commit: {commit.hash} - {commit.message}")
    
    # Get file version info
    version_info = vc_manager.get_file_version_info("config_manager.py")
    print(f"File version: {version_info.git_hash} - {version_info.commit_message}")
    
    # Create experiment version control
    exp_vc = create_experiment_vc()
    
    # Example experiment
    configs = {
        "model": {"name": "test_model", "type": "transformer"},
        "training": {"batch_size": 32, "learning_rate": 1e-4}
    }
    
    # Start experiment
    commit_hash = exp_vc.start_experiment("exp_001", "Test Experiment", configs)
    print(f"Started experiment with commit: {commit_hash}")
    
    # End experiment
    final_metrics = {"accuracy": 0.95, "loss": 0.05}
    final_commit = exp_vc.end_experiment("exp_001", "Test Experiment", configs, final_metrics, "v1.0")
    print(f"Ended experiment with commit: {final_commit}") 