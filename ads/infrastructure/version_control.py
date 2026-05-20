"""
Unified Version Control System for the ads feature.

This module consolidates all version control functionality from the scattered implementations:
- version_control_manager.py (comprehensive git integration)

The new structure follows Clean Architecture principles with clear separation of concerns.
"""

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
import asyncio

from ...config import ConfigManager, ConfigType
from ...training import ExperimentTracker, ExperimentMetadata


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
        """Initialize the version control manager."""
        self.project_root = Path(project_root).resolve()
        self.auto_commit = auto_commit
        self.logger = logging.getLogger(__name__)
        
        # Verify git repository
        if not self._is_git_repository():
            self._initialize_git_repository()
        
        # Initialize config manager
        self.config_manager = ConfigManager(str(self.project_root / "configs"))
        
    def _is_git_repository(self) -> bool:
        """Check if the directory is a git repository."""
        git_dir = self.project_root / ".git"
        return git_dir.exists() and git_dir.is_dir()
    
    def _initialize_git_repository(self) -> None:
        """Initialize a new git repository."""
        try:
            subprocess.run(
                ["git", "init"],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            self.logger.info(f"Initialized git repository in {self.project_root}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to initialize git repository: {e}")
            raise
    
    def get_status(self) -> GitStatus:
        """Get the current git status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                return GitStatus.MODIFIED
            else:
                return GitStatus.CLEAN
        except subprocess.CalledProcessError:
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
        except subprocess.CalledProcessError:
            return "main"
    
    def get_latest_commit(self) -> Optional[GitCommit]:
        """Get information about the latest commit."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%H|%an|%ad|%s", "--date=iso"],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                parts = result.stdout.strip().split("|")
                if len(parts) >= 4:
                    return GitCommit(
                        hash=parts[0],
                        author=parts[1],
                        date=datetime.fromisoformat(parts[2]),
                        message=parts[3]
                    )
            return None
        except subprocess.CalledProcessError:
            return None
    
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> bool:
        """Commit changes to git."""
        # Avoid failing hard if running outside a repo or with no staged changes
        try:
            add_cmd = ["git", "add"] + (files or ["."])
            subprocess.run(add_cmd, cwd=self.project_root, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Continue even if add fails (e.g., not a git repo)
            pass

        try:
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            self.logger.info(f"Committed changes: {message}")
            return True
        except subprocess.CalledProcessError:
            # No changes to commit or not a repo
            self.logger.info("No changes committed (possibly not a git repo or nothing to commit)")
            return False
    
    def create_branch(self, branch_name: str, checkout: bool = True) -> bool:
        """Create a new git branch."""
        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            self.logger.info(f"Created and checked out branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create branch {branch_name}: {e}")
            return False
    
    def checkout_branch(self, branch_name: str) -> bool:
        """Checkout an existing git branch."""
        try:
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            self.logger.info(f"Checked out branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to checkout branch {branch_name}: {e}")
            return False
    
    def get_file_version(self, file_path: str) -> Optional[VersionInfo]:
        """Get version information for a specific file."""
        try:
            # Get git log for the file
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%H|%an|%ad|%s", "--follow", file_path],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                parts = result.stdout.strip().split("|")
                if len(parts) >= 4:
                    return VersionInfo(
                        file_path=file_path,
                        git_hash=parts[0],
                        commit_date=datetime.fromisoformat(parts[2]),
                        commit_message=parts[3],
                        branch=self.get_current_branch()
                    )
            return None
        except subprocess.CalledProcessError:
            return None
    
    def create_experiment_version(self, experiment_id: str, config_hash: str) -> ExperimentVersion:
        """Create a version record for an experiment."""
        latest_commit = self.get_latest_commit()
        if not latest_commit:
            raise ValueError("No git commits found")
        
        return ExperimentVersion(
            experiment_id=experiment_id,
            git_hash=latest_commit.hash,
            branch=self.get_current_branch(),
            config_hash=config_hash,
            code_hash=latest_commit.hash,
            commit_date=latest_commit.date,
            commit_message=latest_commit.message
        )
    
    def get_experiment_versions(self, experiment_id: str) -> List[ExperimentVersion]:
        """Get all versions for a specific experiment."""
        # This would typically query a database or file system
        # For now, return empty list as placeholder
        return []
    
    def rollback_to_version(self, commit_hash: str) -> bool:
        """Rollback to a specific commit version."""
        try:
            subprocess.run(
                ["git", "reset", "--hard", commit_hash],
                cwd=self.project_root,
                check=True,
                capture_output=True
            )
            self.logger.info(f"Rolled back to commit: {commit_hash}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to rollback to {commit_hash}: {e}")
            return False
    
    def get_diff(self, file_path: str, commit1: str = "HEAD", commit2: str = "HEAD~1") -> str:
        """Get diff between two commits for a specific file."""
        try:
            result = subprocess.run(
                ["git", "diff", commit1, commit2, "--", file_path],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""
    
    def get_repository_info(self) -> Dict[str, Any]:
        """Get comprehensive repository information."""
        return {
            "project_root": str(self.project_root),
            "current_branch": self.get_current_branch(),
            "status": self.get_status().value,
            "latest_commit": asdict(self.get_latest_commit()) if self.get_latest_commit() else None,
            "is_git_repository": self._is_git_repository(),
            "auto_commit": self.auto_commit
        }


class VersionControlService:
    """High-level service for version control operations."""
    
    def __init__(self, project_root: str = "."):
        """Initialize the version control service."""
        self.manager = VersionControlManager(project_root)
        self.logger = logging.getLogger(__name__)
    
    async def track_experiment(self, experiment_id: str, config: Dict[str, Any]) -> ExperimentVersion:
        """Track an experiment with version control."""
        try:
            # Create config hash
            config_hash = str(hash(json.dumps(config, sort_keys=True)))
            
            # Create experiment version
            version = self.manager.create_experiment_version(experiment_id, config_hash)
            
            # Auto-commit if enabled
            if self.manager.auto_commit:
                self.manager.commit_changes(f"Experiment {experiment_id}: {version.commit_message}")
            
            self.logger.info(f"Tracked experiment {experiment_id} with version {version.git_hash}")
            return version
        except Exception as e:
            self.logger.error(f"Failed to track experiment {experiment_id}: {e}")
            raise
    
    async def get_experiment_reproducibility(self, experiment_id: str) -> Dict[str, Any]:
        """Get reproducibility information for an experiment."""
        try:
            versions = self.manager.get_experiment_versions(experiment_id)
            if not versions:
                return {"reproducible": False, "reason": "No versions found"}
            
            latest_version = versions[-1]
            return {
                "reproducible": latest_version.is_reproducible,
                "git_hash": latest_version.git_hash,
                "branch": latest_version.branch,
                "commit_date": latest_version.commit_date.isoformat(),
                "dependencies": latest_version.dependencies
            }
        except Exception as e:
            self.logger.error(f"Failed to get reproducibility for {experiment_id}: {e}")
            return {"reproducible": False, "reason": str(e)}
    
    async def create_experiment_branch(self, experiment_id: str) -> bool:
        """Create a dedicated branch for an experiment."""
        try:
            branch_name = f"experiment/{experiment_id}"
            return self.manager.create_branch(branch_name)
        except Exception as e:
            self.logger.error(f"Failed to create experiment branch for {experiment_id}: {e}")
            return False


# Global utility functions
def get_version_control_manager(project_root: str = ".") -> VersionControlManager:
    """Get a global version control manager instance."""
    return VersionControlManager(project_root)


def get_version_control_service(project_root: str = ".") -> VersionControlService:
    """Get a global version control service instance."""
    return VersionControlService(project_root)
