from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from version_control import GitManager, ConfigVersioning, ChangeTracker
from .git_manager import (
from .config_versioning import (
from .change_tracker import (
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional 
from typing import Any, List, Dict, Optional
import asyncio
"""
Version Control Module
=====================

This module provides comprehensive version control for AI video generation projects.

Features:
- Git integration and management
- Configuration versioning
- Change tracking and monitoring
- Diff generation and visualization
- Version history and rollback
- Automatic backup and restore
- Experiment tracking integration

Quick Start:
-----------
```python

# Initialize git manager
git_mgr = GitManager()
git_mgr.create_branch("feature/new_model", "feature")

# Version configurations
config_versioning = ConfigVersioning()
version_id = config_versioning.create_version(
    "config.yaml", 
    "Updated model parameters",
    author="developer"
)

# Track changes
change_tracker = ChangeTracker()
change_tracker.start_monitoring(["./models", "./config"])
```

Components:
----------
- GitManager: Git repository management
- ConfigVersioning: Configuration file versioning
- ChangeTracker: File change monitoring and tracking
"""

    GitManager,
    GitConfig,
    ChangeInfo,
    CommitInfo,
    create_git_manager
)

    ConfigVersioning,
    ConfigVersion,
    ConfigDiff,
    create_config_versioning,
    version_config
)

    ChangeTracker,
    FileChange,
    ChangeSet,
    FileChangeHandler,
    create_change_tracker,
    track_file_change
)

# Version control utilities
class VersionControlSystem:
    """Integrated version control system."""
    
    def __init__(
        self,
        repo_path: str = ".",
        config_dir: str = "config_versions",
        change_dir: str = "change_history",
        auto_commit: bool = True
    ):
        
    """__init__ function."""
self.git_manager = create_git_manager(repo_path, auto_commit)
        self.config_versioning = create_config_versioning(config_dir)
        self.change_tracker = create_change_tracker(change_dir)
        
        logger.info("Version control system initialized")
    
    def version_config_and_commit(
        self,
        config_file: str,
        description: str,
        author: str = "system",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[str, str]:
        """Version a configuration and commit to git."""
        # Create configuration version
        version_id = self.config_versioning.create_version(
            config_file, description, author, tags, metadata
        )
        
        # Commit to git
        commit_success = self.git_manager.commit_changes(
            f"Version configuration: {description}",
            "config",
            {"version_id": version_id, "config_file": config_file}
        )
        
        return version_id, commit_success
    
    def track_and_commit_changes(
        self,
        description: str,
        author: str = "system",
        tags: List[str] = None
    ) -> str:
        """Track changes and create a commit."""
        # Create change set
        change_set_id = self.change_tracker.create_change_set(
            description, author, tags
        )
        
        # Commit to git
        self.git_manager.commit_changes(
            f"Track changes: {description}",
            "track",
            {"change_set_id": change_set_id}
        )
        
        return change_set_id
    
    def get_project_history(
        self,
        since: str = None,
        until: str = None
    ) -> Dict[str, Any]:
        """Get comprehensive project history."""
        # Git commits
        commits = self.git_manager.get_recent_commits(50)
        
        # Configuration versions
        config_versions = self.config_versioning.list_versions(limit=50)
        
        # Change sets
        change_sets = self.change_tracker.get_change_sets(limit=50)
        
        # Filter by time if specified
        if since:
            since_dt = datetime.fromisoformat(since)
            commits = [c for c in commits if datetime.fromisoformat(c.timestamp) >= since_dt]
            config_versions = [v for v in config_versions if datetime.fromisoformat(v.timestamp) >= since_dt]
            change_sets = [cs for cs in change_sets if datetime.fromisoformat(cs.timestamp) >= since_dt]
        
        if until:
            until_dt = datetime.fromisoformat(until)
            commits = [c for c in commits if datetime.fromisoformat(c.timestamp) <= until_dt]
            config_versions = [v for v in config_versions if datetime.fromisoformat(v.timestamp) <= until_dt]
            change_sets = [cs for cs in change_sets if datetime.fromisoformat(cs.timestamp) <= until_dt]
        
        return {
            "git_commits": [c.to_dict() for c in commits],
            "config_versions": [v.to_dict() for v in config_versions],
            "change_sets": [cs.to_dict() for cs in change_sets],
            "summary": {
                "total_commits": len(commits),
                "total_config_versions": len(config_versions),
                "total_change_sets": len(change_sets)
            }
        }
    
    def create_experiment_branch(
        self,
        experiment_name: str,
        config_file: str,
        description: str
    ) -> str:
        """Create an experiment branch with configuration versioning."""
        # Create experiment branch
        branch_name = f"experiment/{experiment_name}"
        success = self.git_manager.create_branch(branch_name, "experiment")
        
        if not success:
            raise ValueError(f"Failed to create branch: {branch_name}")
        
        # Version the configuration
        version_id = self.config_versioning.create_version(
            config_file,
            f"Experiment: {description}",
            author="experiment",
            tags=["experiment", experiment_name],
            metadata={"experiment_name": experiment_name}
        )
        
        # Commit changes
        self.git_manager.commit_changes(
            f"Start experiment: {experiment_name}",
            "experiment",
            {
                "experiment_name": experiment_name,
                "config_version": version_id,
                "description": description
            }
        )
        
        return version_id
    
    def finish_experiment(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        merge_to_main: bool = False
    ) -> bool:
        """Finish an experiment and optionally merge to main."""
        # Create results commit
        self.git_manager.commit_changes(
            f"Experiment results: {experiment_name}",
            "results",
            {"experiment_name": experiment_name, "results": results}
        )
        
        # Merge to main if requested
        if merge_to_main:
            current_branch = self.git_manager.get_current_branch()
            success = self.git_manager.merge_branch(current_branch, "main")
            
            if success:
                # Switch back to main
                self.git_manager.switch_branch("main")
                return True
            else:
                logger.error(f"Failed to merge experiment {experiment_name}")
                return False
        
        return True
    
    def get_experiment_history(self, experiment_name: str) -> Dict[str, Any]:
        """Get history for a specific experiment."""
        # Get experiment-related commits
        commits = self.git_manager.get_recent_commits(100)
        experiment_commits = [
            c for c in commits
            if experiment_name in c.message or 
               (c.message and "experiment" in c.message.lower())
        ]
        
        # Get experiment-related config versions
        config_versions = self.config_versioning.search_versions(experiment_name)
        
        # Get experiment-related change sets
        change_sets = self.change_tracker.get_change_sets(tags=[experiment_name])
        
        return {
            "experiment_name": experiment_name,
            "commits": [c.to_dict() for c in experiment_commits],
            "config_versions": [v.to_dict() for v in config_versions],
            "change_sets": [cs.to_dict() for cs in change_sets],
            "summary": {
                "total_commits": len(experiment_commits),
                "total_config_versions": len(config_versions),
                "total_change_sets": len(change_sets)
            }
        }


# Convenience functions
def create_version_control_system(
    repo_path: str = ".",
    config_dir: str = "config_versions",
    change_dir: str = "change_history",
    auto_commit: bool = True
) -> VersionControlSystem:
    """Create an integrated version control system."""
    return VersionControlSystem(repo_path, config_dir, change_dir, auto_commit)


def quick_version_config(
    config_file: str,
    description: str,
    author: str = "system",
    auto_commit: bool = True
) -> str:
    """Quick function to version a configuration and commit to git."""
    vcs = create_version_control_system(auto_commit=auto_commit)
    version_id, _ = vcs.version_config_and_commit(config_file, description, author)
    return version_id


def start_experiment(
    experiment_name: str,
    config_file: str,
    description: str
) -> str:
    """Start a new experiment with version control."""
    vcs = create_version_control_system()
    return vcs.create_experiment_branch(experiment_name, config_file, description)


def finish_experiment(
    experiment_name: str,
    results: Dict[str, Any],
    merge_to_main: bool = False
) -> bool:
    """Finish an experiment with version control."""
    vcs = create_version_control_system()
    return vcs.finish_experiment(experiment_name, results, merge_to_main)


# Export main components
__all__ = [
    # Core classes
    "GitManager",
    "ConfigVersioning", 
    "ChangeTracker",
    "VersionControlSystem",
    
    # Data classes
    "GitConfig",
    "ChangeInfo",
    "CommitInfo",
    "ConfigVersion",
    "ConfigDiff",
    "FileChange",
    "ChangeSet",
    
    # Convenience functions
    "create_git_manager",
    "create_config_versioning",
    "create_change_tracker",
    "create_version_control_system",
    "version_config",
    "track_file_change",
    "quick_version_config",
    "start_experiment",
    "finish_experiment"
]

# Import logging for convenience
logger = logging.getLogger(__name__)

# Import datetime for convenience functions