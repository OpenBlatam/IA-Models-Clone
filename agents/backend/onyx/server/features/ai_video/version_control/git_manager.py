from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import subprocess
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import difflib
from typing import Any, List, Dict, Optional
import asyncio
"""
Git Version Control Manager
==========================

This module provides comprehensive git version control management for AI video generation projects.

Features:
- Automated git operations
- Configuration versioning
- Change tracking and diff generation
- Branch management
- Commit message generation
- Experiment tracking integration
- Configuration backup and restore
"""


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GitConfig:
    """Configuration for git operations."""
    
    # Repository settings
    repo_path: str = "."
    auto_commit: bool = True
    auto_push: bool = False
    commit_message_template: str = "[{type}] {description}"
    
    # Branch settings
    main_branch: str = "main"
    feature_branch_prefix: str = "feature/"
    experiment_branch_prefix: str = "experiment/"
    hotfix_branch_prefix: str = "hotfix/"
    
    # File tracking
    tracked_files: List[str] = field(default_factory=lambda: [
        "*.py", "*.yaml", "*.yml", "*.json", "*.md", "*.txt"
    ])
    ignored_files: List[str] = field(default_factory=lambda: [
        "__pycache__", "*.pyc", "*.pyo", "*.pyd", ".pytest_cache",
        ".coverage", "htmlcov", ".tox", ".venv", "venv", "env",
        "node_modules", ".next", "dist", "build", "*.log",
        "checkpoints", "outputs", "logs", "artifacts", "runs"
    ])
    
    # Commit settings
    min_changes_for_commit: int = 1
    max_commit_message_length: int = 72
    include_timestamp: bool = True
    include_experiment_info: bool = True


@dataclass
class ChangeInfo:
    """Information about a change."""
    
    file_path: str
    change_type: str  # "added", "modified", "deleted", "renamed"
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    diff: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeInfo':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CommitInfo:
    """Information about a commit."""
    
    commit_hash: str
    author: str
    timestamp: str
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    branch: str
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommitInfo':
        """Create from dictionary."""
        return cls(**data)


class GitManager:
    """Main git version control manager."""
    
    def __init__(self, config: GitConfig):
        
    """__init__ function."""
self.config = config
        self.repo_path = Path(config.repo_path).resolve()
        self.changes: List[ChangeInfo] = []
        
        # Initialize git repository if needed
        self._ensure_git_repo()
        
        logger.info(f"Git manager initialized: {self.repo_path}")
    
    def _ensure_git_repo(self) -> Any:
        """Ensure git repository exists."""
        if not (self.repo_path / ".git").exists():
            logger.info("Initializing git repository")
            self._run_git_command(["init"])
            
            # Create initial .gitignore
            self._create_gitignore()
            
            # Make initial commit
            self._run_git_command(["add", "."])
            self._run_git_command(["commit", "-m", "Initial commit"])
    
    def _create_gitignore(self) -> Any:
        """Create .gitignore file."""
        gitignore_content = """# Python
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
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# AI Video specific
checkpoints/
outputs/
logs/
artifacts/
runs/
experiment_exports/
*.pth
*.ckpt
*.safetensors

# Temporary files
*.tmp
*.temp
*.log
*.out

# Configuration backups
config_backups/
"""
        
        gitignore_path = self.repo_path / ".gitignore"
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
    
    def _run_git_command(self, args: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Run git command and return result."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            logger.error(f"Failed to run git command: {e}")
            return -1, "", str(e)
    
    def get_status(self) -> Dict[str, Any]:
        """Get git repository status."""
        returncode, stdout, stderr = self._run_git_command(["status", "--porcelain"])
        
        if returncode != 0:
            logger.error(f"Failed to get git status: {stderr}")
            return {}
        
        status = {
            "modified": [],
            "added": [],
            "deleted": [],
            "renamed": [],
            "untracked": []
        }
        
        for line in stdout.strip().split('\n'):
            if not line:
                continue
            
            status_code = line[:2]
            file_path = line[3:]
            
            if status_code == "M ":
                status["modified"].append(file_path)
            elif status_code == "A ":
                status["added"].append(file_path)
            elif status_code == "D ":
                status["deleted"].append(file_path)
            elif status_code == "R ":
                status["renamed"].append(file_path)
            elif status_code == "??":
                status["untracked"].append(file_path)
        
        return status
    
    def get_current_branch(self) -> str:
        """Get current branch name."""
        returncode, stdout, stderr = self._run_git_command(["branch", "--show-current"])
        
        if returncode != 0:
            logger.error(f"Failed to get current branch: {stderr}")
            return "main"
        
        return stdout.strip() or "main"
    
    def create_branch(self, branch_name: str, branch_type: str = "feature") -> bool:
        """Create a new branch."""
        if branch_type == "feature":
            full_branch_name = f"{self.config.feature_branch_prefix}{branch_name}"
        elif branch_type == "experiment":
            full_branch_name = f"{self.config.experiment_branch_prefix}{branch_name}"
        elif branch_type == "hotfix":
            full_branch_name = f"{self.config.hotfix_branch_prefix}{branch_name}"
        else:
            full_branch_name = branch_name
        
        # Check if branch exists
        returncode, stdout, stderr = self._run_git_command(["branch", "--list", full_branch_name])
        
        if full_branch_name in stdout:
            logger.warning(f"Branch {full_branch_name} already exists")
            return False
        
        # Create and checkout branch
        returncode, stdout, stderr = self._run_git_command(["checkout", "-b", full_branch_name])
        
        if returncode == 0:
            logger.info(f"Created and switched to branch: {full_branch_name}")
            return True
        else:
            logger.error(f"Failed to create branch: {stderr}")
            return False
    
    def switch_branch(self, branch_name: str) -> bool:
        """Switch to a different branch."""
        returncode, stdout, stderr = self._run_git_command(["checkout", branch_name])
        
        if returncode == 0:
            logger.info(f"Switched to branch: {branch_name}")
            return True
        else:
            logger.error(f"Failed to switch branch: {stderr}")
            return False
    
    def merge_branch(self, source_branch: str, target_branch: str = None) -> bool:
        """Merge a branch into the current branch."""
        if target_branch is None:
            target_branch = self.get_current_branch()
        
        # Switch to target branch
        if not self.switch_branch(target_branch):
            return False
        
        # Merge source branch
        returncode, stdout, stderr = self._run_git_command(["merge", source_branch])
        
        if returncode == 0:
            logger.info(f"Merged {source_branch} into {target_branch}")
            return True
        else:
            logger.error(f"Failed to merge branch: {stderr}")
            return False
    
    def get_file_diff(self, file_path: str) -> Optional[str]:
        """Get diff for a specific file."""
        returncode, stdout, stderr = self._run_git_command(["diff", file_path])
        
        if returncode == 0:
            return stdout
        else:
            logger.warning(f"Failed to get diff for {file_path}: {stderr}")
            return None
    
    def get_file_history(self, file_path: str, max_commits: int = 10) -> List[CommitInfo]:
        """Get commit history for a specific file."""
        returncode, stdout, stderr = self._run_git_command([
            "log", f"-{max_commits}", "--pretty=format:%H|%an|%ad|%s", "--date=iso", file_path
        ])
        
        if returncode != 0:
            logger.error(f"Failed to get file history: {stderr}")
            return []
        
        commits = []
        for line in stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 4:
                commit = CommitInfo(
                    commit_hash=parts[0],
                    author=parts[1],
                    timestamp=parts[2],
                    message=parts[3],
                    files_changed=[file_path],
                    insertions=0,
                    deletions=0,
                    branch=self.get_current_branch()
                )
                commits.append(commit)
        
        return commits
    
    def stage_file(self, file_path: str) -> bool:
        """Stage a file for commit."""
        returncode, stdout, stderr = self._run_git_command(["add", file_path])
        
        if returncode == 0:
            logger.info(f"Staged file: {file_path}")
            return True
        else:
            logger.error(f"Failed to stage file {file_path}: {stderr}")
            return False
    
    def stage_all_changes(self) -> bool:
        """Stage all changes."""
        returncode, stdout, stderr = self._run_git_command(["add", "."])
        
        if returncode == 0:
            logger.info("Staged all changes")
            return True
        else:
            logger.error(f"Failed to stage changes: {stderr}")
            return False
    
    def commit_changes(
        self,
        message: str,
        change_type: str = "update",
        experiment_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Commit staged changes."""
        # Generate commit message
        commit_message = self._generate_commit_message(message, change_type, experiment_info)
        
        # Check if there are changes to commit
        status = self.get_status()
        total_changes = sum(len(files) for files in status.values())
        
        if total_changes < self.config.min_changes_for_commit:
            logger.info("No changes to commit")
            return True
        
        # Commit changes
        returncode, stdout, stderr = self._run_git_command(["commit", "-m", commit_message])
        
        if returncode == 0:
            logger.info(f"Committed changes: {commit_message}")
            
            # Auto push if enabled
            if self.config.auto_push:
                self.push_changes()
            
            return True
        else:
            logger.error(f"Failed to commit changes: {stderr}")
            return False
    
    def _generate_commit_message(
        self,
        message: str,
        change_type: str,
        experiment_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate commit message."""f"
        # Base message
        commit_message = self.config.commit_message_template",
            description=message
        )
        
        # Add timestamp if enabled
        if self.config.include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message += f" [{timestamp}]"
        
        # Add experiment info if provided
        if self.config.include_experiment_info and experiment_info:
            exp_info = f" | Exp: {experiment_info.get('name', 'unknown')}"
            if 'metrics' in experiment_info:
                exp_info += f" | Loss: {experiment_info['metrics'].get('loss', 'N/A')}"
            commit_message += exp_info
        
        # Truncate if too long
        if len(commit_message) > self.config.max_commit_message_length:
            commit_message = commit_message[:self.config.max_commit_message_length-3] + "..."
        
        return commit_message
    
    def push_changes(self, remote: str = "origin", branch: str = None) -> bool:
        """Push changes to remote repository."""
        if branch is None:
            branch = self.get_current_branch()
        
        returncode, stdout, stderr = self._run_git_command(["push", remote, branch])
        
        if returncode == 0:
            logger.info(f"Pushed changes to {remote}/{branch}")
            return True
        else:
            logger.error(f"Failed to push changes: {stderr}")
            return False
    
    def pull_changes(self, remote: str = "origin", branch: str = None) -> bool:
        """Pull changes from remote repository."""
        if branch is None:
            branch = self.get_current_branch()
        
        returncode, stdout, stderr = self._run_git_command(["pull", remote, branch])
        
        if returncode == 0:
            logger.info(f"Pulled changes from {remote}/{branch}")
            return True
        else:
            logger.error(f"Failed to pull changes: {stderr}")
            return False
    
    def create_tag(self, tag_name: str, message: str = "") -> bool:
        """Create a git tag."""
        args = ["tag", tag_name]
        if message:
            args.extend(["-m", message])
        
        returncode, stdout, stderr = self._run_git_command(args)
        
        if returncode == 0:
            logger.info(f"Created tag: {tag_name}")
            return True
        else:
            logger.error(f"Failed to create tag: {stderr}")
            return False
    
    def get_recent_commits(self, max_commits: int = 10) -> List[CommitInfo]:
        """Get recent commits."""
        returncode, stdout, stderr = self._run_git_command([
            "log", f"-{max_commits}", "--pretty=format:%H|%an|%ad|%s|%b", "--date=iso"
        ])
        
        if returncode != 0:
            logger.error(f"Failed to get recent commits: {stderr}")
            return []
        
        commits = []
        for line in stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 4:
                commit = CommitInfo(
                    commit_hash=parts[0],
                    author=parts[1],
                    timestamp=parts[2],
                    message=parts[3],
                    files_changed=[],
                    insertions=0,
                    deletions=0,
                    branch=self.get_current_branch()
                )
                commits.append(commit)
        
        return commits
    
    def track_config_changes(self, config_file: str, config_data: Dict[str, Any]) -> bool:
        """Track configuration changes."""
        config_path = Path(config_file)
        
        # Check if file exists and has changed
        if config_path.exists():
            with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                old_content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        else:
            old_content = ""
        
        # Write new content
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config_data, f, indent=2)
        
        # Stage and commit if auto-commit is enabled
        if self.config.auto_commit:
            self.stage_file(str(config_path))
            self.commit_changes(
                f"Update configuration: {config_path.name}",
                "config",
                {"config_file": config_path.name}
            )
        
        return True
    
    def backup_config(self, config_file: str, backup_dir: str = "config_backups") -> str:
        """Create a backup of a configuration file."""
        config_path = Path(config_file)
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{config_path.stem}_{timestamp}{config_path.suffix}"
        backup_file = backup_path / backup_filename
        
        # Copy file
        shutil.copy2(config_path, backup_file)
        
        # Commit backup
        if self.config.auto_commit:
            self.stage_file(str(backup_file))
            self.commit_changes(f"Backup configuration: {config_path.name}", "backup")
        
        logger.info(f"Configuration backed up to: {backup_file}")
        return str(backup_file)
    
    def restore_config(self, backup_file: str, target_file: str) -> bool:
        """Restore a configuration from backup."""
        backup_path = Path(backup_file)
        target_path = Path(target_file)
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False
        
        # Create backup of current file
        if target_path.exists():
            self.backup_config(str(target_path))
        
        # Restore from backup
        shutil.copy2(backup_path, target_path)
        
        # Commit restoration
        if self.config.auto_commit:
            self.stage_file(str(target_path))
            self.commit_changes(f"Restore configuration: {target_path.name}", "restore")
        
        logger.info(f"Configuration restored from: {backup_file}")
        return True
    
    def get_repository_info(self) -> Dict[str, Any]:
        """Get repository information."""
        current_branch = self.get_current_branch()
        status = self.get_status()
        recent_commits = self.get_recent_commits(5)
        
        return {
            "repository_path": str(self.repo_path),
            "current_branch": current_branch,
            "status": status,
            "recent_commits": [commit.to_dict() for commit in recent_commits],
            "total_changes": sum(len(files) for files in status.values()),
            "last_commit": recent_commits[0].to_dict() if recent_commits else None
        }


# Convenience functions
def create_git_manager(
    repo_path: str = ".",
    auto_commit: bool = True,
    auto_push: bool = False
) -> GitManager:
    """Create git manager with default settings."""
    config = GitConfig(
        repo_path=repo_path,
        auto_commit=auto_commit,
        auto_push=auto_push
    )
    return GitManager(config)


if __name__ == "__main__":
    # Example usage
    print("🔧 Git Version Control Manager")
    print("=" * 40)
    
    # Create git manager
    git_mgr = create_git_manager(auto_commit=True)
    
    # Get repository status
    status = git_mgr.get_status()
    print(f"Repository status: {status}")
    
    # Get current branch
    current_branch = git_mgr.get_current_branch()
    print(f"Current branch: {current_branch}")
    
    # Create a new feature branch
    git_mgr.create_branch("test_feature", "feature")
    
    # Get repository info
    repo_info = git_mgr.get_repository_info()
    print(f"Repository info: {repo_info}")
    
    print("✅ Git manager example completed!") 