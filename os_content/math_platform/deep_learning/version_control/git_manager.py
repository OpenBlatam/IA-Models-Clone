from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import subprocess
import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import git
from git import Repo, GitCommandError
import yaml
import hashlib
import difflib
import tempfile
import zipfile
        import argparse
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Git Version Control Manager
Comprehensive version control system for tracking changes in code and configurations.
"""


logger = logging.getLogger(__name__)


@dataclass
class GitConfig:
    """Configuration for Git version control."""
    # Repository settings
    repo_path: str = "."
    remote_url: Optional[str] = None
    branch: str = "main"
    
    # Commit settings
    author_name: str = "Deep Learning Team"
    author_email: str = "team@example.com"
    commit_message_template: str = "[{type}] {description}"
    
    # File tracking
    track_configs: bool = True
    track_models: bool = True
    track_data: bool = False
    track_logs: bool = False
    track_checkpoints: bool = False
    
    # Ignore patterns
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", "*.log", "*.tmp", "*.cache",
        "checkpoints/*", "logs/*", "data/*", "results/*",
        "*.pth", "*.pt", "*.ckpt", "*.h5", "*.pkl",
        "wandb/*", "runs/*", ".ipynb_checkpoints/*"
    ])
    
    # Auto-commit settings
    auto_commit: bool = False
    auto_commit_interval: int = 3600  # seconds
    auto_push: bool = False
    
    # Backup settings
    create_backups: bool = True
    backup_interval: int = 86400  # seconds (24 hours)
    max_backups: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'repo_path': self.repo_path,
            'remote_url': self.remote_url,
            'branch': self.branch,
            'author_name': self.author_name,
            'author_email': self.author_email,
            'commit_message_template': self.commit_message_template,
            'track_configs': self.track_configs,
            'track_models': self.track_models,
            'track_data': self.track_data,
            'track_logs': self.track_logs,
            'track_checkpoints': self.track_checkpoints,
            'ignore_patterns': self.ignore_patterns,
            'auto_commit': self.auto_commit,
            'auto_commit_interval': self.auto_commit_interval,
            'auto_push': self.auto_push,
            'create_backups': self.create_backups,
            'backup_interval': self.backup_interval,
            'max_backups': self.max_backups
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GitConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'GitConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class CommitInfo:
    """Information about a Git commit."""
    hash: str
    author: str
    author_email: str
    date: datetime
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    branch: str
    tags: List[str] = field(default_factory=list)


@dataclass
class FileChange:
    """Information about a file change."""
    filepath: str
    change_type: str  # added, modified, deleted, renamed
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    diff: Optional[str] = None
    size_change: int = 0


class GitManager:
    """Comprehensive Git version control manager."""
    
    def __init__(self, config: GitConfig):
        
    """__init__ function."""
self.config = config
        self.repo_path = Path(config.repo_path)
        self.repo = None
        self._initialize_repo()
    
    def _initialize_repo(self) -> Any:
        """Initialize Git repository."""
        try:
            if (self.repo_path / ".git").exists():
                self.repo = Repo(self.repo_path)
                logger.info(f"Opened existing Git repository: {self.repo_path}")
            else:
                self.repo = Repo.init(self.repo_path)
                logger.info(f"Initialized new Git repository: {self.repo_path}")
                self._setup_initial_config()
        except Exception as e:
            logger.error(f"Failed to initialize Git repository: {e}")
            raise
    
    def _setup_initial_config(self) -> Any:
        """Setup initial Git configuration."""
        try:
            # Set user configuration
            self.repo.config_writer().set_value("user", "name", self.config.author_name).release()
            self.repo.config_writer().set_value("user", "email", self.config.author_email).release()
            
            # Create .gitignore
            self._create_gitignore()
            
            # Initial commit
            self._add_all_files()
            self.commit("Initial commit", commit_type="init")
            
            logger.info("Initial Git setup completed")
        except Exception as e:
            logger.error(f"Failed to setup initial Git configuration: {e}")
    
    def _create_gitignore(self) -> Any:
        """Create .gitignore file."""
        gitignore_content = [
            "# Python",
            "*.pyc",
            "__pycache__/",
            "*.pyo",
            "*.pyd",
            ".Python",
            "*.so",
            "",
            "# Distribution / packaging",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            "",
            "# PyInstaller",
            "*.manifest",
            "*.spec",
            "",
            "# Unit test / coverage reports",
            "htmlcov/",
            ".tox/",
            ".coverage",
            ".coverage.*",
            ".cache",
            "nosetests.xml",
            "coverage.xml",
            "*.cover",
            ".hypothesis/",
            ".pytest_cache/",
            "",
            "# Jupyter Notebook",
            ".ipynb_checkpoints",
            "",
            "# Environment",
            ".env",
            ".venv",
            "env/",
            "venv/",
            "ENV/",
            "",
            "# Deep Learning specific",
            "checkpoints/",
            "logs/",
            "results/",
            "wandb/",
            "runs/",
            "*.pth",
            "*.pt",
            "*.ckpt",
            "*.h5",
            "*.pkl",
            "",
            "# Data files",
            "data/raw/",
            "data/processed/",
            "*.csv",
            "*.json",
            "*.parquet",
            "",
            "# IDE",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "",
            "# OS",
            ".DS_Store",
            ".DS_Store?",
            "._*",
            ".Spotlight-V100",
            ".Trashes",
            "ehthumbs.db",
            "Thumbs.db",
            "",
            "# Custom ignore patterns"
        ]
        
        # Add custom ignore patterns
        for pattern in self.config.ignore_patterns:
            gitignore_content.append(pattern)
        
        gitignore_path = self.repo_path / ".gitignore"
        with open(gitignore_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write('\n'.join(gitignore_content))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info("Created .gitignore file")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Git repository status."""
        try:
            status = {
                'repo_path': str(self.repo_path),
                'active_branch': self.repo.active_branch.name,
                'is_dirty': self.repo.is_dirty(),
                'untracked_files': [],
                'modified_files': [],
                'staged_files': [],
                'last_commit': None,
                'remote_info': {}
            }
            
            # Get untracked files
            status['untracked_files'] = [f for f in self.repo.untracked_files]
            
            # Get modified files
            for diff in self.repo.index.diff(None):
                status['modified_files'].append(diff.a_path)
            
            # Get staged files
            for diff in self.repo.index.diff('HEAD'):
                status['staged_files'].append(diff.a_path)
            
            # Get last commit info
            if self.repo.head.is_valid():
                last_commit = self.repo.head.commit
                status['last_commit'] = {
                    'hash': last_commit.hexsha,
                    'author': last_commit.author.name,
                    'date': last_commit.committed_datetime.isoformat(),
                    'message': last_commit.message.strip()
                }
            
            # Get remote info
            for remote in self.repo.remotes:
                status['remote_info'][remote.name] = {
                    'url': remote.url,
                    'refs': [ref.name for ref in remote.refs]
                }
            
            return status
        except Exception as e:
            logger.error(f"Failed to get Git status: {e}")
            return {}
    
    def add_file(self, filepath: str, message: str = None):
        """Add a specific file to Git."""
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                logger.warning(f"File does not exist: {filepath}")
                return
            
            # Add file to index
            self.repo.index.add([str(file_path)])
            
            # Commit if message provided
            if message:
                self.commit(message, commit_type="add")
            else:
                logger.info(f"Added file to staging: {filepath}")
        except Exception as e:
            logger.error(f"Failed to add file {filepath}: {e}")
    
    def add_all_files(self, message: str = None):
        """Add all tracked files to Git."""
        try:
            self._add_all_files()
            
            if message:
                self.commit(message, commit_type="update")
            else:
                logger.info("Added all files to staging")
        except Exception as e:
            logger.error(f"Failed to add all files: {e}")
    
    def _add_all_files(self) -> Any:
        """Internal method to add all files."""
        # Add all files except those in .gitignore
        self.repo.index.add('*')
    
    def commit(self, message: str, commit_type: str = "update", files: List[str] = None):
        """Create a Git commit."""f"
        try:
            # Format commit message
            formatted_message = self.config.commit_message_template",
                description=message
            )
            
            # Check if there are changes to commit
            if not self.repo.is_dirty() and not self.repo.untracked_files:
                logger.info("No changes to commit")
                return
            
            # Add specific files if provided
            if files:
                for file_path in files:
                    if Path(file_path).exists():
                        self.repo.index.add([file_path])
            
            # Create commit
            commit = self.repo.index.commit(
                formatted_message,
                author=git.Actor(self.config.author_name, self.config.author_email)
            )
            
            logger.info(f"Created commit: {commit.hexsha[:8]} - {formatted_message}")
            return commit
        except Exception as e:
            logger.error(f"Failed to create commit: {e}")
            return None
    
    def get_commit_history(self, limit: int = 10) -> List[CommitInfo]:
        """Get commit history."""
        try:
            commits = []
            for commit in self.repo.iter_commits(self.config.branch, max_count=limit):
                # Get file changes
                files_changed = []
                insertions = 0
                deletions = 0
                
                if commit.parents:
                    diff = commit.diff(commit.parents[0])
                    for change in diff:
                        files_changed.append(change.a_path or change.b_path)
                        insertions += change.stats.get('insertions', 0)
                        deletions += change.stats.get('deletions', 0)
                else:
                    # Initial commit
                    for file_path in commit.stats.files.keys():
                        files_changed.append(file_path)
                        insertions += commit.stats.files[file_path]['insertions']
                        deletions += commit.stats.files[file_path]['deletions']
                
                # Get tags
                tags = [tag.name for tag in self.repo.tags if tag.commit == commit]
                
                commit_info = CommitInfo(
                    hash=commit.hexsha,
                    author=commit.author.name,
                    author_email=commit.author.email,
                    date=commit.committed_datetime,
                    message=commit.message.strip(),
                    files_changed=files_changed,
                    insertions=insertions,
                    deletions=deletions,
                    branch=self.config.branch,
                    tags=tags
                )
                commits.append(commit_info)
            
            return commits
        except Exception as e:
            logger.error(f"Failed to get commit history: {e}")
            return []
    
    def get_file_changes(self, commit_hash: str) -> List[FileChange]:
        """Get changes for a specific commit."""
        try:
            commit = self.repo.commit(commit_hash)
            changes = []
            
            if commit.parents:
                diff = commit.diff(commit.parents[0])
                for change in diff:
                    file_change = FileChange(
                        filepath=change.a_path or change.b_path,
                        change_type=change.change_type,
                        old_hash=change.a_blob.hexsha if change.a_blob else None,
                        new_hash=change.b_blob.hexsha if change.b_blob else None,
                        diff=change.diff.decode('utf-8') if change.diff else None,
                        size_change=change.stats.get('insertions', 0) - change.stats.get('deletions', 0)
                    )
                    changes.append(file_change)
            
            return changes
        except Exception as e:
            logger.error(f"Failed to get file changes: {e}")
            return []
    
    def create_branch(self, branch_name: str, checkout: bool = True):
        """Create a new branch."""
        try:
            new_branch = self.repo.create_head(branch_name)
            if checkout:
                new_branch.checkout()
                self.config.branch = branch_name
            
            logger.info(f"Created and checked out branch: {branch_name}")
            return new_branch
        except Exception as e:
            logger.error(f"Failed to create branch {branch_name}: {e}")
            return None
    
    def checkout_branch(self, branch_name: str):
        """Checkout a branch."""
        try:
            branch = self.repo.heads[branch_name]
            branch.checkout()
            self.config.branch = branch_name
            logger.info(f"Checked out branch: {branch_name}")
        except Exception as e:
            logger.error(f"Failed to checkout branch {branch_name}: {e}")
    
    def merge_branch(self, source_branch: str, target_branch: str = None):
        """Merge a branch into the current branch."""
        try:
            if target_branch is None:
                target_branch = self.config.branch
            
            # Checkout target branch
            self.checkout_branch(target_branch)
            
            # Merge source branch
            source = self.repo.heads[source_branch]
            self.repo.index.merge_tree(source)
            
            logger.info(f"Merged {source_branch} into {target_branch}")
        except Exception as e:
            logger.error(f"Failed to merge branch {source_branch}: {e}")
    
    def create_tag(self, tag_name: str, message: str = None, commit_hash: str = None):
        """Create a Git tag."""
        try:
            if commit_hash:
                commit = self.repo.commit(commit_hash)
            else:
                commit = self.repo.head.commit
            
            tag = self.repo.create_tag(tag_name, ref=commit, message=message)
            logger.info(f"Created tag: {tag_name}")
            return tag
        except Exception as e:
            logger.error(f"Failed to create tag {tag_name}: {e}")
            return None
    
    def push(self, remote_name: str = "origin", branch: str = None):
        """Push changes to remote repository."""
        try:
            if branch is None:
                branch = self.config.branch
            
            remote = self.repo.remote(remote_name)
            remote.push(branch)
            
            logger.info(f"Pushed {branch} to {remote_name}")
        except Exception as e:
            logger.error(f"Failed to push to {remote_name}: {e}")
    
    def pull(self, remote_name: str = "origin", branch: str = None):
        """Pull changes from remote repository."""
        try:
            if branch is None:
                branch = self.config.branch
            
            remote = self.repo.remote(remote_name)
            remote.pull(branch)
            
            logger.info(f"Pulled {branch} from {remote_name}")
        except Exception as e:
            logger.error(f"Failed to pull from {remote_name}: {e}")
    
    def add_remote(self, name: str, url: str):
        """Add a remote repository."""
        try:
            remote = self.repo.create_remote(name, url)
            logger.info(f"Added remote: {name} -> {url}")
            return remote
        except Exception as e:
            logger.error(f"Failed to add remote {name}: {e}")
            return None
    
    def create_backup(self, backup_name: str = None) -> str:
        """Create a backup of the repository."""
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"
            
            backup_dir = self.repo_path / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create zip archive
            backup_file = backup_dir / f"{backup_name}.zip"
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.repo_path.rglob('*'):
                    if file_path.is_file() and '.git' not in str(file_path):
                        arcname = file_path.relative_to(self.repo_path)
                        zipf.write(file_path, arcname)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            logger.info(f"Created backup: {backup_file}")
            return str(backup_file)
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def restore_backup(self, backup_file: str, restore_path: str = None):
        """Restore from a backup."""
        try:
            if restore_path is None:
                restore_path = self.repo_path / "restored"
            
            restore_path = Path(restore_path)
            restore_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(restore_path)
            
            logger.info(f"Restored backup to: {restore_path}")
            return str(restore_path)
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return None
    
    def get_diff(self, file_path: str, commit1: str = None, commit2: str = None) -> str:
        """Get diff for a file between two commits."""
        try:
            if commit1 is None:
                commit1 = 'HEAD~1'
            if commit2 is None:
                commit2 = 'HEAD'
            
            diff = self.repo.git.diff(commit1, commit2, '--', file_path)
            return diff
        except Exception as e:
            logger.error(f"Failed to get diff for {file_path}: {e}")
            return ""
    
    def stash_changes(self, message: str = None):
        """Stash current changes."""
        try:
            stash = self.repo.git.stash('push', '-m', message or "Stashed changes")
            logger.info("Stashed changes")
            return stash
        except Exception as e:
            logger.error(f"Failed to stash changes: {e}")
            return None
    
    def pop_stash(self, stash_index: int = 0):
        """Pop stashed changes."""
        try:
            self.repo.git.stash('pop', f'stash@{{{stash_index}}}')
            logger.info(f"Popped stash {stash_index}")
        except Exception as e:
            logger.error(f"Failed to pop stash: {e}")
    
    def list_stashes(self) -> List[Dict[str, Any]]:
        """List all stashes."""
        try:
            stashes = []
            for i, stash in enumerate(self.repo.git.stash('list').split('\n')):
                if stash:
                    stashes.append({
                        'index': i,
                        'description': stash
                    })
            return stashes
        except Exception as e:
            logger.error(f"Failed to list stashes: {e}")
            return []
    
    def clean_untracked_files(self, dry_run: bool = True):
        """Clean untracked files."""
        try:
            if dry_run:
                untracked = self.repo.git.clean('-n', '-d')
                logger.info(f"Would remove untracked files:\n{untracked}")
                return untracked
            else:
                self.repo.git.clean('-f', '-d')
                logger.info("Removed untracked files")
        except Exception as e:
            logger.error(f"Failed to clean untracked files: {e}")
    
    def get_repository_info(self) -> Dict[str, Any]:
        """Get comprehensive repository information."""
        try:
            info = {
                'repository_path': str(self.repo_path),
                'active_branch': self.repo.active_branch.name,
                'head_commit': {
                    'hash': self.repo.head.commit.hexsha,
                    'message': self.repo.head.commit.message.strip(),
                    'author': self.repo.head.commit.author.name,
                    'date': self.repo.head.commit.committed_datetime.isoformat()
                },
                'remotes': {},
                'branches': [],
                'tags': [],
                'status': self.get_status()
            }
            
            # Get remotes
            for remote in self.repo.remotes:
                info['remotes'][remote.name] = {
                    'url': remote.url,
                    'refs': [ref.name for ref in remote.refs]
                }
            
            # Get branches
            for branch in self.repo.heads:
                info['branches'].append({
                    'name': branch.name,
                    'commit': branch.commit.hexsha,
                    'is_active': branch.name == self.repo.active_branch.name
                })
            
            # Get tags
            for tag in self.repo.tags:
                info['tags'].append({
                    'name': tag.name,
                    'commit': tag.commit.hexsha,
                    'message': tag.tag.message if tag.tag else None
                })
            
            return info
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return {}


class VersionControlCLI:
    """Command-line interface for version control management."""
    
    def __init__(self, git_manager: GitManager):
        
    """__init__ function."""
self.git_manager = git_manager
    
    def create_parser(self) -> Any:
        """Create command-line argument parser."""
        
        parser = argparse.ArgumentParser(description="Version Control Management CLI")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show repository status')
        
        # Add command
        add_parser = subparsers.add_parser('add', help='Add files to staging')
        add_parser.add_argument('files', nargs='*', help='Files to add (empty for all)')
        add_parser.add_argument('--message', '-m', help='Commit message')
        
        # Commit command
        commit_parser = subparsers.add_parser('commit', help='Create a commit')
        commit_parser.add_argument('message', help='Commit message')
        commit_parser.add_argument('--type', '-t', default='update', help='Commit type')
        
        # Log command
        log_parser = subparsers.add_parser('log', help='Show commit history')
        log_parser.add_argument('--limit', '-n', type=int, default=10, help='Number of commits to show')
        
        # Branch commands
        branch_parser = subparsers.add_parser('branch', help='Branch operations')
        branch_parser.add_argument('action', choices=['list', 'create', 'checkout', 'merge'])
        branch_parser.add_argument('name', nargs='?', help='Branch name')
        branch_parser.add_argument('--source', help='Source branch for merge')
        
        # Tag commands
        tag_parser = subparsers.add_parser('tag', help='Tag operations')
        tag_parser.add_argument('action', choices=['list', 'create'])
        tag_parser.add_argument('name', nargs='?', help='Tag name')
        tag_parser.add_argument('--message', '-m', help='Tag message')
        
        # Remote commands
        remote_parser = subparsers.add_parser('remote', help='Remote operations')
        remote_parser.add_argument('action', choices=['list', 'add', 'push', 'pull'])
        remote_parser.add_argument('name', nargs='?', help='Remote name')
        remote_parser.add_argument('url', nargs='?', help='Remote URL')
        
        # Backup commands
        backup_parser = subparsers.add_parser('backup', help='Backup operations')
        backup_parser.add_argument('action', choices=['create', 'restore', 'list'])
        backup_parser.add_argument('name', nargs='?', help='Backup name or file')
        
        return parser
    
    def run(self, args=None) -> Any:
        """Run the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if parsed_args.command == 'status':
            self._show_status()
        elif parsed_args.command == 'add':
            self._add_files(parsed_args.files, parsed_args.message)
        elif parsed_args.command == 'commit':
            self._commit(parsed_args.message, parsed_args.type)
        elif parsed_args.command == 'log':
            self._show_log(parsed_args.limit)
        elif parsed_args.command == 'branch':
            self._handle_branch(parsed_args.action, parsed_args.name, parsed_args.source)
        elif parsed_args.command == 'tag':
            self._handle_tag(parsed_args.action, parsed_args.name, parsed_args.message)
        elif parsed_args.command == 'remote':
            self._handle_remote(parsed_args.action, parsed_args.name, parsed_args.url)
        elif parsed_args.command == 'backup':
            self._handle_backup(parsed_args.action, parsed_args.name)
        else:
            parser.print_help()
    
    def _show_status(self) -> Any:
        """Show repository status."""
        status = self.git_manager.get_status()
        print("Repository Status:")
        print(f"  Path: {status.get('repo_path', 'N/A')}")
        print(f"  Branch: {status.get('active_branch', 'N/A')}")
        print(f"  Dirty: {status.get('is_dirty', False)}")
        print(f"  Untracked files: {len(status.get('untracked_files', []))}")
        print(f"  Modified files: {len(status.get('modified_files', []))}")
        print(f"  Staged files: {len(status.get('staged_files', []))}")
    
    def _add_files(self, files, message) -> Any:
        """Add files to staging."""
        if files:
            for file_path in files:
                self.git_manager.add_file(file_path, message)
        else:
            self.git_manager.add_all_files(message)
    
    def _commit(self, message, commit_type) -> Any:
        """Create a commit."""
        self.git_manager.commit(message, commit_type)
    
    def _show_log(self, limit) -> Any:
        """Show commit history."""
        commits = self.git_manager.get_commit_history(limit)
        for commit in commits:
            print(f"Commit: {commit.hash[:8]}")
            print(f"Author: {commit.author}")
            print(f"Date: {commit.date}")
            print(f"Message: {commit.message}")
            print(f"Files: {len(commit.files_changed)}")
            print(f"Changes: +{commit.insertions} -{commit.deletions}")
            print("-" * 50)
    
    def _handle_branch(self, action, name, source) -> Any:
        """Handle branch operations."""
        if action == 'list':
            info = self.git_manager.get_repository_info()
            for branch in info.get('branches', []):
                active = " *" if branch['is_active'] else ""
                print(f"{branch['name']}{active}")
        elif action == 'create':
            if name:
                self.git_manager.create_branch(name)
        elif action == 'checkout':
            if name:
                self.git_manager.checkout_branch(name)
        elif action == 'merge':
            if name and source:
                self.git_manager.merge_branch(source, name)
    
    def _handle_tag(self, action, name, message) -> Any:
        """Handle tag operations."""
        if action == 'list':
            info = self.git_manager.get_repository_info()
            for tag in info.get('tags', []):
                print(f"{tag['name']} -> {tag['commit'][:8]}")
        elif action == 'create':
            if name:
                self.git_manager.create_tag(name, message)
    
    def _handle_remote(self, action, name, url) -> Any:
        """Handle remote operations."""
        if action == 'list':
            info = self.git_manager.get_repository_info()
            for remote_name, remote_info in info.get('remotes', {}).items():
                print(f"{remote_name}: {remote_info['url']}")
        elif action == 'add':
            if name and url:
                self.git_manager.add_remote(name, url)
        elif action == 'push':
            if name:
                self.git_manager.push(name)
        elif action == 'pull':
            if name:
                self.git_manager.pull(name)
    
    def _handle_backup(self, action, name) -> Any:
        """Handle backup operations."""
        if action == 'create':
            backup_file = self.git_manager.create_backup(name)
            if backup_file:
                print(f"Created backup: {backup_file}")
        elif action == 'restore':
            if name:
                restore_path = self.git_manager.restore_backup(name)
                if restore_path:
                    print(f"Restored to: {restore_path}")


# Example usage
if __name__ == "__main__":
    # Create Git configuration
    config = GitConfig(
        repo_path="./",
        author_name="Deep Learning Team",
        author_email="team@example.com",
        track_configs=True,
        track_models=True,
        auto_commit=False
    )
    
    # Create Git manager
    git_manager = GitManager(config)
    
    # Create CLI
    cli = VersionControlCLI(git_manager)
    
    # Run CLI
    cli.run() 