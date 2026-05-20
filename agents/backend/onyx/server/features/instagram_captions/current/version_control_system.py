"""
Version Control System
Follows key convention: Use version control (e.g., git) for tracking changes in code and configurations
"""

import subprocess
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import shutil
import hashlib
import difflib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# VERSION CONTROL CONFIGURATION
# ============================================================================

@dataclass
class GitConfig:
    """Git configuration settings"""
    repo_path: str = "."
    author_name: str = "NLP System"
    author_email: str = "nlp@system.com"
    default_branch: str = "main"
    commit_message_template: str = "[{type}] {description}"
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "*.pyo", "__pycache__/", "*.log", "*.tmp",
        "*.pth", "*.ckpt", "*.pt", "*.bin", "*.h5",
        "data/", "cache/", "outputs/", "logs/", "runs/",
        ".ipynb_checkpoints/", ".DS_Store", "*.swp", "*.swo"
    ])
    
    def validate(self) -> bool:
        """Validate git configuration"""
        if not self.repo_path:
            raise ValueError("repo_path is required")
        if not self.author_name:
            raise ValueError("author_name is required")
        if not self.author_email:
            raise ValueError("author_email is required")
        return True

# ============================================================================
# GIT REPOSITORY MANAGER
# ============================================================================

class GitRepositoryManager:
    """Manages git repository operations"""
    
    def __init__(self, config: GitConfig):
        self.config = config
        self.repo_path = Path(config.repo_path)
        self._ensure_git_repo()
    
    def _ensure_git_repo(self):
        """Ensure git repository exists and is initialized"""
        if not (self.repo_path / ".git").exists():
            self._init_repository()
        else:
            self._configure_git()
    
    def _init_repository(self):
        """Initialize new git repository"""
        try:
            subprocess.run(["git", "init"], cwd=self.repo_path, check=True)
            logger.info(f"Initialized git repository at {self.repo_path}")
            self._configure_git()
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize git repository: {e}")
            raise
    
    def _configure_git(self):
        """Configure git user and default branch"""
        try:
            subprocess.run([
                "git", "config", "user.name", self.config.author_name
            ], cwd=self.repo_path, check=True)
            
            subprocess.run([
                "git", "config", "user.email", self.config.author_email
            ], cwd=self.repo_path, check=True)
            
            # Set default branch
            subprocess.run([
                "git", "config", "init.defaultBranch", self.config.default_branch
            ], cwd=self.repo_path, check=True)
            
            logger.info("Git repository configured successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure git: {e}")
            raise
    
    def create_gitignore(self):
        """Create .gitignore file with common patterns"""
        gitignore_path = self.repo_path / ".gitignore"
        
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                for pattern in self.config.ignore_patterns:
                    f.write(f"{pattern}\n")
            
            logger.info("Created .gitignore file")
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get git repository status"""
        try:
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path, capture_output=True, text=True, check=True
            )
            current_branch = branch_result.stdout.strip()
            
            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path, capture_output=True, text=True, check=True
            )
            
            # Parse status
            status_lines = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
            staged = []
            unstaged = []
            
            for line in status_lines:
                if line.startswith('M ') or line.startswith('A '):
                    staged.append(line[3:])
                elif line.startswith(' M') or line.startswith('??'):
                    unstaged.append(line[3:])
            
            return {
                "current_branch": current_branch,
                "staged_files": staged,
                "unstaged_files": unstaged,
                "total_staged": len(staged),
                "total_unstaged": len(unstaged)
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get git status: {e}")
            return {}
    
    def add_files(self, files: List[str] = None, all_files: bool = False):
        """Add files to staging area"""
        try:
            if all_files:
                subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
                logger.info("Added all files to staging area")
            elif files:
                subprocess.run(["git", "add"] + files, cwd=self.repo_path, check=True)
                logger.info(f"Added files to staging area: {files}")
            else:
                logger.warning("No files specified for staging")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add files: {e}")
            raise
    
    def commit(self, message: str, commit_type: str = "update"):
        """Commit staged changes"""
        try:
            formatted_message = self.config.commit_message_template.format(
                type=commit_type, description=message
            )
            
            subprocess.run([
                "git", "commit", "-m", formatted_message
            ], cwd=self.repo_path, check=True)
            
            logger.info(f"Committed changes: {formatted_message}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")
            return False
    
    def create_branch(self, branch_name: str, checkout: bool = True):
        """Create and optionally checkout new branch"""
        try:
            subprocess.run([
                "git", "checkout", "-b", branch_name
            ], cwd=self.repo_path, check=True)
            
            logger.info(f"Created and checked out branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False
    
    def checkout_branch(self, branch_name: str):
        """Checkout existing branch"""
        try:
            subprocess.run([
                "git", "checkout", branch_name
            ], cwd=self.repo_path, check=True)
            
            logger.info(f"Checked out branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout branch: {e}")
            return False
    
    def get_commit_history(self, max_count: int = 10) -> List[Dict[str, Any]]:
        """Get commit history"""
        try:
            result = subprocess.run([
                "git", "log", f"--max-count={max_count}", 
                "--pretty=format:%H|%an|%ae|%ad|%s", "--date=iso"
            ], cwd=self.repo_path, capture_output=True, text=True, check=True)
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) == 5:
                        commits.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "email": parts[2],
                            "date": parts[3],
                            "message": parts[4]
                        })
            
            return commits
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get commit history: {e}")
            return []

# ============================================================================
# CONFIGURATION VERSION TRACKER
# ============================================================================

class ConfigurationVersionTracker:
    """Tracks changes in configuration files"""
    
    def __init__(self, repo_manager: GitRepositoryManager):
        self.repo_manager = repo_manager
        self.config_dir = Path("configs")
        self.backup_dir = Path("config_backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def track_config_changes(self, config_file: str, description: str = "Configuration update"):
        """Track changes in configuration file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return False
        
        try:
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{config_path.stem}_{timestamp}{config_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(config_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            # Add to git
            self.repo_manager.add_files([str(config_path)])
            
            # Commit changes
            commit_message = f"Update {config_path.name}: {description}"
            success = self.repo_manager.commit(commit_message, "config")
            
            if success:
                logger.info(f"Configuration changes tracked: {config_file}")
                return True
            else:
                logger.error("Failed to commit configuration changes")
                return False
                
        except Exception as e:
            logger.error(f"Failed to track configuration changes: {e}")
            return False
    
    def get_config_history(self, config_file: str) -> List[Dict[str, Any]]:
        """Get version history of configuration file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            return []
        
        try:
            result = subprocess.run([
                "git", "log", "--follow", "--pretty=format:%H|%an|%ad|%s", 
                "--date=iso", str(config_path)
            ], cwd=self.repo_manager.repo_path, capture_output=True, text=True, check=True)
            
            history = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) == 4:
                        history.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "date": parts[2],
                            "message": parts[3]
                        })
            
            return history
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get configuration history: {e}")
            return []
    
    def compare_config_versions(self, config_file: str, commit1: str, commit2: str) -> str:
        """Compare two versions of a configuration file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            return "Configuration file not found"
        
        try:
            # Get content from both commits
            result1 = subprocess.run([
                "git", "show", f"{commit1}:{config_path}", 
            ], cwd=self.repo_manager.repo_path, capture_output=True, text=True)
            
            result2 = subprocess.run([
                "git", "show", f"{commit2}:{config_path}", 
            ], cwd=self.repo_manager.repo_path, capture_output=True, text=True)
            
            if result1.returncode != 0 or result2.returncode != 0:
                return "Failed to retrieve commit contents"
            
            content1 = result1.stdout
            content2 = result2.stdout
            
            # Generate diff
            diff = difflib.unified_diff(
                content1.splitlines(keepends=True),
                content2.splitlines(keepends=True),
                fromfile=f"{config_file} ({commit1[:8]})",
                tofile=f"{config_file} ({commit2[:8]})"
            )
            
            return ''.join(diff)
            
        except Exception as e:
            logger.error(f"Failed to compare configuration versions: {e}")
            return f"Error comparing versions: {e}"

# ============================================================================
# CODE VERSION TRACKER
# ============================================================================

class CodeVersionTracker:
    """Tracks changes in code files"""
    
    def __init__(self, repo_manager: GitRepositoryManager):
        self.repo_manager = repo_manager
        self.code_extensions = ['.py', '.tsx', '.ts', '.js', '.jsx', '.cpp', '.c', '.h']
    
    def track_code_changes(self, files: List[str] = None, description: str = "Code update"):
        """Track changes in code files"""
        if files is None:
            # Track all code files
            files = self._find_code_files()
        
        if not files:
            logger.warning("No code files found to track")
            return False
        
        try:
            # Add files to git
            self.repo_manager.add_files(files)
            
            # Commit changes
            commit_message = f"Code update: {description}"
            success = self.repo_manager.commit(commit_message, "code")
            
            if success:
                logger.info(f"Code changes tracked for {len(files)} files")
                return True
            else:
                logger.error("Failed to commit code changes")
                return False
                
        except Exception as e:
            logger.error(f"Failed to track code changes: {e}")
            return False
    
    def _find_code_files(self) -> List[str]:
        """Find all code files in the repository"""
        code_files = []
        
        for ext in self.code_extensions:
            pattern = f"**/*{ext}"
            files = list(self.repo_manager.repo_path.glob(pattern))
            code_files.extend([str(f.relative_to(self.repo_manager.repo_path)) for f in files])
        
        return code_files
    
    def get_file_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Get version history of a specific file"""
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return []
        
        try:
            result = subprocess.run([
                "git", "log", "--follow", "--pretty=format:%H|%an|%ad|%s", 
                "--date=iso", str(file_path_obj)
            ], cwd=self.repo_manager.repo_path, capture_output=True, text=True, check=True)
            
            history = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) == 4:
                        history.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "date": parts[2],
                            "message": parts[3]
                        })
            
            return history
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get file history: {e}")
            return []
    
    def create_feature_branch(self, feature_name: str, description: str = ""):
        """Create a feature branch for development"""
        try:
            # Create and checkout feature branch
            branch_name = f"feature/{feature_name}"
            success = self.repo_manager.create_branch(branch_name)
            
            if success:
                # Create initial commit
                commit_message = f"Start feature: {feature_name}"
                if description:
                    commit_message += f" - {description}"
                
                self.repo_manager.commit(commit_message, "feature")
                logger.info(f"Created feature branch: {branch_name}")
                return True
            else:
                logger.error("Failed to create feature branch")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create feature branch: {e}")
            return False

# ============================================================================
# EXPERIMENT VERSION TRACKER
# ============================================================================

class ExperimentVersionTracker:
    """Tracks experiment versions and results"""
    
    def __init__(self, repo_manager: GitRepositoryManager):
        self.repo_manager = repo_manager
        self.experiments_dir = Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)
    
    def create_experiment_branch(self, experiment_name: str, description: str = ""):
        """Create a branch for a specific experiment"""
        try:
            # Create and checkout experiment branch
            branch_name = f"experiment/{experiment_name}"
            success = self.repo_manager.create_branch(branch_name)
            
            if success:
                # Create experiment directory
                exp_dir = self.experiments_dir / experiment_name
                exp_dir.mkdir(exist_ok=True)
                
                # Create experiment metadata
                metadata = {
                    "name": experiment_name,
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                    "branch": branch_name,
                    "status": "created"
                }
                
                metadata_file = exp_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Add and commit
                self.repo_manager.add_files([str(metadata_file)])
                self.repo_manager.commit(f"Create experiment: {experiment_name}", "experiment")
                
                logger.info(f"Created experiment branch: {branch_name}")
                return True
            else:
                logger.error("Failed to create experiment branch")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create experiment branch: {e}")
            return False
    
    def track_experiment_results(self, experiment_name: str, results_file: str, description: str = ""):
        """Track experiment results"""
        try:
            exp_dir = self.experiments_dir / experiment_name
            if not exp_dir.exists():
                logger.error(f"Experiment directory not found: {experiment_name}")
                return False
            
            # Copy results to experiment directory
            results_path = Path(results_file)
            if results_path.exists():
                dest_path = exp_dir / results_path.name
                shutil.copy2(results_path, dest_path)
                
                # Update metadata
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["last_updated"] = datetime.now().isoformat()
                    metadata["results_file"] = str(dest_path)
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                # Add and commit
                self.repo_manager.add_files([str(dest_path), str(metadata_file)])
                commit_message = f"Update experiment results: {experiment_name}"
                if description:
                    commit_message += f" - {description}"
                
                success = self.repo_manager.commit(commit_message, "experiment")
                
                if success:
                    logger.info(f"Experiment results tracked: {experiment_name}")
                    return True
                else:
                    logger.error("Failed to commit experiment results")
                    return False
            else:
                logger.error(f"Results file not found: {results_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to track experiment results: {e}")
            return False

# ============================================================================
# MAIN VERSION CONTROL SYSTEM
# ============================================================================

class VersionControlSystem:
    """Main version control system for tracking code and configuration changes"""
    
    def __init__(self, config: GitConfig):
        self.config = config
        self.repo_manager = GitRepositoryManager(config)
        self.config_tracker = ConfigurationVersionTracker(self.repo_manager)
        self.code_tracker = CodeVersionTracker(self.repo_manager)
        self.experiment_tracker = ExperimentVersionTracker(self.repo_manager)
        
        # Initialize repository
        self._initialize_repository()
    
    def _initialize_repository(self):
        """Initialize the version control system"""
        try:
            # Create .gitignore
            self.repo_manager.create_gitignore()
            
            # Initial commit if repository is empty
            status = self.repo_manager.get_status()
            if not status.get("staged_files") and not status.get("unstaged_files"):
                # Add all files and make initial commit
                self.repo_manager.add_files(all_files=True)
                self.repo_manager.commit("Initial commit", "init")
                logger.info("Initialized version control system")
            
        except Exception as e:
            logger.error(f"Failed to initialize version control system: {e}")
    
    def track_changes(self, files: List[str] = None, description: str = "Update", 
                     change_type: str = "update"):
        """Track changes in files"""
        try:
            if files is None:
                # Track all changes
                self.repo_manager.add_files(all_files=True)
            else:
                self.repo_manager.add_files(files)
            
            # Commit changes
            success = self.repo_manager.commit(description, change_type)
            
            if success:
                logger.info(f"Changes tracked successfully: {description}")
                return True
            else:
                logger.error("Failed to track changes")
                return False
                
        except Exception as e:
            logger.error(f"Failed to track changes: {e}")
            return False
    
    def get_repository_info(self) -> Dict[str, Any]:
        """Get comprehensive repository information"""
        try:
            status = self.repo_manager.get_status()
            history = self.repo_manager.get_commit_history(5)
            
            return {
                "repository_path": str(self.repo_manager.repo_path),
                "current_branch": status.get("current_branch", "unknown"),
                "staged_files": status.get("staged_files", []),
                "unstaged_files": status.get("unstaged_files", []),
                "recent_commits": history,
                "total_commits": len(history)
            }
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return {}
    
    def create_release_tag(self, version: str, message: str = ""):
        """Create a release tag"""
        try:
            tag_message = f"Release {version}"
            if message:
                tag_message += f": {message}"
            
            subprocess.run([
                "git", "tag", "-a", version, "-m", tag_message
            ], cwd=self.repo_manager.repo_path, check=True)
            
            logger.info(f"Created release tag: {version}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create release tag: {e}")
            return False

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def main():
    """Example usage of the version control system"""
    
    # Create git configuration
    git_config = GitConfig(
        repo_path=".",
        author_name="NLP System",
        author_email="nlp@system.com"
    )
    
    # Initialize version control system
    print("Initializing version control system...")
    vcs = VersionControlSystem(git_config)
    
    # Get repository information
    print("Getting repository information...")
    repo_info = vcs.get_repository_info()
    
    print(f"Repository: {repo_info['repository_path']}")
    print(f"Current branch: {repo_info['current_branch']}")
    print(f"Staged files: {repo_info['total_staged']}")
    print(f"Unstaged files: {repo_info['total_unstaged']}")
    
    # Track configuration changes
    print("\nTracking configuration changes...")
    success = vcs.config_tracker.track_config_changes(
        "config/nlp_config.yaml",
        "Update model parameters"
    )
    
    if success:
        print("Configuration changes tracked successfully")
    else:
        print("Failed to track configuration changes")
    
    # Create feature branch
    print("\nCreating feature branch...")
    success = vcs.code_tracker.create_feature_branch(
        "nlp-optimization",
        "Implement advanced NLP features"
    )
    
    if success:
        print("Feature branch created successfully")
    else:
        print("Failed to create feature branch")
    
    # Create experiment branch
    print("\nCreating experiment branch...")
    success = vcs.experiment_tracker.create_experiment_branch(
        "transformer-fine-tuning",
        "Fine-tune transformer model on custom dataset"
    )
    
    if success:
        print("Experiment branch created successfully")
    else:
        print("Failed to create experiment branch")
    
    # Get final status
    print("\nFinal repository status:")
    final_status = vcs.repo_manager.get_status()
    print(f"Current branch: {final_status['current_branch']}")
    print(f"Staged files: {final_status['total_staged']}")
    print(f"Unstaged files: {final_status['total_unstaged']}")
    
    print("\nVersion control system ready!")

if __name__ == "__main__":
    main()


