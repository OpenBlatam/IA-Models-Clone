from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import subprocess
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from dataclasses import dataclass
from datetime import datetime
        import re
                import shutil
        import json
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Git Workflow Automation for Product Descriptions Feature
Follows PyTorch, Transformers, Diffusers, and Gradio best practices
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GitConfig:
    """Git configuration for ML projects"""
    repo_path: Path
    main_branch: str = "main"
    develop_branch: str = "develop"
    feature_prefix: str = "feature/"
    hotfix_prefix: str = "hotfix/"
    release_prefix: str = "release/"

class GitWorkflow:
    """Automated git workflow for ML projects"""
    
    def __init__(self, config: GitConfig):
        
    """__init__ function."""
self.config = config
        self.repo_path = config.repo_path
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Execute git command with error handling"""
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Command executed: {' '.join(cmd)}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def get_current_branch(self) -> str:
        """Get current branch name"""
        result = self.run_command(["git", "branch", "--show-current"])
        return result.stdout.strip()
    
    def get_status(self) -> Dict[str, Any]:
        """Get git status information"""
        result = self.run_command(["git", "status", "--porcelain"])
        files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        return {
            "modified": [f for f in files if f.startswith('M')],
            "added": [f for f in files if f.startswith('A')],
            "deleted": [f for f in files if f.startswith('D')],
            "untracked": [f for f in files if f.startswith('??')]
        }
    
    def create_feature_branch(self, feature_name: str) -> str:
        """Create and switch to feature branch"""
        branch_name = f"{self.config.feature_prefix}{feature_name}"
        
        # Ensure we're on main branch
        self.run_command(["git", "checkout", self.config.main_branch])
        self.run_command(["git", "pull", "origin", self.config.main_branch])
        
        # Create and switch to feature branch
        self.run_command(["git", "checkout", "-b", branch_name])
        
        logger.info(f"Created feature branch: {branch_name}")
        return branch_name
    
    def commit_changes(self, message: str, files: Optional[List[str]] = None) -> str:
        """Commit changes with conventional commit format"""
        if files:
            self.run_command(["git", "add"] + files)
        else:
            self.run_command(["git", "add", "."])
        
        # Validate commit message format
        if not self._validate_commit_message(message):
            raise ValueError("Invalid commit message format")
        
        self.run_command(["git", "commit", "-m", message])
        
        # Get commit hash
        result = self.run_command(["git", "rev-parse", "HEAD"])
        commit_hash = result.stdout.strip()[:8]
        
        logger.info(f"Committed changes: {commit_hash} - {message}")
        return commit_hash
    
    def _validate_commit_message(self, message: str) -> bool:
        """Validate conventional commit message format"""
        pattern = r'^(feat|fix|docs|style|refactor|test|chore)(\([a-z-]+\))?: .+'
        return bool(re.match(pattern, message))
    
    def push_branch(self, branch_name: Optional[str] = None) -> None:
        """Push current branch to remote"""
        branch = branch_name or self.get_current_branch()
        self.run_command(["git", "push", "origin", branch])
        logger.info(f"Pushed branch: {branch}")
    
    async def create_pull_request(self, title: str, description: str) -> None:
        """Create pull request using GitHub CLI"""
        try:
            self.run_command([
                "gh", "pr", "create",
                "--title", title,
                "--body", description,
                "--base", self.config.main_branch
            ])
            logger.info("Pull request created successfully")
        except subprocess.CalledProcessError:
            logger.warning("GitHub CLI not available, please create PR manually")
    
    def merge_feature_branch(self, feature_name: str) -> None:
        """Merge feature branch to main"""
        branch_name = f"{self.config.feature_prefix}{feature_name}"
        
        # Switch to main branch
        self.run_command(["git", "checkout", self.config.main_branch])
        self.run_command(["git", "pull", "origin", self.config.main_branch])
        
        # Merge feature branch
        self.run_command(["git", "merge", "--no-ff", branch_name])
        
        # Delete feature branch
        self.run_command(["git", "branch", "-d", branch_name])
        self.run_command(["git", "push", "origin", "--delete", branch_name])
        
        logger.info(f"Merged feature branch: {branch_name}")
    
    def create_release_tag(self, version: str, message: str) -> None:
        """Create and push release tag"""
        self.run_command(["git", "tag", "-a", version, "-m", message])
        self.run_command(["git", "push", "origin", version])
        logger.info(f"Created release tag: {version}")
    
    def stash_changes(self, message: str = "") -> None:
        """Stash current changes"""
        cmd = ["git", "stash", "push", "-m", message] if message else ["git", "stash"]
        self.run_command(cmd)
        logger.info("Changes stashed")
    
    def apply_stash(self, stash_index: int = 0) -> None:
        """Apply stashed changes"""
        self.run_command(["git", "stash", "apply", f"stash@{{{stash_index}}}"])
        logger.info(f"Applied stash {stash_index}")

class MLModelVersioning:
    """ML model versioning utilities"""
    
    def __init__(self, models_dir: Path):
        
    """__init__ function."""
self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
    
    def create_model_version(self, version: str, model_files: List[Path]) -> Path:
        """Create versioned model directory"""
        version_dir = self.models_dir / version
        version_dir.mkdir(exist_ok=True)
        
        # Copy model files
        for file_path in model_files:
            if file_path.exists():
                shutil.copy2(file_path, version_dir / file_path.name)
        
        # Create metadata
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "files": [f.name for f in model_files],
            "git_commit": self._get_git_commit()
        }
        
        with open(version_dir / "metadata.json", "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created model version: {version}")
        return version_dir
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

def main():
    """Main workflow execution"""
    # Initialize configuration
    config = GitConfig(
        repo_path=Path.cwd(),
        main_branch="main",
        feature_prefix="feature/product-desc-"
    )
    
    # Initialize workflow
    workflow = GitWorkflow(config)
    model_versioning = MLModelVersioning(Path("models"))
    
    # Example workflow
    try:
        # Create feature branch
        branch = workflow.create_feature_branch("gpt4-integration")
        
        # Check status
        status = workflow.get_status()
        logger.info(f"Git status: {status}")
        
        # Commit changes
        commit_hash = workflow.commit_changes(
            "feat(product-desc): integrate GPT-4 for enhanced generation"
        )
        
        # Push branch
        workflow.push_branch(branch)
        
        # Create PR
        workflow.create_pull_request(
            "Add GPT-4 Integration",
            "Integrates GPT-4 model for enhanced product description generation"
        )
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)

match __name__:
    case "__main__":
    main() 