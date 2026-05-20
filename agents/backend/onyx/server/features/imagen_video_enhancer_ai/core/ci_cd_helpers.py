"""
CI/CD Helpers
=============

Helpers for CI/CD pipelines and automation.
"""

import logging
import subprocess
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BuildInfo:
    """Build information."""
    version: str
    commit_hash: str
    branch: str
    build_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CIHelper:
    """CI/CD helper utilities."""
    
    @staticmethod
    def get_git_info() -> Dict[str, Any]:
        """
        Get git information.
        
        Returns:
            Git information dictionary
        """
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            return {
                "commit_hash": commit_hash,
                "branch": branch,
                "available": True
            }
        except Exception as e:
            logger.warning(f"Could not get git info: {e}")
            return {
                "commit_hash": "unknown",
                "branch": "unknown",
                "available": False
            }
    
    @staticmethod
    def get_version() -> str:
        """
        Get version from version file or git.
        
        Returns:
            Version string
        """
        # Try to read from version file
        version_file = Path("VERSION")
        if version_file.exists():
            return version_file.read_text().strip()
        
        # Try git tag
        try:
            tag = subprocess.check_output(
                ["git", "describe", "--tags", "--always"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return tag
        except Exception:
            return "0.0.0-dev"
    
    @staticmethod
    def create_build_info() -> BuildInfo:
        """
        Create build information.
        
        Returns:
            Build info object
        """
        git_info = CIHelper.get_git_info()
        version = CIHelper.get_version()
        
        return BuildInfo(
            version=version,
            commit_hash=git_info.get("commit_hash", "unknown"),
            branch=git_info.get("branch", "unknown"),
            metadata={
                "git_available": git_info.get("available", False)
            }
        )
    
    @staticmethod
    def save_build_info(output_path: Path):
        """
        Save build info to file.
        
        Args:
            output_path: Output file path
        """
        build_info = CIHelper.create_build_info()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "version": build_info.version,
                "commit_hash": build_info.commit_hash,
                "branch": build_info.branch,
                "build_time": build_info.build_time.isoformat(),
                "metadata": build_info.metadata
            }, f, indent=2)
        
        logger.info(f"Build info saved to {output_path}")


class DeploymentHelper:
    """Deployment helper utilities."""
    
    @staticmethod
    def validate_environment(required_vars: List[str]) -> Dict[str, bool]:
        """
        Validate environment variables.
        
        Args:
            required_vars: List of required variable names
            
        Returns:
            Dictionary of variable availability
        """
        import os
        result = {}
        
        for var in required_vars:
            result[var] = var in os.environ
        
        return result
    
    @staticmethod
    def check_dependencies() -> Dict[str, Any]:
        """
        Check system dependencies.
        
        Returns:
            Dependency status dictionary
        """
        dependencies = {
            "python": {"command": "python", "version_flag": "--version"},
            "pip": {"command": "pip", "version_flag": "--version"},
        }
        
        status = {}
        
        for name, config in dependencies.items():
            try:
                result = subprocess.run(
                    [config["command"], config["version_flag"]],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                status[name] = {
                    "available": result.returncode == 0,
                    "version": result.stdout.strip() if result.returncode == 0 else None
                }
            except Exception as e:
                status[name] = {
                    "available": False,
                    "error": str(e)
                }
        
        return status
    
    @staticmethod
    def create_deployment_manifest(output_path: Path, metadata: Optional[Dict[str, Any]] = None):
        """
        Create deployment manifest.
        
        Args:
            output_path: Output file path
            metadata: Optional metadata
        """
        build_info = CIHelper.create_build_info()
        deps = DeploymentHelper.check_dependencies()
        
        manifest = {
            "build": {
                "version": build_info.version,
                "commit_hash": build_info.commit_hash,
                "branch": build_info.branch,
                "build_time": build_info.build_time.isoformat()
            },
            "dependencies": deps,
            "metadata": metadata or {}
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Deployment manifest saved to {output_path}")




