"""
Deployment Utilities
====================

Utilities for deployment, environment setup, and production readiness.
"""

import asyncio
import logging
import subprocess
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Deployment stages."""
    PRE_DEPLOY = "pre_deploy"
    DEPLOY = "deploy"
    POST_DEPLOY = "post_deploy"
    ROLLBACK = "rollback"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentStep:
    """Deployment step definition."""
    name: str
    command: Optional[str] = None
    script: Optional[Path] = None
    async_function: Optional[Callable] = None
    description: str = ""
    required: bool = True
    timeout: Optional[int] = None
    retry_count: int = 0
    rollback_command: Optional[str] = None
    rollback_script: Optional[Path] = None


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    environment: str
    steps: List[DeploymentStep]
    pre_deploy_checks: List[str] = field(default_factory=list)
    post_deploy_checks: List[str] = field(default_factory=list)
    rollback_steps: List[DeploymentStep] = field(default_factory=list)


class DeploymentManager:
    """Deployment manager for production deployments."""
    
    def __init__(self, deployment_dir: Optional[Path] = None):
        """
        Initialize deployment manager.
        
        Args:
            deployment_dir: Directory for deployment files
        """
        self.deployment_dir = deployment_dir or Path("deployments")
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.configs: Dict[str, DeploymentConfig] = {}
        self.history: List[Dict[str, Any]] = []
        self.history_file = self.deployment_dir / "deployment_history.json"
        self._load_history()
    
    def _load_history(self):
        """Load deployment history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading deployment history: {e}")
                self.history = []
    
    def _save_history(self):
        """Save deployment history."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving deployment history: {e}")
    
    def register_config(self, config: DeploymentConfig):
        """
        Register a deployment configuration.
        
        Args:
            config: Deployment configuration
        """
        self.configs[config.name] = config
        logger.info(f"Registered deployment config: {config.name}")
    
    async def run_step(self, step: DeploymentStep) -> bool:
        """
        Run a deployment step.
        
        Args:
            step: Deployment step
            
        Returns:
            True if successful
        """
        logger.info(f"Running step: {step.name}")
        
        try:
            if step.async_function:
                if step.timeout:
                    await asyncio.wait_for(step.async_function(), timeout=step.timeout)
                else:
                    await step.async_function()
            elif step.script:
                result = subprocess.run(
                    ["python", str(step.script)],
                    capture_output=True,
                    text=True,
                    timeout=step.timeout
                )
                if result.returncode != 0:
                    raise Exception(f"Script failed: {result.stderr}")
            elif step.command:
                result = subprocess.run(
                    step.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=step.timeout
                )
                if result.returncode != 0:
                    raise Exception(f"Command failed: {result.stderr}")
            else:
                logger.warning(f"No action defined for step: {step.name}")
            
            logger.info(f"Step completed: {step.name}")
            return True
            
        except Exception as e:
            logger.error(f"Step failed: {step.name} - {e}")
            if step.retry_count > 0:
                logger.info(f"Retrying step: {step.name}")
                for i in range(step.retry_count):
                    try:
                        await asyncio.sleep(2 ** i)  # Exponential backoff
                        if step.async_function:
                            await step.async_function()
                        elif step.script:
                            result = subprocess.run(
                                ["python", str(step.script)],
                                capture_output=True,
                                text=True,
                                timeout=step.timeout
                            )
                            if result.returncode == 0:
                                return True
                        elif step.command:
                            result = subprocess.run(
                                step.command,
                                shell=True,
                                capture_output=True,
                                text=True,
                                timeout=step.timeout
                            )
                            if result.returncode == 0:
                                return True
                    except Exception:
                        continue
            return False
    
    async def run_deployment(
        self,
        config_name: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run a deployment.
        
        Args:
            config_name: Configuration name
            dry_run: If True, only simulate deployment
            
        Returns:
            Deployment result
        """
        if config_name not in self.configs:
            raise ValueError(f"Deployment config not found: {config_name}")
        
        config = self.configs[config_name]
        deployment_id = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = {
            "deployment_id": deployment_id,
            "config_name": config_name,
            "environment": config.environment,
            "status": DeploymentStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "steps": [],
            "error": None
        }
        
        self.history.append(result)
        self._save_history()
        
        try:
            # Pre-deploy checks
            if config.pre_deploy_checks:
                logger.info("Running pre-deploy checks")
                for check in config.pre_deploy_checks:
                    check_result = subprocess.run(
                        check,
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    if check_result.returncode != 0:
                        raise Exception(f"Pre-deploy check failed: {check}")
            
            if dry_run:
                logger.info("Dry run mode - simulating deployment")
                result["status"] = DeploymentStatus.COMPLETED.value
                result["dry_run"] = True
            else:
                # Run deployment steps
                for step in config.steps:
                    step_result = {
                        "name": step.name,
                        "status": "pending",
                        "started_at": datetime.now().isoformat(),
                        "completed_at": None,
                        "error": None
                    }
                    result["steps"].append(step_result)
                    
                    success = await self.run_step(step)
                    step_result["status"] = "completed" if success else "failed"
                    step_result["completed_at"] = datetime.now().isoformat()
                    
                    if not success and step.required:
                        raise Exception(f"Required step failed: {step.name}")
                
                # Post-deploy checks
                if config.post_deploy_checks:
                    logger.info("Running post-deploy checks")
                    for check in config.post_deploy_checks:
                        check_result = subprocess.run(
                            check,
                            shell=True,
                            capture_output=True,
                            text=True
                        )
                        if check_result.returncode != 0:
                            raise Exception(f"Post-deploy check failed: {check}")
                
                result["status"] = DeploymentStatus.COMPLETED.value
            
            result["completed_at"] = datetime.now().isoformat()
            logger.info(f"Deployment completed: {deployment_id}")
            
        except Exception as e:
            result["status"] = DeploymentStatus.FAILED.value
            result["error"] = str(e)
            result["completed_at"] = datetime.now().isoformat()
            logger.error(f"Deployment failed: {deployment_id} - {e}")
        
        self._save_history()
        return result
    
    async def rollback(self, deployment_id: str) -> bool:
        """
        Rollback a deployment.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            True if successful
        """
        # Find deployment
        deployment = None
        for d in self.history:
            if d.get("deployment_id") == deployment_id:
                deployment = d
                break
        
        if not deployment:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        config_name = deployment.get("config_name")
        if config_name not in self.configs:
            logger.error(f"Config not found: {config_name}")
            return False
        
        config = self.configs[config_name]
        
        try:
            # Run rollback steps
            for step in config.rollback_steps:
                success = await self.run_step(step)
                if not success:
                    logger.warning(f"Rollback step failed: {step.name}")
            
            deployment["status"] = DeploymentStatus.ROLLED_BACK.value
            deployment["rolled_back_at"] = datetime.now().isoformat()
            self._save_history()
            
            logger.info(f"Deployment rolled back: {deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {deployment_id} - {e}")
            return False


class EnvironmentChecker:
    """Check deployment environment readiness."""
    
    @staticmethod
    def check_python_version(min_version: str = "3.8") -> bool:
        """
        Check Python version.
        
        Args:
            min_version: Minimum version required
            
        Returns:
            True if version is sufficient
        """
        import sys
        version_parts = sys.version_info[:2]
        min_parts = tuple(map(int, min_version.split('.')))
        return version_parts >= min_parts
    
    @staticmethod
    def check_dependencies(requirements_file: Path) -> Dict[str, bool]:
        """
        Check if dependencies are installed.
        
        Args:
            requirements_file: Path to requirements file
            
        Returns:
            Dictionary of package -> installed
        """
        if not requirements_file.exists():
            return {}
        
        results = {}
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    package = line.split('==')[0].split('>=')[0].split('<=')[0]
                    try:
                        __import__(package.replace('-', '_'))
                        results[package] = True
                    except ImportError:
                        results[package] = False
        
        return results
    
    @staticmethod
    def check_directories(directories: List[str]) -> Dict[str, bool]:
        """
        Check if directories exist and are writable.
        
        Args:
            directories: List of directory paths
            
        Returns:
            Dictionary of directory -> exists and writable
        """
        results = {}
        for directory in directories:
            path = Path(directory)
            exists = path.exists()
            writable = path.exists() and os.access(path, os.W_OK)
            results[directory] = exists and writable
        return results
    
    @staticmethod
    def check_environment_variables(variables: List[str]) -> Dict[str, bool]:
        """
        Check if environment variables are set.
        
        Args:
            variables: List of variable names
            
        Returns:
            Dictionary of variable -> set
        """
        import os
        return {var: var in os.environ for var in variables}


# Import os for directory checks
import os


