"""
Ultimate BUL System - CI/CD Pipeline & Deployment Automation
Comprehensive deployment automation with testing, security scanning, and monitoring
"""

import asyncio
import subprocess
import json
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import shutil
import tempfile
from pathlib import Path
import docker
import requests
import time

logger = logging.getLogger(__name__)

class PipelineStage(str, Enum):
    """CI/CD Pipeline stages"""
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    QUALITY_CHECK = "quality_check"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    ROLLBACK = "rollback"

class DeploymentEnvironment(str, Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class PipelineStatus(str, Enum):
    """Pipeline status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"

@dataclass
class PipelineStep:
    """Pipeline step configuration"""
    name: str
    stage: PipelineStage
    command: str
    timeout: int = 300
    retries: int = 3
    parallel: bool = False
    dependencies: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

@dataclass
class PipelineExecution:
    """Pipeline execution tracking"""
    id: str
    status: PipelineStatus
    environment: DeploymentEnvironment
    started_at: datetime
    completed_at: Optional[datetime] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None

class CICDPipeline:
    """Comprehensive CI/CD Pipeline for BUL Ultimate System"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = docker.from_env()
        self.pipeline_executions = {}
        self.deployment_history = []
        
        # Pipeline configuration
        self.pipeline_steps = self._initialize_pipeline_steps()
        self.environment_configs = self._initialize_environment_configs()
        
        # Deployment targets
        self.deployment_targets = {
            DeploymentEnvironment.DEVELOPMENT: {
                "host": "dev.bul.local",
                "port": 8000,
                "replicas": 1,
                "resources": {"cpu": "0.5", "memory": "1Gi"}
            },
            DeploymentEnvironment.STAGING: {
                "host": "staging.bul.local",
                "port": 8000,
                "replicas": 2,
                "resources": {"cpu": "1", "memory": "2Gi"}
            },
            DeploymentEnvironment.PRODUCTION: {
                "host": "api.bul.local",
                "port": 8000,
                "replicas": 5,
                "resources": {"cpu": "2", "memory": "4Gi"}
            }
        }
        
        # Monitoring and alerting
        self.monitoring_config = {
            "prometheus_url": "http://localhost:9090",
            "grafana_url": "http://localhost:3000",
            "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
            "email_notifications": os.getenv("EMAIL_NOTIFICATIONS", "false").lower() == "true"
        }
    
    def _initialize_pipeline_steps(self) -> List[PipelineStep]:
        """Initialize CI/CD pipeline steps"""
        return [
            # Build Stage
            PipelineStep(
                name="build_docker_images",
                stage=PipelineStage.BUILD,
                command="docker-compose build --no-cache",
                timeout=600,
                artifacts=["docker-images"]
            ),
            PipelineStep(
                name="build_frontend",
                stage=PipelineStage.BUILD,
                command="npm run build --prefix frontend/",
                timeout=300,
                artifacts=["frontend-dist"]
            ),
            
            # Test Stage
            PipelineStep(
                name="unit_tests",
                stage=PipelineStage.TEST,
                command="pytest tests/unit/ -v --cov=bul --cov-report=xml",
                timeout=600,
                dependencies=["build_docker_images"]
            ),
            PipelineStep(
                name="integration_tests",
                stage=PipelineStage.TEST,
                command="pytest tests/integration/ -v",
                timeout=900,
                dependencies=["build_docker_images"]
            ),
            PipelineStep(
                name="performance_tests",
                stage=PipelineStage.TEST,
                command="pytest tests/performance/ -v",
                timeout=1200,
                dependencies=["build_docker_images"]
            ),
            PipelineStep(
                name="security_tests",
                stage=PipelineStage.TEST,
                command="pytest tests/security/ -v",
                timeout=600,
                dependencies=["build_docker_images"]
            ),
            
            # Security Scan Stage
            PipelineStep(
                name="vulnerability_scan",
                stage=PipelineStage.SECURITY_SCAN,
                command="trivy image bul-api:latest",
                timeout=300,
                dependencies=["build_docker_images"]
            ),
            PipelineStep(
                name="dependency_scan",
                stage=PipelineStage.SECURITY_SCAN,
                command="safety check --json",
                timeout=300
            ),
            PipelineStep(
                name="code_quality_scan",
                stage=PipelineStage.SECURITY_SCAN,
                command="bandit -r bul/ -f json",
                timeout=300
            ),
            
            # Quality Check Stage
            PipelineStep(
                name="lint_check",
                stage=PipelineStage.QUALITY_CHECK,
                command="flake8 bul/ --max-line-length=100",
                timeout=300
            ),
            PipelineStep(
                name="type_check",
                stage=PipelineStage.QUALITY_CHECK,
                command="mypy bul/ --ignore-missing-imports",
                timeout=300
            ),
            PipelineStep(
                name="format_check",
                stage=PipelineStage.QUALITY_CHECK,
                command="black --check bul/",
                timeout=300
            ),
            
            # Deploy Stage
            PipelineStep(
                name="deploy_infrastructure",
                stage=PipelineStage.DEPLOY,
                command="terraform apply -auto-approve",
                timeout=1800,
                dependencies=["vulnerability_scan", "unit_tests"]
            ),
            PipelineStep(
                name="deploy_application",
                stage=PipelineStage.DEPLOY,
                command="kubectl apply -f k8s/",
                timeout=600,
                dependencies=["deploy_infrastructure"]
            ),
            PipelineStep(
                name="run_migrations",
                stage=PipelineStage.DEPLOY,
                command="kubectl exec -it bul-api-0 -- alembic upgrade head",
                timeout=300,
                dependencies=["deploy_application"]
            ),
            PipelineStep(
                name="health_check",
                stage=PipelineStage.DEPLOY,
                command="curl -f http://localhost:8000/health",
                timeout=60,
                dependencies=["run_migrations"]
            ),
            
            # Monitor Stage
            PipelineStep(
                name="monitor_deployment",
                stage=PipelineStage.MONITOR,
                command="python scripts/monitor_deployment.py",
                timeout=3600,
                dependencies=["health_check"]
            )
        ]
    
    def _initialize_environment_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize environment-specific configurations"""
        return {
            DeploymentEnvironment.DEVELOPMENT: {
                "docker_compose_file": "docker-compose.dev.yml",
                "environment": "development",
                "debug": True,
                "log_level": "DEBUG",
                "replicas": 1,
                "resources": {"cpu": "0.5", "memory": "1Gi"}
            },
            DeploymentEnvironment.STAGING: {
                "docker_compose_file": "docker-compose.staging.yml",
                "environment": "staging",
                "debug": False,
                "log_level": "INFO",
                "replicas": 2,
                "resources": {"cpu": "1", "memory": "2Gi"}
            },
            DeploymentEnvironment.PRODUCTION: {
                "docker_compose_file": "docker-compose.prod.yml",
                "environment": "production",
                "debug": False,
                "log_level": "WARNING",
                "replicas": 5,
                "resources": {"cpu": "2", "memory": "4Gi"}
            }
        }
    
    async def run_pipeline(self, environment: DeploymentEnvironment, 
                          branch: str = "main", 
                          commit_sha: str = None) -> PipelineExecution:
        """Run the complete CI/CD pipeline"""
        execution_id = f"pipeline_{int(time.time())}"
        
        execution = PipelineExecution(
            id=execution_id,
            status=PipelineStatus.PENDING,
            environment=environment,
            started_at=datetime.utcnow()
        )
        
        self.pipeline_executions[execution_id] = execution
        
        try:
            logger.info(f"Starting pipeline execution {execution_id} for {environment.value}")
            execution.status = PipelineStatus.RUNNING
            
            # Run pipeline stages
            await self._run_build_stage(execution)
            await self._run_test_stage(execution)
            await self._run_security_scan_stage(execution)
            await self._run_quality_check_stage(execution)
            await self._run_deploy_stage(execution, environment)
            await self._run_monitor_stage(execution)
            
            execution.status = PipelineStatus.SUCCESS
            execution.completed_at = datetime.utcnow()
            
            logger.info(f"Pipeline execution {execution_id} completed successfully")
            await self._send_notification(execution, "success")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            
            logger.error(f"Pipeline execution {execution_id} failed: {e}")
            await self._send_notification(execution, "failure")
            
            # Attempt rollback if in production
            if environment == DeploymentEnvironment.PRODUCTION:
                await self._rollback_deployment(execution)
        
        return execution
    
    async def _run_build_stage(self, execution: PipelineExecution):
        """Run build stage"""
        logger.info("Running build stage")
        
        build_steps = [step for step in self.pipeline_steps if step.stage == PipelineStage.BUILD]
        
        for step in build_steps:
            await self._execute_step(execution, step)
    
    async def _run_test_stage(self, execution: PipelineExecution):
        """Run test stage"""
        logger.info("Running test stage")
        
        test_steps = [step for step in self.pipeline_steps if step.stage == PipelineStage.TEST]
        
        # Run tests in parallel
        parallel_tasks = []
        for step in test_steps:
            if step.parallel:
                parallel_tasks.append(self._execute_step(execution, step))
            else:
                await self._execute_step(execution, step)
        
        if parallel_tasks:
            await asyncio.gather(*parallel_tasks, return_exceptions=True)
    
    async def _run_security_scan_stage(self, execution: PipelineExecution):
        """Run security scan stage"""
        logger.info("Running security scan stage")
        
        security_steps = [step for step in self.pipeline_steps if step.stage == PipelineStage.SECURITY_SCAN]
        
        for step in security_steps:
            await self._execute_step(execution, step)
    
    async def _run_quality_check_stage(self, execution: PipelineExecution):
        """Run quality check stage"""
        logger.info("Running quality check stage")
        
        quality_steps = [step for step in self.pipeline_steps if step.stage == PipelineStage.QUALITY_CHECK]
        
        for step in quality_steps:
            await self._execute_step(execution, step)
    
    async def _run_deploy_stage(self, execution: PipelineExecution, environment: DeploymentEnvironment):
        """Run deploy stage"""
        logger.info(f"Running deploy stage for {environment.value}")
        
        deploy_steps = [step for step in self.pipeline_steps if step.stage == PipelineStage.DEPLOY]
        
        for step in deploy_steps:
            await self._execute_step(execution, step, environment)
    
    async def _run_monitor_stage(self, execution: PipelineExecution):
        """Run monitor stage"""
        logger.info("Running monitor stage")
        
        monitor_steps = [step for step in self.pipeline_steps if step.stage == PipelineStage.MONITOR]
        
        for step in monitor_steps:
            await self._execute_step(execution, step)
    
    async def _execute_step(self, execution: PipelineExecution, step: PipelineStep, 
                          environment: Optional[DeploymentEnvironment] = None):
        """Execute a pipeline step"""
        step_start_time = datetime.utcnow()
        
        try:
            logger.info(f"Executing step: {step.name}")
            
            # Set environment variables
            env = os.environ.copy()
            env.update(step.environment_variables)
            
            if environment:
                env.update(self.environment_configs[environment])
            
            # Execute command
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=step.timeout,
                env=env
            )
            
            step_result = {
                "name": step.name,
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": (datetime.utcnow() - step_start_time).total_seconds(),
                "started_at": step_start_time.isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }
            
            execution.steps.append(step_result)
            
            if result.returncode != 0:
                raise Exception(f"Step {step.name} failed with return code {result.returncode}")
            
            logger.info(f"Step {step.name} completed successfully")
            
        except subprocess.TimeoutExpired:
            step_result = {
                "name": step.name,
                "status": "timeout",
                "return_code": -1,
                "stdout": "",
                "stderr": f"Step timed out after {step.timeout} seconds",
                "duration": step.timeout,
                "started_at": step_start_time.isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }
            
            execution.steps.append(step_result)
            raise Exception(f"Step {step.name} timed out after {step.timeout} seconds")
            
        except Exception as e:
            step_result = {
                "name": step.name,
                "status": "error",
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": (datetime.utcnow() - step_start_time).total_seconds(),
                "started_at": step_start_time.isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }
            
            execution.steps.append(step_result)
            raise
    
    async def _rollback_deployment(self, execution: PipelineExecution):
        """Rollback deployment"""
        logger.info("Rolling back deployment")
        
        try:
            # Get previous deployment
            previous_deployment = self._get_previous_deployment(execution.environment)
            
            if previous_deployment:
                # Rollback to previous version
                rollback_command = f"kubectl rollout undo deployment/bul-api -n {execution.environment.value}"
                result = subprocess.run(rollback_command, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    execution.status = PipelineStatus.ROLLED_BACK
                    logger.info("Deployment rolled back successfully")
                    await self._send_notification(execution, "rollback")
                else:
                    logger.error(f"Rollback failed: {result.stderr}")
            else:
                logger.warning("No previous deployment found for rollback")
                
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
    
    def _get_previous_deployment(self, environment: DeploymentEnvironment) -> Optional[Dict[str, Any]]:
        """Get previous deployment information"""
        # This would typically query the deployment history
        # For now, return None
        return None
    
    async def _send_notification(self, execution: PipelineExecution, status: str):
        """Send notification about pipeline status"""
        try:
            if status == "success":
                message = f"✅ Pipeline {execution.id} completed successfully for {execution.environment.value}"
            elif status == "failure":
                message = f"❌ Pipeline {execution.id} failed for {execution.environment.value}"
            elif status == "rollback":
                message = f"🔄 Pipeline {execution.id} rolled back for {execution.environment.value}"
            else:
                message = f"ℹ️ Pipeline {execution.id} status: {status} for {execution.environment.value}"
            
            # Send Slack notification
            if self.monitoring_config["slack_webhook"]:
                await self._send_slack_notification(message, execution)
            
            # Send email notification
            if self.monitoring_config["email_notifications"]:
                await self._send_email_notification(message, execution)
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def _send_slack_notification(self, message: str, execution: PipelineExecution):
        """Send Slack notification"""
        try:
            webhook_url = self.monitoring_config["slack_webhook"]
            
            payload = {
                "text": message,
                "attachments": [
                    {
                        "color": "good" if execution.status == PipelineStatus.SUCCESS else "danger",
                        "fields": [
                            {"title": "Pipeline ID", "value": execution.id, "short": True},
                            {"title": "Environment", "value": execution.environment.value, "short": True},
                            {"title": "Status", "value": execution.status.value, "short": True},
                            {"title": "Duration", "value": f"{(execution.completed_at - execution.started_at).total_seconds():.2f}s", "short": True}
                        ]
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack notification sent successfully")
                    else:
                        logger.error(f"Failed to send Slack notification: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_email_notification(self, message: str, execution: PipelineExecution):
        """Send email notification"""
        try:
            # This would typically use an email service like SendGrid or SES
            # For now, just log the notification
            logger.info(f"Email notification: {message}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def get_pipeline_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get pipeline execution status"""
        return self.pipeline_executions.get(execution_id)
    
    def get_deployment_history(self, environment: Optional[DeploymentEnvironment] = None) -> List[PipelineExecution]:
        """Get deployment history"""
        if environment:
            return [exec for exec in self.pipeline_executions.values() if exec.environment == environment]
        return list(self.pipeline_executions.values())
    
    def cancel_pipeline(self, execution_id: str) -> bool:
        """Cancel running pipeline"""
        execution = self.pipeline_executions.get(execution_id)
        if execution and execution.status == PipelineStatus.RUNNING:
            execution.status = PipelineStatus.CANCELLED
            execution.completed_at = datetime.utcnow()
            logger.info(f"Pipeline {execution_id} cancelled")
            return True
        return False
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        executions = list(self.pipeline_executions.values())
        
        if not executions:
            return {"status": "no_data"}
        
        total_executions = len(executions)
        successful_executions = len([e for e in executions if e.status == PipelineStatus.SUCCESS])
        failed_executions = len([e for e in executions if e.status == PipelineStatus.FAILED])
        
        # Calculate average duration
        completed_executions = [e for e in executions if e.completed_at]
        if completed_executions:
            avg_duration = sum(
                (e.completed_at - e.started_at).total_seconds() 
                for e in completed_executions
            ) / len(completed_executions)
        else:
            avg_duration = 0
        
        # Success rate by environment
        success_rate_by_env = {}
        for env in DeploymentEnvironment:
            env_executions = [e for e in executions if e.environment == env]
            if env_executions:
                env_successful = len([e for e in env_executions if e.status == PipelineStatus.SUCCESS])
                success_rate_by_env[env.value] = (env_successful / len(env_executions)) * 100
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": (successful_executions / total_executions) * 100 if total_executions > 0 else 0,
            "average_duration_seconds": avg_duration,
            "success_rate_by_environment": success_rate_by_env,
            "last_execution": executions[-1].started_at.isoformat() if executions else None
        }
    
    def export_pipeline_data(self) -> Dict[str, Any]:
        """Export pipeline data for analysis"""
        return {
            "executions": [
                {
                    "id": exec.id,
                    "status": exec.status.value,
                    "environment": exec.environment.value,
                    "started_at": exec.started_at.isoformat(),
                    "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                    "steps": exec.steps,
                    "error": exec.error
                }
                for exec in self.pipeline_executions.values()
            ],
            "metrics": self.get_pipeline_metrics(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global CI/CD pipeline instance
ci_cd_pipeline = None

def get_ci_cd_pipeline() -> CICDPipeline:
    """Get the global CI/CD pipeline instance"""
    global ci_cd_pipeline
    if ci_cd_pipeline is None:
        config = {
            "docker_registry": "registry.bul.local",
            "namespace": "bul-system",
            "monitoring_enabled": True
        }
        ci_cd_pipeline = CICDPipeline(config)
    return ci_cd_pipeline

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "docker_registry": "registry.bul.local",
            "namespace": "bul-system",
            "monitoring_enabled": True
        }
        
        pipeline = CICDPipeline(config)
        
        # Run pipeline for staging
        execution = await pipeline.run_pipeline(
            environment=DeploymentEnvironment.STAGING,
            branch="feature/new-feature",
            commit_sha="abc123"
        )
        
        print(f"Pipeline execution {execution.id} completed with status: {execution.status}")
        
        # Get pipeline metrics
        metrics = pipeline.get_pipeline_metrics()
        print("Pipeline Metrics:")
        print(json.dumps(metrics, indent=2))
    
    asyncio.run(main())













