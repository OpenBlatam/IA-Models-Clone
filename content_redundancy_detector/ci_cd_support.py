"""
CI/CD Support for Continuous Integration and Deployment
Sistema CI/CD para integración y despliegue continuo ultra-optimizado
"""

import asyncio
import logging
import time
import json
import subprocess
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Estados de pipeline"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class StageType(Enum):
    """Tipos de stage"""
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    SECURITY = "security"
    QUALITY = "quality"
    NOTIFICATION = "notification"


class TriggerType(Enum):
    """Tipos de trigger"""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    SCHEDULE = "schedule"
    MANUAL = "manual"
    WEBHOOK = "webhook"


@dataclass
class PipelineInfo:
    """Información de pipeline"""
    id: str
    name: str
    status: PipelineStatus
    trigger_type: TriggerType
    branch: str
    commit_sha: str
    author: str
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    duration: Optional[float]
    stages: List[str]
    metadata: Dict[str, Any]


@dataclass
class StageInfo:
    """Información de stage"""
    id: str
    name: str
    type: StageType
    status: PipelineStatus
    pipeline_id: str
    started_at: Optional[float]
    completed_at: Optional[float]
    duration: Optional[float]
    jobs: List[str]
    logs: str
    metadata: Dict[str, Any]


@dataclass
class JobInfo:
    """Información de job"""
    id: str
    name: str
    stage_id: str
    status: PipelineStatus
    started_at: Optional[float]
    completed_at: Optional[float]
    duration: Optional[float]
    logs: str
    exit_code: Optional[int]
    metadata: Dict[str, Any]


@dataclass
class BuildInfo:
    """Información de build"""
    id: str
    name: str
    version: str
    status: PipelineStatus
    created_at: float
    completed_at: Optional[float]
    duration: Optional[float]
    artifacts: List[str]
    metadata: Dict[str, Any]


@dataclass
class DeploymentInfo:
    """Información de deployment"""
    id: str
    name: str
    environment: str
    version: str
    status: PipelineStatus
    created_at: float
    completed_at: Optional[float]
    duration: Optional[float]
    rollback_version: Optional[str]
    metadata: Dict[str, Any]


class GitManager:
    """Manager de Git"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
    
    async def get_current_branch(self) -> str:
        """Obtener rama actual"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception as e:
            logger.error(f"Error getting current branch: {e}")
            return "unknown"
    
    async def get_current_commit(self) -> str:
        """Obtener commit actual"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception as e:
            logger.error(f"Error getting current commit: {e}")
            return "unknown"
    
    async def get_commit_author(self, commit_sha: str) -> str:
        """Obtener autor del commit"""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%an", commit_sha],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception as e:
            logger.error(f"Error getting commit author: {e}")
            return "unknown"
    
    async def get_changed_files(self, commit_sha: str) -> List[str]:
        """Obtener archivos cambiados en commit"""
        try:
            result = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except Exception as e:
            logger.error(f"Error getting changed files: {e}")
            return []
    
    async def get_diff(self, commit_sha: str) -> str:
        """Obtener diff del commit"""
        try:
            result = subprocess.run(
                ["git", "show", commit_sha],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout
        except Exception as e:
            logger.error(f"Error getting diff: {e}")
            return ""


class TestRunner:
    """Runner de tests"""
    
    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """Ejecutar tests unitarios"""
        try:
            start_time = time.time()
            
            # Simular ejecución de tests
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            duration = time.time() - start_time
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_count": len([line for line in result.stdout.split('\n') if 'PASSED' in line or 'FAILED' in line])
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "exit_code": -1,
                "duration": 300,
                "stdout": "",
                "stderr": "Test execution timed out",
                "test_count": 0
            }
        except Exception as e:
            logger.error(f"Error running unit tests: {e}")
            return {
                "status": "failed",
                "exit_code": -1,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "test_count": 0
            }
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Ejecutar tests de integración"""
        try:
            start_time = time.time()
            
            # Simular ejecución de tests de integración
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/integration/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            duration = time.time() - start_time
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_count": len([line for line in result.stdout.split('\n') if 'PASSED' in line or 'FAILED' in line])
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "exit_code": -1,
                "duration": 600,
                "stdout": "",
                "stderr": "Integration test execution timed out",
                "test_count": 0
            }
        except Exception as e:
            logger.error(f"Error running integration tests: {e}")
            return {
                "status": "failed",
                "exit_code": -1,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "test_count": 0
            }
    
    async def run_security_tests(self) -> Dict[str, Any]:
        """Ejecutar tests de seguridad"""
        try:
            start_time = time.time()
            
            # Simular ejecución de tests de seguridad
            result = subprocess.run(
                ["python", "-m", "bandit", "-r", ".", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            duration = time.time() - start_time
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "vulnerabilities": json.loads(result.stdout) if result.stdout else []
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "exit_code": -1,
                "duration": 120,
                "stdout": "",
                "stderr": "Security test execution timed out",
                "vulnerabilities": []
            }
        except Exception as e:
            logger.error(f"Error running security tests: {e}")
            return {
                "status": "failed",
                "exit_code": -1,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "vulnerabilities": []
            }


class BuildManager:
    """Manager de builds"""
    
    def __init__(self):
        self.builds: Dict[str, BuildInfo] = {}
    
    async def create_build(self, name: str, version: str) -> BuildInfo:
        """Crear build"""
        build_id = f"build_{int(time.time())}"
        
        build = BuildInfo(
            id=build_id,
            name=name,
            version=version,
            status=PipelineStatus.PENDING,
            created_at=time.time(),
            completed_at=None,
            duration=None,
            artifacts=[],
            metadata={}
        )
        
        self.builds[build_id] = build
        return build
    
    async def start_build(self, build_id: str) -> bool:
        """Iniciar build"""
        if build_id not in self.builds:
            return False
        
        build = self.builds[build_id]
        build.status = PipelineStatus.RUNNING
        build.metadata["started_at"] = time.time()
        
        # Simular proceso de build
        try:
            # Docker build
            result = subprocess.run(
                ["docker", "build", "-t", f"{build.name}:{build.version}", "."],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                build.status = PipelineStatus.SUCCESS
                build.artifacts.append(f"{build.name}:{build.version}")
            else:
                build.status = PipelineStatus.FAILED
            
            build.completed_at = time.time()
            build.duration = build.completed_at - build.metadata["started_at"]
            
            return build.status == PipelineStatus.SUCCESS
            
        except subprocess.TimeoutExpired:
            build.status = PipelineStatus.FAILED
            build.completed_at = time.time()
            build.duration = 300
            return False
        except Exception as e:
            logger.error(f"Error in build process: {e}")
            build.status = PipelineStatus.FAILED
            build.completed_at = time.time()
            build.duration = time.time() - build.metadata["started_at"]
            return False
    
    async def get_build(self, build_id: str) -> Optional[BuildInfo]:
        """Obtener build"""
        return self.builds.get(build_id)
    
    async def list_builds(self) -> List[BuildInfo]:
        """Listar builds"""
        return list(self.builds.values())


class DeploymentManager:
    """Manager de deployments"""
    
    def __init__(self):
        self.deployments: Dict[str, DeploymentInfo] = {}
    
    async def create_deployment(self, name: str, environment: str, version: str) -> DeploymentInfo:
        """Crear deployment"""
        deployment_id = f"deploy_{int(time.time())}"
        
        deployment = DeploymentInfo(
            id=deployment_id,
            name=name,
            environment=environment,
            version=version,
            status=PipelineStatus.PENDING,
            created_at=time.time(),
            completed_at=None,
            duration=None,
            rollback_version=None,
            metadata={}
        )
        
        self.deployments[deployment_id] = deployment
        return deployment
    
    async def start_deployment(self, deployment_id: str) -> bool:
        """Iniciar deployment"""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        deployment.status = PipelineStatus.RUNNING
        deployment.metadata["started_at"] = time.time()
        
        # Simular proceso de deployment
        try:
            # Kubernetes deployment
            result = subprocess.run(
                ["kubectl", "apply", "-f", f"k8s/{deployment.environment}/"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                deployment.status = PipelineStatus.SUCCESS
            else:
                deployment.status = PipelineStatus.FAILED
            
            deployment.completed_at = time.time()
            deployment.duration = deployment.completed_at - deployment.metadata["started_at"]
            
            return deployment.status == PipelineStatus.SUCCESS
            
        except subprocess.TimeoutExpired:
            deployment.status = PipelineStatus.FAILED
            deployment.completed_at = time.time()
            deployment.duration = 120
            return False
        except Exception as e:
            logger.error(f"Error in deployment process: {e}")
            deployment.status = PipelineStatus.FAILED
            deployment.completed_at = time.time()
            deployment.duration = time.time() - deployment.metadata["started_at"]
            return False
    
    async def rollback_deployment(self, deployment_id: str, version: str) -> bool:
        """Rollback deployment"""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        deployment.rollback_version = version
        deployment.status = PipelineStatus.RUNNING
        
        # Simular rollback
        try:
            result = subprocess.run(
                ["kubectl", "rollout", "undo", f"deployment/{deployment.name}"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                deployment.status = PipelineStatus.SUCCESS
            else:
                deployment.status = PipelineStatus.FAILED
            
            return deployment.status == PipelineStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Error in rollback process: {e}")
            deployment.status = PipelineStatus.FAILED
            return False
    
    async def get_deployment(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """Obtener deployment"""
        return self.deployments.get(deployment_id)
    
    async def list_deployments(self) -> List[DeploymentInfo]:
        """Listar deployments"""
        return list(self.deployments.values())


class PipelineManager:
    """Manager de pipelines"""
    
    def __init__(self):
        self.pipelines: Dict[str, PipelineInfo] = {}
        self.stages: Dict[str, StageInfo] = {}
        self.jobs: Dict[str, JobInfo] = {}
        self.git_manager = GitManager()
        self.test_runner = TestRunner()
        self.build_manager = BuildManager()
        self.deployment_manager = DeploymentManager()
    
    async def create_pipeline(self, name: str, trigger_type: TriggerType, 
                            branch: str = None, commit_sha: str = None) -> PipelineInfo:
        """Crear pipeline"""
        pipeline_id = f"pipeline_{int(time.time())}"
        
        if not branch:
            branch = await self.git_manager.get_current_branch()
        if not commit_sha:
            commit_sha = await self.git_manager.get_current_commit()
        
        author = await self.git_manager.get_commit_author(commit_sha)
        
        pipeline = PipelineInfo(
            id=pipeline_id,
            name=name,
            status=PipelineStatus.PENDING,
            trigger_type=trigger_type,
            branch=branch,
            commit_sha=commit_sha,
            author=author,
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            duration=None,
            stages=[],
            metadata={}
        )
        
        self.pipelines[pipeline_id] = pipeline
        return pipeline
    
    async def start_pipeline(self, pipeline_id: str) -> bool:
        """Iniciar pipeline"""
        if pipeline_id not in self.pipelines:
            return False
        
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = PipelineStatus.RUNNING
        pipeline.started_at = time.time()
        
        try:
            # Ejecutar stages del pipeline
            success = await self._execute_pipeline_stages(pipeline)
            
            pipeline.completed_at = time.time()
            pipeline.duration = pipeline.completed_at - pipeline.started_at
            pipeline.status = PipelineStatus.SUCCESS if success else PipelineStatus.FAILED
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing pipeline: {e}")
            pipeline.status = PipelineStatus.FAILED
            pipeline.completed_at = time.time()
            pipeline.duration = time.time() - pipeline.started_at
            return False
    
    async def _execute_pipeline_stages(self, pipeline: PipelineInfo) -> bool:
        """Ejecutar stages del pipeline"""
        # Stage 1: Build
        build_stage = await self._create_stage("build", StageType.BUILD, pipeline.id)
        pipeline.stages.append(build_stage.id)
        
        if not await self._execute_stage(build_stage):
            return False
        
        # Stage 2: Test
        test_stage = await self._create_stage("test", StageType.TEST, pipeline.id)
        pipeline.stages.append(test_stage.id)
        
        if not await self._execute_stage(test_stage):
            return False
        
        # Stage 3: Security
        security_stage = await self._create_stage("security", StageType.SECURITY, pipeline.id)
        pipeline.stages.append(security_stage.id)
        
        if not await self._execute_stage(security_stage):
            return False
        
        # Stage 4: Deploy (solo si es main branch)
        if pipeline.branch == "main":
            deploy_stage = await self._create_stage("deploy", StageType.DEPLOY, pipeline.id)
            pipeline.stages.append(deploy_stage.id)
            
            if not await self._execute_stage(deploy_stage):
                return False
        
        return True
    
    async def _create_stage(self, name: str, stage_type: StageType, pipeline_id: str) -> StageInfo:
        """Crear stage"""
        stage_id = f"stage_{int(time.time())}"
        
        stage = StageInfo(
            id=stage_id,
            name=name,
            type=stage_type,
            status=PipelineStatus.PENDING,
            pipeline_id=pipeline_id,
            started_at=None,
            completed_at=None,
            duration=None,
            jobs=[],
            logs="",
            metadata={}
        )
        
        self.stages[stage_id] = stage
        return stage
    
    async def _execute_stage(self, stage: StageInfo) -> bool:
        """Ejecutar stage"""
        stage.status = PipelineStatus.RUNNING
        stage.started_at = time.time()
        
        try:
            if stage.type == StageType.BUILD:
                # Crear y ejecutar build
                build = await self.build_manager.create_build("content-redundancy-detector", "latest")
                success = await self.build_manager.start_build(build.id)
                stage.logs = f"Build {build.id}: {'SUCCESS' if success else 'FAILED'}"
                
            elif stage.type == StageType.TEST:
                # Ejecutar tests
                unit_tests = await self.test_runner.run_unit_tests()
                integration_tests = await self.test_runner.run_integration_tests()
                
                success = (unit_tests["status"] == "success" and 
                          integration_tests["status"] == "success")
                
                stage.logs = f"Unit Tests: {unit_tests['status']}, Integration Tests: {integration_tests['status']}"
                
            elif stage.type == StageType.SECURITY:
                # Ejecutar tests de seguridad
                security_tests = await self.test_runner.run_security_tests()
                success = security_tests["status"] == "success"
                stage.logs = f"Security Tests: {security_tests['status']}"
                
            elif stage.type == StageType.DEPLOY:
                # Crear y ejecutar deployment
                deployment = await self.deployment_manager.create_deployment(
                    "content-redundancy-detector", "production", "latest"
                )
                success = await self.deployment_manager.start_deployment(deployment.id)
                stage.logs = f"Deployment {deployment.id}: {'SUCCESS' if success else 'FAILED'}"
                
            else:
                success = True
                stage.logs = f"Stage {stage.name} completed"
            
            stage.completed_at = time.time()
            stage.duration = stage.completed_at - stage.started_at
            stage.status = PipelineStatus.SUCCESS if success else PipelineStatus.FAILED
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing stage {stage.name}: {e}")
            stage.status = PipelineStatus.FAILED
            stage.completed_at = time.time()
            stage.duration = time.time() - stage.started_at
            stage.logs = f"Error: {str(e)}"
            return False
    
    async def get_pipeline(self, pipeline_id: str) -> Optional[PipelineInfo]:
        """Obtener pipeline"""
        return self.pipelines.get(pipeline_id)
    
    async def list_pipelines(self) -> List[PipelineInfo]:
        """Listar pipelines"""
        return list(self.pipelines.values())
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de pipelines"""
        total_pipelines = len(self.pipelines)
        successful_pipelines = sum(1 for p in self.pipelines.values() if p.status == PipelineStatus.SUCCESS)
        failed_pipelines = sum(1 for p in self.pipelines.values() if p.status == PipelineStatus.FAILED)
        running_pipelines = sum(1 for p in self.pipelines.values() if p.status == PipelineStatus.RUNNING)
        
        return {
            "total_pipelines": total_pipelines,
            "successful_pipelines": successful_pipelines,
            "failed_pipelines": failed_pipelines,
            "running_pipelines": running_pipelines,
            "success_rate": (successful_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0
        }


# Instancia global del manager CI/CD
cicd_manager = PipelineManager()


# Router para endpoints CI/CD
cicd_router = APIRouter()


@cicd_router.post("/cicd/pipeline/create")
async def create_pipeline_endpoint(pipeline_data: dict):
    """Crear pipeline"""
    try:
        name = pipeline_data.get("name")
        trigger_type_str = pipeline_data.get("trigger_type", "manual")
        branch = pipeline_data.get("branch")
        commit_sha = pipeline_data.get("commit_sha")
        
        if not name:
            raise HTTPException(status_code=400, detail="Pipeline name is required")
        
        trigger_type = TriggerType(trigger_type_str)
        pipeline = await cicd_manager.create_pipeline(name, trigger_type, branch, commit_sha)
        
        return {
            "message": "Pipeline created successfully",
            "pipeline_id": pipeline.id,
            "name": pipeline.name,
            "status": pipeline.status.value,
            "branch": pipeline.branch,
            "commit_sha": pipeline.commit_sha
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid trigger type: {e}")
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")


@cicd_router.post("/cicd/pipeline/{pipeline_id}/start")
async def start_pipeline_endpoint(pipeline_id: str):
    """Iniciar pipeline"""
    try:
        success = await cicd_manager.start_pipeline(pipeline_id)
        
        if success:
            return {"message": "Pipeline started successfully", "pipeline_id": pipeline_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to start pipeline")
            
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")


@cicd_router.get("/cicd/pipeline/{pipeline_id}")
async def get_pipeline_endpoint(pipeline_id: str):
    """Obtener pipeline"""
    try:
        pipeline = await cicd_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {
            "id": pipeline.id,
            "name": pipeline.name,
            "status": pipeline.status.value,
            "trigger_type": pipeline.trigger_type.value,
            "branch": pipeline.branch,
            "commit_sha": pipeline.commit_sha,
            "author": pipeline.author,
            "created_at": pipeline.created_at,
            "started_at": pipeline.started_at,
            "completed_at": pipeline.completed_at,
            "duration": pipeline.duration,
            "stages": pipeline.stages,
            "metadata": pipeline.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")


@cicd_router.get("/cicd/pipelines")
async def list_pipelines_endpoint():
    """Listar pipelines"""
    try:
        pipelines = await cicd_manager.list_pipelines()
        return {
            "pipelines": [
                {
                    "id": pipeline.id,
                    "name": pipeline.name,
                    "status": pipeline.status.value,
                    "trigger_type": pipeline.trigger_type.value,
                    "branch": pipeline.branch,
                    "commit_sha": pipeline.commit_sha,
                    "author": pipeline.author,
                    "created_at": pipeline.created_at,
                    "duration": pipeline.duration
                }
                for pipeline in pipelines
            ]
        }
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@cicd_router.get("/cicd/stats")
async def get_cicd_stats_endpoint():
    """Obtener estadísticas CI/CD"""
    try:
        stats = await cicd_manager.get_pipeline_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting CI/CD stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get CI/CD stats: {str(e)}")


@cicd_router.post("/cicd/build/create")
async def create_build_endpoint(build_data: dict):
    """Crear build"""
    try:
        name = build_data.get("name")
        version = build_data.get("version", "latest")
        
        if not name:
            raise HTTPException(status_code=400, detail="Build name is required")
        
        build = await cicd_manager.build_manager.create_build(name, version)
        
        return {
            "message": "Build created successfully",
            "build_id": build.id,
            "name": build.name,
            "version": build.version,
            "status": build.status.value
        }
        
    except Exception as e:
        logger.error(f"Error creating build: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create build: {str(e)}")


@cicd_router.post("/cicd/build/{build_id}/start")
async def start_build_endpoint(build_id: str):
    """Iniciar build"""
    try:
        success = await cicd_manager.build_manager.start_build(build_id)
        
        if success:
            return {"message": "Build started successfully", "build_id": build_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to start build")
            
    except Exception as e:
        logger.error(f"Error starting build: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start build: {str(e)}")


@cicd_router.get("/cicd/builds")
async def list_builds_endpoint():
    """Listar builds"""
    try:
        builds = await cicd_manager.build_manager.list_builds()
        return {
            "builds": [
                {
                    "id": build.id,
                    "name": build.name,
                    "version": build.version,
                    "status": build.status.value,
                    "created_at": build.created_at,
                    "duration": build.duration,
                    "artifacts": build.artifacts
                }
                for build in builds
            ]
        }
    except Exception as e:
        logger.error(f"Error listing builds: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list builds: {str(e)}")


@cicd_router.post("/cicd/deployment/create")
async def create_deployment_endpoint(deployment_data: dict):
    """Crear deployment"""
    try:
        name = deployment_data.get("name")
        environment = deployment_data.get("environment", "production")
        version = deployment_data.get("version", "latest")
        
        if not name:
            raise HTTPException(status_code=400, detail="Deployment name is required")
        
        deployment = await cicd_manager.deployment_manager.create_deployment(name, environment, version)
        
        return {
            "message": "Deployment created successfully",
            "deployment_id": deployment.id,
            "name": deployment.name,
            "environment": deployment.environment,
            "version": deployment.version,
            "status": deployment.status.value
        }
        
    except Exception as e:
        logger.error(f"Error creating deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create deployment: {str(e)}")


@cicd_router.post("/cicd/deployment/{deployment_id}/start")
async def start_deployment_endpoint(deployment_id: str):
    """Iniciar deployment"""
    try:
        success = await cicd_manager.deployment_manager.start_deployment(deployment_id)
        
        if success:
            return {"message": "Deployment started successfully", "deployment_id": deployment_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to start deployment")
            
    except Exception as e:
        logger.error(f"Error starting deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start deployment: {str(e)}")


@cicd_router.post("/cicd/deployment/{deployment_id}/rollback")
async def rollback_deployment_endpoint(deployment_id: str, rollback_data: dict):
    """Rollback deployment"""
    try:
        version = rollback_data.get("version")
        
        if not version:
            raise HTTPException(status_code=400, detail="Rollback version is required")
        
        success = await cicd_manager.deployment_manager.rollback_deployment(deployment_id, version)
        
        if success:
            return {"message": "Deployment rollback started successfully", "deployment_id": deployment_id, "version": version}
        else:
            raise HTTPException(status_code=500, detail="Failed to rollback deployment")
            
    except Exception as e:
        logger.error(f"Error rolling back deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rollback deployment: {str(e)}")


@cicd_router.get("/cicd/deployments")
async def list_deployments_endpoint():
    """Listar deployments"""
    try:
        deployments = await cicd_manager.deployment_manager.list_deployments()
        return {
            "deployments": [
                {
                    "id": deployment.id,
                    "name": deployment.name,
                    "environment": deployment.environment,
                    "version": deployment.version,
                    "status": deployment.status.value,
                    "created_at": deployment.created_at,
                    "duration": deployment.duration,
                    "rollback_version": deployment.rollback_version
                }
                for deployment in deployments
            ]
        }
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list deployments: {str(e)}")


# Funciones de utilidad para integración
async def create_pipeline(name: str, trigger_type: TriggerType, branch: str = None, commit_sha: str = None) -> PipelineInfo:
    """Crear pipeline"""
    return await cicd_manager.create_pipeline(name, trigger_type, branch, commit_sha)


async def start_pipeline(pipeline_id: str) -> bool:
    """Iniciar pipeline"""
    return await cicd_manager.start_pipeline(pipeline_id)


async def get_pipeline(pipeline_id: str) -> Optional[PipelineInfo]:
    """Obtener pipeline"""
    return await cicd_manager.get_pipeline(pipeline_id)


async def get_cicd_stats() -> Dict[str, Any]:
    """Obtener estadísticas CI/CD"""
    return await cicd_manager.get_pipeline_stats()


logger.info("CI/CD support module loaded successfully")

