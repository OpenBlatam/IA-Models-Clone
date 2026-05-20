#!/usr/bin/env python3
"""
Advanced AI/ML Pipeline Automation for Frontier Model Training
Provides comprehensive pipeline orchestration, workflow management, and automation.
"""

import os
import json
import yaml
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import docker
import kubernetes
from kubernetes import client, config
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
import mlflow
import wandb
import optuna
import ray
from ray import tune
import dask
from dask.distributed import Client
import prefect
from prefect import flow, task
import celery
from celery import Celery
import redis
import sqlite3
from contextlib import contextmanager

console = Console()

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class ResourceType(Enum):
    """Resource types for task execution."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

class ExecutionEnvironment(Enum):
    """Execution environments."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    RAY = "ray"
    DASK = "dask"

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    pipeline_name: str
    description: str
    version: str = "1.0.0"
    environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL
    max_parallel_tasks: int = 4
    timeout_minutes: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 30
    resource_limits: Dict[ResourceType, float] = None
    environment_variables: Dict[str, str] = None
    secrets: Dict[str, str] = None
    notifications: Dict[str, Any] = None
    monitoring: bool = True
    logging_level: str = "INFO"

@dataclass
class TaskConfig:
    """Task configuration."""
    task_name: str
    task_type: str
    command: Optional[str] = None
    script_path: Optional[str] = None
    function: Optional[Callable] = None
    dependencies: List[str] = None
    resources: Dict[ResourceType, float] = None
    timeout_minutes: int = 30
    retry_attempts: int = 2
    retry_delay_seconds: int = 10
    environment_variables: Dict[str, str] = None
    inputs: Dict[str, Any] = None
    outputs: Dict[str, Any] = None
    condition: Optional[Callable] = None
    parallel_execution: bool = False

@dataclass
class PipelineExecution:
    """Pipeline execution information."""
    execution_id: str
    pipeline_name: str
    status: PipelineStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    tasks: List[Dict[str, Any]] = None
    logs: List[str] = None
    metrics: Dict[str, Any] = None
    error_message: Optional[str] = None

class TaskExecutor:
    """Base task executor."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task."""
        raise NotImplementedError
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate task inputs."""
        return True
    
    def prepare_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare task outputs."""
        return outputs

class LocalTaskExecutor(TaskExecutor):
    """Local task executor."""
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task locally."""
        try:
            if self.config.command:
                return self._execute_command()
            elif self.config.script_path:
                return self._execute_script()
            elif self.config.function:
                return self._execute_function(context)
            else:
                raise ValueError("No execution method specified")
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            raise
    
    def _execute_command(self) -> Dict[str, Any]:
        """Execute shell command."""
        import subprocess
        
        result = subprocess.run(
            self.config.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_minutes * 60
        )
        
        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    
    def _execute_script(self) -> Dict[str, Any]:
        """Execute Python script."""
        import subprocess
        
        result = subprocess.run(
            ["python", self.config.script_path],
            capture_output=True,
            text=True,
            timeout=self.config.timeout_minutes * 60
        )
        
        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    
    def _execute_function(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python function."""
        try:
            result = self.config.function(**context)
            return {
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

class DockerTaskExecutor(TaskExecutor):
    """Docker task executor."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.docker_client = docker.from_env()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task in Docker container."""
        try:
            # Build or pull image
            image = self._get_or_build_image()
            
            # Run container
            container = self.docker_client.containers.run(
                image,
                command=self.config.command,
                environment=self.config.environment_variables,
                volumes=self._prepare_volumes(),
                network_mode="host",
                detach=True,
                remove=True
            )
            
            # Wait for completion
            result = container.wait(timeout=self.config.timeout_minutes * 60)
            
            # Get logs
            logs = container.logs().decode('utf-8')
            
            return {
                "return_code": result["StatusCode"],
                "logs": logs,
                "success": result["StatusCode"] == 0
            }
            
        except Exception as e:
            self.logger.error(f"Docker task execution failed: {e}")
            raise
    
    def _get_or_build_image(self) -> str:
        """Get or build Docker image."""
        # Simplified - in practice, you'd implement proper image management
        return "python:3.9"
    
    def _prepare_volumes(self) -> Dict[str, Dict[str, str]]:
        """Prepare Docker volumes."""
        return {}

class KubernetesTaskExecutor(TaskExecutor):
    """Kubernetes task executor."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.batch_v1 = client.BatchV1Api()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task in Kubernetes."""
        try:
            # Create job
            job = self._create_job()
            
            # Submit job
            job_response = self.batch_v1.create_namespaced_job(
                namespace="default",
                body=job
            )
            
            # Wait for completion
            self._wait_for_job_completion(job_response.metadata.name)
            
            # Get logs
            logs = self._get_job_logs(job_response.metadata.name)
            
            return {
                "job_name": job_response.metadata.name,
                "logs": logs,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Kubernetes task execution failed: {e}")
            raise
    
    def _create_job(self) -> client.V1Job:
        """Create Kubernetes job."""
        # Simplified job creation
        job = client.V1Job(
            metadata=client.V1ObjectMeta(name=f"task-{self.config.task_name}"),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="task-container",
                                image="python:3.9",
                                command=["python", "-c", self.config.command or "print('Hello World')"]
                            )
                        ],
                        restart_policy="Never"
                    )
                )
            )
        )
        return job
    
    def _wait_for_job_completion(self, job_name: str):
        """Wait for job completion."""
        # Simplified - in practice, you'd implement proper waiting logic
        time.sleep(10)
    
    def _get_job_logs(self, job_name: str) -> str:
        """Get job logs."""
        # Simplified - in practice, you'd implement proper log retrieval
        return "Job completed successfully"

class RayTaskExecutor(TaskExecutor):
    """Ray task executor."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        if not ray.is_initialized():
            ray.init()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using Ray."""
        try:
            if self.config.function:
                # Execute function remotely
                remote_func = ray.remote(self.config.function)
                result = ray.get(remote_func.remote(**context))
                
                return {
                    "result": result,
                    "success": True
                }
            else:
                raise ValueError("Ray executor requires a function")
                
        except Exception as e:
            self.logger.error(f"Ray task execution failed: {e}")
            raise

class PipelineOrchestrator:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.task_graph = nx.DiGraph()
        self.execution_history = []
        self.active_executions = {}
        
        # Initialize execution environment
        self.executor_factory = self._init_executor_factory()
        
        # Initialize monitoring
        if config.monitoring:
            self._init_monitoring()
    
    def _init_executor_factory(self) -> Dict[ExecutionEnvironment, Callable]:
        """Initialize executor factory."""
        return {
            ExecutionEnvironment.LOCAL: LocalTaskExecutor,
            ExecutionEnvironment.DOCKER: DockerTaskExecutor,
            ExecutionEnvironment.KUBERNETES: KubernetesTaskExecutor,
            ExecutionEnvironment.RAY: RayTaskExecutor
        }
    
    def _init_monitoring(self):
        """Initialize monitoring."""
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.config.pipeline_name)
        
        # Initialize Weights & Biases
        wandb.init(project=self.config.pipeline_name, mode="disabled")
    
    def add_task(self, task_config: TaskConfig):
        """Add task to pipeline."""
        self.task_graph.add_node(
            task_config.task_name,
            config=task_config,
            status=TaskStatus.PENDING
        )
        
        # Add dependencies
        if task_config.dependencies:
            for dep in task_config.dependencies:
                self.task_graph.add_edge(dep, task_config.task_name)
    
    def validate_pipeline(self) -> Tuple[bool, List[str]]:
        """Validate pipeline configuration."""
        errors = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.task_graph):
            errors.append("Pipeline contains cycles")
        
        # Check for orphaned tasks
        orphaned = [node for node in self.task_graph.nodes() 
                   if self.task_graph.in_degree(node) == 0 and self.task_graph.out_degree(node) == 0]
        if len(orphaned) > 1:
            errors.append(f"Multiple orphaned tasks found: {orphaned}")
        
        # Validate task configurations
        for node in self.task_graph.nodes():
            task_config = self.task_graph.nodes[node]['config']
            if not self._validate_task_config(task_config):
                errors.append(f"Invalid configuration for task: {task_config.task_name}")
        
        return len(errors) == 0, errors
    
    def _validate_task_config(self, task_config: TaskConfig) -> bool:
        """Validate individual task configuration."""
        # Check required fields
        if not task_config.task_name:
            return False
        
        # Check execution method
        execution_methods = [task_config.command, task_config.script_path, task_config.function]
        if not any(execution_methods):
            return False
        
        return True
    
    def execute_pipeline(self, context: Dict[str, Any] = None) -> str:
        """Execute the entire pipeline."""
        context = context or {}
        
        # Validate pipeline
        is_valid, errors = self.validate_pipeline()
        if not is_valid:
            raise ValueError(f"Pipeline validation failed: {errors}")
        
        # Create execution
        execution_id = self._generate_execution_id()
        execution = PipelineExecution(
            execution_id=execution_id,
            pipeline_name=self.config.pipeline_name,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
            tasks=[],
            logs=[],
            metrics={}
        )
        
        self.active_executions[execution_id] = execution
        
        # Start execution in background
        execution_thread = threading.Thread(
            target=self._execute_pipeline_thread,
            args=(execution, context),
            daemon=True
        )
        execution_thread.start()
        
        console.print(f"[blue]Pipeline execution started: {execution_id}[/blue]")
        return execution_id
    
    def _execute_pipeline_thread(self, execution: PipelineExecution, context: Dict[str, Any]):
        """Execute pipeline in background thread."""
        try:
            # Get topological order
            task_order = list(nx.topological_sort(self.task_graph))
            
            # Execute tasks
            for task_name in task_order:
                if execution.status == PipelineStatus.CANCELLED:
                    break
                
                task_config = self.task_graph.nodes[task_name]['config']
                
                # Check task condition
                if task_config.condition and not task_config.condition(context):
                    self.logger.info(f"Skipping task {task_name} due to condition")
                    continue
                
                # Execute task
                task_result = self._execute_task(task_name, task_config, context)
                
                # Update execution
                execution.tasks.append({
                    "task_name": task_name,
                    "status": task_result["status"],
                    "started_at": task_result["started_at"],
                    "completed_at": task_result["completed_at"],
                    "duration": task_result["duration"],
                    "result": task_result["result"]
                })
                
                # Update context with task outputs
                if task_result["result"] and "outputs" in task_result["result"]:
                    context.update(task_result["result"]["outputs"])
                
                # Check for failure
                if task_result["status"] == TaskStatus.FAILED:
                    execution.status = PipelineStatus.FAILED
                    execution.error_message = task_result["result"].get("error", "Task failed")
                    break
            
            # Mark as completed if not failed or cancelled
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.COMPLETED
            
            execution.completed_at = datetime.now()
            
            # Log completion
            if execution.status == PipelineStatus.COMPLETED:
                console.print(f"[green]Pipeline execution completed: {execution.execution_id}[/green]")
            else:
                console.print(f"[red]Pipeline execution failed: {execution.execution_id}[/red]")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            self.logger.error(f"Pipeline execution failed: {e}")
        
        finally:
            # Remove from active executions
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
            
            # Add to history
            self.execution_history.append(execution)
    
    def _execute_task(self, task_name: str, task_config: TaskConfig, 
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual task."""
        start_time = datetime.now()
        
        try:
            # Create executor
            executor_class = self.executor_factory[self.config.environment]
            executor = executor_class(task_config)
            
            # Execute task
            result = executor.execute(context)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": TaskStatus.COMPLETED,
                "started_at": start_time,
                "completed_at": datetime.now(),
                "duration": duration,
                "result": result
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": TaskStatus.FAILED,
                "started_at": start_time,
                "completed_at": datetime.now(),
                "duration": duration,
                "result": {"error": str(e)}
            }
    
    def get_execution_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get execution status."""
        # Check active executions
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Check history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return execution
        
        return None
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel pipeline execution."""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = PipelineStatus.CANCELLED
            execution.completed_at = datetime.now()
            return True
        return False
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for e in self.execution_history 
                                 if e.status == PipelineStatus.COMPLETED)
        failed_executions = sum(1 for e in self.execution_history 
                             if e.status == PipelineStatus.FAILED)
        
        avg_duration = 0
        if total_executions > 0:
            durations = [(e.completed_at - e.started_at).total_seconds() 
                        for e in self.execution_history if e.completed_at]
            avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "average_duration_seconds": avg_duration,
            "active_executions": len(self.active_executions)
        }
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = secrets.token_hex(4)
        return f"{self.config.pipeline_name}_{timestamp}_{random_suffix}"
    
    def export_pipeline(self, file_path: str):
        """Export pipeline configuration."""
        pipeline_data = {
            "config": asdict(self.config),
            "tasks": []
        }
        
        for node in self.task_graph.nodes():
            task_config = self.task_graph.nodes[node]['config']
            task_data = asdict(task_config)
            task_data["dependencies"] = list(self.task_graph.predecessors(node))
            pipeline_data["tasks"].append(task_data)
        
        with open(file_path, 'w') as f:
            yaml.dump(pipeline_data, f, default_flow_style=False)
    
    def import_pipeline(self, file_path: str):
        """Import pipeline configuration."""
        with open(file_path, 'r') as f:
            pipeline_data = yaml.safe_load(f)
        
        # Load config
        self.config = PipelineConfig(**pipeline_data["config"])
        
        # Clear existing tasks
        self.task_graph.clear()
        
        # Load tasks
        for task_data in pipeline_data["tasks"]:
            task_config = TaskConfig(**task_data)
            self.add_task(task_config)

class MLPipelineBuilder:
    """Builder for ML-specific pipelines."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.orchestrator = None
    
    def create_data_pipeline(self, source_path: str, output_path: str) -> 'MLPipelineBuilder':
        """Create data processing pipeline."""
        config = PipelineConfig(
            pipeline_name=f"{self.pipeline_name}_data",
            description="Data processing pipeline"
        )
        
        self.orchestrator = PipelineOrchestrator(config)
        
        # Add data processing tasks
        self.orchestrator.add_task(TaskConfig(
            task_name="load_data",
            task_type="data_loading",
            script_path="./scripts/data_processor.py",
            inputs={"source_path": source_path},
            outputs={"data_path": output_path}
        ))
        
        self.orchestrator.add_task(TaskConfig(
            task_name="preprocess_data",
            task_type="preprocessing",
            script_path="./scripts/data_processor.py",
            dependencies=["load_data"],
            inputs={"data_path": output_path}
        ))
        
        return self
    
    def create_training_pipeline(self, model_config: Dict[str, Any]) -> 'MLPipelineBuilder':
        """Create model training pipeline."""
        if not self.orchestrator:
            raise ValueError("Must create data pipeline first")
        
        # Add training tasks
        self.orchestrator.add_task(TaskConfig(
            task_name="train_model",
            task_type="training",
            script_path="./scripts/grpo_train.py",
            dependencies=["preprocess_data"],
            inputs=model_config,
            outputs={"model_path": "./models/trained_model"}
        ))
        
        self.orchestrator.add_task(TaskConfig(
            task_name="evaluate_model",
            task_type="evaluation",
            script_path="./scripts/model_evaluator.py",
            dependencies=["train_model"],
            inputs={"model_path": "./models/trained_model"}
        ))
        
        return self
    
    def create_deployment_pipeline(self, deployment_config: Dict[str, Any]) -> 'MLPipelineBuilder':
        """Create model deployment pipeline."""
        if not self.orchestrator:
            raise ValueError("Must create training pipeline first")
        
        # Add deployment tasks
        self.orchestrator.add_task(TaskConfig(
            task_name="optimize_model",
            task_type="optimization",
            script_path="./scripts/model_optimizer.py",
            dependencies=["evaluate_model"],
            inputs={"model_path": "./models/trained_model"}
        ))
        
        self.orchestrator.add_task(TaskConfig(
            task_name="deploy_model",
            task_type="deployment",
            script_path="./scripts/model_server.py",
            dependencies=["optimize_model"],
            inputs=deployment_config
        ))
        
        return self
    
    def build(self) -> PipelineOrchestrator:
        """Build the pipeline."""
        if not self.orchestrator:
            raise ValueError("No pipeline created")
        
        return self.orchestrator

def main():
    """Main function for pipeline CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    parser.add_argument("--action", type=str,
                       choices=["create", "execute", "status", "cancel", "export", "import"],
                       required=True, help="Action to perform")
    parser.add_argument("--pipeline-name", type=str, default="ml_pipeline",
                       help="Pipeline name")
    parser.add_argument("--config-file", type=str, help="Pipeline configuration file")
    parser.add_argument("--execution-id", type=str, help="Execution ID")
    parser.add_argument("--environment", type=str,
                       choices=["local", "docker", "kubernetes", "ray"],
                       default="local", help="Execution environment")
    
    args = parser.parse_args()
    
    if args.action == "create":
        # Create ML pipeline
        builder = MLPipelineBuilder(args.pipeline_name)
        orchestrator = (builder
                       .create_data_pipeline("./data", "./processed_data")
                       .create_training_pipeline({"epochs": 10, "batch_size": 32})
                       .create_deployment_pipeline({"host": "0.0.0.0", "port": 8000})
                       .build())
        
        # Export pipeline
        orchestrator.export_pipeline(f"{args.pipeline_name}.yaml")
        console.print(f"[green]Pipeline created: {args.pipeline_name}.yaml[/green]")
    
    elif args.action == "execute":
        if not args.config_file:
            console.print("[red]Configuration file required for execution[/red]")
            return
        
        # Load pipeline
        config = PipelineConfig(
            pipeline_name=args.pipeline_name,
            environment=ExecutionEnvironment(args.environment)
        )
        orchestrator = PipelineOrchestrator(config)
        orchestrator.import_pipeline(args.config_file)
        
        # Execute pipeline
        execution_id = orchestrator.execute_pipeline()
        console.print(f"[green]Pipeline execution started: {execution_id}[/green]")
    
    elif args.action == "status":
        if not args.execution_id:
            console.print("[red]Execution ID required for status check[/red]")
            return
        
        # Get execution status
        config = PipelineConfig(pipeline_name=args.pipeline_name)
        orchestrator = PipelineOrchestrator(config)
        execution = orchestrator.get_execution_status(args.execution_id)
        
        if execution:
            console.print(f"[blue]Execution Status: {execution.status.value}[/blue]")
            console.print(f"[blue]Started: {execution.started_at}[/blue]")
            if execution.completed_at:
                console.print(f"[blue]Completed: {execution.completed_at}[/blue]")
        else:
            console.print("[red]Execution not found[/red]")
    
    elif args.action == "cancel":
        if not args.execution_id:
            console.print("[red]Execution ID required for cancellation[/red]")
            return
        
        # Cancel execution
        config = PipelineConfig(pipeline_name=args.pipeline_name)
        orchestrator = PipelineOrchestrator(config)
        success = orchestrator.cancel_execution(args.execution_id)
        
        if success:
            console.print("[green]Execution cancelled[/green]")
        else:
            console.print("[red]Execution not found or not running[/red]")
    
    elif args.action == "export":
        if not args.config_file:
            console.print("[red]Configuration file required for export[/red]")
            return
        
        # Export pipeline
        config = PipelineConfig(pipeline_name=args.pipeline_name)
        orchestrator = PipelineOrchestrator(config)
        orchestrator.import_pipeline(args.config_file)
        orchestrator.export_pipeline(f"{args.pipeline_name}_exported.yaml")
        console.print(f"[green]Pipeline exported: {args.pipeline_name}_exported.yaml[/green]")
    
    elif args.action == "import":
        if not args.config_file:
            console.print("[red]Configuration file required for import[/red]")
            return
        
        # Import pipeline
        config = PipelineConfig(pipeline_name=args.pipeline_name)
        orchestrator = PipelineOrchestrator(config)
        orchestrator.import_pipeline(args.config_file)
        console.print(f"[green]Pipeline imported: {args.config_file}[/green]")

if __name__ == "__main__":
    main()
