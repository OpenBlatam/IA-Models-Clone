#!/usr/bin/env python3
"""
Advanced Distributed Training System for Frontier Model Training
Provides multi-GPU, multi-node, and federated learning capabilities.
"""

import os
import json
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
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import ray
from ray import tune
import horovod.torch as hvd
import deepspeed
import fairscale
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import ShardedDataParallel as SDP
import accelerate
from accelerate import Accelerator
import wandb
import mlflow
import optuna
import dask
from dask.distributed import Client
import kubernetes
from kubernetes import client, config
import docker
import paramiko
import redis
import sqlite3
from contextlib import contextmanager

console = Console()

class DistributionStrategy(Enum):
    """Distribution strategies."""
    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    DISTRIBUTED = "distributed"
    HOROVOD = "horovod"
    DEEPSPEED = "deepspeed"
    FAIRSCALE = "fairscale"
    ACCELERATE = "accelerate"
    RAY = "ray"
    FEDERATED = "federated"

class NodeRole(Enum):
    """Node roles in distributed training."""
    MASTER = "master"
    WORKER = "worker"
    EVALUATOR = "evaluator"
    COORDINATOR = "coordinator"

class CommunicationBackend(Enum):
    """Communication backends."""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"
    TCP = "tcp"

@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    strategy: DistributionStrategy = DistributionStrategy.SINGLE_GPU
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500
    node_rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: CommunicationBackend = CommunicationBackend.NCCL
    init_method: str = "env://"
    timeout_minutes: int = 30
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    broadcast_buffers: bool = True
    static_graph: bool = False

@dataclass
class NodeInfo:
    """Node information."""
    node_id: str
    role: NodeRole
    address: str
    port: int
    gpu_count: int
    memory_gb: float
    cpu_count: int
    status: str = "active"
    last_heartbeat: datetime = None

@dataclass
class TrainingMetrics:
    """Training metrics."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    throughput_samples_per_sec: float
    gpu_utilization: float
    memory_usage_gb: float
    communication_time_ms: float
    computation_time_ms: float
    timestamp: datetime

class DistributedTrainer:
    """Main distributed training coordinator."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.nodes: Dict[str, NodeInfo] = {}
        self.training_metrics: List[TrainingMetrics] = []
        self.is_initialized = False
        
        # Initialize strategy-specific components
        self.strategy_handler = self._init_strategy_handler()
        
        # Initialize monitoring
        self._init_monitoring()
    
    def _init_strategy_handler(self):
        """Initialize strategy-specific handler."""
        if self.config.strategy == DistributionStrategy.SINGLE_GPU:
            return SingleGPUTrainer(self.config)
        elif self.config.strategy == DistributionStrategy.MULTI_GPU:
            return MultiGPUTrainer(self.config)
        elif self.config.strategy == DistributionStrategy.DISTRIBUTED:
            return DistributedTrainerHandler(self.config)
        elif self.config.strategy == DistributionStrategy.HOROVOD:
            return HorovodTrainer(self.config)
        elif self.config.strategy == DistributionStrategy.DEEPSPEED:
            return DeepSpeedTrainer(self.config)
        elif self.config.strategy == DistributionStrategy.FAIRSCALE:
            return FairScaleTrainer(self.config)
        elif self.config.strategy == DistributionStrategy.ACCELERATE:
            return AccelerateTrainer(self.config)
        elif self.config.strategy == DistributionStrategy.RAY:
            return RayTrainer(self.config)
        elif self.config.strategy == DistributionStrategy.FEDERATED:
            return FederatedTrainer(self.config)
        else:
            raise ValueError(f"Unsupported strategy: {self.config.strategy}")
    
    def _init_monitoring(self):
        """Initialize monitoring."""
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///distributed_training.db")
        mlflow.set_experiment("distributed_training")
        
        # Initialize Weights & Biases
        wandb.init(project="distributed_training", mode="disabled")
    
    def add_node(self, node_info: NodeInfo):
        """Add node to cluster."""
        self.nodes[node_info.node_id] = node_info
        console.print(f"[blue]Added node: {node_info.node_id} ({node_info.role.value})[/blue]")
    
    def remove_node(self, node_id: str):
        """Remove node from cluster."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            console.print(f"[yellow]Removed node: {node_id}[/yellow]")
    
    def discover_nodes(self) -> List[NodeInfo]:
        """Discover available nodes."""
        discovered_nodes = []
        
        # Kubernetes discovery
        try:
            k8s_nodes = self._discover_kubernetes_nodes()
            discovered_nodes.extend(k8s_nodes)
        except Exception as e:
            self.logger.warning(f"Kubernetes discovery failed: {e}")
        
        # Docker discovery
        try:
            docker_nodes = self._discover_docker_nodes()
            discovered_nodes.extend(docker_nodes)
        except Exception as e:
            self.logger.warning(f"Docker discovery failed: {e}")
        
        # Manual discovery
        manual_nodes = self._discover_manual_nodes()
        discovered_nodes.extend(manual_nodes)
        
        return discovered_nodes
    
    def _discover_kubernetes_nodes(self) -> List[NodeInfo]:
        """Discover nodes in Kubernetes cluster."""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        v1 = client.CoreV1Api()
        nodes = v1.list_node()
        
        discovered_nodes = []
        for node in nodes.items:
            node_info = NodeInfo(
                node_id=node.metadata.name,
                role=NodeRole.WORKER,
                address=node.status.addresses[0].address,
                port=29500,
                gpu_count=self._get_gpu_count_from_node(node),
                memory_gb=self._get_memory_from_node(node),
                cpu_count=self._get_cpu_count_from_node(node)
            )
            discovered_nodes.append(node_info)
        
        return discovered_nodes
    
    def _discover_docker_nodes(self) -> List[NodeInfo]:
        """Discover Docker containers."""
        docker_client = docker.from_env()
        containers = docker_client.containers.list()
        
        discovered_nodes = []
        for container in containers:
            if "training" in container.name.lower():
                node_info = NodeInfo(
                    node_id=container.name,
                    role=NodeRole.WORKER,
                    address=container.attrs["NetworkSettings"]["IPAddress"],
                    port=29500,
                    gpu_count=1,  # Simplified
                    memory_gb=4.0,  # Simplified
                    cpu_count=2  # Simplified
                )
                discovered_nodes.append(node_info)
        
        return discovered_nodes
    
    def _discover_manual_nodes(self) -> List[NodeInfo]:
        """Discover manually configured nodes."""
        # This would read from a configuration file
        return []
    
    def _get_gpu_count_from_node(self, node) -> int:
        """Get GPU count from Kubernetes node."""
        # Simplified - in practice, you'd parse node labels/annotations
        return 1
    
    def _get_memory_from_node(self, node) -> float:
        """Get memory from Kubernetes node."""
        # Simplified - in practice, you'd parse node capacity
        return 8.0
    
    def _get_cpu_count_from_node(self, node) -> int:
        """Get CPU count from Kubernetes node."""
        # Simplified - in practice, you'd parse node capacity
        return 4
    
    def initialize_distributed(self):
        """Initialize distributed training."""
        if self.is_initialized:
            return
        
        try:
            # Initialize strategy-specific distributed setup
            self.strategy_handler.initialize()
            
            self.is_initialized = True
            console.print("[green]Distributed training initialized[/green]")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute distributed training."""
        if not self.is_initialized:
            self.initialize_distributed()
        
        # Start training
        console.print(f"[blue]Starting distributed training for {num_epochs} epochs[/blue]")
        
        # Execute strategy-specific training
        results = self.strategy_handler.train(
            model, train_loader, val_loader, optimizer, 
            scheduler, num_epochs, loss_fn
        )
        
        # Log results
        self._log_training_results(results)
        
        return results
    
    def _log_training_results(self, results: Dict[str, Any]):
        """Log training results."""
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_params(results.get("params", {}))
            mlflow.log_metrics(results.get("metrics", {}))
        
        # Log to Weights & Biases
        wandb.log(results.get("metrics", {}))
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.is_initialized:
            self.strategy_handler.cleanup()
            self.is_initialized = False
            console.print("[yellow]Distributed training cleaned up[/yellow]")

class SingleGPUTrainer:
    """Single GPU trainer."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize single GPU training."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute single GPU training."""
        model = model.to(self.device)
        
        results = {
            "metrics": {},
            "params": {},
            "artifacts": {}
        }
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    console.print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += loss_fn(output, target).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            results["metrics"][f"epoch_{epoch}_train_loss"] = avg_train_loss
            results["metrics"][f"epoch_{epoch}_val_loss"] = avg_val_loss
            
            console.print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return results
    
    def cleanup(self):
        """Cleanup single GPU training."""
        torch.cuda.empty_cache()

class MultiGPUTrainer:
    """Multi-GPU trainer using DataParallel."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize multi-GPU training."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device("cuda")
        self.num_gpus = torch.cuda.device_count()
        console.print(f"[blue]Using {self.num_gpus} GPUs[/blue]")
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute multi-GPU training."""
        model = model.to(self.device)
        model = nn.DataParallel(model)
        
        results = {
            "metrics": {},
            "params": {"num_gpus": self.num_gpus},
            "artifacts": {}
        }
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    console.print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += loss_fn(output, target).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            results["metrics"][f"epoch_{epoch}_train_loss"] = avg_train_loss
            results["metrics"][f"epoch_{epoch}_val_loss"] = avg_val_loss
            
            console.print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return results
    
    def cleanup(self):
        """Cleanup multi-GPU training."""
        torch.cuda.empty_cache()

class DistributedTrainerHandler:
    """Distributed trainer using PyTorch DDP."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize distributed training."""
        # Set environment variables
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = str(self.config.master_port)
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.node_rank)
        os.environ['LOCAL_RANK'] = str(self.config.local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend.value,
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.node_rank,
            timeout=timedelta(minutes=self.config.timeout_minutes)
        )
        
        # Set device
        self.device = torch.device(f"cuda:{self.config.local_rank}")
        torch.cuda.set_device(self.device)
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute distributed training."""
        model = model.to(self.device)
        model = DDP(model, device_ids=[self.config.local_rank])
        
        # Use distributed sampler
        train_sampler = DistributedSampler(train_loader.dataset)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=train_sampler,
            num_workers=train_loader.num_workers
        )
        
        results = {
            "metrics": {},
            "params": {"world_size": self.config.world_size},
            "artifacts": {}
        }
        
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0 and self.config.local_rank == 0:
                    console.print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Synchronize
            dist.barrier()
            
            # Validation (only on rank 0)
            if self.config.local_rank == 0:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        val_loss += loss_fn(output, target).item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                results["metrics"][f"epoch_{epoch}_train_loss"] = avg_train_loss
                results["metrics"][f"epoch_{epoch}_val_loss"] = avg_val_loss
                
                console.print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return results
    
    def cleanup(self):
        """Cleanup distributed training."""
        dist.destroy_process_group()
        torch.cuda.empty_cache()

class HorovodTrainer:
    """Horovod trainer."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize Horovod."""
        hvd.init()
        
        # Set device
        self.device = torch.device(f"cuda:{hvd.local_rank()}")
        torch.cuda.set_device(self.device)
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute Horovod training."""
        model = model.to(self.device)
        
        # Wrap optimizer with Horovod
        optimizer = hvd.DistributedOptimizer(optimizer)
        
        # Broadcast parameters
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        
        results = {
            "metrics": {},
            "params": {"world_size": hvd.size()},
            "artifacts": {}
        }
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0 and hvd.rank() == 0:
                    console.print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Validation (only on rank 0)
            if hvd.rank() == 0:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        val_loss += loss_fn(output, target).item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                results["metrics"][f"epoch_{epoch}_train_loss"] = avg_train_loss
                results["metrics"][f"epoch_{epoch}_val_loss"] = avg_val_loss
                
                console.print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return results
    
    def cleanup(self):
        """Cleanup Horovod."""
        torch.cuda.empty_cache()

class DeepSpeedTrainer:
    """DeepSpeed trainer."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize DeepSpeed."""
        # DeepSpeed initialization is typically done in the training script
        pass
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute DeepSpeed training."""
        # DeepSpeed training implementation
        # This would integrate with DeepSpeed's training loop
        pass
    
    def cleanup(self):
        """Cleanup DeepSpeed."""
        pass

class FairScaleTrainer:
    """FairScale trainer."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize FairScale."""
        pass
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute FairScale training."""
        # FairScale training implementation
        pass
    
    def cleanup(self):
        """Cleanup FairScale."""
        pass

class AccelerateTrainer:
    """Accelerate trainer."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.accelerator = Accelerator()
    
    def initialize(self):
        """Initialize Accelerate."""
        pass
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute Accelerate training."""
        # Prepare model, optimizer, and data loaders
        model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        
        results = {
            "metrics": {},
            "params": {},
            "artifacts": {}
        }
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                self.accelerator.backward(loss)
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    console.print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss += loss_fn(output, target).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            results["metrics"][f"epoch_{epoch}_train_loss"] = avg_train_loss
            results["metrics"][f"epoch_{epoch}_val_loss"] = avg_val_loss
            
            console.print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return results
    
    def cleanup(self):
        """Cleanup Accelerate."""
        pass

class RayTrainer:
    """Ray trainer."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize Ray."""
        if not ray.is_initialized():
            ray.init()
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute Ray training."""
        # Ray training implementation using Ray Tune
        pass
    
    def cleanup(self):
        """Cleanup Ray."""
        if ray.is_initialized():
            ray.shutdown()

class FederatedTrainer:
    """Federated learning trainer."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def initialize(self):
        """Initialize federated learning."""
        pass
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader, optimizer, scheduler, 
              num_epochs: int, loss_fn) -> Dict[str, Any]:
        """Execute federated training."""
        # Federated learning implementation
        pass
    
    def cleanup(self):
        """Cleanup federated learning."""
        pass

def main():
    """Main function for distributed training CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Training System")
    parser.add_argument("--strategy", type=str,
                       choices=["single_gpu", "multi_gpu", "distributed", "horovod", 
                               "deepspeed", "fairscale", "accelerate", "ray", "federated"],
                       default="single_gpu", help="Distribution strategy")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--num-gpus-per-node", type=int, default=1, help="GPUs per node")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master-port", type=int, default=29500, help="Master port")
    parser.add_argument("--node-rank", type=int, default=0, help="Node rank")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank")
    parser.add_argument("--world-size", type=int, default=1, help="World size")
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo", "mpi"], 
                       default="nccl", help="Communication backend")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision")
    parser.add_argument("--discover-nodes", action="store_true", help="Discover available nodes")
    
    args = parser.parse_args()
    
    # Create distributed configuration
    config = DistributedConfig(
        strategy=DistributionStrategy(args.strategy),
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
        master_addr=args.master_addr,
        master_port=args.master_port,
        node_rank=args.node_rank,
        local_rank=args.local_rank,
        world_size=args.world_size,
        backend=CommunicationBackend(args.backend),
        mixed_precision=args.mixed_precision
    )
    
    # Create distributed trainer
    trainer = DistributedTrainer(config)
    
    if args.discover_nodes:
        # Discover nodes
        nodes = trainer.discover_nodes()
        
        table = Table(title="Discovered Nodes")
        table.add_column("Node ID", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Address", style="green")
        table.add_column("GPUs", style="yellow")
        table.add_column("Memory (GB)", style="blue")
        table.add_column("CPUs", style="red")
        
        for node in nodes:
            table.add_row(
                node.node_id,
                node.role.value,
                node.address,
                str(node.gpu_count),
                f"{node.memory_gb:.1f}",
                str(node.cpu_count)
            )
        
        console.print(table)
        
        # Add discovered nodes
        for node in nodes:
            trainer.add_node(node)
    
    # Initialize distributed training
    trainer.initialize_distributed()
    
    console.print(f"[green]Distributed training initialized with strategy: {args.strategy}[/green]")
    
    # Cleanup
    trainer.cleanup()

if __name__ == "__main__":
    main()
