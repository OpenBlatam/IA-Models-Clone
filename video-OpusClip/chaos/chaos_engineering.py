#!/usr/bin/env python3
"""
Chaos Engineering Framework

Advanced chaos engineering with:
- Fault injection and testing
- Resilience validation
- Failure simulation
- Recovery testing
- Performance degradation testing
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import random
import signal
import psutil
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque

logger = structlog.get_logger("chaos_engineering")

# =============================================================================
# CHAOS ENGINEERING MODELS
# =============================================================================

class ChaosType(Enum):
    """Types of chaos experiments."""
    LATENCY = "latency"
    ERROR = "error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    SERVICE_SHUTDOWN = "service_shutdown"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    RANDOM = "random"

class ChaosState(Enum):
    """Chaos experiment states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ChaosExperiment:
    """Chaos experiment configuration."""
    experiment_id: str
    name: str
    description: str
    chaos_type: ChaosType
    target_service: str
    duration: int  # seconds
    intensity: float  # 0.0 to 1.0
    parameters: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "chaos_type": self.chaos_type.value,
            "target_service": self.target_service,
            "duration": self.duration,
            "intensity": self.intensity,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class ChaosResult:
    """Chaos experiment result."""
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime]
    state: ChaosState
    metrics_before: Dict[str, Any]
    metrics_during: Dict[str, Any]
    metrics_after: Dict[str, Any]
    errors: List[str]
    recovery_time: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "state": self.state.value,
            "metrics_before": self.metrics_before,
            "metrics_during": self.metrics_during,
            "metrics_after": self.metrics_after,
            "errors": self.errors,
            "recovery_time": self.recovery_time,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }

# =============================================================================
# CHAOS ENGINEERING FRAMEWORK
# =============================================================================

class ChaosEngineeringFramework:
    """Advanced chaos engineering framework."""
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.active_experiments: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, ChaosResult] = {}
        self.monitoring_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # System metrics tracking
        self.baseline_metrics: Dict[str, Any] = {}
        self.current_metrics: Dict[str, Any] = {}
        
        # Recovery mechanisms
        self.recovery_handlers: Dict[str, Callable[[], Awaitable[bool]]] = {}
        
        # Statistics
        self.stats = {
            'total_experiments': 0,
            'successful_experiments': 0,
            'failed_experiments': 0,
            'total_downtime': 0.0,
            'average_recovery_time': 0.0
        }
    
    async def start(self) -> None:
        """Start the chaos engineering framework."""
        await self._collect_baseline_metrics()
        logger.info("Chaos engineering framework started")
    
    async def stop(self) -> None:
        """Stop the chaos engineering framework."""
        # Stop all active experiments
        for experiment_id, task in self.active_experiments.items():
            task.cancel()
            await self._cleanup_experiment(experiment_id)
        
        logger.info("Chaos engineering framework stopped")
    
    async def create_experiment(self, experiment: ChaosExperiment) -> str:
        """Create a new chaos experiment."""
        self.experiments[experiment.experiment_id] = experiment
        self.stats['total_experiments'] += 1
        
        logger.info(
            "Chaos experiment created",
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            chaos_type=experiment.chaos_type.value,
            target_service=experiment.target_service
        )
        
        return experiment.experiment_id
    
    async def run_experiment(self, experiment_id: str) -> ChaosResult:
        """Run a chaos experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if not experiment.enabled:
            raise ValueError(f"Experiment {experiment_id} is disabled")
        
        if experiment_id in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} is already running")
        
        # Create result object
        result = ChaosResult(
            experiment_id=experiment_id,
            start_time=datetime.utcnow(),
            end_time=None,
            state=ChaosState.RUNNING,
            metrics_before={},
            metrics_during={},
            metrics_after={},
            errors=[],
            recovery_time=None
        )
        
        self.results[experiment_id] = result
        
        try:
            # Collect metrics before experiment
            result.metrics_before = await self._collect_system_metrics()
            
            # Start experiment
            task = asyncio.create_task(self._execute_experiment(experiment, result))
            self.active_experiments[experiment_id] = task
            
            # Wait for experiment to complete
            await task
            
            # Collect metrics after experiment
            result.metrics_after = await self._collect_system_metrics()
            result.end_time = datetime.utcnow()
            result.state = ChaosState.COMPLETED
            
            # Calculate recovery time
            if experiment_id in self.recovery_handlers:
                recovery_start = time.time()
                recovery_success = await self.recovery_handlers[experiment_id]()
                result.recovery_time = time.time() - recovery_start
                
                if not recovery_success:
                    result.errors.append("Recovery failed")
            
            self.stats['successful_experiments'] += 1
            
            logger.info(
                "Chaos experiment completed",
                experiment_id=experiment_id,
                duration=result.duration,
                recovery_time=result.recovery_time
            )
            
        except Exception as e:
            result.state = ChaosState.FAILED
            result.errors.append(str(e))
            result.end_time = datetime.utcnow()
            
            self.stats['failed_experiments'] += 1
            
            logger.error(
                "Chaos experiment failed",
                experiment_id=experiment_id,
                error=str(e)
            )
            
            # Attempt recovery
            if experiment_id in self.recovery_handlers:
                try:
                    await self.recovery_handlers[experiment_id]()
                except Exception as recovery_error:
                    result.errors.append(f"Recovery error: {str(recovery_error)}")
        
        finally:
            # Cleanup
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            await self._cleanup_experiment(experiment_id)
        
        return result
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment."""
        if experiment_id not in self.active_experiments:
            return False
        
        task = self.active_experiments[experiment_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self.active_experiments[experiment_id]
        
        # Update result
        if experiment_id in self.results:
            self.results[experiment_id].state = ChaosState.COMPLETED
            self.results[experiment_id].end_time = datetime.utcnow()
        
        await self._cleanup_experiment(experiment_id)
        
        logger.info("Chaos experiment stopped", experiment_id=experiment_id)
        return True
    
    async def _execute_experiment(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Execute a chaos experiment."""
        start_time = time.time()
        
        try:
            # Apply chaos based on type
            if experiment.chaos_type == ChaosType.LATENCY:
                await self._inject_latency(experiment, result)
            elif experiment.chaos_type == ChaosType.ERROR:
                await self._inject_errors(experiment, result)
            elif experiment.chaos_type == ChaosType.RESOURCE_EXHAUSTION:
                await self._exhaust_resources(experiment, result)
            elif experiment.chaos_type == ChaosType.NETWORK_PARTITION:
                await self._partition_network(experiment, result)
            elif experiment.chaos_type == ChaosType.SERVICE_SHUTDOWN:
                await self._shutdown_service(experiment, result)
            elif experiment.chaos_type == ChaosType.MEMORY_LEAK:
                await self._simulate_memory_leak(experiment, result)
            elif experiment.chaos_type == ChaosType.CPU_SPIKE:
                await self._spike_cpu(experiment, result)
            elif experiment.chaos_type == ChaosType.DISK_FULL:
                await self._fill_disk(experiment, result)
            elif experiment.chaos_type == ChaosType.RANDOM:
                await self._random_chaos(experiment, result)
            else:
                raise ValueError(f"Unknown chaos type: {experiment.chaos_type}")
            
            # Monitor during experiment
            await self._monitor_during_experiment(experiment, result)
            
        except Exception as e:
            result.errors.append(f"Experiment execution error: {str(e)}")
            raise
    
    async def _inject_latency(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Inject latency into the system."""
        latency_ms = experiment.parameters.get('latency_ms', 1000)
        jitter = experiment.parameters.get('jitter', 0.1)
        
        # Apply intensity
        actual_latency = latency_ms * experiment.intensity
        
        # Add jitter
        if jitter > 0:
            jitter_amount = actual_latency * jitter
            actual_latency += random.uniform(-jitter_amount, jitter_amount)
        
        # Inject latency by sleeping
        await asyncio.sleep(actual_latency / 1000.0)
        
        logger.info(
            "Latency injected",
            experiment_id=experiment.experiment_id,
            latency_ms=actual_latency
        )
    
    async def _inject_errors(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Inject errors into the system."""
        error_rate = experiment.parameters.get('error_rate', 0.1)
        error_types = experiment.parameters.get('error_types', ['timeout', 'connection_error'])
        
        # Apply intensity
        actual_error_rate = error_rate * experiment.intensity
        
        if random.random() < actual_error_rate:
            error_type = random.choice(error_types)
            
            if error_type == 'timeout':
                raise TimeoutError("Chaos engineering timeout injection")
            elif error_type == 'connection_error':
                raise ConnectionError("Chaos engineering connection error injection")
            elif error_type == 'value_error':
                raise ValueError("Chaos engineering value error injection")
            else:
                raise Exception(f"Chaos engineering {error_type} injection")
        
        logger.info(
            "Error injection attempted",
            experiment_id=experiment.experiment_id,
            error_rate=actual_error_rate
        )
    
    async def _exhaust_resources(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Exhaust system resources."""
        resource_type = experiment.parameters.get('resource_type', 'memory')
        duration = experiment.parameters.get('duration', 30)
        
        if resource_type == 'memory':
            # Allocate memory
            memory_mb = int(experiment.parameters.get('memory_mb', 100) * experiment.intensity)
            memory_data = []
            
            try:
                for _ in range(memory_mb):
                    memory_data.append(bytearray(1024 * 1024))  # 1MB chunks
                    await asyncio.sleep(0.001)  # Small delay to prevent blocking
                
                # Hold memory for duration
                await asyncio.sleep(duration)
                
            finally:
                # Clean up memory
                memory_data.clear()
                del memory_data
        
        elif resource_type == 'cpu':
            # Spawn CPU-intensive tasks
            cpu_cores = int(experiment.parameters.get('cpu_cores', 1) * experiment.intensity)
            tasks = []
            
            try:
                for _ in range(cpu_cores):
                    task = asyncio.create_task(self._cpu_intensive_task(duration))
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                
            finally:
                for task in tasks:
                    task.cancel()
        
        logger.info(
            "Resource exhaustion completed",
            experiment_id=experiment.experiment_id,
            resource_type=resource_type
        )
    
    async def _partition_network(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Simulate network partition."""
        # This would typically involve network configuration changes
        # For simulation, we'll just log the action
        logger.info(
            "Network partition simulated",
            experiment_id=experiment.experiment_id,
            target_service=experiment.target_service
        )
        
        # Simulate network issues
        await asyncio.sleep(experiment.duration)
    
    async def _shutdown_service(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Simulate service shutdown."""
        # This would typically involve stopping actual services
        # For simulation, we'll just log the action
        logger.info(
            "Service shutdown simulated",
            experiment_id=experiment.experiment_id,
            target_service=experiment.target_service
        )
        
        # Simulate service downtime
        await asyncio.sleep(experiment.duration)
    
    async def _simulate_memory_leak(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Simulate memory leak."""
        leak_rate = experiment.parameters.get('leak_rate', 1)  # MB per second
        duration = experiment.duration
        
        # Apply intensity
        actual_leak_rate = leak_rate * experiment.intensity
        
        leaked_memory = []
        
        try:
            for _ in range(int(duration * actual_leak_rate)):
                # Allocate memory that won't be freed
                leaked_memory.append(bytearray(1024 * 1024))  # 1MB
                await asyncio.sleep(1.0)
        
        finally:
            # Clean up (in real scenario, this wouldn't happen)
            leaked_memory.clear()
            del leaked_memory
        
        logger.info(
            "Memory leak simulated",
            experiment_id=experiment.experiment_id,
            leak_rate=actual_leak_rate
        )
    
    async def _spike_cpu(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Simulate CPU spike."""
        spike_duration = experiment.parameters.get('spike_duration', 10)
        cpu_cores = int(experiment.parameters.get('cpu_cores', 1) * experiment.intensity)
        
        tasks = []
        
        try:
            for _ in range(cpu_cores):
                task = asyncio.create_task(self._cpu_intensive_task(spike_duration))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        finally:
            for task in tasks:
                task.cancel()
        
        logger.info(
            "CPU spike simulated",
            experiment_id=experiment.experiment_id,
            cpu_cores=cpu_cores
        )
    
    async def _fill_disk(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Simulate disk full condition."""
        # This would typically involve creating large files
        # For simulation, we'll just log the action
        logger.info(
            "Disk full condition simulated",
            experiment_id=experiment.experiment_id
        )
        
        await asyncio.sleep(experiment.duration)
    
    async def _random_chaos(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Apply random chaos."""
        chaos_types = [
            ChaosType.LATENCY,
            ChaosType.ERROR,
            ChaosType.RESOURCE_EXHAUSTION
        ]
        
        # Randomly select and apply chaos
        selected_chaos = random.choice(chaos_types)
        
        # Create temporary experiment with selected chaos type
        temp_experiment = ChaosExperiment(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            description=experiment.description,
            chaos_type=selected_chaos,
            target_service=experiment.target_service,
            duration=experiment.duration,
            intensity=experiment.intensity,
            parameters=experiment.parameters
        )
        
        # Execute the selected chaos
        await self._execute_experiment(temp_experiment, result)
    
    async def _cpu_intensive_task(self, duration: int) -> None:
        """CPU-intensive task for resource exhaustion."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # CPU-intensive calculation
            sum(range(1000000))
            await asyncio.sleep(0.001)  # Small yield to prevent blocking
    
    async def _monitor_during_experiment(self, experiment: ChaosExperiment, result: ChaosResult) -> None:
        """Monitor system during experiment."""
        start_time = time.time()
        
        while time.time() - start_time < experiment.duration:
            # Collect metrics
            metrics = await self._collect_system_metrics()
            result.metrics_during.update(metrics)
            
            # Call monitoring callbacks
            for callback in self.monitoring_callbacks:
                try:
                    callback(experiment.experiment_id, metrics)
                except Exception as e:
                    logger.error("Monitoring callback error", error=str(e))
            
            await asyncio.sleep(1)  # Monitor every second
    
    async def _collect_baseline_metrics(self) -> None:
        """Collect baseline system metrics."""
        self.baseline_metrics = await self._collect_system_metrics()
        logger.info("Baseline metrics collected", metrics=self.baseline_metrics)
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            return {}
    
    async def _cleanup_experiment(self, experiment_id: str) -> None:
        """Clean up after experiment."""
        # This would typically involve restoring system state
        logger.info("Experiment cleanup completed", experiment_id=experiment_id)
    
    def add_monitoring_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add monitoring callback."""
        self.monitoring_callbacks.append(callback)
    
    def add_recovery_handler(self, experiment_id: str, handler: Callable[[], Awaitable[bool]]) -> None:
        """Add recovery handler for experiment."""
        self.recovery_handlers[experiment_id] = handler
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """Get experiment statistics."""
        return {
            **self.stats,
            'active_experiments': len(self.active_experiments),
            'total_experiments_created': len(self.experiments),
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': self.current_metrics
        }
    
    def get_experiment_results(self, experiment_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get experiment results."""
        if experiment_id:
            return self.results.get(experiment_id, {}).to_dict() if experiment_id in self.results else {}
        else:
            return [result.to_dict() for result in self.results.values()]

# =============================================================================
# GLOBAL CHAOS ENGINEERING INSTANCE
# =============================================================================

# Global chaos engineering framework
chaos_framework = ChaosEngineeringFramework()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ChaosType',
    'ChaosState',
    'ChaosExperiment',
    'ChaosResult',
    'ChaosEngineeringFramework',
    'chaos_framework'
]





























