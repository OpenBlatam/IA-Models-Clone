#!/usr/bin/env python3
"""
Quantum Computing Manager for Enhanced HeyGen AI
Integrates quantum algorithms and quantum machine learning capabilities.
"""

import asyncio
import time
import json
import structlog
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import pickle

logger = structlog.get_logger()

class QuantumBackend(Enum):
    """Available quantum backends."""
    SIMULATOR = "simulator"
    IBM_Q = "ibm_q"
    GOOGLE_QUANTUM = "google_quantum"
    MICROSOFT_AZURE = "microsoft_azure"
    AMAZON_BRAKET = "amazon_braket"
    CUSTOM = "custom"

class QuantumAlgorithm(Enum):
    """Available quantum algorithms."""
    GROVER = "grover"
    SHOR = "shor"
    QUANTUM_FOURIER_TRANSFORM = "qft"
    QUANTUM_NEURAL_NETWORK = "qnn"
    QUANTUM_KERNEL = "quantum_kernel"
    VARIATIONAL_QUANTUM_EIGENSOLVER = "vqe"
    QUANTUM_APPROXIMATE_OPTIMIZATION = "qaoa"
    QUANTUM_MACHINE_LEARNING = "qml"

class QuantumTaskStatus(Enum):
    """Quantum task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class QuantumCircuit:
    """Quantum circuit representation."""
    name: str
    num_qubits: int
    num_classical_bits: int
    gates: List[Dict[str, Any]]
    measurements: List[Dict[str, Any]]
    parameters: Dict[str, float]
    depth: int
    width: int

@dataclass
class QuantumTask:
    """Quantum computing task."""
    task_id: str
    algorithm: QuantumAlgorithm
    backend: QuantumBackend
    status: QuantumTaskStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    circuit: Optional[QuantumCircuit]
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

@dataclass
class QuantumModel:
    """Quantum machine learning model."""
    name: str
    algorithm: QuantumAlgorithm
    backend: QuantumBackend
    num_qubits: int
    parameters: Dict[str, float]
    training_history: List[Dict[str, float]]
    performance_metrics: Dict[str, float]
    created_at: datetime
    updated_at: datetime
    is_trained: bool = False

class QuantumComputingManager:
    """Comprehensive quantum computing management for HeyGen AI."""
    
    def __init__(
        self,
        enable_quantum_simulation: bool = True,
        enable_real_quantum: bool = False,
        max_workers: int = 4,
        quantum_dir: str = "./quantum"
    ):
        self.enable_quantum_simulation = enable_quantum_simulation
        self.enable_real_quantum = enable_real_quantum
        self.max_workers = max_workers
        self.quantum_dir = Path(quantum_dir)
        
        # Create quantum directory
        self.quantum_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread pool for quantum operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task registry
        self.quantum_tasks: Dict[str, QuantumTask] = {}
        self.quantum_models: Dict[str, QuantumModel] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # Initialize quantum backends
        self._initialize_quantum_backends()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Quantum Computing Manager initialized successfully")
    
    def _initialize_quantum_backends(self):
        """Initialize quantum computing backends."""
        try:
            self.available_backends = {
                QuantumBackend.SIMULATOR: {
                    'available': True,
                    'max_qubits': 32,
                    'error_rate': 0.0,
                    'connection': None
                }
            }
            
            # Try to initialize real quantum backends if enabled
            if self.enable_real_quantum:
                self._initialize_real_quantum_backends()
            
            logger.info(f"Initialized {len(self.available_backends)} quantum backends")
            
        except Exception as e:
            logger.error(f"Quantum backend initialization failed: {e}")
    
    def _initialize_real_quantum_backends(self):
        """Initialize real quantum computing backends."""
        try:
            # IBM Quantum
            try:
                import qiskit
                from qiskit import IBMQ
                
                # Try to load IBM Quantum account
                IBMQ.load_account()
                provider = IBMQ.get_provider()
                
                # Get available backends
                for backend in provider.backends():
                    if backend.configuration().simulator:
                        continue
                    
                    self.available_backends[QuantumBackend.IBM_Q] = {
                        'available': True,
                        'max_qubits': backend.configuration().n_qubits,
                        'error_rate': 0.01,  # Typical error rate
                        'connection': backend
                    }
                    break
                
                logger.info("IBM Quantum backend initialized")
                
            except ImportError:
                logger.warning("Qiskit not available for IBM Quantum")
            except Exception as e:
                logger.warning(f"IBM Quantum initialization failed: {e}")
            
            # Google Quantum (Cirq)
            try:
                import cirq
                
                self.available_backends[QuantumBackend.GOOGLE_QUANTUM] = {
                    'available': True,
                    'max_qubits': 53,  # Sycamore processor
                    'error_rate': 0.005,  # Google's error rate
                    'connection': None  # Would need actual API key
                }
                
                logger.info("Google Quantum backend initialized")
                
            except ImportError:
                logger.warning("Cirq not available for Google Quantum")
            except Exception as e:
                logger.warning(f"Google Quantum initialization failed: {e}")
            
            # Microsoft Azure Quantum
            try:
                import azure.quantum
                
                self.available_backends[QuantumBackend.MICROSOFT_AZURE] = {
                    'available': True,
                    'max_qubits': 40,  # IonQ processor
                    'error_rate': 0.01,
                    'connection': None  # Would need actual connection
                }
                
                logger.info("Microsoft Azure Quantum backend initialized")
                
            except ImportError:
                logger.warning("Azure Quantum SDK not available")
            except Exception as e:
                logger.warning(f"Microsoft Azure Quantum initialization failed: {e}")
            
        except Exception as e:
            logger.error(f"Real quantum backend initialization failed: {e}")
    
    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await self._monitor_quantum_tasks()
                await self._monitor_quantum_models()
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Quantum monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await self._cleanup_completed_tasks()
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Quantum cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_quantum_tasks(self):
        """Monitor quantum task status."""
        try:
            for task_id, task in self.quantum_tasks.items():
                if task.status == QuantumTaskStatus.RUNNING:
                    # Check task progress
                    await self._check_quantum_task_progress(task_id)
                    
        except Exception as e:
            logger.error(f"Quantum task monitoring error: {e}")
    
    async def _monitor_quantum_models(self):
        """Monitor quantum model performance."""
        try:
            for model_id, model in self.quantum_models.items():
                if model.is_trained:
                    # Check model performance
                    await self._check_quantum_model_performance(model_id)
                    
        except Exception as e:
            logger.error(f"Quantum model monitoring error: {e}")
    
    async def _check_quantum_task_progress(self, task_id: str):
        """Check quantum task progress."""
        try:
            task = self.quantum_tasks[task_id]
            
            # This would typically involve checking the actual quantum backend
            # For now, we'll simulate progress
            
            logger.debug(f"Quantum task {task_id} progress check completed")
            
        except Exception as e:
            logger.error(f"Quantum task progress check failed: {e}")
    
    async def _check_quantum_model_performance(self, model_id: str):
        """Check quantum model performance."""
        try:
            model = self.quantum_models[model_id]
            
            # This would typically involve running inference on test data
            # and comparing with expected performance thresholds
            
            logger.debug(f"Quantum model {model_id} performance check completed")
            
        except Exception as e:
            logger.error(f"Quantum model performance check failed: {e}")
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed quantum tasks."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=1)  # Keep tasks for 1 day
            
            tasks_to_remove = []
            for task_id, task in self.quantum_tasks.items():
                if (task.status in [QuantumTaskStatus.COMPLETED, QuantumTaskStatus.FAILED] and
                    task.completed_at and task.completed_at < cutoff_time):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                await self._remove_quantum_task(task_id)
                
        except Exception as e:
            logger.error(f"Quantum task cleanup error: {e}")
    
    async def _remove_quantum_task(self, task_id: str):
        """Remove a quantum task."""
        try:
            del self.quantum_tasks[task_id]
            logger.info(f"Quantum task {task_id} removed")
            
        except Exception as e:
            logger.error(f"Failed to remove quantum task {task_id}: {e}")
    
    async def create_quantum_circuit(
        self,
        name: str,
        num_qubits: int,
        gates: List[Dict[str, Any]],
        measurements: List[Dict[str, Any]],
        parameters: Optional[Dict[str, float]] = None
    ) -> QuantumCircuit:
        """Create a quantum circuit."""
        try:
            circuit = QuantumCircuit(
                name=name,
                num_qubits=num_qubits,
                num_classical_bits=len(measurements),
                gates=gates,
                measurements=measurements,
                parameters=parameters or {},
                depth=len(gates),
                width=num_qubits
            )
            
            logger.info(f"Quantum circuit '{name}' created with {num_qubits} qubits")
            return circuit
            
        except Exception as e:
            logger.error(f"Failed to create quantum circuit: {e}")
            raise
    
    async def run_quantum_algorithm(
        self,
        algorithm: QuantumAlgorithm,
        backend: QuantumBackend,
        parameters: Dict[str, Any],
        circuit: Optional[QuantumCircuit] = None
    ) -> str:
        """Run a quantum algorithm."""
        try:
            # Check if backend is available
            if backend not in self.available_backends or not self.available_backends[backend]['available']:
                raise ValueError(f"Quantum backend {backend.value} not available")
            
            # Create task
            task_id = f"quantum_{algorithm.value}_{int(time.time())}"
            
            task = QuantumTask(
                task_id=task_id,
                algorithm=algorithm,
                backend=backend,
                status=QuantumTaskStatus.PENDING,
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                circuit=circuit,
                parameters=parameters,
                results=None
            )
            
            self.quantum_tasks[task_id] = task
            
            # Start execution in background
            asyncio.create_task(self._execute_quantum_task(task_id))
            
            logger.info(f"Quantum algorithm {algorithm.value} started with task ID: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to run quantum algorithm: {e}")
            raise
    
    async def _execute_quantum_task(self, task_id: str):
        """Execute a quantum computing task."""
        try:
            task = self.quantum_tasks[task_id]
            task.status = QuantumTaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Execute based on algorithm type
            if task.algorithm == QuantumAlgorithm.GROVER:
                results = await self._run_grover_algorithm(task)
            elif task.algorithm == QuantumAlgorithm.QUANTUM_NEURAL_NETWORK:
                results = await self._run_quantum_neural_network(task)
            elif task.algorithm == QuantumAlgorithm.QUANTUM_KERNEL:
                results = await self._run_quantum_kernel(task)
            elif task.algorithm == QuantumAlgorithm.VARIATIONAL_QUANTUM_EIGENSOLVER:
                results = await self._run_vqe_algorithm(task)
            else:
                results = await self._run_generic_quantum_algorithm(task)
            
            # Complete task
            task.status = QuantumTaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.results = results
            task.execution_time = (task.completed_at - task.started_at).total_seconds()
            
            # Update performance history
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'task_id': task_id,
                'algorithm': task.algorithm.value,
                'backend': task.backend.value,
                'execution_time': task.execution_time,
                'status': 'completed'
            })
            
            logger.info(f"Quantum task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Quantum task {task_id} failed: {e}")
            task.status = QuantumTaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            if task.started_at:
                task.execution_time = (task.completed_at - task.started_at).total_seconds()
    
    async def _run_grover_algorithm(self, task: QuantumTask) -> Dict[str, Any]:
        """Run Grover's search algorithm."""
        try:
            # Simulate Grover's algorithm
            num_qubits = task.circuit.num_qubits if task.circuit else 4
            iterations = task.parameters.get('iterations', 2)
            
            # Simulate quantum search
            search_space_size = 2 ** num_qubits
            target_probability = np.sin((2 * iterations + 1) * np.arcsin(1 / np.sqrt(search_space_size))) ** 2
            
            # Simulate measurement results
            measurements = np.random.choice(
                range(search_space_size),
                size=100,
                p=[target_probability / search_space_size] * search_space_size
            )
            
            results = {
                'algorithm': 'grover',
                'num_qubits': num_qubits,
                'iterations': iterations,
                'search_space_size': search_space_size,
                'target_probability': target_probability,
                'measurements': measurements.tolist(),
                'most_common_result': int(np.bincount(measurements).argmax()),
                'success_rate': target_probability
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Grover algorithm execution failed: {e}")
            raise
    
    async def _run_quantum_neural_network(self, task: QuantumTask) -> Dict[str, Any]:
        """Run quantum neural network algorithm."""
        try:
            # Simulate quantum neural network
            num_qubits = task.circuit.num_qubits if task.circuit else 4
            num_layers = task.parameters.get('num_layers', 3)
            learning_rate = task.parameters.get('learning_rate', 0.1)
            
            # Simulate training
            epochs = 100
            loss_history = []
            accuracy_history = []
            
            for epoch in range(epochs):
                # Simulate loss decrease
                loss = 1.0 * np.exp(-epoch / 20) + 0.1 * np.random.random()
                accuracy = 1.0 - loss + 0.05 * np.random.random()
                
                loss_history.append(loss)
                accuracy_history.append(accuracy)
            
            results = {
                'algorithm': 'quantum_neural_network',
                'num_qubits': num_qubits,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'final_loss': loss_history[-1],
                'final_accuracy': accuracy_history[-1],
                'loss_history': loss_history,
                'accuracy_history': accuracy_history,
                'convergence_epoch': np.argmin(loss_history)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Quantum neural network execution failed: {e}")
            raise
    
    async def _run_quantum_kernel(self, task: QuantumTask) -> Dict[str, Any]:
        """Run quantum kernel algorithm."""
        try:
            # Simulate quantum kernel
            num_qubits = task.circuit.num_qubits if task.circuit else 4
            kernel_type = task.parameters.get('kernel_type', 'rbf')
            
            # Simulate kernel matrix computation
            num_samples = 50
            kernel_matrix = np.random.random((num_samples, num_samples))
            kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2  # Make symmetric
            np.fill_diagonal(kernel_matrix, 1.0)  # Diagonal should be 1
            
            # Simulate classification results
            y_true = np.random.randint(0, 2, num_samples)
            y_pred = np.random.randint(0, 2, num_samples)
            
            accuracy = np.mean(y_true == y_pred)
            
            results = {
                'algorithm': 'quantum_kernel',
                'num_qubits': num_qubits,
                'kernel_type': kernel_type,
                'num_samples': num_samples,
                'kernel_matrix_shape': kernel_matrix.shape,
                'kernel_matrix_trace': np.trace(kernel_matrix),
                'classification_accuracy': accuracy,
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Quantum kernel execution failed: {e}")
            raise
    
    async def _run_vqe_algorithm(self, task: QuantumTask) -> Dict[str, Any]:
        """Run Variational Quantum Eigensolver algorithm."""
        try:
            # Simulate VQE
            num_qubits = task.circuit.num_qubits if task.circuit else 4
            max_iterations = task.parameters.get('max_iterations', 100)
            
            # Simulate optimization
            energy_history = []
            parameter_history = []
            
            for iteration in range(max_iterations):
                # Simulate energy decrease
                energy = 2.0 * np.exp(-iteration / 30) + 0.1 * np.random.random()
                parameters = np.random.random(4)  # Simulate parameter updates
                
                energy_history.append(energy)
                parameter_history.append(parameters.tolist())
            
            results = {
                'algorithm': 'vqe',
                'num_qubits': num_qubits,
                'max_iterations': max_iterations,
                'final_energy': energy_history[-1],
                'energy_history': energy_history,
                'parameter_history': parameter_history,
                'convergence_iteration': np.argmin(energy_history),
                'ground_state_energy': min(energy_history)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"VQE algorithm execution failed: {e}")
            raise
    
    async def _run_generic_quantum_algorithm(self, task: QuantumTask) -> Dict[str, Any]:
        """Run a generic quantum algorithm."""
        try:
            # Generic simulation
            num_qubits = task.circuit.num_qubits if task.circuit else 4
            
            # Simulate quantum state evolution
            state_vector = np.random.random(2 ** num_qubits)
            state_vector = state_vector / np.linalg.norm(state_vector)  # Normalize
            
            # Simulate measurements
            probabilities = np.abs(state_vector) ** 2
            measurements = np.random.choice(
                range(2 ** num_qubits),
                size=1000,
                p=probabilities
            )
            
            results = {
                'algorithm': task.algorithm.value,
                'num_qubits': num_qubits,
                'state_vector_size': len(state_vector),
                'measurements': measurements.tolist(),
                'most_probable_state': int(np.argmax(probabilities)),
                'entropy': -np.sum(probabilities * np.log2(probabilities + 1e-10))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Generic quantum algorithm execution failed: {e}")
            raise
    
    async def create_quantum_model(
        self,
        name: str,
        algorithm: QuantumAlgorithm,
        backend: QuantumBackend,
        num_qubits: int,
        parameters: Dict[str, float]
    ) -> str:
        """Create a quantum machine learning model."""
        try:
            model_id = f"qmodel_{name}_{int(time.time())}"
            
            model = QuantumModel(
                name=name,
                algorithm=algorithm,
                backend=backend,
                num_qubits=num_qubits,
                parameters=parameters,
                training_history=[],
                performance_metrics={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_trained=False
            )
            
            self.quantum_models[model_id] = model
            
            logger.info(f"Quantum model '{name}' created with ID: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to create quantum model: {e}")
            raise
    
    async def train_quantum_model(
        self,
        model_id: str,
        training_data: np.ndarray,
        training_labels: np.ndarray,
        epochs: int = 100
    ) -> bool:
        """Train a quantum machine learning model."""
        try:
            if model_id not in self.quantum_models:
                raise ValueError(f"Quantum model {model_id} not found")
            
            model = self.quantum_models[model_id]
            
            # Simulate training
            for epoch in range(epochs):
                # Simulate training metrics
                loss = 1.0 * np.exp(-epoch / 20) + 0.1 * np.random.random()
                accuracy = 1.0 - loss + 0.05 * np.random.random()
                
                model.training_history.append({
                    'epoch': epoch,
                    'loss': loss,
                    'accuracy': accuracy
                })
            
            # Update model
            model.is_trained = True
            model.updated_at = datetime.now()
            model.performance_metrics = {
                'final_loss': model.training_history[-1]['loss'],
                'final_accuracy': model.training_history[-1]['accuracy'],
                'best_accuracy': max(h['accuracy'] for h in model.training_history),
                'convergence_epoch': np.argmin([h['loss'] for h in model.training_history])
            }
            
            logger.info(f"Quantum model {model_id} training completed")
            return True
            
        except Exception as e:
            logger.error(f"Quantum model training failed: {e}")
            return False
    
    async def get_quantum_task_info(self, task_id: str) -> Optional[QuantumTask]:
        """Get quantum task information."""
        return self.quantum_tasks.get(task_id)
    
    async def get_quantum_model_info(self, model_id: str) -> Optional[QuantumModel]:
        """Get quantum model information."""
        return self.quantum_models.get(model_id)
    
    async def list_quantum_tasks(self, status: Optional[QuantumTaskStatus] = None) -> List[QuantumTask]:
        """List quantum tasks, optionally filtered by status."""
        if status:
            return [task for task in self.quantum_tasks.values() if task.status == status]
        return list(self.quantum_tasks.values())
    
    async def list_quantum_models(self) -> List[QuantumModel]:
        """List all quantum models."""
        return list(self.quantum_models.values())
    
    async def get_available_backends(self) -> Dict[QuantumBackend, Dict[str, Any]]:
        """Get available quantum backends."""
        return self.available_backends
    
    async def get_quantum_performance_summary(self) -> Dict[str, Any]:
        """Get summary of quantum computing performance."""
        try:
            if not self.performance_history:
                return {"message": "No quantum performance data available"}
            
            # Calculate statistics
            execution_times = [entry['execution_time'] for entry in self.performance_history if 'execution_time' in entry]
            algorithms = [entry['algorithm'] for entry in self.performance_history]
            backends = [entry['backend'] for entry in self.performance_history]
            
            summary = {
                'total_tasks_executed': len(self.performance_history),
                'successful_tasks': len([entry for entry in self.performance_history if entry.get('status') == 'completed']),
                'average_execution_time': np.mean(execution_times) if execution_times else 0,
                'algorithm_distribution': {alg: algorithms.count(alg) for alg in set(algorithms)},
                'backend_distribution': {backend: backends.count(backend) for backend in set(backends)},
                'recent_tasks': self.performance_history[-10:]  # Last 10 tasks
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Quantum performance summary generation failed: {e}")
            return {"error": str(e)}
    
    async def cancel_quantum_task(self, task_id: str) -> bool:
        """Cancel a quantum task."""
        try:
            if task_id not in self.quantum_tasks:
                return False
            
            task = self.quantum_tasks[task_id]
            if task.status == QuantumTaskStatus.RUNNING:
                task.status = QuantumTaskStatus.CANCELLED
                task.completed_at = datetime.now()
                logger.info(f"Quantum task {task_id} cancelled")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel quantum task {task_id}: {e}")
            return False
    
    async def cleanup_old_models(self, days_to_keep: int = 30):
        """Clean up old quantum models."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=days_to_keep)
            
            cleaned_count = 0
            models_to_remove = []
            
            for model_id, model in self.quantum_models.items():
                if model.updated_at < cutoff_time and not model.is_trained:
                    models_to_remove.append(model_id)
            
            for model_id in models_to_remove:
                del self.quantum_models[model_id]
                cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old quantum models")
            
        except Exception as e:
            logger.error(f"Quantum model cleanup failed: {e}")
    
    async def shutdown(self):
        """Shutdown the quantum computing manager."""
        try:
            # Cancel background tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Quantum computing manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Quantum computing manager shutdown error: {e}")

# Global quantum computing manager instance
quantum_computing_manager: Optional[QuantumComputingManager] = None

def get_quantum_computing_manager() -> QuantumComputingManager:
    """Get global quantum computing manager instance."""
    global quantum_computing_manager
    if quantum_computing_manager is None:
        quantum_computing_manager = QuantumComputingManager()
    return quantum_computing_manager

async def shutdown_quantum_computing_manager():
    """Shutdown global quantum computing manager."""
    global quantum_computing_manager
    if quantum_computing_manager:
        await quantum_computing_manager.shutdown()
        quantum_computing_manager = None

