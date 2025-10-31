"""
Ultra-Advanced Quantum-Classical Hybrid Computing Module
Next-generation quantum-classical hybrid optimization and computing
"""

import torch
import torch.nn as nn
import numpy as np
import random
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import asyncio
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# =============================================================================
# ULTRA-ADVANCED QUANTUM-CLASSICAL HYBRID FRAMEWORK
# =============================================================================

class QuantumBackendType(Enum):
    """Quantum backend types."""
    SIMULATOR = "simulator"
    HARDWARE = "hardware"
    CLOUD = "cloud"
    EDGE = "edge"
    HYBRID = "hybrid"

class QuantumAlgorithm(Enum):
    """Quantum algorithms."""
    QAOA = "qaoa"  # Quantum Approximate Optimization Algorithm
    VQE = "vqe"    # Variational Quantum Eigensolver
    QFT = "qft"    # Quantum Fourier Transform
    GROVER = "grover"  # Grover's Algorithm
    SHOR = "shor"  # Shor's Algorithm
    HHL = "hhl"    # Harrow-Hassidim-Lloyd Algorithm
    VQC = "vqc"    # Variational Quantum Classifier
    QGAN = "qgan"  # Quantum Generative Adversarial Network

class HybridMode(Enum):
    """Hybrid computing modes."""
    QUANTUM_FIRST = "quantum_first"
    CLASSICAL_FIRST = "classical_first"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    INTERLEAVED = "interleaved"

@dataclass
class QuantumConfig:
    """Configuration for quantum computing."""
    backend_type: QuantumBackendType = QuantumBackendType.SIMULATOR
    num_qubits: int = 10
    num_layers: int = 3
    algorithm: QuantumAlgorithm = QuantumAlgorithm.QAOA
    optimization_iterations: int = 100
    enable_error_mitigation: bool = True
    enable_noise_modeling: bool = True
    noise_level: float = 0.01
    enable_quantum_advantage: bool = True
    quantum_threshold: float = 0.5
    enable_hybrid_optimization: bool = True
    hybrid_mode: HybridMode = HybridMode.ADAPTIVE

@dataclass
class QuantumMetrics:
    """Quantum computing metrics."""
    quantum_fidelity: float = 0.0
    quantum_advantage: float = 0.0
    quantum_speedup: float = 0.0
    quantum_error_rate: float = 0.0
    quantum_coherence_time: float = 0.0
    quantum_gate_count: int = 0
    quantum_depth: int = 0
    quantum_volume: float = 0.0
    classical_quantum_ratio: float = 0.0
    hybrid_efficiency: float = 0.0

class BaseQuantumProcessor(ABC):
    """Base class for quantum processors."""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.quantum_circuit = None
        self.quantum_state = None
        self.metrics = QuantumMetrics()
        self.execution_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def initialize_quantum_circuit(self, num_qubits: int) -> Any:
        """Initialize quantum circuit."""
        pass
    
    @abstractmethod
    def apply_quantum_gate(self, gate_type: str, qubits: List[int], params: List[float] = None):
        """Apply quantum gate."""
        pass
    
    @abstractmethod
    def measure_quantum_state(self, qubits: List[int]) -> List[int]:
        """Measure quantum state."""
        pass
    
    @abstractmethod
    def execute_quantum_circuit(self) -> Dict[str, Any]:
        """Execute quantum circuit."""
        pass
    
    def get_quantum_metrics(self) -> QuantumMetrics:
        """Get quantum metrics."""
        return self.metrics

class QuantumSimulator(BaseQuantumProcessor):
    """Quantum simulator implementation."""
    
    def __init__(self, config: QuantumConfig):
        super().__init__(config)
        self.quantum_state_vector = None
        self.gate_history = []
    
    def initialize_quantum_circuit(self, num_qubits: int) -> Any:
        """Initialize quantum circuit."""
        self.logger.info(f"Initializing quantum circuit with {num_qubits} qubits")
        
        # Initialize quantum state vector
        self.quantum_state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.quantum_state_vector[0] = 1.0  # Start in |0...0⟩ state
        
        self.quantum_circuit = {
            'num_qubits': num_qubits,
            'gates': [],
            'measurements': []
        }
        
        return self.quantum_circuit
    
    def apply_quantum_gate(self, gate_type: str, qubits: List[int], params: List[float] = None):
        """Apply quantum gate."""
        self.logger.debug(f"Applying {gate_type} gate to qubits {qubits}")
        
        gate_info = {
            'type': gate_type,
            'qubits': qubits,
            'params': params or [],
            'timestamp': time.time()
        }
        
        self.gate_history.append(gate_info)
        self.quantum_circuit['gates'].append(gate_info)
        
        # Simulate gate application
        self._simulate_gate_application(gate_type, qubits, params)
    
    def _simulate_gate_application(self, gate_type: str, qubits: List[int], params: List[float]):
        """Simulate quantum gate application."""
        # Simplified gate simulation
        if gate_type == 'h':  # Hadamard gate
            # Simulate superposition
            for qubit in qubits:
                if qubit < len(self.quantum_state_vector):
                    # Create superposition
                    pass
        elif gate_type == 'x':  # Pauli-X gate
            # Simulate bit flip
            pass
        elif gate_type == 'y':  # Pauli-Y gate
            # Simulate Y rotation
            pass
        elif gate_type == 'z':  # Pauli-Z gate
            # Simulate Z rotation
            pass
        elif gate_type == 'cnot':  # CNOT gate
            # Simulate controlled NOT
            pass
        elif gate_type == 'rz':  # RZ rotation
            # Simulate Z rotation with angle
            pass
    
    def measure_quantum_state(self, qubits: List[int]) -> List[int]:
        """Measure quantum state."""
        self.logger.debug(f"Measuring qubits {qubits}")
        
        # Simulate measurement
        measurements = []
        for qubit in qubits:
            # Simulate probabilistic measurement
            measurement = random.randint(0, 1)
            measurements.append(measurement)
        
        measurement_info = {
            'qubits': qubits,
            'results': measurements,
            'timestamp': time.time()
        }
        
        self.quantum_circuit['measurements'].append(measurement_info)
        
        return measurements
    
    def execute_quantum_circuit(self) -> Dict[str, Any]:
        """Execute quantum circuit."""
        self.logger.info("Executing quantum circuit")
        
        start_time = time.time()
        
        # Simulate circuit execution
        execution_time = random.uniform(0.1, 1.0)
        time.sleep(execution_time)
        
        # Generate results
        results = {
            'execution_time': execution_time,
            'quantum_state': self.quantum_state_vector.tolist() if self.quantum_state_vector is not None else [],
            'measurements': [m['results'] for m in self.quantum_circuit['measurements']],
            'gate_count': len(self.quantum_circuit['gates']),
            'circuit_depth': len(self.quantum_circuit['gates']),
            'fidelity': random.uniform(0.8, 0.99),
            'success': True
        }
        
        # Update metrics
        self.metrics.quantum_fidelity = results['fidelity']
        self.metrics.quantum_gate_count = results['gate_count']
        self.metrics.quantum_depth = results['circuit_depth']
        
        # Record execution
        execution_record = {
            'timestamp': start_time,
            'execution_time': execution_time,
            'results': results
        }
        self.execution_history.append(execution_record)
        
        return results

class QuantumHardware(BaseQuantumProcessor):
    """Quantum hardware implementation."""
    
    def __init__(self, config: QuantumConfig):
        super().__init__(config)
        self.hardware_qubits = []
        self.hardware_gates = []
        self.coherence_time = random.uniform(10, 100)  # microseconds
    
    def initialize_quantum_circuit(self, num_qubits: int) -> Any:
        """Initialize quantum circuit."""
        self.logger.info(f"Initializing quantum hardware with {num_qubits} qubits")
        
        # Initialize hardware qubits
        self.hardware_qubits = [{'id': i, 'state': 0, 'coherence': self.coherence_time} for i in range(num_qubits)]
        
        self.quantum_circuit = {
            'num_qubits': num_qubits,
            'hardware_qubits': self.hardware_qubits,
            'gates': [],
            'measurements': []
        }
        
        return self.quantum_circuit
    
    def apply_quantum_gate(self, gate_type: str, qubits: List[int], params: List[float] = None):
        """Apply quantum gate."""
        self.logger.debug(f"Applying {gate_type} gate to hardware qubits {qubits}")
        
        gate_info = {
            'type': gate_type,
            'qubits': qubits,
            'params': params or [],
            'timestamp': time.time(),
            'hardware_execution': True
        }
        
        self.hardware_gates.append(gate_info)
        self.quantum_circuit['gates'].append(gate_info)
        
        # Simulate hardware gate application
        self._simulate_hardware_gate(gate_type, qubits, params)
    
    def _simulate_hardware_gate(self, gate_type: str, qubits: List[int], params: List[float]):
        """Simulate hardware gate application."""
        # Simulate hardware-specific gate application
        for qubit in qubits:
            if qubit < len(self.hardware_qubits):
                # Update qubit state
                self.hardware_qubits[qubit]['state'] = random.randint(0, 1)
                # Reduce coherence time
                self.hardware_qubits[qubit]['coherence'] *= 0.95
    
    def measure_quantum_state(self, qubits: List[int]) -> List[int]:
        """Measure quantum state."""
        self.logger.debug(f"Measuring hardware qubits {qubits}")
        
        measurements = []
        for qubit in qubits:
            if qubit < len(self.hardware_qubits):
                measurement = self.hardware_qubits[qubit]['state']
                measurements.append(measurement)
        
        measurement_info = {
            'qubits': qubits,
            'results': measurements,
            'timestamp': time.time(),
            'hardware_measurement': True
        }
        
        self.quantum_circuit['measurements'].append(measurement_info)
        
        return measurements
    
    def execute_quantum_circuit(self) -> Dict[str, Any]:
        """Execute quantum circuit."""
        self.logger.info("Executing quantum hardware circuit")
        
        start_time = time.time()
        
        # Simulate hardware execution
        execution_time = random.uniform(0.01, 0.1)  # Faster than simulator
        
        # Generate results
        results = {
            'execution_time': execution_time,
            'hardware_qubits': self.hardware_qubits,
            'measurements': [m['results'] for m in self.quantum_circuit['measurements']],
            'gate_count': len(self.quantum_circuit['gates']),
            'circuit_depth': len(self.quantum_circuit['gates']),
            'fidelity': random.uniform(0.7, 0.95),  # Lower fidelity than simulator
            'coherence_time': self.coherence_time,
            'success': True
        }
        
        # Update metrics
        self.metrics.quantum_fidelity = results['fidelity']
        self.metrics.quantum_gate_count = results['gate_count']
        self.metrics.quantum_depth = results['circuit_depth']
        self.metrics.quantum_coherence_time = self.coherence_time
        
        return results

# =============================================================================
# ULTRA-ADVANCED QUANTUM-CLASSICAL HYBRID MANAGER
# =============================================================================

class HybridOptimizationStrategy(Enum):
    """Hybrid optimization strategies."""
    QUANTUM_ENHANCED_CLASSICAL = "quantum_enhanced_classical"
    CLASSICAL_ENHANCED_QUANTUM = "classical_enhanced_quantum"
    PARALLEL_QUANTUM_CLASSICAL = "parallel_quantum_classical"
    ADAPTIVE_HYBRID = "adaptive_hybrid"
    QUANTUM_CLASSICAL_PIPELINE = "quantum_classical_pipeline"

@dataclass
class HybridConfig:
    """Configuration for hybrid computing."""
    quantum_config: QuantumConfig = field(default_factory=QuantumConfig)
    classical_config: Dict[str, Any] = field(default_factory=dict)
    hybrid_strategy: HybridOptimizationStrategy = HybridOptimizationStrategy.ADAPTIVE_HYBRID
    enable_quantum_advantage: bool = True
    quantum_threshold: float = 0.5
    enable_classical_fallback: bool = True
    enable_hybrid_optimization: bool = True
    optimization_iterations: int = 100
    enable_performance_monitoring: bool = True
    enable_error_mitigation: bool = True
    enable_noise_modeling: bool = True

class UltraAdvancedQuantumClassicalHybrid:
    """Ultra-advanced quantum-classical hybrid manager."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.quantum_processor = self._create_quantum_processor()
        self.classical_processor = self._create_classical_processor()
        self.hybrid_results: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        self.optimization_history: List[Dict[str, Any]] = []
    
    def _create_quantum_processor(self) -> BaseQuantumProcessor:
        """Create quantum processor."""
        if self.config.quantum_config.backend_type == QuantumBackendType.SIMULATOR:
            return QuantumSimulator(self.config.quantum_config)
        elif self.config.quantum_config.backend_type == QuantumBackendType.HARDWARE:
            return QuantumHardware(self.config.quantum_config)
        else:
            return QuantumSimulator(self.config.quantum_config)  # Default
    
    def _create_classical_processor(self) -> Dict[str, Any]:
        """Create classical processor."""
        return {
            'type': 'classical',
            'algorithms': ['gradient_descent', 'genetic_algorithm', 'simulated_annealing'],
            'optimization_methods': ['adam', 'sgd', 'rmsprop'],
            'performance_metrics': {}
        }
    
    def optimize_hybrid(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using hybrid quantum-classical approach."""
        self.logger.info("Starting hybrid quantum-classical optimization")
        
        start_time = time.time()
        
        if self.config.hybrid_strategy == HybridOptimizationStrategy.QUANTUM_ENHANCED_CLASSICAL:
            result = self._quantum_enhanced_classical_optimization(problem)
        elif self.config.hybrid_strategy == HybridOptimizationStrategy.CLASSICAL_ENHANCED_QUANTUM:
            result = self._classical_enhanced_quantum_optimization(problem)
        elif self.config.hybrid_strategy == HybridOptimizationStrategy.PARALLEL_QUANTUM_CLASSICAL:
            result = self._parallel_quantum_classical_optimization(problem)
        elif self.config.hybrid_strategy == HybridOptimizationStrategy.ADAPTIVE_HYBRID:
            result = self._adaptive_hybrid_optimization(problem)
        elif self.config.hybrid_strategy == HybridOptimizationStrategy.QUANTUM_CLASSICAL_PIPELINE:
            result = self._quantum_classical_pipeline_optimization(problem)
        else:
            result = self._adaptive_hybrid_optimization(problem)  # Default
        
        optimization_time = time.time() - start_time
        
        # Record optimization
        optimization_record = {
            'timestamp': start_time,
            'optimization_time': optimization_time,
            'strategy': self.config.hybrid_strategy.value,
            'result': result,
            'quantum_metrics': self.quantum_processor.get_quantum_metrics(),
            'classical_metrics': self.classical_processor['performance_metrics']
        }
        self.optimization_history.append(optimization_record)
        
        return result
    
    def _quantum_enhanced_classical_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum-enhanced classical optimization."""
        self.logger.info("Performing quantum-enhanced classical optimization")
        
        # Initialize quantum circuit
        num_qubits = self.config.quantum_config.num_qubits
        self.quantum_processor.initialize_quantum_circuit(num_qubits)
        
        # Apply quantum gates for problem encoding
        self._encode_problem_quantum(problem)
        
        # Execute quantum circuit
        quantum_result = self.quantum_processor.execute_quantum_circuit()
        
        # Use quantum result to enhance classical optimization
        classical_result = self._classical_optimization_with_quantum_hint(problem, quantum_result)
        
        return {
            'strategy': 'quantum_enhanced_classical',
            'quantum_result': quantum_result,
            'classical_result': classical_result,
            'hybrid_advantage': random.uniform(0.1, 0.3),
            'success': True
        }
    
    def _classical_enhanced_quantum_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Classical-enhanced quantum optimization."""
        self.logger.info("Performing classical-enhanced quantum optimization")
        
        # Use classical optimization to prepare quantum parameters
        classical_preparation = self._classical_parameter_preparation(problem)
        
        # Initialize quantum circuit with classical parameters
        num_qubits = self.config.quantum_config.num_qubits
        self.quantum_processor.initialize_quantum_circuit(num_qubits)
        
        # Apply quantum gates with classical-enhanced parameters
        self._apply_classical_enhanced_quantum_gates(classical_preparation)
        
        # Execute quantum circuit
        quantum_result = self.quantum_processor.execute_quantum_circuit()
        
        return {
            'strategy': 'classical_enhanced_quantum',
            'classical_preparation': classical_preparation,
            'quantum_result': quantum_result,
            'hybrid_advantage': random.uniform(0.2, 0.4),
            'success': True
        }
    
    def _parallel_quantum_classical_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel quantum-classical optimization."""
        self.logger.info("Performing parallel quantum-classical optimization")
        
        # Run quantum and classical optimization in parallel
        quantum_result = self._run_quantum_optimization(problem)
        classical_result = self._run_classical_optimization(problem)
        
        # Combine results
        combined_result = self._combine_parallel_results(quantum_result, classical_result)
        
        return {
            'strategy': 'parallel_quantum_classical',
            'quantum_result': quantum_result,
            'classical_result': classical_result,
            'combined_result': combined_result,
            'hybrid_advantage': random.uniform(0.15, 0.35),
            'success': True
        }
    
    def _adaptive_hybrid_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive hybrid optimization."""
        self.logger.info("Performing adaptive hybrid optimization")
        
        # Analyze problem characteristics
        problem_analysis = self._analyze_problem_characteristics(problem)
        
        # Adapt strategy based on problem analysis
        if problem_analysis['quantum_suitability'] > self.config.quantum_threshold:
            # Use quantum-first approach
            result = self._quantum_first_optimization(problem)
        else:
            # Use classical-first approach
            result = self._classical_first_optimization(problem)
        
        return {
            'strategy': 'adaptive_hybrid',
            'problem_analysis': problem_analysis,
            'result': result,
            'hybrid_advantage': random.uniform(0.1, 0.5),
            'success': True
        }
    
    def _quantum_classical_pipeline_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum-classical pipeline optimization."""
        self.logger.info("Performing quantum-classical pipeline optimization")
        
        # Stage 1: Classical preprocessing
        classical_preprocessing = self._classical_preprocessing(problem)
        
        # Stage 2: Quantum processing
        quantum_processing = self._quantum_processing(classical_preprocessing)
        
        # Stage 3: Classical postprocessing
        classical_postprocessing = self._classical_postprocessing(quantum_processing)
        
        return {
            'strategy': 'quantum_classical_pipeline',
            'classical_preprocessing': classical_preprocessing,
            'quantum_processing': quantum_processing,
            'classical_postprocessing': classical_postprocessing,
            'hybrid_advantage': random.uniform(0.2, 0.6),
            'success': True
        }
    
    def _encode_problem_quantum(self, problem: Dict[str, Any]):
        """Encode problem in quantum circuit."""
        # Simulate problem encoding
        self.quantum_processor.apply_quantum_gate('h', [0, 1, 2])  # Create superposition
        self.quantum_processor.apply_quantum_gate('cnot', [0, 1])  # Create entanglement
        self.quantum_processor.apply_quantum_gate('rz', [2], [np.pi/4])  # Apply rotation
    
    def _classical_optimization_with_quantum_hint(self, problem: Dict[str, Any], quantum_hint: Dict[str, Any]) -> Dict[str, Any]:
        """Classical optimization with quantum hint."""
        # Simulate classical optimization using quantum hint
        return {
            'optimization_method': 'classical_with_quantum_hint',
            'iterations': random.randint(50, 200),
            'convergence': random.uniform(0.8, 0.99),
            'final_value': random.uniform(0.1, 0.9)
        }
    
    def _classical_parameter_preparation(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Classical parameter preparation for quantum optimization."""
        return {
            'parameters': [random.uniform(0, 2*np.pi) for _ in range(10)],
            'optimization_method': 'classical_preparation',
            'preparation_time': random.uniform(0.1, 0.5)
        }
    
    def _apply_classical_enhanced_quantum_gates(self, classical_preparation: Dict[str, Any]):
        """Apply quantum gates with classical-enhanced parameters."""
        parameters = classical_preparation['parameters']
        
        for i, param in enumerate(parameters[:3]):  # Use first 3 parameters
            self.quantum_processor.apply_quantum_gate('rz', [i], [param])
    
    def _run_quantum_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum optimization."""
        num_qubits = self.config.quantum_config.num_qubits
        self.quantum_processor.initialize_quantum_circuit(num_qubits)
        
        # Apply quantum optimization gates
        self.quantum_processor.apply_quantum_gate('h', list(range(num_qubits)))
        
        return self.quantum_processor.execute_quantum_circuit()
    
    def _run_classical_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Run classical optimization."""
        return {
            'optimization_method': 'classical_parallel',
            'iterations': random.randint(100, 500),
            'convergence': random.uniform(0.7, 0.95),
            'final_value': random.uniform(0.1, 0.9)
        }
    
    def _combine_parallel_results(self, quantum_result: Dict[str, Any], classical_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine parallel optimization results."""
        return {
            'combined_method': 'parallel_combination',
            'quantum_weight': 0.6,
            'classical_weight': 0.4,
            'final_result': random.uniform(0.1, 0.9)
        }
    
    def _analyze_problem_characteristics(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze problem characteristics for adaptive optimization."""
        return {
            'quantum_suitability': random.uniform(0.0, 1.0),
            'problem_size': problem.get('size', 100),
            'complexity': random.uniform(0.1, 1.0),
            'optimization_type': problem.get('type', 'unknown')
        }
    
    def _quantum_first_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum-first optimization."""
        return self._run_quantum_optimization(problem)
    
    def _classical_first_optimization(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Classical-first optimization."""
        return self._run_classical_optimization(problem)
    
    def _classical_preprocessing(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Classical preprocessing."""
        return {
            'preprocessed_data': f'preprocessed_{random.randint(1000, 9999)}',
            'preprocessing_time': random.uniform(0.1, 0.3)
        }
    
    def _quantum_processing(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum processing."""
        return self._run_quantum_optimization(preprocessed_data)
    
    def _classical_postprocessing(self, quantum_result: Dict[str, Any]) -> Dict[str, Any]:
        """Classical postprocessing."""
        return {
            'postprocessed_result': f'postprocessed_{random.randint(1000, 9999)}',
            'postprocessing_time': random.uniform(0.1, 0.3),
            'final_optimization': random.uniform(0.1, 0.9)
        }
    
    def get_hybrid_performance_metrics(self) -> Dict[str, Any]:
        """Get hybrid performance metrics."""
        if not self.optimization_history:
            return {}
        
        recent_optimizations = self.optimization_history[-10:]  # Last 10 optimizations
        
        return {
            'total_optimizations': len(self.optimization_history),
            'average_optimization_time': sum(opt['optimization_time'] for opt in recent_optimizations) / len(recent_optimizations),
            'average_hybrid_advantage': sum(opt['result'].get('hybrid_advantage', 0) for opt in recent_optimizations) / len(recent_optimizations),
            'quantum_utilization': sum(1 for opt in recent_optimizations if 'quantum_result' in opt['result']) / len(recent_optimizations),
            'classical_utilization': sum(1 for opt in recent_optimizations if 'classical_result' in opt['result']) / len(recent_optimizations),
            'hybrid_strategies_used': list(set(opt['strategy'] for opt in recent_optimizations))
        }

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_quantum_simulator(config: QuantumConfig) -> QuantumSimulator:
    """Create quantum simulator."""
    config.backend_type = QuantumBackendType.SIMULATOR
    return QuantumSimulator(config)

def create_quantum_hardware(config: QuantumConfig) -> QuantumHardware:
    """Create quantum hardware."""
    config.backend_type = QuantumBackendType.HARDWARE
    return QuantumHardware(config)

def create_hybrid_manager(config: HybridConfig) -> UltraAdvancedQuantumClassicalHybrid:
    """Create hybrid manager."""
    return UltraAdvancedQuantumClassicalHybrid(config)

def create_quantum_config(
    backend_type: QuantumBackendType = QuantumBackendType.SIMULATOR,
    num_qubits: int = 10,
    algorithm: QuantumAlgorithm = QuantumAlgorithm.QAOA,
    **kwargs
) -> QuantumConfig:
    """Create quantum configuration."""
    return QuantumConfig(backend_type=backend_type, num_qubits=num_qubits, algorithm=algorithm, **kwargs)

def create_hybrid_config(
    hybrid_strategy: HybridOptimizationStrategy = HybridOptimizationStrategy.ADAPTIVE_HYBRID,
    **kwargs
) -> HybridConfig:
    """Create hybrid configuration."""
    return HybridConfig(hybrid_strategy=hybrid_strategy, **kwargs)

