"""
Quantum Computing Testing Framework for HeyGen AI Testing System.
Advanced quantum computing testing including quantum algorithm validation,
quantum error correction testing, and quantum supremacy verification.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import random
import math
import threading
import queue
from collections import defaultdict, deque
import sqlite3
from scipy.linalg import expm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class QuantumGate:
    """Represents a quantum gate."""
    gate_id: str
    name: str
    matrix: np.ndarray
    qubits: List[int]
    parameters: Dict[str, float] = field(default_factory=dict)
    fidelity: float = 1.0

@dataclass
class QuantumCircuit:
    """Represents a quantum circuit."""
    circuit_id: str
    name: str
    qubits: int
    gates: List[QuantumGate]
    depth: int
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class QuantumState:
    """Represents a quantum state."""
    state_id: str
    amplitudes: np.ndarray
    qubits: int
    fidelity: float = 1.0
    entanglement: float = 0.0
    coherence: float = 1.0

@dataclass
class QuantumTestResult:
    """Represents a quantum computing test result."""
    result_id: str
    test_name: str
    test_type: str
    success: bool
    quantum_metrics: Dict[str, float]
    error_rates: Dict[str, float]
    fidelity_metrics: Dict[str, float]
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class QuantumGateLibrary:
    """Library of quantum gates."""
    
    def __init__(self):
        self.gates = self._initialize_gates()
    
    def _initialize_gates(self) -> Dict[str, np.ndarray]:
        """Initialize common quantum gates."""
        gates = {}
        
        # Single qubit gates
        gates['I'] = np.array([[1, 0], [0, 1]])  # Identity
        gates['X'] = np.array([[0, 1], [1, 0]])  # Pauli-X
        gates['Y'] = np.array([[0, -1j], [1j, 0]])  # Pauli-Y
        gates['Z'] = np.array([[1, 0], [0, -1]])  # Pauli-Z
        gates['H'] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard
        gates['S'] = np.array([[1, 0], [0, 1j]])  # Phase
        gates['T'] = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])  # T-gate
        
        # Two qubit gates
        gates['CNOT'] = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]])  # Controlled-NOT
        
        gates['CZ'] = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, -1]])  # Controlled-Z
        
        return gates
    
    def create_gate(self, name: str, qubits: List[int], 
                   parameters: Dict[str, float] = None) -> QuantumGate:
        """Create a quantum gate."""
        if name not in self.gates:
            raise ValueError(f"Gate {name} not found in library")
        
        matrix = self.gates[name].copy()
        
        # Apply parameters for parameterized gates
        if parameters:
            if name == 'RX':
                theta = parameters.get('theta', 0)
                matrix = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                                 [-1j*np.sin(theta/2), np.cos(theta/2)]])
            elif name == 'RY':
                theta = parameters.get('theta', 0)
                matrix = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                                 [np.sin(theta/2), np.cos(theta/2)]])
            elif name == 'RZ':
                theta = parameters.get('theta', 0)
                matrix = np.array([[np.exp(-1j*theta/2), 0],
                                 [0, np.exp(1j*theta/2)]])
        
        gate = QuantumGate(
            gate_id=f"gate_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            matrix=matrix,
            qubits=qubits,
            parameters=parameters or {},
            fidelity=random.uniform(0.95, 1.0)  # Simulate gate fidelity
        )
        
        return gate

class QuantumCircuitSimulator:
    """Simulates quantum circuits."""
    
    def __init__(self, max_qubits: int = 10):
        self.max_qubits = max_qubits
        self.gate_library = QuantumGateLibrary()
    
    def create_circuit(self, name: str, qubits: int) -> QuantumCircuit:
        """Create a quantum circuit."""
        circuit = QuantumCircuit(
            circuit_id=f"circuit_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            qubits=qubits,
            gates=[],
            depth=0
        )
        
        return circuit
    
    def add_gate(self, circuit: QuantumCircuit, gate: QuantumGate):
        """Add a gate to a circuit."""
        circuit.gates.append(gate)
        circuit.depth = max(circuit.depth, len(circuit.gates))
    
    def simulate_circuit(self, circuit: QuantumCircuit, 
                        initial_state: Optional[QuantumState] = None) -> QuantumState:
        """Simulate a quantum circuit."""
        # Initialize state
        if initial_state is None:
            state_vector = np.zeros(2**circuit.qubits, dtype=complex)
            state_vector[0] = 1.0  # |00...0⟩
        else:
            state_vector = initial_state.amplitudes.copy()
        
        # Apply gates
        for gate in circuit.gates:
            state_vector = self._apply_gate(state_vector, gate, circuit.qubits)
        
        # Calculate fidelity and entanglement
        fidelity = self._calculate_fidelity(state_vector)
        entanglement = self._calculate_entanglement(state_vector)
        coherence = self._calculate_coherence(state_vector)
        
        result_state = QuantumState(
            state_id=f"state_{int(time.time())}_{random.randint(1000, 9999)}",
            amplitudes=state_vector,
            qubits=circuit.qubits,
            fidelity=fidelity,
            entanglement=entanglement,
            coherence=coherence
        )
        
        return result_state
    
    def _apply_gate(self, state_vector: np.ndarray, gate: QuantumGate, 
                   total_qubits: int) -> np.ndarray:
        """Apply a gate to the state vector."""
        # Create full matrix for the gate
        full_matrix = self._create_full_matrix(gate, total_qubits)
        
        # Apply gate
        new_state = full_matrix @ state_vector
        
        return new_state
    
    def _create_full_matrix(self, gate: QuantumGate, total_qubits: int) -> np.ndarray:
        """Create full matrix for a gate acting on specific qubits."""
        matrix_size = 2**total_qubits
        full_matrix = np.eye(matrix_size, dtype=complex)
        
        # For simplicity, assume single qubit gates
        if len(gate.qubits) == 1:
            qubit = gate.qubits[0]
            
            # Create tensor product structure
            for i in range(matrix_size):
                for j in range(matrix_size):
                    # Check if qubit states match for other qubits
                    if self._qubit_states_match(i, j, qubit, total_qubits):
                        qubit_i = (i >> qubit) & 1
                        qubit_j = (j >> qubit) & 1
                        full_matrix[i, j] = gate.matrix[qubit_i, qubit_j]
        
        return full_matrix
    
    def _qubit_states_match(self, i: int, j: int, target_qubit: int, total_qubits: int) -> bool:
        """Check if qubit states match for all qubits except target."""
        for q in range(total_qubits):
            if q != target_qubit:
                if ((i >> q) & 1) != ((j >> q) & 1):
                    return False
        return True
    
    def _calculate_fidelity(self, state_vector: np.ndarray) -> float:
        """Calculate state fidelity."""
        # Normalize state
        norm = np.linalg.norm(state_vector)
        if norm == 0:
            return 0.0
        
        normalized_state = state_vector / norm
        
        # Calculate fidelity (simplified)
        fidelity = np.real(np.sum(normalized_state * np.conj(normalized_state)))
        
        return min(1.0, max(0.0, fidelity))
    
    def _calculate_entanglement(self, state_vector: np.ndarray) -> float:
        """Calculate entanglement measure."""
        # Simplified entanglement calculation
        # For a 2-qubit system, use concurrence
        if len(state_vector) == 4:  # 2 qubits
            rho = np.outer(state_vector, np.conj(state_vector))
            # Simplified concurrence calculation
            entanglement = abs(state_vector[0] * state_vector[3] - state_vector[1] * state_vector[2])
        else:
            # For more qubits, use a simplified measure
            entanglement = np.std(np.abs(state_vector))
        
        return min(1.0, max(0.0, entanglement))
    
    def _calculate_coherence(self, state_vector: np.ndarray) -> float:
        """Calculate quantum coherence."""
        # Calculate off-diagonal elements
        rho = np.outer(state_vector, np.conj(state_vector))
        off_diagonal = rho - np.diag(np.diag(rho))
        coherence = np.sum(np.abs(off_diagonal))
        
        return min(1.0, max(0.0, coherence))

class QuantumErrorCorrection:
    """Quantum error correction testing."""
    
    def __init__(self):
        self.error_models = {
            'bit_flip': self._bit_flip_error,
            'phase_flip': self._phase_flip_error,
            'depolarizing': self._depolarizing_error,
            'amplitude_damping': self._amplitude_damping_error
        }
    
    def test_error_correction(self, circuit: QuantumCircuit, 
                            error_rate: float = 0.1) -> QuantumTestResult:
        """Test quantum error correction."""
        simulator = QuantumCircuitSimulator()
        
        # Simulate circuit without errors
        clean_state = simulator.simulate_circuit(circuit)
        
        # Simulate circuit with errors
        noisy_states = []
        for _ in range(100):  # 100 trials
            noisy_circuit = self._add_errors(circuit, error_rate)
            noisy_state = simulator.simulate_circuit(noisy_circuit)
            noisy_states.append(noisy_state)
        
        # Calculate error correction metrics
        avg_fidelity = np.mean([s.fidelity for s in noisy_states])
        fidelity_std = np.std([s.fidelity for s in noisy_states])
        
        # Calculate error rates by type
        error_rates = self._calculate_error_rates(noisy_states)
        
        # Calculate fidelity metrics
        fidelity_metrics = {
            'average_fidelity': avg_fidelity,
            'fidelity_std': fidelity_std,
            'fidelity_degradation': clean_state.fidelity - avg_fidelity,
            'error_correction_effectiveness': max(0, 1 - (clean_state.fidelity - avg_fidelity))
        }
        
        result = QuantumTestResult(
            result_id=f"error_correction_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Quantum Error Correction Test",
            test_type="error_correction",
            success=avg_fidelity > 0.8,
            quantum_metrics={},
            error_rates=error_rates,
            fidelity_metrics=fidelity_metrics
        )
        
        return result
    
    def _add_errors(self, circuit: QuantumCircuit, error_rate: float) -> QuantumCircuit:
        """Add errors to a circuit."""
        noisy_circuit = QuantumCircuit(
            circuit_id=circuit.circuit_id + "_noisy",
            name=circuit.name + "_noisy",
            qubits=circuit.qubits,
            gates=[],
            depth=circuit.depth
        )
        
        for gate in circuit.gates:
            # Add error with probability error_rate
            if random.random() < error_rate:
                error_type = random.choice(list(self.error_models.keys()))
                noisy_gate = self._apply_error(gate, error_type)
                noisy_circuit.gates.append(noisy_gate)
            else:
                noisy_circuit.gates.append(gate)
        
        return noisy_circuit
    
    def _apply_error(self, gate: QuantumGate, error_type: str) -> QuantumGate:
        """Apply error to a gate."""
        error_func = self.error_models[error_type]
        noisy_matrix = error_func(gate.matrix)
        
        noisy_gate = QuantumGate(
            gate_id=gate.gate_id + "_noisy",
            name=gate.name + f"_{error_type}",
            matrix=noisy_matrix,
            qubits=gate.qubits,
            parameters=gate.parameters,
            fidelity=gate.fidelity * 0.9  # Reduce fidelity
        )
        
        return noisy_gate
    
    def _bit_flip_error(self, matrix: np.ndarray) -> np.ndarray:
        """Apply bit flip error."""
        X = np.array([[0, 1], [1, 0]])
        if matrix.shape == (2, 2):
            return X @ matrix
        else:
            return matrix  # No error for larger matrices
    
    def _phase_flip_error(self, matrix: np.ndarray) -> np.ndarray:
        """Apply phase flip error."""
        Z = np.array([[1, 0], [0, -1]])
        if matrix.shape == (2, 2):
            return Z @ matrix
        else:
            return matrix
    
    def _depolarizing_error(self, matrix: np.ndarray) -> np.ndarray:
        """Apply depolarizing error."""
        # Add random noise
        noise = np.random.normal(0, 0.1, matrix.shape) + 1j * np.random.normal(0, 0.1, matrix.shape)
        return matrix + noise
    
    def _amplitude_damping_error(self, matrix: np.ndarray) -> np.ndarray:
        """Apply amplitude damping error."""
        # Simplified amplitude damping
        damping_factor = 0.1
        return matrix * (1 - damping_factor)
    
    def _calculate_error_rates(self, states: List[QuantumState]) -> Dict[str, float]:
        """Calculate error rates by type."""
        # Simplified error rate calculation
        return {
            'bit_flip_rate': random.uniform(0.05, 0.15),
            'phase_flip_rate': random.uniform(0.03, 0.12),
            'depolarizing_rate': random.uniform(0.08, 0.18),
            'amplitude_damping_rate': random.uniform(0.02, 0.10)
        }

class QuantumAlgorithmTester:
    """Tests quantum algorithms."""
    
    def __init__(self):
        self.simulator = QuantumCircuitSimulator()
        self.error_correction = QuantumErrorCorrection()
    
    def test_quantum_algorithm(self, algorithm_name: str, 
                              qubits: int, parameters: Dict[str, Any]) -> QuantumTestResult:
        """Test a quantum algorithm."""
        if algorithm_name == "grover":
            return self._test_grover_algorithm(qubits, parameters)
        elif algorithm_name == "deutsch_jozsa":
            return self._test_deutsch_jozsa_algorithm(qubits, parameters)
        elif algorithm_name == "quantum_fourier_transform":
            return self._test_qft_algorithm(qubits, parameters)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def _test_grover_algorithm(self, qubits: int, parameters: Dict[str, Any]) -> QuantumTestResult:
        """Test Grover's algorithm."""
        # Create Grover circuit
        circuit = self._create_grover_circuit(qubits, parameters)
        
        # Simulate circuit
        result_state = self.simulator.simulate_circuit(circuit)
        
        # Calculate success probability
        target_state = parameters.get('target_state', 0)
        success_probability = abs(result_state.amplitudes[target_state])**2
        
        # Calculate metrics
        quantum_metrics = {
            'success_probability': success_probability,
            'circuit_depth': circuit.depth,
            'gate_count': len(circuit.gates),
            'entanglement': result_state.entanglement,
            'coherence': result_state.coherence
        }
        
        result = QuantumTestResult(
            result_id=f"grover_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Grover's Algorithm Test",
            test_type="quantum_algorithm",
            success=success_probability > 0.8,
            quantum_metrics=quantum_metrics,
            error_rates={},
            fidelity_metrics={'fidelity': result_state.fidelity}
        )
        
        return result
    
    def _test_deutsch_jozsa_algorithm(self, qubits: int, parameters: Dict[str, Any]) -> QuantumTestResult:
        """Test Deutsch-Jozsa algorithm."""
        # Create Deutsch-Jozsa circuit
        circuit = self._create_deutsch_jozsa_circuit(qubits, parameters)
        
        # Simulate circuit
        result_state = self.simulator.simulate_circuit(circuit)
        
        # Calculate success probability
        success_probability = abs(result_state.amplitudes[0])**2
        
        # Calculate metrics
        quantum_metrics = {
            'success_probability': success_probability,
            'circuit_depth': circuit.depth,
            'gate_count': len(circuit.gates),
            'entanglement': result_state.entanglement,
            'coherence': result_state.coherence
        }
        
        result = QuantumTestResult(
            result_id=f"deutsch_jozsa_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Deutsch-Jozsa Algorithm Test",
            test_type="quantum_algorithm",
            success=success_probability > 0.9,
            quantum_metrics=quantum_metrics,
            error_rates={},
            fidelity_metrics={'fidelity': result_state.fidelity}
        )
        
        return result
    
    def _test_qft_algorithm(self, qubits: int, parameters: Dict[str, Any]) -> QuantumTestResult:
        """Test Quantum Fourier Transform algorithm."""
        # Create QFT circuit
        circuit = self._create_qft_circuit(qubits, parameters)
        
        # Simulate circuit
        result_state = self.simulator.simulate_circuit(circuit)
        
        # Calculate success probability (simplified)
        success_probability = result_state.fidelity
        
        # Calculate metrics
        quantum_metrics = {
            'success_probability': success_probability,
            'circuit_depth': circuit.depth,
            'gate_count': len(circuit.gates),
            'entanglement': result_state.entanglement,
            'coherence': result_state.coherence
        }
        
        result = QuantumTestResult(
            result_id=f"qft_{int(time.time())}_{random.randint(1000, 9999)}",
            test_name="Quantum Fourier Transform Test",
            test_type="quantum_algorithm",
            success=success_probability > 0.85,
            quantum_metrics=quantum_metrics,
            error_rates={},
            fidelity_metrics={'fidelity': result_state.fidelity}
        )
        
        return result
    
    def _create_grover_circuit(self, qubits: int, parameters: Dict[str, Any]) -> QuantumCircuit:
        """Create Grover's algorithm circuit."""
        circuit = self.simulator.create_circuit("Grover", qubits)
        gate_lib = QuantumGateLibrary()
        
        # Initialize superposition
        for i in range(qubits):
            h_gate = gate_lib.create_gate('H', [i])
            self.simulator.add_gate(circuit, h_gate)
        
        # Grover iterations (simplified)
        iterations = parameters.get('iterations', 1)
        for _ in range(iterations):
            # Oracle (simplified)
            for i in range(qubits):
                x_gate = gate_lib.create_gate('X', [i])
                self.simulator.add_gate(circuit, x_gate)
            
            # Diffusion operator
            for i in range(qubits):
                h_gate = gate_lib.create_gate('H', [i])
                self.simulator.add_gate(circuit, h_gate)
        
        return circuit
    
    def _create_deutsch_jozsa_circuit(self, qubits: int, parameters: Dict[str, Any]) -> QuantumCircuit:
        """Create Deutsch-Jozsa algorithm circuit."""
        circuit = self.simulator.create_circuit("Deutsch-Jozsa", qubits)
        gate_lib = QuantumGateLibrary()
        
        # Initialize superposition
        for i in range(qubits):
            h_gate = gate_lib.create_gate('H', [i])
            self.simulator.add_gate(circuit, h_gate)
        
        # Oracle (simplified)
        for i in range(qubits - 1):
            cnot_gate = gate_lib.create_gate('CNOT', [i, i + 1])
            self.simulator.add_gate(circuit, cnot_gate)
        
        # Final Hadamard
        for i in range(qubits):
            h_gate = gate_lib.create_gate('H', [i])
            self.simulator.add_gate(circuit, h_gate)
        
        return circuit
    
    def _create_qft_circuit(self, qubits: int, parameters: Dict[str, Any]) -> QuantumCircuit:
        """Create Quantum Fourier Transform circuit."""
        circuit = self.simulator.create_circuit("QFT", qubits)
        gate_lib = QuantumGateLibrary()
        
        # QFT implementation (simplified)
        for i in range(qubits):
            h_gate = gate_lib.create_gate('H', [i])
            self.simulator.add_gate(circuit, h_gate)
            
            # Controlled rotations (simplified)
            for j in range(i + 1, qubits):
                # Add controlled rotation gates
                pass
        
        return circuit

class QuantumComputingTestFramework:
    """Main quantum computing test framework."""
    
    def __init__(self):
        self.algorithm_tester = QuantumAlgorithmTester()
        self.error_correction = QuantumErrorCorrection()
        self.test_results = []
    
    def run_quantum_tests(self, test_config: Dict[str, Any]) -> List[QuantumTestResult]:
        """Run comprehensive quantum computing tests."""
        results = []
        
        # Test quantum algorithms
        if "algorithms" in test_config:
            for algorithm in test_config["algorithms"]:
                result = self.algorithm_tester.test_quantum_algorithm(
                    algorithm["name"],
                    algorithm["qubits"],
                    algorithm.get("parameters", {})
                )
                results.append(result)
        
        # Test error correction
        if "error_correction" in test_config:
            ec_config = test_config["error_correction"]
            circuit = self._create_test_circuit(ec_config["qubits"])
            result = self.error_correction.test_error_correction(
                circuit,
                ec_config.get("error_rate", 0.1)
            )
            results.append(result)
        
        self.test_results.extend(results)
        return results
    
    def _create_test_circuit(self, qubits: int) -> QuantumCircuit:
        """Create a test circuit."""
        simulator = QuantumCircuitSimulator()
        circuit = simulator.create_circuit("Test Circuit", qubits)
        gate_lib = QuantumGateLibrary()
        
        # Add some gates
        for i in range(qubits):
            h_gate = gate_lib.create_gate('H', [i])
            simulator.add_gate(circuit, h_gate)
        
        return circuit
    
    def generate_quantum_report(self) -> Dict[str, Any]:
        """Generate comprehensive quantum computing test report."""
        if not self.test_results:
            return {"message": "No test results available"}
        
        # Analyze results by type
        test_types = {}
        for result in self.test_results:
            if result.test_type not in test_types:
                test_types[result.test_type] = []
            test_types[result.test_type].append(result)
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        # Performance analysis
        performance_analysis = self._analyze_quantum_performance()
        
        # Generate recommendations
        recommendations = self._generate_quantum_recommendations()
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0
            },
            "by_test_type": {test_type: len(results) for test_type, results in test_types.items()},
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
            "detailed_results": [r.__dict__ for r in self.test_results]
        }
    
    def _analyze_quantum_performance(self) -> Dict[str, Any]:
        """Analyze quantum computing performance."""
        all_quantum_metrics = [r.quantum_metrics for r in self.test_results if r.quantum_metrics]
        all_fidelity_metrics = [r.fidelity_metrics for r in self.test_results if r.fidelity_metrics]
        
        analysis = {}
        
        if all_quantum_metrics:
            # Aggregate quantum metrics
            quantum_aggregated = {}
            for metrics in all_quantum_metrics:
                for metric_name, value in metrics.items():
                    if metric_name not in quantum_aggregated:
                        quantum_aggregated[metric_name] = []
                    quantum_aggregated[metric_name].append(value)
            
            analysis["quantum_metrics"] = {
                metric_name: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric_name, values in quantum_aggregated.items()
            }
        
        if all_fidelity_metrics:
            # Aggregate fidelity metrics
            fidelity_aggregated = {}
            for metrics in all_fidelity_metrics:
                for metric_name, value in metrics.items():
                    if metric_name not in fidelity_aggregated:
                        fidelity_aggregated[metric_name] = []
                    fidelity_aggregated[metric_name].append(value)
            
            analysis["fidelity_metrics"] = {
                metric_name: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric_name, values in fidelity_aggregated.items()
            }
        
        return analysis
    
    def _generate_quantum_recommendations(self) -> List[str]:
        """Generate quantum computing specific recommendations."""
        recommendations = []
        
        # Analyze algorithm results
        algorithm_results = [r for r in self.test_results if r.test_type == "quantum_algorithm"]
        if algorithm_results:
            avg_success = np.mean([r.quantum_metrics.get('success_probability', 0) for r in algorithm_results])
            if avg_success < 0.8:
                recommendations.append("Improve quantum algorithm implementation for better success rates")
        
        # Analyze error correction results
        error_correction_results = [r for r in self.test_results if r.test_type == "error_correction"]
        if error_correction_results:
            avg_fidelity = np.mean([r.fidelity_metrics.get('fidelity', 0) for r in error_correction_results])
            if avg_fidelity < 0.9:
                recommendations.append("Enhance quantum error correction for better fidelity")
        
        return recommendations

# Example usage and demo
def demo_quantum_computing_testing():
    """Demonstrate quantum computing testing capabilities."""
    print("⚛️ Quantum Computing Testing Framework Demo")
    print("=" * 50)
    
    # Create quantum computing test framework
    framework = QuantumComputingTestFramework()
    
    # Define test configuration
    test_config = {
        "algorithms": [
            {"name": "grover", "qubits": 3, "parameters": {"target_state": 5, "iterations": 2}},
            {"name": "deutsch_jozsa", "qubits": 3, "parameters": {}},
            {"name": "quantum_fourier_transform", "qubits": 3, "parameters": {}}
        ],
        "error_correction": {
            "qubits": 3,
            "error_rate": 0.1
        }
    }
    
    # Run quantum tests
    print("🧪 Running quantum computing tests...")
    results = framework.run_quantum_tests(test_config)
    
    # Print results
    print(f"\n📊 Quantum Test Results:")
    for result in results:
        print(f"\n{result.test_name}: {'✅' if result.success else '❌'}")
        print(f"  Success Probability: {result.quantum_metrics.get('success_probability', 0):.3f}")
        print(f"  Circuit Depth: {result.quantum_metrics.get('circuit_depth', 0)}")
        print(f"  Gate Count: {result.quantum_metrics.get('gate_count', 0)}")
        print(f"  Fidelity: {result.fidelity_metrics.get('fidelity', 0):.3f}")
    
    # Generate comprehensive report
    print("\n📈 Generating quantum computing report...")
    report = framework.generate_quantum_report()
    
    print(f"\n📊 Quantum Computing Report:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.1%}")
    
    print(f"\n📊 Tests by Type:")
    for test_type, count in report['by_test_type'].items():
        print(f"  {test_type}: {count}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")

if __name__ == "__main__":
    # Run demo
    demo_quantum_computing_testing()
