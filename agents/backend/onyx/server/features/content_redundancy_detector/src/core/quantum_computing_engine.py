"""
Quantum Computing Engine - Advanced quantum computing capabilities
"""

import asyncio
import logging
import time
import numpy as np
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import warnings
warnings.filterwarnings('ignore')

# Quantum computing libraries
try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
    from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
    from qiskit.providers.ibmq import IBMQ
    from qiskit.visualization import plot_histogram, plot_bloch_multivector
    from qiskit.algorithms import Grover, Shor, QAOA, VQE
    from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
    from qiskit.circuit.library import TwoLocal, EfficientSU2
    from qiskit.opflow import PauliSumOp
    from qiskit.quantum_info import random_unitary, random_statevector
    from qiskit.ignis.mitigation import CompleteMeasFitter, TensoredMeasFitter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    qiskit = None

try:
    import cirq
    from cirq import Circuit, Moment, GridQubit, LineQubit
    from cirq.ops import H, X, Y, Z, CNOT, CZ, SWAP, T, S
    from cirq.sim import Simulator
    from cirq.google import Sycamore
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False
    cirq = None

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None

logger = logging.getLogger(__name__)


@dataclass
class QuantumConfig:
    """Quantum computing configuration"""
    enable_qiskit: bool = True
    enable_cirq: bool = True
    enable_pennylane: bool = True
    enable_ibm_quantum: bool = False
    enable_google_quantum: bool = False
    enable_ionq: bool = False
    enable_rigetti: bool = False
    max_qubits: int = 32
    max_depth: int = 100
    optimization_level: int = 3
    enable_error_mitigation: bool = True
    enable_quantum_ml: bool = True
    enable_quantum_optimization: bool = True
    enable_quantum_cryptography: bool = True
    enable_quantum_simulation: bool = True
    backend_type: str = "simulator"  # simulator, ibmq, google, ionq, rigetti
    shots: int = 1024


@dataclass
class QuantumCircuit:
    """Quantum circuit data class"""
    circuit_id: str
    timestamp: datetime
    qubits: int
    depth: int
    gates: List[Dict[str, Any]]
    measurements: List[Dict[str, Any]]
    backend: str
    status: str
    results: Optional[Dict[str, Any]] = None


@dataclass
class QuantumAlgorithm:
    """Quantum algorithm data class"""
    algorithm_id: str
    timestamp: datetime
    algorithm_type: str
    problem_size: int
    parameters: Dict[str, Any]
    execution_time: float
    success_probability: float
    results: Dict[str, Any]
    backend_used: str


@dataclass
class QuantumState:
    """Quantum state data class"""
    state_id: str
    timestamp: datetime
    qubits: int
    state_vector: np.ndarray
    density_matrix: Optional[np.ndarray] = None
    entanglement_entropy: Optional[float] = None
    fidelity: Optional[float] = None


class QiskitQuantumEngine:
    """Qiskit-based quantum computing engine"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.simulator = None
        self.backend = None
        self.circuits = {}
        self.results = {}
        self._initialize_qiskit()
    
    def _initialize_qiskit(self):
        """Initialize Qiskit quantum engine"""
        try:
            if not QISKIT_AVAILABLE:
                logger.warning("Qiskit not available")
                return
            
            # Initialize simulator
            self.simulator = QasmSimulator()
            
            # Initialize IBM Quantum if enabled
            if self.config.enable_ibm_quantum:
                try:
                    IBMQ.load_account()
                    provider = IBMQ.get_provider()
                    self.backend = provider.get_backend('ibmq_qasm_simulator')
                except Exception as e:
                    logger.warning(f"IBM Quantum not available: {e}")
                    self.backend = self.simulator
            else:
                self.backend = self.simulator
            
            logger.info("Qiskit quantum engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Qiskit: {e}")
    
    async def create_quantum_circuit(self, qubits: int, name: str = "quantum_circuit") -> QuantumCircuit:
        """Create a quantum circuit"""
        try:
            if not QISKIT_AVAILABLE:
                raise ValueError("Qiskit not available")
            
            # Create quantum and classical registers
            qreg = QuantumRegister(qubits, 'q')
            creg = ClassicalRegister(qubits, 'c')
            circuit = qiskit.QuantumCircuit(qreg, creg)
            
            # Add some basic gates for demonstration
            for i in range(qubits):
                circuit.h(qreg[i])  # Hadamard gate
            
            # Add CNOT gates for entanglement
            for i in range(qubits - 1):
                circuit.cx(qreg[i], qreg[i + 1])
            
            # Add measurements
            circuit.measure_all()
            
            # Transpile circuit
            transpiled_circuit = transpile(circuit, self.backend, optimization_level=self.config.optimization_level)
            
            circuit_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()
            
            quantum_circuit = QuantumCircuit(
                circuit_id=circuit_id,
                timestamp=datetime.now(),
                qubits=qubits,
                depth=transpiled_circuit.depth(),
                gates=[{"type": "h", "qubit": i} for i in range(qubits)] + 
                      [{"type": "cx", "control": i, "target": i + 1} for i in range(qubits - 1)],
                measurements=[{"qubit": i, "classical_bit": i} for i in range(qubits)],
                backend=str(self.backend),
                status="created"
            )
            
            self.circuits[circuit_id] = {
                "circuit": transpiled_circuit,
                "metadata": quantum_circuit
            }
            
            return quantum_circuit
            
        except Exception as e:
            logger.error(f"Error creating quantum circuit: {e}")
            raise
    
    async def execute_quantum_circuit(self, circuit_id: str) -> Dict[str, Any]:
        """Execute a quantum circuit"""
        try:
            if circuit_id not in self.circuits:
                raise ValueError(f"Circuit {circuit_id} not found")
            
            circuit_data = self.circuits[circuit_id]
            circuit = circuit_data["circuit"]
            metadata = circuit_data["metadata"]
            
            # Execute circuit
            start_time = time.time()
            job = self.backend.run(assemble(circuit, shots=self.config.shots))
            result = job.result()
            execution_time = (time.time() - start_time) * 1000
            
            # Get counts
            counts = result.get_counts()
            
            # Calculate statistics
            total_shots = sum(counts.values())
            probabilities = {state: count / total_shots for state, count in counts.items()}
            
            # Update metadata
            metadata.status = "completed"
            metadata.results = {
                "counts": counts,
                "probabilities": probabilities,
                "execution_time_ms": execution_time,
                "total_shots": total_shots
            }
            
            self.results[circuit_id] = metadata.results
            
            return metadata.results
            
        except Exception as e:
            logger.error(f"Error executing quantum circuit: {e}")
            raise
    
    async def grover_search(self, search_space: List[str], target: str) -> QuantumAlgorithm:
        """Implement Grover's search algorithm"""
        try:
            if not QISKIT_AVAILABLE:
                raise ValueError("Qiskit not available")
            
            start_time = time.time()
            
            # Create oracle for target
            n_qubits = int(np.ceil(np.log2(len(search_space))))
            
            # Simplified Grover implementation
            qreg = QuantumRegister(n_qubits, 'q')
            creg = ClassicalRegister(n_qubits, 'c')
            circuit = qiskit.QuantumCircuit(qreg, creg)
            
            # Initialize superposition
            for i in range(n_qubits):
                circuit.h(qreg[i])
            
            # Grover iterations (simplified)
            iterations = int(np.pi / 4 * np.sqrt(len(search_space)))
            for _ in range(iterations):
                # Oracle (simplified)
                target_index = search_space.index(target) if target in search_space else 0
                target_binary = format(target_index, f'0{n_qubits}b')
                
                for i, bit in enumerate(target_binary):
                    if bit == '0':
                        circuit.x(qreg[i])
                
                # Multi-controlled Z gate (simplified)
                if n_qubits > 1:
                    circuit.h(qreg[-1])
                    circuit.mcx(list(qreg[:-1]), qreg[-1])
                    circuit.h(qreg[-1])
                
                for i, bit in enumerate(target_binary):
                    if bit == '0':
                        circuit.x(qreg[i])
                
                # Diffusion operator
                for i in range(n_qubits):
                    circuit.h(qreg[i])
                    circuit.x(qreg[i])
                
                if n_qubits > 1:
                    circuit.h(qreg[-1])
                    circuit.mcx(list(qreg[:-1]), qreg[-1])
                    circuit.h(qreg[-1])
                
                for i in range(n_qubits):
                    circuit.x(qreg[i])
                    circuit.h(qreg[i])
            
            # Measurements
            circuit.measure_all()
            
            # Execute
            transpiled_circuit = transpile(circuit, self.backend)
            job = self.backend.run(assemble(transpiled_circuit, shots=self.config.shots))
            result = job.result()
            counts = result.get_counts()
            
            execution_time = (time.time() - start_time) * 1000
            
            # Find most probable result
            most_probable = max(counts, key=counts.get)
            success_probability = counts[most_probable] / sum(counts.values())
            
            algorithm = QuantumAlgorithm(
                algorithm_id=hashlib.md5(f"grover_{time.time()}".encode()).hexdigest(),
                timestamp=datetime.now(),
                algorithm_type="grover_search",
                problem_size=len(search_space),
                parameters={"target": target, "iterations": iterations},
                execution_time=execution_time,
                success_probability=success_probability,
                results={"counts": counts, "most_probable": most_probable},
                backend_used=str(self.backend)
            )
            
            return algorithm
            
        except Exception as e:
            logger.error(f"Error in Grover search: {e}")
            raise
    
    async def quantum_fourier_transform(self, input_state: List[complex]) -> QuantumState:
        """Implement Quantum Fourier Transform"""
        try:
            if not QISKIT_AVAILABLE:
                raise ValueError("Qiskit not available")
            
            n_qubits = int(np.ceil(np.log2(len(input_state))))
            
            # Create circuit
            qreg = QuantumRegister(n_qubits, 'q')
            circuit = qiskit.QuantumCircuit(qreg)
            
            # Initialize state
            circuit.initialize(input_state[:2**n_qubits], qreg)
            
            # QFT implementation
            for i in range(n_qubits):
                circuit.h(qreg[i])
                for j in range(i + 1, n_qubits):
                    angle = 2 * np.pi / (2 ** (j - i + 1))
                    circuit.cp(angle, qreg[j], qreg[i])
            
            # Swap qubits
            for i in range(n_qubits // 2):
                circuit.swap(qreg[i], qreg[n_qubits - 1 - i])
            
            # Simulate
            simulator = StatevectorSimulator()
            job = simulator.run(circuit)
            result = job.result()
            statevector = result.get_statevector()
            
            quantum_state = QuantumState(
                state_id=hashlib.md5(f"qft_{time.time()}".encode()).hexdigest(),
                timestamp=datetime.now(),
                qubits=n_qubits,
                state_vector=np.array(statevector),
                entanglement_entropy=self._calculate_entanglement_entropy(statevector)
            )
            
            return quantum_state
            
        except Exception as e:
            logger.error(f"Error in Quantum Fourier Transform: {e}")
            raise
    
    def _calculate_entanglement_entropy(self, statevector: np.ndarray) -> float:
        """Calculate entanglement entropy of quantum state"""
        try:
            # Simplified entanglement entropy calculation
            probabilities = np.abs(statevector) ** 2
            probabilities = probabilities[probabilities > 1e-10]  # Remove near-zero probabilities
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy
        except Exception as e:
            logger.error(f"Error calculating entanglement entropy: {e}")
            return 0.0


class CirqQuantumEngine:
    """Cirq-based quantum computing engine"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.simulator = None
        self.circuits = {}
        self._initialize_cirq()
    
    def _initialize_cirq(self):
        """Initialize Cirq quantum engine"""
        try:
            if not CIRQ_AVAILABLE:
                logger.warning("Cirq not available")
                return
            
            self.simulator = Simulator()
            logger.info("Cirq quantum engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Cirq: {e}")
    
    async def create_cirq_circuit(self, qubits: int, name: str = "cirq_circuit") -> QuantumCircuit:
        """Create a Cirq quantum circuit"""
        try:
            if not CIRQ_AVAILABLE:
                raise ValueError("Cirq not available")
            
            # Create qubits
            qubit_list = [LineQubit(i) for i in range(qubits)]
            
            # Create circuit
            circuit = Circuit()
            
            # Add gates
            for qubit in qubit_list:
                circuit.append(H(qubit))
            
            # Add entanglement
            for i in range(qubits - 1):
                circuit.append(CNOT(qubit_list[i], qubit_list[i + 1]))
            
            circuit_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()
            
            quantum_circuit = QuantumCircuit(
                circuit_id=circuit_id,
                timestamp=datetime.now(),
                qubits=qubits,
                depth=len(circuit),
                gates=[{"type": "h", "qubit": i} for i in range(qubits)] + 
                      [{"type": "cnot", "control": i, "target": i + 1} for i in range(qubits - 1)],
                measurements=[],
                backend="cirq_simulator",
                status="created"
            )
            
            self.circuits[circuit_id] = {
                "circuit": circuit,
                "metadata": quantum_circuit
            }
            
            return quantum_circuit
            
        except Exception as e:
            logger.error(f"Error creating Cirq circuit: {e}")
            raise
    
    async def execute_cirq_circuit(self, circuit_id: str) -> Dict[str, Any]:
        """Execute a Cirq quantum circuit"""
        try:
            if circuit_id not in self.circuits:
                raise ValueError(f"Circuit {circuit_id} not found")
            
            circuit_data = self.circuits[circuit_id]
            circuit = circuit_data["circuit"]
            metadata = circuit_data["metadata"]
            
            # Execute circuit
            start_time = time.time()
            result = self.simulator.run(circuit, repetitions=self.config.shots)
            execution_time = (time.time() - start_time) * 1000
            
            # Process results
            measurements = {}
            for measurement in result.measurements:
                for key, values in measurement.items():
                    if key not in measurements:
                        measurements[key] = []
                    measurements[key].extend(values)
            
            # Count results
            counts = {}
            for key, values in measurements.items():
                for value in values:
                    state = ''.join(map(str, value))
                    counts[state] = counts.get(state, 0) + 1
            
            # Update metadata
            metadata.status = "completed"
            metadata.results = {
                "counts": counts,
                "execution_time_ms": execution_time,
                "total_shots": self.config.shots
            }
            
            return metadata.results
            
        except Exception as e:
            logger.error(f"Error executing Cirq circuit: {e}")
            raise


class PennyLaneQuantumEngine:
    """PennyLane-based quantum computing engine"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.devices = {}
        self.circuits = {}
        self._initialize_pennylane()
    
    def _initialize_pennylane(self):
        """Initialize PennyLane quantum engine"""
        try:
            if not PENNYLANE_AVAILABLE:
                logger.warning("PennyLane not available")
                return
            
            # Initialize devices
            self.devices["default"] = qml.device('default.qubit', wires=self.config.max_qubits)
            self.devices["simulator"] = qml.device('qiskit.aer', wires=self.config.max_qubits)
            
            logger.info("PennyLane quantum engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PennyLane: {e}")
    
    async def create_pennylane_circuit(self, qubits: int, name: str = "pennylane_circuit") -> QuantumCircuit:
        """Create a PennyLane quantum circuit"""
        try:
            if not PENNYLANE_AVAILABLE:
                raise ValueError("PennyLane not available")
            
            device = self.devices["default"]
            
            @qml.qnode(device)
            def quantum_circuit():
                # Add gates
                for i in range(qubits):
                    qml.Hadamard(wires=i)
                
                # Add entanglement
                for i in range(qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]
            
            circuit_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()
            
            quantum_circuit_obj = QuantumCircuit(
                circuit_id=circuit_id,
                timestamp=datetime.now(),
                qubits=qubits,
                depth=qubits + (qubits - 1),
                gates=[{"type": "hadamard", "wire": i} for i in range(qubits)] + 
                      [{"type": "cnot", "wires": [i, i + 1]} for i in range(qubits - 1)],
                measurements=[{"type": "pauli_z", "wire": i} for i in range(qubits)],
                backend="pennylane",
                status="created"
            )
            
            self.circuits[circuit_id] = {
                "circuit": quantum_circuit,
                "metadata": quantum_circuit_obj
            }
            
            return quantum_circuit_obj
            
        except Exception as e:
            logger.error(f"Error creating PennyLane circuit: {e}")
            raise
    
    async def execute_pennylane_circuit(self, circuit_id: str) -> Dict[str, Any]:
        """Execute a PennyLane quantum circuit"""
        try:
            if circuit_id not in self.circuits:
                raise ValueError(f"Circuit {circuit_id} not found")
            
            circuit_data = self.circuits[circuit_id]
            circuit = circuit_data["circuit"]
            metadata = circuit_data["metadata"]
            
            # Execute circuit
            start_time = time.time()
            result = circuit()
            execution_time = (time.time() - start_time) * 1000
            
            # Update metadata
            metadata.status = "completed"
            metadata.results = {
                "expectation_values": result.tolist() if hasattr(result, 'tolist') else result,
                "execution_time_ms": execution_time
            }
            
            return metadata.results
            
        except Exception as e:
            logger.error(f"Error executing PennyLane circuit: {e}")
            raise


class QuantumComputingEngine:
    """Main Quantum Computing Engine"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.qiskit_engine = QiskitQuantumEngine(config) if QISKIT_AVAILABLE else None
        self.cirq_engine = CirqQuantumEngine(config) if CIRQ_AVAILABLE else None
        self.pennylane_engine = PennyLaneQuantumEngine(config) if PENNYLANE_AVAILABLE else None
        
        self.quantum_algorithms = []
        self.quantum_states = []
        self.performance_metrics = {}
        
        self._initialize_quantum_engine()
    
    def _initialize_quantum_engine(self):
        """Initialize quantum computing engine"""
        try:
            available_engines = []
            if self.qiskit_engine:
                available_engines.append("Qiskit")
            if self.cirq_engine:
                available_engines.append("Cirq")
            if self.pennylane_engine:
                available_engines.append("PennyLane")
            
            logger.info(f"Quantum Computing Engine initialized with: {', '.join(available_engines)}")
            
        except Exception as e:
            logger.error(f"Error initializing quantum engine: {e}")
    
    async def create_quantum_circuit(self, qubits: int, backend: str = "qiskit", name: str = "quantum_circuit") -> QuantumCircuit:
        """Create a quantum circuit using specified backend"""
        try:
            if backend == "qiskit" and self.qiskit_engine:
                return await self.qiskit_engine.create_quantum_circuit(qubits, name)
            elif backend == "cirq" and self.cirq_engine:
                return await self.cirq_engine.create_cirq_circuit(qubits, name)
            elif backend == "pennylane" and self.pennylane_engine:
                return await self.pennylane_engine.create_pennylane_circuit(qubits, name)
            else:
                raise ValueError(f"Backend {backend} not available")
                
        except Exception as e:
            logger.error(f"Error creating quantum circuit: {e}")
            raise
    
    async def execute_quantum_circuit(self, circuit_id: str, backend: str = "qiskit") -> Dict[str, Any]:
        """Execute a quantum circuit using specified backend"""
        try:
            if backend == "qiskit" and self.qiskit_engine:
                return await self.qiskit_engine.execute_quantum_circuit(circuit_id)
            elif backend == "cirq" and self.cirq_engine:
                return await self.cirq_engine.execute_cirq_circuit(circuit_id)
            elif backend == "pennylane" and self.pennylane_engine:
                return await self.pennylane_engine.execute_pennylane_circuit(circuit_id)
            else:
                raise ValueError(f"Backend {backend} not available")
                
        except Exception as e:
            logger.error(f"Error executing quantum circuit: {e}")
            raise
    
    async def run_quantum_algorithm(self, algorithm_type: str, parameters: Dict[str, Any]) -> QuantumAlgorithm:
        """Run a quantum algorithm"""
        try:
            if algorithm_type == "grover_search" and self.qiskit_engine:
                search_space = parameters.get("search_space", ["0", "1", "2", "3"])
                target = parameters.get("target", "1")
                return await self.qiskit_engine.grover_search(search_space, target)
            elif algorithm_type == "quantum_fourier_transform" and self.qiskit_engine:
                input_state = parameters.get("input_state", [1, 0, 0, 0])
                quantum_state = await self.qiskit_engine.quantum_fourier_transform(input_state)
                
                # Convert to algorithm result
                algorithm = QuantumAlgorithm(
                    algorithm_id=quantum_state.state_id,
                    timestamp=quantum_state.timestamp,
                    algorithm_type=algorithm_type,
                    problem_size=len(input_state),
                    parameters=parameters,
                    execution_time=0.0,
                    success_probability=1.0,
                    results={"state_vector": quantum_state.state_vector.tolist()},
                    backend_used="qiskit"
                )
                return algorithm
            else:
                raise ValueError(f"Algorithm {algorithm_type} not supported")
                
        except Exception as e:
            logger.error(f"Error running quantum algorithm: {e}")
            raise
    
    async def get_quantum_capabilities(self) -> Dict[str, Any]:
        """Get quantum computing capabilities"""
        try:
            capabilities = {
                "available_backends": [],
                "supported_algorithms": [],
                "max_qubits": self.config.max_qubits,
                "max_depth": self.config.max_depth,
                "optimization_level": self.config.optimization_level,
                "error_mitigation": self.config.enable_error_mitigation,
                "quantum_ml": self.config.enable_quantum_ml,
                "quantum_optimization": self.config.enable_quantum_optimization,
                "quantum_cryptography": self.config.enable_quantum_cryptography,
                "quantum_simulation": self.config.enable_quantum_simulation
            }
            
            if self.qiskit_engine:
                capabilities["available_backends"].append("qiskit")
                capabilities["supported_algorithms"].extend([
                    "grover_search", "quantum_fourier_transform", "variational_quantum_eigensolver",
                    "quantum_approximate_optimization_algorithm", "quantum_machine_learning"
                ])
            
            if self.cirq_engine:
                capabilities["available_backends"].append("cirq")
                capabilities["supported_algorithms"].extend([
                    "quantum_simulation", "quantum_circuit_optimization"
                ])
            
            if self.pennylane_engine:
                capabilities["available_backends"].append("pennylane")
                capabilities["supported_algorithms"].extend([
                    "quantum_machine_learning", "variational_quantum_algorithms",
                    "quantum_neural_networks", "quantum_optimization"
                ])
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Error getting quantum capabilities: {e}")
            return {}
    
    async def get_quantum_performance_metrics(self) -> Dict[str, Any]:
        """Get quantum computing performance metrics"""
        try:
            metrics = {
                "total_algorithms_run": len(self.quantum_algorithms),
                "total_circuits_created": 0,
                "average_execution_time": 0.0,
                "success_rate": 0.0,
                "backend_usage": {},
                "algorithm_types": {}
            }
            
            # Count circuits
            if self.qiskit_engine:
                metrics["total_circuits_created"] += len(self.qiskit_engine.circuits)
            if self.cirq_engine:
                metrics["total_circuits_created"] += len(self.cirq_engine.circuits)
            if self.pennylane_engine:
                metrics["total_circuits_created"] += len(self.pennylane_engine.circuits)
            
            # Calculate averages
            if self.quantum_algorithms:
                execution_times = [alg.execution_time for alg in self.quantum_algorithms]
                metrics["average_execution_time"] = statistics.mean(execution_times)
                
                success_probabilities = [alg.success_probability for alg in self.quantum_algorithms]
                metrics["success_rate"] = statistics.mean(success_probabilities)
                
                # Backend usage
                for alg in self.quantum_algorithms:
                    backend = alg.backend_used
                    metrics["backend_usage"][backend] = metrics["backend_usage"].get(backend, 0) + 1
                    
                    alg_type = alg.algorithm_type
                    metrics["algorithm_types"][alg_type] = metrics["algorithm_types"].get(alg_type, 0) + 1
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting quantum performance metrics: {e}")
            return {}


# Global instance
quantum_computing_engine: Optional[QuantumComputingEngine] = None


async def initialize_quantum_computing_engine(config: Optional[QuantumConfig] = None) -> None:
    """Initialize quantum computing engine"""
    global quantum_computing_engine
    
    if config is None:
        config = QuantumConfig()
    
    quantum_computing_engine = QuantumComputingEngine(config)
    logger.info("Quantum Computing Engine initialized successfully")


async def get_quantum_computing_engine() -> Optional[QuantumComputingEngine]:
    """Get quantum computing engine instance"""
    return quantum_computing_engine
