"""
Quantum Types and Definitions
=============================

Type definitions for quantum computing and post-quantum cryptography.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import uuid
import numpy as np

class QuantumGateType(Enum):
    """Quantum gate types."""
    PAULI_X = "pauli_x"
    PAULI_Y = "pauli_y"
    PAULI_Z = "pauli_z"
    HADAMARD = "hadamard"
    CNOT = "cnot"
    TOFFOLI = "toffoli"
    PHASE = "phase"
    ROTATION_X = "rotation_x"
    ROTATION_Y = "rotation_y"
    ROTATION_Z = "rotation_z"
    SWAP = "swap"
    MEASUREMENT = "measurement"

class QuantumAlgorithmType(Enum):
    """Quantum algorithm types."""
    GROVER_SEARCH = "grover_search"
    SHOR_FACTORIZATION = "shor_factorization"
    QUANTUM_FOURIER_TRANSFORM = "quantum_fourier_transform"
    VARIATIONAL_QUANTUM_EIGENSOLVER = "vqe"
    QUANTUM_APPROXIMATE_OPTIMIZATION = "qaoa"
    QUANTUM_MACHINE_LEARNING = "qml"
    QUANTUM_WALK = "quantum_walk"
    TELEPORTATION = "teleportation"

class PostQuantumAlgorithm(Enum):
    """Post-quantum cryptographic algorithms."""
    LATTICE_BASED = "lattice_based"
    CODE_BASED = "code_based"
    MULTIVARIATE = "multivariate"
    HASH_BASED = "hash_based"
    ISOGENY_BASED = "isogeny_based"
    NTRU = "ntru"
    KYBER = "kyber"
    DILITHIUM = "dilithium"
    FALCON = "falcon"
    SPHINCS = "sphincs"

class QuantumState(Enum):
    """Quantum state types."""
    ZERO = "|0⟩"
    ONE = "|1⟩"
    PLUS = "|+⟩"
    MINUS = "|-⟩"
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    MIXED = "mixed"

@dataclass
class QuantumBit:
    """Quantum bit (qubit) representation."""
    id: str
    state: QuantumState = QuantumState.ZERO
    amplitude_0: complex = 1.0
    amplitude_1: complex = 0.0
    phase: float = 0.0
    coherence_time: float = 0.0  # microseconds
    error_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def __init__(self, qubit_id: str = None):
        self.id = qubit_id or str(uuid.uuid4())
        self.state = QuantumState.ZERO
        self.amplitude_0 = 1.0
        self.amplitude_1 = 0.0
        self.phase = 0.0
        self.coherence_time = 0.0
        self.error_rate = 0.0
        self.created_at = datetime.now()
    
    def get_state_vector(self) -> np.ndarray:
        """Get quantum state vector."""
        return np.array([self.amplitude_0, self.amplitude_1])
    
    def measure(self) -> int:
        """Measure the qubit."""
        prob_0 = abs(self.amplitude_0) ** 2
        prob_1 = abs(self.amplitude_1) ** 2
        
        # Normalize probabilities
        total_prob = prob_0 + prob_1
        if total_prob > 0:
            prob_0 /= total_prob
            prob_1 /= total_prob
        
        # Simulate measurement
        import random
        return 0 if random.random() < prob_0 else 1
    
    def apply_gate(self, gate_type: QuantumGateType, parameters: Dict[str, Any] = None):
        """Apply quantum gate to qubit."""
        if gate_type == QuantumGateType.PAULI_X:
            # X gate: |0⟩ ↔ |1⟩
            self.amplitude_0, self.amplitude_1 = self.amplitude_1, self.amplitude_0
        elif gate_type == QuantumGateType.HADAMARD:
            # H gate: creates superposition
            new_amp_0 = (self.amplitude_0 + self.amplitude_1) / np.sqrt(2)
            new_amp_1 = (self.amplitude_0 - self.amplitude_1) / np.sqrt(2)
            self.amplitude_0 = new_amp_0
            self.amplitude_1 = new_amp_1
        elif gate_type == QuantumGateType.PHASE:
            # Phase gate
            angle = parameters.get("angle", 0) if parameters else 0
            self.amplitude_1 *= np.exp(1j * angle)

@dataclass
class QuantumRegister:
    """Quantum register containing multiple qubits."""
    id: str
    qubits: List[QuantumBit] = field(default_factory=list)
    size: int = 0
    entangled_pairs: List[Tuple[int, int]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __init__(self, register_id: str = None, size: int = 1):
        self.id = register_id or str(uuid.uuid4())
        self.size = size
        self.qubits = [QuantumBit() for _ in range(size)]
        self.entangled_pairs = []
        self.created_at = datetime.now()
    
    def get_qubit(self, index: int) -> Optional[QuantumBit]:
        """Get qubit by index."""
        if 0 <= index < len(self.qubits):
            return self.qubits[index]
        return None
    
    def entangle_qubits(self, qubit1_index: int, qubit2_index: int):
        """Create entanglement between two qubits."""
        if (0 <= qubit1_index < len(self.qubits) and 
            0 <= qubit2_index < len(self.qubits)):
            self.entangled_pairs.append((qubit1_index, qubit2_index))
    
    def measure_all(self) -> List[int]:
        """Measure all qubits in the register."""
        return [qubit.measure() for qubit in self.qubits]

@dataclass
class QuantumGate:
    """Quantum gate definition."""
    id: str
    gate_type: QuantumGateType
    qubits: List[int] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    matrix: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __init__(self, gate_type: QuantumGateType, qubits: List[int] = None, parameters: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.gate_type = gate_type
        self.qubits = qubits or [0]
        self.parameters = parameters or {}
        self.matrix = self._generate_matrix()
        self.created_at = datetime.now()
    
    def _generate_matrix(self) -> np.ndarray:
        """Generate gate matrix."""
        if self.gate_type == QuantumGateType.PAULI_X:
            return np.array([[0, 1], [1, 0]], dtype=complex)
        elif self.gate_type == QuantumGateType.PAULI_Y:
            return np.array([[0, -1j], [1j, 0]], dtype=complex)
        elif self.gate_type == QuantumGateType.PAULI_Z:
            return np.array([[1, 0], [0, -1]], dtype=complex)
        elif self.gate_type == QuantumGateType.HADAMARD:
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        elif self.gate_type == QuantumGateType.CNOT:
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        else:
            return np.eye(2, dtype=complex)

@dataclass
class QuantumCircuit:
    """Quantum circuit definition."""
    id: str
    name: str
    description: str
    qubits: int
    gates: List[QuantumGate] = field(default_factory=list)
    measurements: List[int] = field(default_factory=list)
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_gate(self, gate: QuantumGate):
        """Add gate to circuit."""
        self.gates.append(gate)
        self.depth = max(self.depth, len(self.gates))
    
    def add_measurement(self, qubit_index: int):
        """Add measurement to circuit."""
        if qubit_index not in self.measurements:
            self.measurements.append(qubit_index)
    
    def get_circuit_matrix(self) -> np.ndarray:
        """Get overall circuit matrix."""
        if not self.gates:
            return np.eye(2 ** self.qubits, dtype=complex)
        
        # Start with identity matrix
        circuit_matrix = np.eye(2 ** self.qubits, dtype=complex)
        
        # Apply each gate
        for gate in self.gates:
            if gate.matrix is not None:
                # Tensor product with identity for other qubits
                full_matrix = self._expand_gate_matrix(gate)
                circuit_matrix = full_matrix @ circuit_matrix
        
        return circuit_matrix
    
    def _expand_gate_matrix(self, gate: QuantumGate) -> np.ndarray:
        """Expand gate matrix to full circuit size."""
        # Simplified implementation
        return gate.matrix

@dataclass
class QuantumKey:
    """Quantum key for encryption."""
    id: str
    key_data: bytes
    algorithm: PostQuantumAlgorithm
    key_size: int
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    
    def is_valid(self) -> bool:
        """Check if key is valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        if self.max_usage and self.usage_count >= self.max_usage:
            return False
        return True
    
    def use_key(self):
        """Mark key as used."""
        self.usage_count += 1

@dataclass
class QuantumAlgorithm:
    """Quantum algorithm definition."""
    id: str
    name: str
    algorithm_type: QuantumAlgorithmType
    description: str
    circuit: QuantumCircuit
    parameters: Dict[str, Any] = field(default_factory=dict)
    complexity: str = "unknown"  # polynomial, exponential, etc.
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class QuantumSimulation:
    """Quantum simulation result."""
    algorithm_id: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    quantum_advantage: bool = False
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PostQuantumKeyPair:
    """Post-quantum cryptographic key pair."""
    id: str
    algorithm: PostQuantumAlgorithm
    public_key: bytes
    private_key: bytes
    key_size: int
    security_level: int  # 128, 192, 256 bits
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

@dataclass
class QuantumCryptographicOperation:
    """Quantum cryptographic operation."""
    id: str
    operation_type: str  # encrypt, decrypt, sign, verify, key_exchange
    algorithm: PostQuantumAlgorithm
    input_data: bytes
    output_data: Optional[bytes] = None
    key_id: str
    success: bool = False
    execution_time: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class QuantumSecurityMetrics:
    """Quantum security metrics."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_execution_time: float = 0.0
    key_usage_count: int = 0
    key_rotation_count: int = 0
    quantum_advantage_detected: int = 0
    post_quantum_algorithm_usage: Dict[str, int] = field(default_factory=dict)
    security_level_distribution: Dict[int, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
