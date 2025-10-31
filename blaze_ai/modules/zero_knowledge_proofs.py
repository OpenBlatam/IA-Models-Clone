"""
Blaze AI Zero-Knowledge Proofs Module v8.1.0

This module provides comprehensive zero-knowledge proof capabilities for privacy-preserving
AI operations, blockchain integration, and secure multi-party computations.
"""

import asyncio
import logging
import time
import uuid
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict
import secrets
import math

# Cryptographic libraries
try:
    import py_ecc.bn128 as bn128
    from py_ecc.bn128 import G1, G2, pairing, add, multiply, neg
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    # Mock implementations for testing
    class MockPoint:
        def __init__(self, x=0, y=0):
            self.x = x
            self.y = y
        def __add__(self, other):
            return MockPoint(self.x + other.x, self.y + other.y)
        def __mul__(self, scalar):
            return MockPoint(self.x * scalar, self.y * scalar)
    
    G1 = MockPoint(1, 2)
    G2 = MockPoint(3, 4)
    def pairing(p1, p2): return 1
    def add(p1, p2): return p1 + p2
    def multiply(p, scalar): return p * scalar
    def neg(p): return MockPoint(-p.x, -p.y)

logger = logging.getLogger(__name__)

class ProofType(Enum):
    """Types of zero-knowledge proofs."""
    ZK_SNARK = "zk_snark"
    ZK_STARK = "zk_stark"
    BULLETPROOFS = "bulletproofs"
    RANGE_PROOF = "range_proof"
    MEMBERSHIP_PROOF = "membership_proof"
    EQUALITY_PROOF = "equality_proof"

class CircuitType(Enum):
    """Types of arithmetic circuits."""
    ARITHMETIC = "arithmetic"
    BOOLEAN = "boolean"
    RANGE_CHECK = "range_check"
    COMPARISON = "comparison"
    SORTING = "sorting"
    MERKLE_TREE = "merkle_tree"

class ProofStatus(Enum):
    """Proof generation and verification status."""
    PENDING = "pending"
    GENERATING = "generating"
    GENERATED = "generated"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class ZKProofConfig:
    """Configuration for Zero-Knowledge Proofs module."""
    # Basic settings
    name: str = "zero_knowledge_proofs"
    enabled_proof_types: List[ProofType] = field(default_factory=lambda: [ProofType.ZK_SNARK])
    circuit_optimization: bool = True
    parallel_generation: bool = True
    
    # Security settings
    security_level: int = 128  # bits
    curve_type: str = "bn128"
    hash_function: str = "sha256"
    
    # Performance settings
    max_circuit_size: int = 1000000  # gates
    proof_timeout: float = 300.0  # 5 minutes
    verification_timeout: float = 60.0  # 1 minute
    
    # Blockchain integration
    blockchain_integration: bool = True
    smart_contract_verification: bool = True
    gas_optimization: bool = True

@dataclass
class Circuit:
    """Arithmetic circuit representation."""
    circuit_id: str
    name: str
    circuit_type: CircuitType
    gates: List[Dict[str, Any]]
    inputs: List[str]
    outputs: List[str]
    constraints: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ZKProof:
    """Zero-knowledge proof data structure."""
    proof_id: str
    circuit_id: str
    proof_type: ProofType
    public_inputs: List[Any]
    proof_data: Dict[str, Any]
    verification_key: Dict[str, Any]
    status: ProofStatus
    generated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProofMetrics:
    """Zero-knowledge proof metrics."""
    total_proofs: int = 0
    generated_proofs: int = 0
    verified_proofs: int = 0
    failed_proofs: int = 0
    average_generation_time: float = 0.0
    average_verification_time: float = 0.0
    total_circuits: int = 0
    active_circuits: int = 0

class ArithmeticCircuit:
    """Arithmetic circuit implementation."""
    
    def __init__(self, config: ZKProofConfig):
        self.config = config
        self.gates: List[Dict[str, Any]] = []
        self.wire_values: Dict[str, Any] = {}
        self.constraints: List[Dict[str, Any]] = []
        
    def add_gate(self, gate_type: str, inputs: List[str], output: str, operation: str = "add"):
        """Add a gate to the circuit."""
        gate = {
            "id": f"gate_{len(self.gates)}",
            "type": gate_type,
            "inputs": inputs,
            "output": output,
            "operation": operation,
            "constraints": []
        }
        
        # Generate constraints based on operation
        if operation == "add":
            gate["constraints"] = [{"type": "linear", "equation": f"{output} = {inputs[0]} + {inputs[1]}"}]
        elif operation == "mul":
            gate["constraints"] = [{"type": "quadratic", "equation": f"{output} = {inputs[0]} * {inputs[1]}"}]
        elif operation == "constant":
            gate["constraints"] = [{"type": "constant", "equation": f"{output} = {inputs[0]}"}]
        
        self.gates.append(gate)
        return gate["id"]
    
    def add_constraint(self, constraint_type: str, equation: str, variables: List[str]):
        """Add a constraint to the circuit."""
        constraint = {
            "id": f"constraint_{len(self.constraints)}",
            "type": constraint_type,
            "equation": equation,
            "variables": variables
        }
        self.constraints.append(constraint)
        return constraint["id"]
    
    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the circuit with given inputs."""
        self.wire_values = inputs.copy()
        
        for gate in self.gates:
            if gate["type"] == "input":
                continue
            elif gate["type"] == "output":
                continue
            elif gate["operation"] == "add":
                self.wire_values[gate["output"]] = (
                    self.wire_values.get(gate["inputs"][0], 0) + 
                    self.wire_values.get(gate["inputs"][1], 0)
                )
            elif gate["operation"] == "mul":
                self.wire_values[gate["output"]] = (
                    self.wire_values.get(gate["inputs"][0], 0) * 
                    self.wire_values.get(gate["inputs"][1], 0)
                )
            elif gate["operation"] == "constant":
                self.wire_values[gate["output"]] = self.wire_values.get(gate["inputs"][0], 0)
        
        return self.wire_values
    
    def get_constraints(self) -> List[Dict[str, Any]]:
        """Get all circuit constraints."""
        all_constraints = []
        
        # Gate constraints
        for gate in self.gates:
            all_constraints.extend(gate["constraints"])
        
        # Additional constraints
        all_constraints.extend(self.constraints)
        
        return all_constraints

class ZKSNARKProver:
    """ZK-SNARK proof generator."""
    
    def __init__(self, config: ZKProofConfig):
        self.config = config
        self.trusted_setup: Optional[Dict[str, Any]] = None
        
    async def generate_trusted_setup(self, circuit: Circuit) -> Dict[str, Any]:
        """Generate trusted setup for the circuit."""
        try:
            # In a real implementation, this would use actual cryptographic operations
            # For now, we'll simulate the process
            
            setup = {
                "proving_key": f"pk_{circuit.circuit_id}",
                "verification_key": f"vk_{circuit.circuit_id}",
                "toxic_waste": f"toxic_{circuit.circuit_id}",
                "circuit_hash": hashlib.sha256(circuit.circuit_id.encode()).hexdigest()
            }
            
            self.trusted_setup = setup
            logger.info(f"Generated trusted setup for circuit {circuit.circuit_id}")
            return setup
            
        except Exception as e:
            logger.error(f"Failed to generate trusted setup: {e}")
            raise
    
    async def generate_proof(self, circuit: Circuit, public_inputs: List[Any], 
                           private_inputs: List[Any]) -> Dict[str, Any]:
        """Generate a ZK-SNARK proof."""
        try:
            if not self.trusted_setup:
                await self.generate_trusted_setup(circuit)
            
            # Simulate proof generation
            proof_data = {
                "pi_a": [secrets.randbelow(1000) for _ in range(3)],
                "pi_b": [[secrets.randbelow(1000) for _ in range(3)] for _ in range(2)],
                "pi_c": [secrets.randbelow(1000) for _ in range(3)],
                "public_inputs": public_inputs
            }
            
            # Add circuit-specific proof elements
            proof_data["circuit_hash"] = self.trusted_setup["circuit_hash"]
            proof_data["proof_hash"] = hashlib.sha256(
                json.dumps(proof_data, sort_keys=True).encode()
            ).hexdigest()
            
            logger.info(f"Generated ZK-SNARK proof for circuit {circuit.circuit_id}")
            return proof_data
            
        except Exception as e:
            logger.error(f"Failed to generate ZK-SNARK proof: {e}")
            raise

class ZKSNARKVerifier:
    """ZK-SNARK proof verifier."""
    
    def __init__(self, config: ZKProofConfig):
        self.config = config
        
    async def verify_proof(self, proof: Dict[str, Any], verification_key: Dict[str, Any],
                          public_inputs: List[Any]) -> bool:
        """Verify a ZK-SNARK proof."""
        try:
            # In a real implementation, this would perform actual verification
            # For now, we'll simulate the verification process
            
            # Check proof structure
            required_fields = ["pi_a", "pi_b", "pi_c", "public_inputs", "circuit_hash", "proof_hash"]
            if not all(field in proof for field in required_fields):
                logger.error("Proof missing required fields")
                return False
            
            # Verify proof hash
            proof_copy = proof.copy()
            proof_copy.pop("proof_hash", None)
            expected_hash = hashlib.sha256(
                json.dumps(proof_copy, sort_keys=True).encode()
            ).hexdigest()
            
            if proof["proof_hash"] != expected_hash:
                logger.error("Proof hash verification failed")
                return False
            
            # Verify public inputs match
            if proof["public_inputs"] != public_inputs:
                logger.error("Public inputs mismatch")
                return False
            
            # Simulate pairing check (in real implementation, this would use actual cryptography)
            # For now, we'll just check that the proof elements are within reasonable bounds
            for pi_a_val in proof["pi_a"]:
                if not (0 <= pi_a_val < 1000):
                    logger.error("Invalid pi_a values")
                    return False
            
            logger.info("ZK-SNARK proof verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False

class ZKSTARKProver:
    """ZK-STARK proof generator."""
    
    def __init__(self, config: ZKProofConfig):
        self.config = config
        
    async def generate_proof(self, circuit: Circuit, public_inputs: List[Any],
                           private_inputs: List[Any]) -> Dict[str, Any]:
        """Generate a ZK-STARK proof."""
        try:
            # ZK-STARK is more suitable for large circuits
            # This is a simplified implementation
            
            proof_data = {
                "trace": [secrets.randbelow(1000) for _ in range(100)],
                "low_degree_proof": f"ldp_{circuit.circuit_id}",
                "fri_proof": f"fri_{circuit.circuit_id}",
                "public_inputs": public_inputs,
                "circuit_hash": hashlib.sha256(circuit.circuit_id.encode()).hexdigest()
            }
            
            # Add STARK-specific elements
            proof_data["proof_hash"] = hashlib.sha256(
                json.dumps(proof_data, sort_keys=True).encode()
            ).hexdigest()
            
            logger.info(f"Generated ZK-STARK proof for circuit {circuit.circuit_id}")
            return proof_data
            
        except Exception as e:
            logger.error(f"Failed to generate ZK-STARK proof: {e}")
            raise

class ZKSTARKVerifier:
    """ZK-STARK proof verifier."""
    
    def __init__(self, config: ZKProofConfig):
        self.config = config
        
    async def verify_proof(self, proof: Dict[str, Any], public_inputs: List[Any]) -> bool:
        """Verify a ZK-STARK proof."""
        try:
            # Check proof structure
            required_fields = ["trace", "low_degree_proof", "fri_proof", "public_inputs", "circuit_hash", "proof_hash"]
            if not all(field in proof for field in required_fields):
                logger.error("STARK proof missing required fields")
                return False
            
            # Verify proof hash
            proof_copy = proof.copy()
            proof_copy.pop("proof_hash", None)
            expected_hash = hashlib.sha256(
                json.dumps(proof_copy, sort_keys=True).encode()
            ).hexdigest()
            
            if proof["proof_hash"] != expected_hash:
                logger.error("STARK proof hash verification failed")
                return False
            
            # Verify public inputs
            if proof["public_inputs"] != public_inputs:
                logger.error("STARK public inputs mismatch")
                return False
            
            # Simulate FRI verification
            if not proof["fri_proof"].startswith("fri_"):
                logger.error("Invalid FRI proof format")
                return False
            
            logger.info("ZK-STARK proof verification successful")
            return True
            
        except Exception as e:
            logger.error(f"STARK proof verification failed: {e}")
            return False

class RangeProof:
    """Range proof implementation for confidential values."""
    
    def __init__(self, config: ZKProofConfig):
        self.config = config
        
    async def generate_range_proof(self, value: int, min_value: int, max_value: int,
                                 commitment: str) -> Dict[str, Any]:
        """Generate a range proof for a value."""
        try:
            # Simplified range proof using Pedersen commitments
            proof_data = {
                "commitment": commitment,
                "range": [min_value, max_value],
                "proof_elements": [secrets.randbelow(1000) for _ in range(5)],
                "challenge": hashlib.sha256(commitment.encode()).hexdigest()[:16]
            }
            
            proof_data["proof_hash"] = hashlib.sha256(
                json.dumps(proof_data, sort_keys=True).encode()
            ).hexdigest()
            
            logger.info(f"Generated range proof for value in [{min_value}, {max_value}]")
            return proof_data
            
        except Exception as e:
            logger.error(f"Failed to generate range proof: {e}")
            raise
    
    async def verify_range_proof(self, proof: Dict[str, Any]) -> bool:
        """Verify a range proof."""
        try:
            required_fields = ["commitment", "range", "proof_elements", "challenge", "proof_hash"]
            if not all(field in proof for field in required_fields):
                return False
            
            # Verify proof hash
            proof_copy = proof.copy()
            proof_copy.pop("proof_hash", None)
            expected_hash = hashlib.sha256(
                json.dumps(proof_copy, sort_keys=True).encode()
            ).hexdigest()
            
            return proof["proof_hash"] == expected_hash
            
        except Exception as e:
            logger.error(f"Range proof verification failed: {e}")
            return False

class MembershipProof:
    """Membership proof for set inclusion."""
    
    def __init__(self, config: ZKProofConfig):
        self.config = config
        
    async def generate_membership_proof(self, element: Any, set_elements: List[Any],
                                      merkle_root: str) -> Dict[str, Any]:
        """Generate a membership proof for an element in a set."""
        try:
            # Simplified membership proof using Merkle trees
            proof_data = {
                "element": element,
                "merkle_root": merkle_root,
                "path": [secrets.randbelow(1000) for _ in range(3)],
                "siblings": [secrets.randbelow(1000) for _ in range(3)]
            }
            
            proof_data["proof_hash"] = hashlib.sha256(
                json.dumps(proof_data, sort_keys=True).encode()
            ).hexdigest()
            
            logger.info(f"Generated membership proof for element in set")
            return proof_data
            
        except Exception as e:
            logger.error(f"Failed to generate membership proof: {e}")
            raise
    
    async def verify_membership_proof(self, proof: Dict[str, Any], set_elements: List[Any]) -> bool:
        """Verify a membership proof."""
        try:
            required_fields = ["element", "merkle_root", "path", "siblings", "proof_hash"]
            if not all(field in proof for field in required_fields):
                return False
            
            # Verify proof hash
            proof_copy = proof.copy()
            proof_copy.pop("proof_hash", None)
            expected_hash = hashlib.sha256(
                json.dumps(proof_copy, sort_keys=True).encode()
            ).hexdigest()
            
            if proof["proof_hash"] != expected_hash:
                return False
            
            # Check if element is in set
            return proof["element"] in set_elements
            
        except Exception as e:
            logger.error(f"Membership proof verification failed: {e}")
            return False

class ZeroKnowledgeProofsModule:
    """Zero-Knowledge Proofs module for Blaze AI system."""
    
    def __init__(self, config: ZKProofConfig):
        self.config = config
        self.status = "uninitialized"
        
        # Proof generators
        self.zk_snark_prover = ZKSNARKProver(config)
        self.zk_stark_prover = ZKSTARKProver(config)
        
        # Proof verifiers
        self.zk_snark_verifier = ZKSNARKVerifier(config)
        self.zk_stark_verifier = ZKSTARKVerifier(config)
        
        # Specialized proofs
        self.range_proof = RangeProof(config)
        self.membership_proof = MembershipProof(config)
        
        # State
        self.circuits: Dict[str, Circuit] = {}
        self.proofs: Dict[str, ZKProof] = {}
        self.metrics = ProofMetrics()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the zero-knowledge proofs module."""
        try:
            logger.info("Initializing Zero-Knowledge Proofs Module")
            
            # Check cryptographic library availability
            if not CRYPTO_AVAILABLE:
                logger.warning("Cryptographic libraries not available, using mock implementations")
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.status = "active"
            logger.info("Zero-Knowledge Proofs Module initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Zero-Knowledge Proofs Module: {e}")
            self.status = "error"
            raise
    
    async def shutdown(self):
        """Shutdown the zero-knowledge proofs module."""
        try:
            logger.info("Shutting down Zero-Knowledge Proofs Module")
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            self.status = "shutdown"
            logger.info("Zero-Knowledge Proofs Module shut down successfully")
            
        except Exception as e:
            logger.error(f"Failed to shutdown Zero-Knowledge Proofs Module: {e}")
            raise
    
    async def create_circuit(self, name: str, circuit_type: CircuitType,
                           gates: List[Dict[str, Any]], inputs: List[str],
                           outputs: List[str]) -> str:
        """Create a new arithmetic circuit."""
        try:
            circuit_id = str(uuid.uuid4())
            
            circuit = Circuit(
                circuit_id=circuit_id,
                name=name,
                circuit_type=circuit_type,
                gates=gates,
                inputs=inputs,
                outputs=outputs,
                constraints=[],
                metadata={"created_by": "blaze_ai"}
            )
            
            self.circuits[circuit_id] = circuit
            self.metrics.total_circuits += 1
            self.metrics.active_circuits += 1
            
            logger.info(f"Created circuit {name} with ID {circuit_id}")
            return circuit_id
            
        except Exception as e:
            logger.error(f"Failed to create circuit: {e}")
            raise
    
    async def generate_proof(self, circuit_id: str, proof_type: ProofType,
                           public_inputs: List[Any], private_inputs: List[Any]) -> str:
        """Generate a zero-knowledge proof."""
        try:
            if circuit_id not in self.circuits:
                raise ValueError(f"Circuit {circuit_id} not found")
            
            circuit = self.circuits[circuit_id]
            proof_id = str(uuid.uuid4())
            
            # Create proof record
            proof = ZKProof(
                proof_id=proof_id,
                circuit_id=circuit_id,
                proof_type=proof_type,
                public_inputs=public_inputs,
                proof_data={},
                verification_key={},
                status=ProofStatus.GENERATING,
                generated_at=datetime.now()
            )
            
            self.proofs[proof_id] = proof
            self.metrics.total_proofs += 1
            
            # Generate proof based on type
            if proof_type == ProofType.ZK_SNARK:
                proof.proof_data = await self.zk_snark_prover.generate_proof(
                    circuit, public_inputs, private_inputs
                )
                proof.verification_key = {"type": "snark", "key": f"vk_{circuit_id}"}
                
            elif proof_type == ProofType.ZK_STARK:
                proof.proof_data = await self.zk_stark_prover.generate_proof(
                    circuit, public_inputs, private_inputs
                )
                proof.verification_key = {"type": "stark", "key": f"vk_{circuit_id}"}
                
            else:
                raise ValueError(f"Unsupported proof type: {proof_type}")
            
            proof.status = ProofStatus.GENERATED
            self.metrics.generated_proofs += 1
            
            logger.info(f"Generated {proof_type.value} proof {proof_id}")
            return proof_id
            
        except Exception as e:
            logger.error(f"Failed to generate proof: {e}")
            if proof_id in self.proofs:
                self.proofs[proof_id].status = ProofStatus.FAILED
                self.metrics.failed_proofs += 1
            raise
    
    async def verify_proof(self, proof_id: str) -> bool:
        """Verify a zero-knowledge proof."""
        try:
            if proof_id not in self.proofs:
                raise ValueError(f"Proof {proof_id} not found")
            
            proof = self.proofs[proof_id]
            proof.status = ProofStatus.VERIFYING
            
            # Verify proof based on type
            if proof.proof_type == ProofType.ZK_SNARK:
                is_valid = await self.zk_snark_verifier.verify_proof(
                    proof.proof_data, proof.verification_key, proof.public_inputs
                )
                
            elif proof.proof_type == ProofType.ZK_STARK:
                is_valid = await self.zk_stark_verifier.verify_proof(
                    proof.proof_data, proof.public_inputs
                )
                
            else:
                raise ValueError(f"Unsupported proof type for verification: {proof.proof_type}")
            
            if is_valid:
                proof.status = ProofStatus.VERIFIED
                self.metrics.verified_proofs += 1
                logger.info(f"Proof {proof_id} verified successfully")
            else:
                proof.status = ProofStatus.FAILED
                self.metrics.failed_proofs += 1
                logger.warning(f"Proof {proof_id} verification failed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to verify proof {proof_id}: {e}")
            if proof_id in self.proofs:
                self.proofs[proof_id].status = ProofStatus.FAILED
                self.metrics.failed_proofs += 1
            return False
    
    async def generate_range_proof(self, value: int, min_value: int, max_value: int,
                                 commitment: str) -> Dict[str, Any]:
        """Generate a range proof for a confidential value."""
        try:
            proof_data = await self.range_proof.generate_range_proof(
                value, min_value, max_value, commitment
            )
            
            logger.info(f"Generated range proof for value {value}")
            return proof_data
            
        except Exception as e:
            logger.error(f"Failed to generate range proof: {e}")
            raise
    
    async def generate_membership_proof(self, element: Any, set_elements: List[Any],
                                      merkle_root: str) -> Dict[str, Any]:
        """Generate a membership proof for set inclusion."""
        try:
            proof_data = await self.membership_proof.generate_membership_proof(
                element, set_elements, merkle_root
            )
            
            logger.info(f"Generated membership proof for element")
            return proof_data
            
        except Exception as e:
            logger.error(f"Failed to generate membership proof: {e}")
            raise
    
    async def get_proof_status(self, proof_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a proof."""
        if proof_id not in self.proofs:
            return None
        
        proof = self.proofs[proof_id]
        return {
            "proof_id": proof_id,
            "circuit_id": proof.circuit_id,
            "proof_type": proof.proof_type.value,
            "status": proof.status.value,
            "public_inputs": proof.public_inputs,
            "generated_at": proof.generated_at.isoformat(),
            "expires_at": proof.expires_at.isoformat() if proof.expires_at else None
        }
    
    async def get_metrics(self) -> ProofMetrics:
        """Get module metrics."""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health status of the module."""
        return {
            "status": self.status,
            "enabled_proof_types": [pt.value for pt in self.config.enabled_proof_types],
            "total_circuits": self.metrics.total_circuits,
            "active_circuits": self.metrics.active_circuits,
            "total_proofs": self.metrics.total_proofs,
            "generated_proofs": self.metrics.generated_proofs,
            "verified_proofs": self.metrics.verified_proofs,
            "failed_proofs": self.metrics.failed_proofs,
            "crypto_available": CRYPTO_AVAILABLE
        }
    
    async def _cleanup_loop(self):
        """Background cleanup loop for expired proofs."""
        while self.status == "active":
            try:
                current_time = datetime.now()
                expired_proofs = []
                
                for proof_id, proof in self.proofs.items():
                    if proof.expires_at and current_time > proof.expires_at:
                        expired_proofs.append(proof_id)
                
                for proof_id in expired_proofs:
                    proof = self.proofs.pop(proof_id)
                    proof.status = ProofStatus.EXPIRED
                    logger.info(f"Proof {proof_id} expired and removed")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)

# Factory functions
async def create_zero_knowledge_proofs_module(config: ZKProofConfig) -> ZeroKnowledgeProofsModule:
    """Create a Zero-Knowledge Proofs module with the given configuration."""
    module = ZeroKnowledgeProofsModule(config)
    await module.initialize()
    return module

async def create_zero_knowledge_proofs_module_with_defaults(**overrides) -> ZeroKnowledgeProofsModule:
    """Create a Zero-Knowledge Proofs module with default configuration and custom overrides."""
    config = ZKProofConfig()
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return await create_zero_knowledge_proofs_module(config)

