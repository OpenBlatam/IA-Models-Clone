"""
Blaze AI Zero-Knowledge Proofs Module Tests

This file provides comprehensive tests for the Zero-Knowledge Proofs Module.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import secrets

from blaze_ai.modules.zero_knowledge_proofs import (
    ZeroKnowledgeProofsModule,
    ZKProofConfig,
    ProofType,
    CircuitType,
    ProofStatus,
    Circuit,
    ZKProof,
    ProofMetrics,
    ArithmeticCircuit,
    ZKSNARKProver,
    ZKSNARKVerifier,
    ZKSTARKProver,
    ZKSTARKVerifier,
    RangeProof,
    MembershipProof,
    create_zero_knowledge_proofs_module,
    create_zero_knowledge_proofs_module_with_defaults
)

@pytest.fixture
def zk_config():
    """Create a basic zero-knowledge proofs configuration."""
    return ZKProofConfig(
        name="test_zk_proofs",
        enabled_proof_types=[ProofType.ZK_SNARK],
        security_level=128,
        circuit_optimization=True,
        parallel_generation=True
    )

@pytest.fixture
async def zk_module(zk_config):
    """Create a zero-knowledge proofs module for testing."""
    module = ZeroKnowledgeProofsModule(zk_config)
    await module.initialize()
    yield module
    await module.shutdown()

class TestZKProofConfig:
    """Test ZKProofConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ZKProofConfig()
        assert config.name == "zero_knowledge_proofs"
        assert ProofType.ZK_SNARK in config.enabled_proof_types
        assert config.circuit_optimization is True
        assert config.parallel_generation is True
        assert config.security_level == 128
        assert config.curve_type == "bn128"
        assert config.hash_function == "sha256"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ZKProofConfig(
            name="custom_zk",
            enabled_proof_types=[ProofType.ZK_STARK],
            security_level=256,
            circuit_optimization=False,
            blockchain_integration=False
        )
        assert config.name == "custom_zk"
        assert ProofType.ZK_STARK in config.enabled_proof_types
        assert config.security_level == 256
        assert config.circuit_optimization is False
        assert config.blockchain_integration is False

class TestCircuit:
    """Test Circuit dataclass."""
    
    def test_circuit_creation(self):
        """Test circuit creation."""
        circuit = Circuit(
            circuit_id="test_circuit",
            name="test_circuit",
            circuit_type=CircuitType.ARITHMETIC,
            gates=[],
            inputs=["a", "b"],
            outputs=["c"],
            constraints=[]
        )
        
        assert circuit.circuit_id == "test_circuit"
        assert circuit.name == "test_circuit"
        assert circuit.circuit_type == CircuitType.ARITHMETIC
        assert circuit.inputs == ["a", "b"]
        assert circuit.outputs == ["c"]
        assert circuit.gates == []
        assert circuit.constraints == []

class TestZKProof:
    """Test ZKProof dataclass."""
    
    def test_proof_creation(self):
        """Test proof creation."""
        proof = ZKProof(
            proof_id="test_proof",
            circuit_id="test_circuit",
            proof_type=ProofType.ZK_SNARK,
            public_inputs=[1, 2],
            proof_data={"test": "data"},
            verification_key={"key": "value"},
            status=ProofStatus.GENERATED,
            generated_at=datetime.now()
        )
        
        assert proof.proof_id == "test_proof"
        assert proof.circuit_id == "test_circuit"
        assert proof.proof_type == ProofType.ZK_SNARK
        assert proof.public_inputs == [1, 2]
        assert proof.proof_data == {"test": "data"}
        assert proof.status == ProofStatus.GENERATED

class TestArithmeticCircuit:
    """Test ArithmeticCircuit class."""
    
    def test_circuit_initialization(self, zk_config):
        """Test arithmetic circuit initialization."""
        circuit = ArithmeticCircuit(zk_config)
        assert circuit.gates == []
        assert circuit.wire_values == {}
        assert circuit.constraints == []
    
    def test_add_gate(self, zk_config):
        """Test adding gates to circuit."""
        circuit = ArithmeticCircuit(zk_config)
        
        gate_id = circuit.add_gate("add", ["a", "b"], "c", "add")
        assert gate_id == "gate_0"
        assert len(circuit.gates) == 1
        
        gate = circuit.gates[0]
        assert gate["type"] == "add"
        assert gate["inputs"] == ["a", "b"]
        assert gate["output"] == "c"
        assert gate["operation"] == "add"
    
    def test_add_constraint(self, zk_config):
        """Test adding constraints to circuit."""
        circuit = ArithmeticCircuit(zk_config)
        
        constraint_id = circuit.add_constraint("linear", "a + b = c", ["a", "b", "c"])
        assert constraint_id == "constraint_0"
        assert len(circuit.constraints) == 1
        
        constraint = circuit.constraints[0]
        assert constraint["type"] == "linear"
        assert constraint["equation"] == "a + b = c"
        assert constraint["variables"] == ["a", "b", "c"]
    
    def test_circuit_evaluation(self, zk_config):
        """Test circuit evaluation."""
        circuit = ArithmeticCircuit(zk_config)
        
        # Add gates for y = a * b + c
        circuit.add_gate("input", ["a"], "a", "input")
        circuit.add_gate("input", ["b"], "b", "input")
        circuit.add_gate("input", ["c"], "c", "input")
        circuit.add_gate("mul", ["a", "b"], "temp1", "mul")
        circuit.add_gate("add", ["temp1", "c"], "y", "add")
        
        inputs = {"a": 3, "b": 4, "c": 5}
        result = circuit.evaluate(inputs)
        
        assert result["y"] == 17  # 3 * 4 + 5 = 17
    
    def test_get_constraints(self, zk_config):
        """Test getting all circuit constraints."""
        circuit = ArithmeticCircuit(zk_config)
        
        circuit.add_gate("add", ["a", "b"], "c", "add")
        circuit.add_constraint("linear", "a + b = c", ["a", "b", "c"])
        
        constraints = circuit.get_constraints()
        assert len(constraints) == 2  # 1 gate constraint + 1 additional constraint

class TestZKSNARKProver:
    """Test ZK-SNARK prover."""
    
    def test_prover_initialization(self, zk_config):
        """Test ZK-SNARK prover initialization."""
        prover = ZKSNARKProver(zk_config)
        assert prover.config == zk_config
        assert prover.trusted_setup is None
    
    @pytest.mark.asyncio
    async def test_generate_trusted_setup(self, zk_config):
        """Test trusted setup generation."""
        prover = ZKSNARKProver(zk_config)
        
        circuit = Circuit(
            circuit_id="test_circuit",
            name="test",
            circuit_type=CircuitType.ARITHMETIC,
            gates=[],
            inputs=[],
            outputs=[],
            constraints=[]
        )
        
        setup = await prover.generate_trusted_setup(circuit)
        
        assert "proving_key" in setup
        assert "verification_key" in setup
        assert "toxic_waste" in setup
        assert "circuit_hash" in setup
        assert prover.trusted_setup == setup
    
    @pytest.mark.asyncio
    async def test_generate_proof(self, zk_config):
        """Test ZK-SNARK proof generation."""
        prover = ZKSNARKProver(zk_config)
        
        circuit = Circuit(
            circuit_id="test_circuit",
            name="test",
            circuit_type=CircuitType.ARITHMETIC,
            gates=[],
            inputs=[],
            outputs=[],
            constraints=[]
        )
        
        public_inputs = [1, 2]
        private_inputs = [3, 4]
        
        proof_data = await prover.generate_proof(circuit, public_inputs, private_inputs)
        
        assert "pi_a" in proof_data
        assert "pi_b" in proof_data
        assert "pi_c" in proof_data
        assert "public_inputs" in proof_data
        assert "circuit_hash" in proof_data
        assert "proof_hash" in proof_data

class TestZKSNARKVerifier:
    """Test ZK-SNARK verifier."""
    
    def test_verifier_initialization(self, zk_config):
        """Test ZK-SNARK verifier initialization."""
        verifier = ZKSNARKVerifier(zk_config)
        assert verifier.config == zk_config
    
    @pytest.mark.asyncio
    async def test_verify_proof_success(self, zk_config):
        """Test successful proof verification."""
        verifier = ZKSNARKVerifier(zk_config)
        
        proof = {
            "pi_a": [100, 200, 300],
            "pi_b": [[400, 500, 600], [700, 800, 900]],
            "pi_c": [1000, 1100, 1200],
            "public_inputs": [1, 2],
            "circuit_hash": "test_hash",
            "proof_hash": "proof_hash"
        }
        
        verification_key = {"type": "snark", "key": "test_key"}
        public_inputs = [1, 2]
        
        is_valid = await verifier.verify_proof(proof, verification_key, public_inputs)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_verify_proof_missing_fields(self, zk_config):
        """Test proof verification with missing fields."""
        verifier = ZKSNARKVerifier(zk_config)
        
        proof = {
            "pi_a": [100, 200, 300],
            # Missing required fields
        }
        
        verification_key = {"type": "snark", "key": "test_key"}
        public_inputs = [1, 2]
        
        is_valid = await verifier.verify_proof(proof, verification_key, public_inputs)
        assert is_valid is False

class TestZKSTARKProver:
    """Test ZK-STARK prover."""
    
    def test_prover_initialization(self, zk_config):
        """Test ZK-STARK prover initialization."""
        prover = ZKSTARKProver(zk_config)
        assert prover.config == zk_config
    
    @pytest.mark.asyncio
    async def test_generate_proof(self, zk_config):
        """Test ZK-STARK proof generation."""
        prover = ZKSTARKProver(zk_config)
        
        circuit = Circuit(
            circuit_id="test_circuit",
            name="test",
            circuit_type=CircuitType.ARITHMETIC,
            gates=[],
            inputs=[],
            outputs=[],
            constraints=[]
        )
        
        public_inputs = [1, 2]
        private_inputs = [3, 4]
        
        proof_data = await prover.generate_proof(circuit, public_inputs, private_inputs)
        
        assert "trace" in proof_data
        assert "low_degree_proof" in proof_data
        assert "fri_proof" in proof_data
        assert "public_inputs" in proof_data
        assert "circuit_hash" in proof_data
        assert "proof_hash" in proof_data

class TestRangeProof:
    """Test range proof implementation."""
    
    def test_range_proof_initialization(self, zk_config):
        """Test range proof initialization."""
        range_proof = RangeProof(zk_config)
        assert range_proof.config == zk_config
    
    @pytest.mark.asyncio
    async def test_generate_range_proof(self, zk_config):
        """Test range proof generation."""
        range_proof = RangeProof(zk_config)
        
        value = 30
        min_value = 18
        max_value = 65
        commitment = "test_commitment"
        
        proof_data = await range_proof.generate_range_proof(
            value, min_value, max_value, commitment
        )
        
        assert "commitment" in proof_data
        assert "range" in proof_data
        assert "proof_elements" in proof_data
        assert "challenge" in proof_data
        assert "proof_hash" in proof_data
        assert proof_data["range"] == [min_value, max_value]
    
    @pytest.mark.asyncio
    async def test_verify_range_proof_success(self, zk_config):
        """Test successful range proof verification."""
        range_proof = RangeProof(zk_config)
        
        proof = {
            "commitment": "test_commitment",
            "range": [18, 65],
            "proof_elements": [100, 200, 300, 400, 500],
            "challenge": "test_challenge",
            "proof_hash": "proof_hash"
        }
        
        is_valid = await range_proof.verify_range_proof(proof)
        assert is_valid is True

class TestMembershipProof:
    """Test membership proof implementation."""
    
    def test_membership_proof_initialization(self, zk_config):
        """Test membership proof initialization."""
        membership_proof = MembershipProof(zk_config)
        assert membership_proof.config == zk_config
    
    @pytest.mark.asyncio
    async def test_generate_membership_proof(self, zk_config):
        """Test membership proof generation."""
        membership_proof = MembershipProof(zk_config)
        
        element = "user1"
        set_elements = ["user1", "user2", "user3"]
        merkle_root = "test_merkle_root"
        
        proof_data = await membership_proof.generate_membership_proof(
            element, set_elements, merkle_root
        )
        
        assert "element" in proof_data
        assert "merkle_root" in proof_data
        assert "path" in proof_data
        assert "siblings" in proof_data
        assert "proof_hash" in proof_data
    
    @pytest.mark.asyncio
    async def test_verify_membership_proof_success(self, zk_config):
        """Test successful membership proof verification."""
        membership_proof = MembershipProof(zk_config)
        
        proof = {
            "element": "user1",
            "merkle_root": "test_merkle_root",
            "path": [100, 200, 300],
            "siblings": [400, 500, 600],
            "proof_hash": "proof_hash"
        }
        
        set_elements = ["user1", "user2", "user3"]
        
        is_valid = await membership_proof.verify_membership_proof(proof, set_elements)
        assert is_valid is True

class TestZeroKnowledgeProofsModule:
    """Test the main Zero-Knowledge Proofs module."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, zk_config):
        """Test module initialization."""
        module = ZeroKnowledgeProofsModule(zk_config)
        await module.initialize()
        
        assert module.status == "active"
        assert module.zk_snark_prover is not None
        assert module.zk_stark_prover is not None
        assert module.zk_snark_verifier is not None
        assert module.zk_stark_verifier is not None
        assert module.range_proof is not None
        assert module.membership_proof is not None
    
    @pytest.mark.asyncio
    async def test_shutdown(self, zk_module):
        """Test module shutdown."""
        await zk_module.shutdown()
        assert zk_module.status == "shutdown"
    
    @pytest.mark.asyncio
    async def test_create_circuit(self, zk_module):
        """Test circuit creation."""
        circuit_name = "test_circuit"
        circuit_type = CircuitType.ARITHMETIC
        gates = [{"type": "input", "inputs": ["a"], "output": "a", "operation": "input"}]
        inputs = ["a"]
        outputs = ["a"]
        
        circuit_id = await zk_module.create_circuit(
            name=circuit_name,
            circuit_type=circuit_type,
            gates=gates,
            inputs=inputs,
            outputs=outputs
        )
        
        assert circuit_id in zk_module.circuits
        circuit = zk_module.circuits[circuit_id]
        assert circuit.name == circuit_name
        assert circuit.circuit_type == circuit_type
    
    @pytest.mark.asyncio
    async def test_generate_proof_zk_snark(self, zk_module):
        """Test ZK-SNARK proof generation."""
        # First create a circuit
        circuit_id = await zk_module.create_circuit(
            name="test_circuit",
            circuit_type=CircuitType.ARITHMETIC,
            gates=[],
            inputs=["a"],
            outputs=["b"],
            constraints=[]
        )
        
        # Generate proof
        proof_id = await zk_module.generate_proof(
            circuit_id=circuit_id,
            proof_type=ProofType.ZK_SNARK,
            public_inputs=[1],
            private_inputs=[2]
        )
        
        assert proof_id in zk_module.proofs
        proof = zk_module.proofs[proof_id]
        assert proof.proof_type == ProofType.ZK_SNARK
        assert proof.status == ProofStatus.GENERATED
    
    @pytest.mark.asyncio
    async def test_verify_proof(self, zk_module):
        """Test proof verification."""
        # First create a circuit and generate a proof
        circuit_id = await zk_module.create_circuit(
            name="test_circuit",
            circuit_type=CircuitType.ARITHMETIC,
            gates=[],
            inputs=["a"],
            outputs=["b"],
            constraints=[]
        )
        
        proof_id = await zk_module.generate_proof(
            circuit_id=circuit_id,
            proof_type=ProofType.ZK_SNARK,
            public_inputs=[1],
            private_inputs=[2]
        )
        
        # Verify the proof
        is_valid = await zk_module.verify_proof(proof_id)
        assert is_valid is True
        
        # Check that the proof status was updated
        proof = zk_module.proofs[proof_id]
        assert proof.status == ProofStatus.VERIFIED
    
    @pytest.mark.asyncio
    async def test_generate_range_proof(self, zk_module):
        """Test range proof generation."""
        value = 25
        min_value = 18
        max_value = 65
        commitment = "test_commitment"
        
        proof_data = await zk_module.generate_range_proof(
            value, min_value, max_value, commitment
        )
        
        assert "commitment" in proof_data
        assert "range" in proof_data
        assert "proof_hash" in proof_data
    
    @pytest.mark.asyncio
    async def test_generate_membership_proof(self, zk_module):
        """Test membership proof generation."""
        element = "user1"
        set_elements = ["user1", "user2", "user3"]
        merkle_root = "test_merkle_root"
        
        proof_data = await zk_module.generate_membership_proof(
            element, set_elements, merkle_root
        )
        
        assert "element" in proof_data
        assert "merkle_root" in proof_data
        assert "proof_hash" in proof_data
    
    @pytest.mark.asyncio
    async def test_get_proof_status(self, zk_module):
        """Test getting proof status."""
        # First create a circuit and generate a proof
        circuit_id = await zk_module.create_circuit(
            name="test_circuit",
            circuit_type=CircuitType.ARITHMETIC,
            gates=[],
            inputs=["a"],
            outputs=["b"],
            constraints=[]
        )
        
        proof_id = await zk_module.generate_proof(
            circuit_id=circuit_id,
            proof_type=ProofType.ZK_SNARK,
            public_inputs=[1],
            private_inputs=[2]
        )
        
        # Get proof status
        status = await zk_module.get_proof_status(proof_id)
        
        assert status is not None
        assert status["proof_id"] == proof_id
        assert status["circuit_id"] == circuit_id
        assert status["proof_type"] == ProofType.ZK_SNARK.value
        assert status["status"] == ProofStatus.GENERATED.value
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, zk_module):
        """Test getting module metrics."""
        metrics = await zk_module.get_metrics()
        
        assert metrics is not None
        assert hasattr(metrics, 'total_circuits')
        assert hasattr(metrics, 'active_circuits')
        assert hasattr(metrics, 'total_proofs')
        assert hasattr(metrics, 'generated_proofs')
        assert hasattr(metrics, 'verified_proofs')
        assert hasattr(metrics, 'failed_proofs')
    
    @pytest.mark.asyncio
    async def test_health_check(self, zk_module):
        """Test module health check."""
        health = await zk_module.health_check()
        
        assert health is not None
        assert "status" in health
        assert "enabled_proof_types" in health
        assert "total_circuits" in health
        assert "total_proofs" in health
        assert "crypto_available" in health

class TestFactoryFunctions:
    """Test factory functions."""
    
    @pytest.mark.asyncio
    async def test_create_zero_knowledge_proofs_module(self, zk_config):
        """Test creating module with explicit config."""
        with patch.object(ZeroKnowledgeProofsModule, 'initialize'):
            module = await create_zero_knowledge_proofs_module(zk_config)
            
            assert isinstance(module, ZeroKnowledgeProofsModule)
            assert module.config == zk_config
    
    @pytest.mark.asyncio
    async def test_create_zero_knowledge_proofs_module_with_defaults(self):
        """Test creating module with default config and overrides."""
        with patch.object(ZeroKnowledgeProofsModule, 'initialize'):
            module = await create_zero_knowledge_proofs_module_with_defaults(
                enabled_proof_types=[ProofType.ZK_STARK],
                security_level=256
            )
            
            assert isinstance(module, ZeroKnowledgeProofsModule)
            assert ProofType.ZK_STARK in module.config.enabled_proof_types
            assert module.config.security_level == 256

@pytest.mark.asyncio
async def test_integration_scenario():
    """Test a complete integration scenario."""
    # Create module
    config = ZKProofConfig(
        enabled_proof_types=[ProofType.ZK_SNARK],
        security_level=128,
        circuit_optimization=True
    )
    
    with patch.object(ZeroKnowledgeProofsModule, 'initialize'):
        module = ZeroKnowledgeProofsModule(config)
        await module.initialize()
        
        # Create circuit
        circuit_id = await module.create_circuit(
            name="integration_test_circuit",
            circuit_type=CircuitType.ARITHMETIC,
            gates=[],
            inputs=["a", "b"],
            outputs=["c"],
            constraints=[]
        )
        
        # Verify circuit creation
        assert circuit_id in module.circuits
        assert module.metrics.total_circuits == 1
        
        # Generate proof
        proof_id = await module.generate_proof(
            circuit_id=circuit_id,
            proof_type=ProofType.ZK_SNARK,
            public_inputs=[1, 2],
            private_inputs=[3, 4]
        )
        
        # Verify proof generation
        assert proof_id in module.proofs
        assert module.metrics.total_proofs == 1
        
        # Verify proof
        is_valid = await module.verify_proof(proof_id)
        assert is_valid is True
        
        # Check final metrics
        final_metrics = await module.get_metrics()
        assert final_metrics.total_circuits == 1
        assert final_metrics.total_proofs == 1
        assert final_metrics.verified_proofs == 1
        
        # Health check
        health = await module.health_check()
        assert health["status"] == "active"
        
        # Cleanup
        await module.shutdown()
        assert module.status == "shutdown"

if __name__ == "__main__":
    pytest.main([__file__])

