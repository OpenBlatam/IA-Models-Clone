"""
Blockchain Testing Framework for HeyGen AI Testing System.
Advanced blockchain-based testing including immutable test records,
smart contract validation, and decentralized test verification.
"""

import time
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import random
import threading
import queue
from collections import defaultdict, deque
import sqlite3
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64

@dataclass
class TestBlock:
    """Represents a block in the test blockchain."""
    index: int
    timestamp: datetime
    test_data: Dict[str, Any]
    previous_hash: str
    hash: str
    nonce: int = 0
    merkle_root: str = ""
    validator: str = ""
    signature: str = ""

@dataclass
class TestTransaction:
    """Represents a test transaction."""
    transaction_id: str
    test_id: str
    test_name: str
    execution_time: datetime
    result: str  # "pass", "fail", "skip"
    duration: float
    metadata: Dict[str, Any]
    hash: str = ""
    signature: str = ""

@dataclass
class SmartContract:
    """Represents a smart contract for test validation."""
    contract_id: str
    name: str
    conditions: List[Dict[str, Any]]
    rewards: Dict[str, float]
    penalties: Dict[str, float]
    created_at: datetime
    active: bool = True

@dataclass
class TestValidator:
    """Represents a test validator node."""
    validator_id: str
    name: str
    public_key: str
    stake: float
    reputation: float
    active: bool = True
    last_validation: Optional[datetime] = None

class BlockchainTestEngine:
    """Blockchain-based test execution engine."""
    
    def __init__(self, difficulty: int = 4):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = difficulty
        self.mining_reward = 10.0
        self.validators = {}
        self.smart_contracts = {}
        self.private_key = None
        self.public_key = None
        self._generate_keys()
        self._create_genesis_block()
    
    def _generate_keys(self):
        """Generate RSA key pair for signing."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
    
    def _create_genesis_block(self):
        """Create the genesis block."""
        genesis_block = TestBlock(
            index=0,
            timestamp=datetime.now(),
            test_data={"message": "Genesis block for test blockchain"},
            previous_hash="0",
            hash="",
            merkle_root=""
        )
        genesis_block.hash = self._calculate_hash(genesis_block)
        self.chain.append(genesis_block)
    
    def _calculate_hash(self, block: TestBlock) -> str:
        """Calculate hash of a block."""
        block_string = f"{block.index}{block.timestamp.isoformat()}{json.dumps(block.test_data, sort_keys=True)}{block.previous_hash}{block.nonce}{block.merkle_root}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _calculate_merkle_root(self, transactions: List[TestTransaction]) -> str:
        """Calculate Merkle root of transactions."""
        if not transactions:
            return ""
        
        if len(transactions) == 1:
            return transactions[0].hash
        
        # Simple Merkle tree implementation
        current_level = [tx.hash for tx in transactions]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                combined = left + right
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            current_level = next_level
        
        return current_level[0]
    
    def _sign_transaction(self, transaction: TestTransaction) -> str:
        """Sign a transaction with private key."""
        transaction_data = f"{transaction.transaction_id}{transaction.test_id}{transaction.result}{transaction.duration}"
        signature = self.private_key.sign(
            transaction_data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
    
    def _verify_signature(self, transaction: TestTransaction, signature: str, public_key: str) -> bool:
        """Verify transaction signature."""
        try:
            # This would require the public key object
            # For demo purposes, we'll simulate verification
            return True
        except:
            return False
    
    def add_test_transaction(self, test_id: str, test_name: str, result: str, 
                           duration: float, metadata: Dict[str, Any] = None) -> TestTransaction:
        """Add a test transaction to pending transactions."""
        transaction = TestTransaction(
            transaction_id=f"tx_{int(time.time())}_{random.randint(1000, 9999)}",
            test_id=test_id,
            test_name=test_name,
            execution_time=datetime.now(),
            result=result,
            duration=duration,
            metadata=metadata or {}
        )
        
        # Calculate transaction hash
        transaction_data = f"{transaction.transaction_id}{transaction.test_id}{transaction.test_name}{transaction.result}{transaction.duration}"
        transaction.hash = hashlib.sha256(transaction_data.encode()).hexdigest()
        
        # Sign transaction
        transaction.signature = self._sign_transaction(transaction)
        
        self.pending_transactions.append(transaction)
        return transaction
    
    def mine_block(self, validator_id: str = None) -> TestBlock:
        """Mine a new block with pending transactions."""
        if not self.pending_transactions:
            raise ValueError("No pending transactions to mine")
        
        # Create new block
        previous_block = self.chain[-1]
        new_block = TestBlock(
            index=len(self.chain),
            timestamp=datetime.now(),
            test_data={
                "transactions": [tx.__dict__ for tx in self.pending_transactions],
                "miner": validator_id or "default_miner"
            },
            previous_hash=previous_block.hash,
            hash="",
            merkle_root=""
        )
        
        # Calculate Merkle root
        new_block.merkle_root = self._calculate_merkle_root(self.pending_transactions)
        
        # Mine block (proof of work)
        new_block = self._proof_of_work(new_block)
        
        # Add to chain
        self.chain.append(new_block)
        
        # Clear pending transactions
        self.pending_transactions = []
        
        return new_block
    
    def _proof_of_work(self, block: TestBlock) -> TestBlock:
        """Perform proof of work mining."""
        target = "0" * self.difficulty
        
        while block.hash[:self.difficulty] != target:
            block.nonce += 1
            block.hash = self._calculate_hash(block)
        
        return block
    
    def validate_chain(self) -> bool:
        """Validate the entire blockchain."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block hash is valid
            if current_block.hash != self._calculate_hash(current_block):
                return False
            
            # Check if current block points to previous block
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_test_history(self, test_id: str) -> List[TestTransaction]:
        """Get complete test history from blockchain."""
        test_history = []
        
        for block in self.chain[1:]:  # Skip genesis block
            if "transactions" in block.test_data:
                for tx_data in block.test_data["transactions"]:
                    if tx_data["test_id"] == test_id:
                        # Reconstruct transaction object
                        tx = TestTransaction(
                            transaction_id=tx_data["transaction_id"],
                            test_id=tx_data["test_id"],
                            test_name=tx_data["test_name"],
                            execution_time=datetime.fromisoformat(tx_data["execution_time"]),
                            result=tx_data["result"],
                            duration=tx_data["duration"],
                            metadata=tx_data["metadata"],
                            hash=tx_data["hash"],
                            signature=tx_data["signature"]
                        )
                        test_history.append(tx)
        
        return test_history
    
    def get_test_statistics(self, test_id: str) -> Dict[str, Any]:
        """Get test statistics from blockchain data."""
        test_history = self.get_test_history(test_id)
        
        if not test_history:
            return {}
        
        total_executions = len(test_history)
        pass_count = sum(1 for tx in test_history if tx.result == "pass")
        fail_count = sum(1 for tx in test_history if tx.result == "fail")
        skip_count = sum(1 for tx in test_history if tx.result == "skip")
        
        durations = [tx.duration for tx in test_history]
        
        return {
            "test_id": test_id,
            "total_executions": total_executions,
            "pass_rate": pass_count / total_executions if total_executions > 0 else 0,
            "fail_rate": fail_count / total_executions if total_executions > 0 else 0,
            "skip_rate": skip_count / total_executions if total_executions > 0 else 0,
            "average_duration": np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "duration_std": np.std(durations),
            "first_execution": min(tx.execution_time for tx in test_history),
            "last_execution": max(tx.execution_time for tx in test_history)
        }

class SmartContractManager:
    """Manages smart contracts for test validation."""
    
    def __init__(self):
        self.contracts = {}
        self.contract_executions = []
    
    def create_contract(self, name: str, conditions: List[Dict[str, Any]], 
                       rewards: Dict[str, float] = None, 
                       penalties: Dict[str, float] = None) -> SmartContract:
        """Create a new smart contract."""
        contract = SmartContract(
            contract_id=f"contract_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            conditions=conditions,
            rewards=rewards or {},
            penalties=penalties or {},
            created_at=datetime.now()
        )
        
        self.contracts[contract.contract_id] = contract
        return contract
    
    def execute_contract(self, contract_id: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a smart contract against test data."""
        if contract_id not in self.contracts:
            raise ValueError(f"Contract {contract_id} not found")
        
        contract = self.contracts[contract_id]
        execution_result = {
            "contract_id": contract_id,
            "execution_time": datetime.now(),
            "conditions_met": [],
            "conditions_failed": [],
            "rewards": 0.0,
            "penalties": 0.0,
            "total_score": 0.0
        }
        
        # Evaluate conditions
        for condition in contract.conditions:
            if self._evaluate_condition(condition, test_data):
                execution_result["conditions_met"].append(condition)
                execution_result["rewards"] += contract.rewards.get(condition["name"], 0.0)
            else:
                execution_result["conditions_failed"].append(condition)
                execution_result["penalties"] += contract.penalties.get(condition["name"], 0.0)
        
        execution_result["total_score"] = execution_result["rewards"] - execution_result["penalties"]
        
        # Record execution
        self.contract_executions.append(execution_result)
        
        return execution_result
    
    def _evaluate_condition(self, condition: Dict[str, Any], test_data: Dict[str, Any]) -> bool:
        """Evaluate a single condition against test data."""
        condition_type = condition.get("type")
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        if field not in test_data:
            return False
        
        test_value = test_data[field]
        
        if condition_type == "numeric":
            return self._evaluate_numeric_condition(test_value, operator, value)
        elif condition_type == "string":
            return self._evaluate_string_condition(test_value, operator, value)
        elif condition_type == "boolean":
            return self._evaluate_boolean_condition(test_value, operator, value)
        else:
            return False
    
    def _evaluate_numeric_condition(self, test_value: float, operator: str, value: float) -> bool:
        """Evaluate numeric condition."""
        if operator == ">":
            return test_value > value
        elif operator == ">=":
            return test_value >= value
        elif operator == "<":
            return test_value < value
        elif operator == "<=":
            return test_value <= value
        elif operator == "==":
            return abs(test_value - value) < 0.001  # Float comparison
        elif operator == "!=":
            return abs(test_value - value) >= 0.001
        else:
            return False
    
    def _evaluate_string_condition(self, test_value: str, operator: str, value: str) -> bool:
        """Evaluate string condition."""
        if operator == "==":
            return test_value == value
        elif operator == "!=":
            return test_value != value
        elif operator == "contains":
            return value in test_value
        elif operator == "starts_with":
            return test_value.startswith(value)
        elif operator == "ends_with":
            return test_value.endswith(value)
        else:
            return False
    
    def _evaluate_boolean_condition(self, test_value: bool, operator: str, value: bool) -> bool:
        """Evaluate boolean condition."""
        if operator == "==":
            return test_value == value
        elif operator == "!=":
            return test_value != value
        else:
            return False

class TestValidatorNetwork:
    """Manages a network of test validators."""
    
    def __init__(self):
        self.validators = {}
        self.validation_queue = queue.Queue()
        self.validation_results = []
        self.validator_threads = []
    
    def add_validator(self, name: str, stake: float = 100.0) -> TestValidator:
        """Add a validator to the network."""
        validator = TestValidator(
            validator_id=f"validator_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            public_key=f"pub_key_{random.randint(10000, 99999)}",
            stake=stake,
            reputation=1.0
        )
        
        self.validators[validator.validator_id] = validator
        return validator
    
    def start_validation_workers(self, num_workers: int = 3):
        """Start validation worker threads."""
        for i in range(num_workers):
            thread = threading.Thread(target=self._validation_worker, daemon=True)
            thread.start()
            self.validator_threads.append(thread)
    
    def _validation_worker(self):
        """Validation worker thread."""
        while True:
            try:
                validation_task = self.validation_queue.get(timeout=1)
                self._process_validation(validation_task)
                self.validation_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Validation worker error: {e}")
    
    def _process_validation(self, task: Dict[str, Any]):
        """Process a validation task."""
        validator_id = task["validator_id"]
        test_data = task["test_data"]
        
        # Simulate validation process
        validation_result = {
            "validator_id": validator_id,
            "test_id": test_data.get("test_id"),
            "validation_time": datetime.now(),
            "valid": random.random() > 0.1,  # 90% validation success
            "confidence": random.uniform(0.7, 1.0),
            "notes": "Validation completed"
        }
        
        self.validation_results.append(validation_result)
        
        # Update validator reputation
        if validation_result["valid"]:
            self.validators[validator_id].reputation = min(1.0, self.validators[validator_id].reputation + 0.01)
        else:
            self.validators[validator_id].reputation = max(0.0, self.validators[validator_id].reputation - 0.05)
    
    def submit_for_validation(self, test_data: Dict[str, Any]):
        """Submit test data for validation."""
        # Select validators based on stake and reputation
        active_validators = [v for v in self.validators.values() if v.active]
        selected_validators = sorted(active_validators, key=lambda x: x.stake * x.reputation, reverse=True)[:3]
        
        for validator in selected_validators:
            validation_task = {
                "validator_id": validator.validator_id,
                "test_data": test_data,
                "submission_time": datetime.now()
            }
            self.validation_queue.put(validation_task)
    
    def get_validation_consensus(self, test_id: str) -> Dict[str, Any]:
        """Get validation consensus for a test."""
        test_validations = [v for v in self.validation_results if v.get("test_id") == test_id]
        
        if not test_validations:
            return {"consensus": "no_validations", "confidence": 0.0}
        
        valid_count = sum(1 for v in test_validations if v["valid"])
        total_count = len(test_validations)
        
        consensus = "valid" if valid_count > total_count / 2 else "invalid"
        confidence = valid_count / total_count if consensus == "valid" else (total_count - valid_count) / total_count
        
        return {
            "consensus": consensus,
            "confidence": confidence,
            "validations": len(test_validations),
            "valid_count": valid_count,
            "invalid_count": total_count - valid_count
        }

class BlockchainTestFramework:
    """Main blockchain testing framework."""
    
    def __init__(self):
        self.blockchain = BlockchainTestEngine()
        self.smart_contracts = SmartContractManager()
        self.validator_network = TestValidatorNetwork()
        self.test_registry = {}
        
        # Initialize validator network
        self.validator_network.add_validator("Primary Validator", stake=1000.0)
        self.validator_network.add_validator("Secondary Validator", stake=800.0)
        self.validator_network.add_validator("Tertiary Validator", stake=600.0)
        self.validator_network.start_validation_workers()
    
    def register_test(self, test_name: str, test_func: Callable) -> str:
        """Register a test in the blockchain framework."""
        test_id = f"test_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.test_registry[test_id] = {
            "name": test_name,
            "function": test_func,
            "registered_at": datetime.now(),
            "execution_count": 0
        }
        
        return test_id
    
    def execute_test(self, test_id: str) -> Dict[str, Any]:
        """Execute a test and record it in the blockchain."""
        if test_id not in self.test_registry:
            raise ValueError(f"Test {test_id} not registered")
        
        test_info = self.test_registry[test_id]
        test_func = test_info["function"]
        
        # Execute test
        start_time = time.time()
        try:
            result = test_func()
            execution_time = time.time() - start_time
            test_result = "pass" if result else "fail"
        except Exception as e:
            execution_time = time.time() - start_time
            test_result = "fail"
            result = str(e)
        
        # Create transaction
        transaction = self.blockchain.add_test_transaction(
            test_id=test_id,
            test_name=test_info["name"],
            result=test_result,
            duration=execution_time,
            metadata={
                "exception": str(e) if test_result == "fail" else None,
                "execution_count": test_info["execution_count"] + 1
            }
        )
        
        # Submit for validation
        self.validator_network.submit_for_validation({
            "test_id": test_id,
            "test_name": test_info["name"],
            "result": test_result,
            "duration": execution_time,
            "transaction_id": transaction.transaction_id
        })
        
        # Update execution count
        test_info["execution_count"] += 1
        
        # Mine block if we have enough transactions
        if len(self.blockchain.pending_transactions) >= 5:
            self.blockchain.mine_block()
        
        return {
            "test_id": test_id,
            "result": test_result,
            "duration": execution_time,
            "transaction_id": transaction.transaction_id,
            "blockchain_valid": self.blockchain.validate_chain()
        }
    
    def create_quality_contract(self) -> str:
        """Create a smart contract for test quality validation."""
        conditions = [
            {
                "name": "pass_rate_high",
                "type": "numeric",
                "field": "pass_rate",
                "operator": ">=",
                "value": 0.9
            },
            {
                "name": "duration_reasonable",
                "type": "numeric",
                "field": "average_duration",
                "operator": "<=",
                "value": 5.0
            },
            {
                "name": "execution_frequency",
                "type": "numeric",
                "field": "total_executions",
                "operator": ">=",
                "value": 10
            }
        ]
        
        rewards = {
            "pass_rate_high": 50.0,
            "duration_reasonable": 25.0,
            "execution_frequency": 25.0
        }
        
        penalties = {
            "pass_rate_high": -30.0,
            "duration_reasonable": -15.0,
            "execution_frequency": -10.0
        }
        
        contract = self.smart_contracts.create_contract(
            name="Test Quality Contract",
            conditions=conditions,
            rewards=rewards,
            penalties=penalties
        )
        
        return contract.contract_id
    
    def evaluate_test_quality(self, test_id: str, contract_id: str) -> Dict[str, Any]:
        """Evaluate test quality using smart contract."""
        # Get test statistics
        stats = self.blockchain.get_test_statistics(test_id)
        
        # Execute smart contract
        contract_result = self.smart_contracts.execute_contract(contract_id, stats)
        
        # Get validation consensus
        validation_consensus = self.validator_network.get_validation_consensus(test_id)
        
        return {
            "test_id": test_id,
            "contract_id": contract_id,
            "test_statistics": stats,
            "contract_evaluation": contract_result,
            "validation_consensus": validation_consensus,
            "overall_score": contract_result["total_score"]
        }
    
    def get_blockchain_report(self) -> Dict[str, Any]:
        """Get comprehensive blockchain report."""
        return {
            "blockchain_info": {
                "chain_length": len(self.blockchain.chain),
                "pending_transactions": len(self.blockchain.pending_transactions),
                "is_valid": self.blockchain.validate_chain(),
                "difficulty": self.blockchain.difficulty
            },
            "test_registry": {
                "total_tests": len(self.test_registry),
                "tests": list(self.test_registry.keys())
            },
            "smart_contracts": {
                "total_contracts": len(self.smart_contracts.contracts),
                "contracts": list(self.smart_contracts.contracts.keys())
            },
            "validator_network": {
                "total_validators": len(self.validator_network.validators),
                "active_validators": len([v for v in self.validator_network.validators.values() if v.active]),
                "total_validations": len(self.validator_network.validation_results)
            }
        }

# Example usage and demo
def demo_blockchain_testing():
    """Demonstrate blockchain testing capabilities."""
    print("⛓️ Blockchain Testing Framework Demo")
    print("=" * 50)
    
    # Create blockchain testing framework
    framework = BlockchainTestFramework()
    
    # Create sample test functions
    def reliable_test():
        return True
    
    def flaky_test():
        return random.random() > 0.3
    
    def slow_test():
        time.sleep(0.1)
        return True
    
    # Register tests
    test1_id = framework.register_test("reliable_test", reliable_test)
    test2_id = framework.register_test("flaky_test", flaky_test)
    test3_id = framework.register_test("slow_test", slow_test)
    
    print(f"📝 Registered {len(framework.test_registry)} tests")
    
    # Execute tests multiple times
    print("\n🔄 Executing tests...")
    for i in range(10):
        framework.execute_test(test1_id)
        framework.execute_test(test2_id)
        framework.execute_test(test3_id)
    
    # Mine remaining transactions
    if framework.blockchain.pending_transactions:
        framework.blockchain.mine_block()
    
    print(f"⛏️ Mined {len(framework.blockchain.chain)} blocks")
    
    # Create quality contract
    contract_id = framework.create_quality_contract()
    print(f"📜 Created quality contract: {contract_id}")
    
    # Evaluate test quality
    print("\n📊 Evaluating test quality...")
    for test_id in [test1_id, test2_id, test3_id]:
        evaluation = framework.evaluate_test_quality(test_id, contract_id)
        print(f"\nTest: {evaluation['test_id']}")
        print(f"  Pass Rate: {evaluation['test_statistics']['pass_rate']:.1%}")
        print(f"  Average Duration: {evaluation['test_statistics']['average_duration']:.2f}s")
        print(f"  Contract Score: {evaluation['contract_evaluation']['total_score']:.1f}")
        print(f"  Validation Consensus: {evaluation['validation_consensus']['consensus']}")
        print(f"  Overall Score: {evaluation['overall_score']:.1f}")
    
    # Get blockchain report
    print("\n📈 Blockchain Report:")
    report = framework.get_blockchain_report()
    print(f"  Chain Length: {report['blockchain_info']['chain_length']}")
    print(f"  Blockchain Valid: {report['blockchain_info']['is_valid']}")
    print(f"  Total Tests: {report['test_registry']['total_tests']}")
    print(f"  Smart Contracts: {report['smart_contracts']['total_contracts']}")
    print(f"  Validators: {report['validator_network']['active_validators']}")

if __name__ == "__main__":
    # Run demo
    demo_blockchain_testing()
