"""
Gamma App - Real Improvement Blockchain
Blockchain system for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class BlockchainType(Enum):
    """Blockchain types"""
    PUBLIC = "public"
    PRIVATE = "private"
    CONSORTIUM = "consortium"
    HYBRID = "hybrid"

class ConsensusAlgorithm(Enum):
    """Consensus algorithms"""
    PROOF_OF_WORK = "proof_of_work"
    PROOF_OF_STAKE = "proof_of_stake"
    PROOF_OF_AUTHORITY = "proof_of_authority"
    DELEGATED_PROOF_OF_STAKE = "delegated_proof_of_stake"
    PRACTICAL_BYZANTINE_FAULT_TOLERANCE = "pbft"

@dataclass
class Block:
    """Blockchain block"""
    index: int
    timestamp: datetime
    data: Dict[str, Any]
    previous_hash: str
    hash: str
    nonce: int = 0
    difficulty: int = 4

    def __post_init__(self):
        if not self.hash:
            self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = f"{self.index}{self.timestamp.isoformat()}{json.dumps(self.data, sort_keys=True)}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int):
        """Mine block with given difficulty"""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

@dataclass
class Transaction:
    """Blockchain transaction"""
    transaction_id: str
    from_address: str
    to_address: str
    amount: float
    data: Dict[str, Any]
    timestamp: datetime = None
    signature: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def calculate_hash(self) -> str:
        """Calculate transaction hash"""
        transaction_string = f"{self.transaction_id}{self.from_address}{self.to_address}{self.amount}{json.dumps(self.data, sort_keys=True)}{self.timestamp.isoformat()}"
        return hashlib.sha256(transaction_string.encode()).hexdigest()

@dataclass
class SmartContract:
    """Smart contract"""
    contract_id: str
    name: str
    code: str
    address: str
    creator: str
    created_at: datetime = None
    status: str = "active"

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RealImprovementBlockchain:
    """
    Blockchain system for real improvements
    """
    
    def __init__(self, project_root: str = ".", blockchain_type: BlockchainType = BlockchainType.PRIVATE):
        """Initialize blockchain system"""
        self.project_root = Path(project_root)
        self.blockchain_type = blockchain_type
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.blockchain_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.mining_reward = 1.0
        self.difficulty = 4
        self.mining_threads = []
        self.mining_active = False
        
        # Initialize database
        self._init_blockchain_database()
        
        # Create genesis block
        self._create_genesis_block()
        
        # Start mining
        self._start_mining()
        
        logger.info(f"Real Improvement Blockchain initialized for {self.project_root}")
    
    def _init_blockchain_database(self):
        """Initialize blockchain database"""
        try:
            conn = sqlite3.connect("blockchain.db")
            cursor = conn.cursor()
            
            # Create blocks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    index INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    nonce INTEGER NOT NULL,
                    difficulty INTEGER NOT NULL
                )
            ''')
            
            # Create transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    from_address TEXT NOT NULL,
                    to_address TEXT NOT NULL,
                    amount REAL NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    signature TEXT,
                    block_index INTEGER,
                    FOREIGN KEY (block_index) REFERENCES blocks (index)
                )
            ''')
            
            # Create smart contracts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS smart_contracts (
                    contract_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    code TEXT NOT NULL,
                    address TEXT NOT NULL,
                    creator TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize blockchain database: {e}")
    
    def _create_genesis_block(self):
        """Create genesis block"""
        try:
            genesis_data = {
                "message": "Genesis block for Real Improvement Blockchain",
                "timestamp": datetime.utcnow().isoformat(),
                "creator": "system"
            }
            
            genesis_block = Block(
                index=0,
                timestamp=datetime.utcnow(),
                data=genesis_data,
                previous_hash="0",
                hash="",
                nonce=0,
                difficulty=self.difficulty
            )
            
            # Mine genesis block
            genesis_block.mine_block(self.difficulty)
            
            self.chain.append(genesis_block)
            self._save_block_to_db(genesis_block)
            
            self._log_blockchain("genesis_created", "Genesis block created")
            
        except Exception as e:
            logger.error(f"Failed to create genesis block: {e}")
    
    def _start_mining(self):
        """Start blockchain mining"""
        try:
            self.mining_active = True
            
            # Start mining thread
            mining_thread = threading.Thread(target=self._mine_blocks, daemon=True)
            mining_thread.start()
            self.mining_threads.append(mining_thread)
            
            self._log_blockchain("mining_started", "Blockchain mining started")
            
        except Exception as e:
            logger.error(f"Failed to start mining: {e}")
    
    def _mine_blocks(self):
        """Mine blocks continuously"""
        while self.mining_active:
            try:
                if self.pending_transactions:
                    # Create new block
                    new_block = self._create_new_block()
                    
                    # Mine block
                    new_block.mine_block(self.difficulty)
                    
                    # Add to chain
                    self.chain.append(new_block)
                    self._save_block_to_db(new_block)
                    
                    # Clear pending transactions
                    self.pending_transactions.clear()
                    
                    self._log_blockchain("block_mined", f"Block {new_block.index} mined")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Failed to mine block: {e}")
                time.sleep(5)
    
    def _create_new_block(self) -> Block:
        """Create new block"""
        try:
            previous_block = self.chain[-1]
            
            # Prepare block data
            block_data = {
                "transactions": [
                    {
                        "transaction_id": t.transaction_id,
                        "from_address": t.from_address,
                        "to_address": t.to_address,
                        "amount": t.amount,
                        "data": t.data,
                        "timestamp": t.timestamp.isoformat()
                    }
                    for t in self.pending_transactions
                ],
                "mining_reward": self.mining_reward,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            new_block = Block(
                index=len(self.chain),
                timestamp=datetime.utcnow(),
                data=block_data,
                previous_hash=previous_block.hash,
                hash="",
                nonce=0,
                difficulty=self.difficulty
            )
            
            return new_block
            
        except Exception as e:
            logger.error(f"Failed to create new block: {e}")
            raise
    
    def add_transaction(self, from_address: str, to_address: str, amount: float, data: Dict[str, Any]) -> str:
        """Add transaction to blockchain"""
        try:
            transaction_id = f"tx_{int(time.time() * 1000)}"
            
            transaction = Transaction(
                transaction_id=transaction_id,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                data=data
            )
            
            self.pending_transactions.append(transaction)
            
            self._log_blockchain("transaction_added", f"Transaction {transaction_id} added")
            
            return transaction_id
            
        except Exception as e:
            logger.error(f"Failed to add transaction: {e}")
            raise
    
    def create_smart_contract(self, name: str, code: str, creator: str) -> str:
        """Create smart contract"""
        try:
            contract_id = f"contract_{int(time.time() * 1000)}"
            address = f"0x{hashlib.sha256(contract_id.encode()).hexdigest()[:40]}"
            
            smart_contract = SmartContract(
                contract_id=contract_id,
                name=name,
                code=code,
                address=address,
                creator=creator
            )
            
            self.smart_contracts[contract_id] = smart_contract
            self._save_smart_contract_to_db(smart_contract)
            
            # Add transaction for contract creation
            self.add_transaction(
                from_address="system",
                to_address=address,
                amount=0.0,
                data={
                    "type": "contract_creation",
                    "contract_id": contract_id,
                    "contract_name": name
                }
            )
            
            self._log_blockchain("contract_created", f"Smart contract {name} created")
            
            return contract_id
            
        except Exception as e:
            logger.error(f"Failed to create smart contract: {e}")
            raise
    
    def execute_smart_contract(self, contract_id: str, function_name: str, parameters: Dict[str, Any], caller: str) -> Dict[str, Any]:
        """Execute smart contract function"""
        try:
            if contract_id not in self.smart_contracts:
                return {"error": "Contract not found"}
            
            contract = self.smart_contracts[contract_id]
            
            # Simulate contract execution
            result = self._simulate_contract_execution(contract, function_name, parameters)
            
            # Add transaction for contract execution
            self.add_transaction(
                from_address=caller,
                to_address=contract.address,
                amount=0.0,
                data={
                    "type": "contract_execution",
                    "contract_id": contract_id,
                    "function": function_name,
                    "parameters": parameters,
                    "result": result
                }
            )
            
            self._log_blockchain("contract_executed", f"Executed {function_name} on contract {contract.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute smart contract: {e}")
            return {"error": str(e)}
    
    def _simulate_contract_execution(self, contract: SmartContract, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate smart contract execution"""
        try:
            # Simple contract execution simulation
            if function_name == "get_balance":
                return {"balance": 1000.0}
            elif function_name == "transfer":
                amount = parameters.get("amount", 0.0)
                return {"success": True, "amount": amount}
            elif function_name == "get_info":
                return {
                    "contract_id": contract.contract_id,
                    "name": contract.name,
                    "address": contract.address,
                    "creator": contract.creator
                }
            else:
                return {"error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_block(self, index: int) -> Optional[Dict[str, Any]]:
        """Get block by index"""
        try:
            if 0 <= index < len(self.chain):
                block = self.chain[index]
                return {
                    "index": block.index,
                    "timestamp": block.timestamp.isoformat(),
                    "data": block.data,
                    "previous_hash": block.previous_hash,
                    "hash": block.hash,
                    "nonce": block.nonce,
                    "difficulty": block.difficulty
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get block: {e}")
            return None
    
    def get_latest_block(self) -> Optional[Dict[str, Any]]:
        """Get latest block"""
        try:
            if self.chain:
                return self.get_block(len(self.chain) - 1)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest block: {e}")
            return None
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Get blockchain information"""
        try:
            total_blocks = len(self.chain)
            pending_transactions = len(self.pending_transactions)
            total_contracts = len(self.smart_contracts)
            
            # Calculate total transactions
            total_transactions = 0
            for block in self.chain:
                if "transactions" in block.data:
                    total_transactions += len(block.data["transactions"])
            
            # Calculate blockchain size (approximate)
            blockchain_size = sum(len(json.dumps(block.data)) for block in self.chain)
            
            return {
                "blockchain_type": self.blockchain_type.value,
                "total_blocks": total_blocks,
                "pending_transactions": pending_transactions,
                "total_transactions": total_transactions,
                "total_contracts": total_contracts,
                "mining_reward": self.mining_reward,
                "difficulty": self.difficulty,
                "blockchain_size_bytes": blockchain_size,
                "mining_active": self.mining_active,
                "consensus_algorithm": "proof_of_work"
            }
            
        except Exception as e:
            logger.error(f"Failed to get blockchain info: {e}")
            return {}
    
    def validate_blockchain(self) -> bool:
        """Validate blockchain integrity"""
        try:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i - 1]
                
                # Check if current block hash is valid
                if current_block.hash != current_block.calculate_hash():
                    self._log_blockchain("validation_failed", f"Invalid hash for block {i}")
                    return False
                
                # Check if current block points to previous block
                if current_block.previous_hash != previous_block.hash:
                    self._log_blockchain("validation_failed", f"Invalid previous hash for block {i}")
                    return False
            
            self._log_blockchain("validation_passed", "Blockchain validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate blockchain: {e}")
            return False
    
    def _save_block_to_db(self, block: Block):
        """Save block to database"""
        try:
            conn = sqlite3.connect("blockchain.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO blocks 
                (index, timestamp, data, previous_hash, hash, nonce, difficulty)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                block.index,
                block.timestamp.isoformat(),
                json.dumps(block.data),
                block.previous_hash,
                block.hash,
                block.nonce,
                block.difficulty
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save block to database: {e}")
    
    def _save_smart_contract_to_db(self, contract: SmartContract):
        """Save smart contract to database"""
        try:
            conn = sqlite3.connect("blockchain.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO smart_contracts 
                (contract_id, name, code, address, creator, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                contract.contract_id,
                contract.name,
                contract.code,
                contract.address,
                contract.creator,
                contract.created_at.isoformat(),
                contract.status
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save smart contract to database: {e}")
    
    def _log_blockchain(self, event: str, message: str):
        """Log blockchain event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if "blockchain_logs" not in self.blockchain_logs:
            self.blockchain_logs["blockchain_logs"] = []
        
        self.blockchain_logs["blockchain_logs"].append(log_entry)
        
        logger.info(f"Blockchain: {event} - {message}")
    
    def get_blockchain_logs(self) -> List[Dict[str, Any]]:
        """Get blockchain logs"""
        return self.blockchain_logs.get("blockchain_logs", [])
    
    def shutdown(self):
        """Shutdown blockchain system"""
        try:
            self.mining_active = False
            
            # Wait for mining threads to finish
            for thread in self.mining_threads:
                thread.join(timeout=5)
            
            self._log_blockchain("shutdown", "Blockchain system shutdown completed")
            
        except Exception as e:
            logger.error(f"Failed to shutdown blockchain system: {e}")

# Global blockchain instance
improvement_blockchain = None

def get_improvement_blockchain() -> RealImprovementBlockchain:
    """Get improvement blockchain instance"""
    global improvement_blockchain
    if not improvement_blockchain:
        improvement_blockchain = RealImprovementBlockchain()
    return improvement_blockchain













