"""
Blockchain Support for Decentralized Content Verification
Sistema Blockchain para verificación de contenido descentralizada ultra-optimizado
"""

import asyncio
import logging
import time
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import threading
import base64

logger = logging.getLogger(__name__)


class BlockchainType(Enum):
    """Tipos de blockchain"""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    POLYGON = "polygon"
    BINANCE_SMART_CHAIN = "binance_smart_chain"
    SOLANA = "solana"
    CARDANO = "cardano"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"


class TransactionType(Enum):
    """Tipos de transacciones"""
    CONTENT_HASH = "content_hash"
    SIMILARITY_PROOF = "similarity_proof"
    QUALITY_CERTIFICATE = "quality_certificate"
    OWNERSHIP_CLAIM = "ownership_claim"
    VERIFICATION_REQUEST = "verification_request"
    AUDIT_LOG = "audit_log"


class HashAlgorithm(Enum):
    """Algoritmos de hash"""
    SHA256 = "sha256"
    SHA1 = "sha1"
    MD5 = "md5"
    BLAKE2B = "blake2b"
    KECCAK256 = "keccak256"
    RIPEMD160 = "ripemd160"


@dataclass
class BlockchainTransaction:
    """Transacción blockchain"""
    id: str
    type: TransactionType
    content_hash: str
    data: Dict[str, Any]
    timestamp: float
    block_number: Optional[int]
    transaction_hash: Optional[str]
    gas_used: Optional[int]
    gas_price: Optional[int]
    status: str
    metadata: Dict[str, Any]


@dataclass
class BlockchainBlock:
    """Bloque blockchain"""
    number: int
    hash: str
    previous_hash: str
    timestamp: float
    transactions: List[str]
    merkle_root: str
    nonce: int
    difficulty: int
    miner: str
    size: int


@dataclass
class SmartContract:
    """Contrato inteligente"""
    address: str
    name: str
    blockchain: BlockchainType
    abi: List[Dict[str, Any]]
    bytecode: str
    deployed_at: float
    creator: str
    functions: List[str]
    events: List[str]


@dataclass
class MerkleTree:
    """Árbol Merkle"""
    root: str
    leaves: List[str]
    levels: List[List[str]]
    depth: int


class HashGenerator:
    """Generador de hashes"""
    
    @staticmethod
    def generate_hash(content: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Generar hash del contenido"""
        content_bytes = content.encode('utf-8')
        
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(content_bytes).hexdigest()
        elif algorithm == HashAlgorithm.SHA1:
            return hashlib.sha1(content_bytes).hexdigest()
        elif algorithm == HashAlgorithm.MD5:
            return hashlib.md5(content_bytes).hexdigest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(content_bytes).hexdigest()
        elif algorithm == HashAlgorithm.KECCAK256:
            # Simulación de Keccak256
            return hashlib.sha3_256(content_bytes).hexdigest()
        elif algorithm == HashAlgorithm.RIPEMD160:
            return hashlib.new('ripemd160', content_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    @staticmethod
    def generate_merkle_tree(transactions: List[str]) -> MerkleTree:
        """Generar árbol Merkle"""
        if not transactions:
            return MerkleTree("", [], [], 0)
        
        # Crear hojas
        leaves = [HashGenerator.generate_hash(tx) for tx in transactions]
        levels = [leaves]
        
        # Construir niveles
        current_level = leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                combined = left + right
                next_level.append(HashGenerator.generate_hash(combined))
            levels.append(next_level)
            current_level = next_level
        
        root = current_level[0] if current_level else ""
        depth = len(levels) - 1
        
        return MerkleTree(root, leaves, levels, depth)
    
    @staticmethod
    def verify_merkle_proof(leaf: str, proof: List[str], root: str) -> bool:
        """Verificar prueba Merkle"""
        current_hash = leaf
        
        for sibling in proof:
            # Determinar orden (izquierda o derecha)
            if current_hash < sibling:
                combined = current_hash + sibling
            else:
                combined = sibling + current_hash
            current_hash = HashGenerator.generate_hash(combined)
        
        return current_hash == root


class BlockchainSimulator:
    """Simulador de blockchain"""
    
    def __init__(self, blockchain_type: BlockchainType):
        self.blockchain_type = blockchain_type
        self.blocks: Dict[int, BlockchainBlock] = {}
        self.transactions: Dict[str, BlockchainTransaction] = {}
        self.pending_transactions: List[str] = []
        self.current_block_number = 0
        self.difficulty = 4  # Número de ceros al inicio del hash
        self._lock = threading.Lock()
    
    def create_transaction(self, tx_type: TransactionType, content_hash: str, 
                          data: Dict[str, Any]) -> str:
        """Crear transacción"""
        tx_id = f"tx_{int(time.time())}_{id(data)}"
        
        transaction = BlockchainTransaction(
            id=tx_id,
            type=tx_type,
            content_hash=content_hash,
            data=data,
            timestamp=time.time(),
            block_number=None,
            transaction_hash=None,
            gas_used=None,
            gas_price=None,
            status="pending",
            metadata={}
        )
        
        with self._lock:
            self.transactions[tx_id] = transaction
            self.pending_transactions.append(tx_id)
        
        return tx_id
    
    def mine_block(self) -> BlockchainBlock:
        """Minar bloque"""
        with self._lock:
            if not self.pending_transactions:
                return None
            
            # Crear nuevo bloque
            block_number = self.current_block_number + 1
            previous_hash = self.blocks[self.current_block_number].hash if self.current_block_number in self.blocks else "0" * 64
            
            # Seleccionar transacciones para el bloque
            block_transactions = self.pending_transactions[:10]  # Máximo 10 transacciones por bloque
            
            # Generar Merkle root
            merkle_tree = HashGenerator.generate_merkle_tree(block_transactions)
            
            # Minar bloque (simulación)
            nonce = 0
            timestamp = time.time()
            
            while True:
                block_data = f"{block_number}{previous_hash}{merkle_tree.root}{nonce}{timestamp}"
                block_hash = HashGenerator.generate_hash(block_data)
                
                if block_hash.startswith("0" * self.difficulty):
                    break
                
                nonce += 1
            
            # Crear bloque
            block = BlockchainBlock(
                number=block_number,
                hash=block_hash,
                previous_hash=previous_hash,
                timestamp=timestamp,
                transactions=block_transactions,
                merkle_root=merkle_tree.root,
                nonce=nonce,
                difficulty=self.difficulty,
                miner="miner_001",
                size=len(json.dumps(block_transactions))
            )
            
            # Actualizar transacciones
            for tx_id in block_transactions:
                if tx_id in self.transactions:
                    self.transactions[tx_id].block_number = block_number
                    self.transactions[tx_id].transaction_hash = HashGenerator.generate_hash(tx_id)
                    self.transactions[tx_id].status = "confirmed"
                    self.transactions[tx_id].gas_used = 21000  # Gas estándar
                    self.transactions[tx_id].gas_price = 20  # Gwei
            
            # Actualizar estado
            self.blocks[block_number] = block
            self.current_block_number = block_number
            self.pending_transactions = [tx for tx in self.pending_transactions if tx not in block_transactions]
            
            return block
    
    def get_transaction(self, tx_id: str) -> Optional[BlockchainTransaction]:
        """Obtener transacción"""
        with self._lock:
            return self.transactions.get(tx_id)
    
    def get_block(self, block_number: int) -> Optional[BlockchainBlock]:
        """Obtener bloque"""
        with self._lock:
            return self.blocks.get(block_number)
    
    def get_latest_block(self) -> Optional[BlockchainBlock]:
        """Obtener último bloque"""
        with self._lock:
            if self.current_block_number in self.blocks:
                return self.blocks[self.current_block_number]
            return None
    
    def verify_transaction(self, tx_id: str) -> bool:
        """Verificar transacción"""
        with self._lock:
            if tx_id not in self.transactions:
                return False
            
            transaction = self.transactions[tx_id]
            return transaction.status == "confirmed" and transaction.block_number is not None


class SmartContractManager:
    """Manager de contratos inteligentes"""
    
    def __init__(self):
        self.contracts: Dict[str, SmartContract] = {}
        self.deployed_contracts: Dict[str, str] = {}  # address -> contract_id
    
    def create_contract(self, name: str, blockchain: BlockchainType, 
                       abi: List[Dict[str, Any]], bytecode: str, creator: str) -> str:
        """Crear contrato inteligente"""
        contract_id = f"contract_{int(time.time())}_{id(name)}"
        
        # Generar dirección del contrato
        address = HashGenerator.generate_hash(contract_id)[:42]  # 40 caracteres + 0x
        
        contract = SmartContract(
            address=address,
            name=name,
            blockchain=blockchain,
            abi=abi,
            bytecode=bytecode,
            deployed_at=time.time(),
            creator=creator,
            functions=[item["name"] for item in abi if item["type"] == "function"],
            events=[item["name"] for item in abi if item["type"] == "event"]
        )
        
        self.contracts[contract_id] = contract
        self.deployed_contracts[address] = contract_id
        
        return contract_id
    
    def get_contract(self, contract_id: str) -> Optional[SmartContract]:
        """Obtener contrato"""
        return self.contracts.get(contract_id)
    
    def get_contract_by_address(self, address: str) -> Optional[SmartContract]:
        """Obtener contrato por dirección"""
        if address in self.deployed_contracts:
            contract_id = self.deployed_contracts[address]
            return self.contracts.get(contract_id)
        return None
    
    def call_contract_function(self, address: str, function_name: str, 
                              parameters: List[Any]) -> Dict[str, Any]:
        """Llamar función del contrato"""
        contract = self.get_contract_by_address(address)
        if not contract:
            raise ValueError(f"Contract not found at address {address}")
        
        if function_name not in contract.functions:
            raise ValueError(f"Function {function_name} not found in contract")
        
        # Simular ejecución de función
        result = {
            "contract_address": address,
            "function_name": function_name,
            "parameters": parameters,
            "result": f"Function {function_name} executed successfully",
            "gas_used": 50000,
            "execution_time": 0.1,
            "timestamp": time.time()
        }
        
        return result


class ContentVerification:
    """Verificación de contenido"""
    
    def __init__(self, blockchain_simulator: BlockchainSimulator):
        self.blockchain = blockchain_simulator
        self.verification_records: Dict[str, Dict[str, Any]] = {}
    
    def verify_content_hash(self, content: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> Dict[str, Any]:
        """Verificar hash de contenido"""
        content_hash = HashGenerator.generate_hash(content, algorithm)
        
        # Buscar transacciones con este hash
        matching_transactions = []
        for tx in self.blockchain.transactions.values():
            if tx.content_hash == content_hash:
                matching_transactions.append(tx)
        
        verification_result = {
            "content_hash": content_hash,
            "algorithm": algorithm.value,
            "is_verified": len(matching_transactions) > 0,
            "matching_transactions": len(matching_transactions),
            "first_verified_at": min([tx.timestamp for tx in matching_transactions]) if matching_transactions else None,
            "verification_count": len(matching_transactions),
            "block_numbers": [tx.block_number for tx in matching_transactions if tx.block_number],
            "timestamp": time.time()
        }
        
        return verification_result
    
    def create_verification_proof(self, content: str, tx_type: TransactionType, 
                                 additional_data: Dict[str, Any] = None) -> str:
        """Crear prueba de verificación"""
        content_hash = HashGenerator.generate_hash(content)
        
        data = {
            "content_length": len(content),
            "verification_timestamp": time.time(),
            "verifier": "content_redundancy_detector",
            **(additional_data or {})
        }
        
        tx_id = self.blockchain.create_transaction(tx_type, content_hash, data)
        
        # Intentar minar bloque inmediatamente
        self.blockchain.mine_block()
        
        return tx_id
    
    def get_verification_history(self, content_hash: str) -> List[Dict[str, Any]]:
        """Obtener historial de verificación"""
        history = []
        
        for tx in self.blockchain.transactions.values():
            if tx.content_hash == content_hash:
                history.append({
                    "transaction_id": tx.id,
                    "type": tx.type.value,
                    "timestamp": tx.timestamp,
                    "block_number": tx.block_number,
                    "status": tx.status,
                    "data": tx.data
                })
        
        return sorted(history, key=lambda x: x["timestamp"])


class BlockchainManager:
    """Manager principal de blockchain"""
    
    def __init__(self):
        self.blockchains: Dict[BlockchainType, BlockchainSimulator] = {}
        self.smart_contracts = SmartContractManager()
        self.content_verification: Dict[BlockchainType, ContentVerification] = {}
        self.is_running = False
        self._mining_task = None
    
    async def start(self):
        """Iniciar blockchain manager"""
        try:
            self.is_running = True
            
            # Inicializar blockchains
            for blockchain_type in BlockchainType:
                self.blockchains[blockchain_type] = BlockchainSimulator(blockchain_type)
                self.content_verification[blockchain_type] = ContentVerification(
                    self.blockchains[blockchain_type]
                )
            
            # Iniciar minería automática
            self._mining_task = asyncio.create_task(self._mining_loop())
            
            logger.info("Blockchain manager started")
            
        except Exception as e:
            logger.error(f"Error starting blockchain manager: {e}")
            raise
    
    async def stop(self):
        """Detener blockchain manager"""
        try:
            self.is_running = False
            
            if self._mining_task:
                self._mining_task.cancel()
                try:
                    await self._mining_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Blockchain manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping blockchain manager: {e}")
    
    async def _mining_loop(self):
        """Loop de minería"""
        while self.is_running:
            try:
                # Minar bloques en todas las blockchains
                for blockchain in self.blockchains.values():
                    if blockchain.pending_transactions:
                        blockchain.mine_block()
                
                await asyncio.sleep(10)  # Minar cada 10 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in mining loop: {e}")
                await asyncio.sleep(10)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        stats = {
            "is_running": self.is_running,
            "blockchains": {},
            "smart_contracts": {
                "total_contracts": len(self.smart_contracts.contracts),
                "deployed_contracts": len(self.smart_contracts.deployed_contracts)
            }
        }
        
        for blockchain_type, blockchain in self.blockchains.items():
            stats["blockchains"][blockchain_type.value] = {
                "blocks": len(blockchain.blocks),
                "transactions": len(blockchain.transactions),
                "pending_transactions": len(blockchain.pending_transactions),
                "current_block_number": blockchain.current_block_number,
                "difficulty": blockchain.difficulty
            }
        
        return stats


# Instancia global del manager de blockchain
blockchain_manager = BlockchainManager()


# Router para endpoints de blockchain
blockchain_router = APIRouter()


@blockchain_router.post("/blockchain/transactions/create")
async def create_blockchain_transaction_endpoint(transaction_data: dict):
    """Crear transacción blockchain"""
    try:
        blockchain_type = BlockchainType(transaction_data["blockchain"])
        tx_type = TransactionType(transaction_data["type"])
        content_hash = transaction_data["content_hash"]
        data = transaction_data.get("data", {})
        
        blockchain = blockchain_manager.blockchains[blockchain_type]
        tx_id = blockchain.create_transaction(tx_type, content_hash, data)
        
        return {
            "message": "Blockchain transaction created successfully",
            "transaction_id": tx_id,
            "blockchain": blockchain_type.value,
            "type": tx_type.value,
            "content_hash": content_hash
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid blockchain type or transaction type: {e}")
    except Exception as e:
        logger.error(f"Error creating blockchain transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create blockchain transaction: {str(e)}")


@blockchain_router.get("/blockchain/transactions/{tx_id}")
async def get_blockchain_transaction_endpoint(tx_id: str, blockchain: str):
    """Obtener transacción blockchain"""
    try:
        blockchain_type = BlockchainType(blockchain)
        blockchain_sim = blockchain_manager.blockchains[blockchain_type]
        transaction = blockchain_sim.get_transaction(tx_id)
        
        if transaction:
            return {
                "id": transaction.id,
                "type": transaction.type.value,
                "content_hash": transaction.content_hash,
                "data": transaction.data,
                "timestamp": transaction.timestamp,
                "block_number": transaction.block_number,
                "transaction_hash": transaction.transaction_hash,
                "gas_used": transaction.gas_used,
                "gas_price": transaction.gas_price,
                "status": transaction.status,
                "metadata": transaction.metadata
            }
        else:
            raise HTTPException(status_code=404, detail="Transaction not found")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid blockchain type: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting blockchain transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain transaction: {str(e)}")


@blockchain_router.get("/blockchain/blocks/{block_number}")
async def get_blockchain_block_endpoint(block_number: int, blockchain: str):
    """Obtener bloque blockchain"""
    try:
        blockchain_type = BlockchainType(blockchain)
        blockchain_sim = blockchain_manager.blockchains[blockchain_type]
        block = blockchain_sim.get_block(block_number)
        
        if block:
            return {
                "number": block.number,
                "hash": block.hash,
                "previous_hash": block.previous_hash,
                "timestamp": block.timestamp,
                "transactions": block.transactions,
                "merkle_root": block.merkle_root,
                "nonce": block.nonce,
                "difficulty": block.difficulty,
                "miner": block.miner,
                "size": block.size
            }
        else:
            raise HTTPException(status_code=404, detail="Block not found")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid blockchain type: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting blockchain block: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain block: {str(e)}")


@blockchain_router.get("/blockchain/blocks/latest")
async def get_latest_blockchain_block_endpoint(blockchain: str):
    """Obtener último bloque blockchain"""
    try:
        blockchain_type = BlockchainType(blockchain)
        blockchain_sim = blockchain_manager.blockchains[blockchain_type]
        block = blockchain_sim.get_latest_block()
        
        if block:
            return {
                "number": block.number,
                "hash": block.hash,
                "previous_hash": block.previous_hash,
                "timestamp": block.timestamp,
                "transactions": block.transactions,
                "merkle_root": block.merkle_root,
                "nonce": block.nonce,
                "difficulty": block.difficulty,
                "miner": block.miner,
                "size": block.size
            }
        else:
            raise HTTPException(status_code=404, detail="No blocks found")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid blockchain type: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest blockchain block: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest blockchain block: {str(e)}")


@blockchain_router.post("/blockchain/content/verify")
async def verify_content_hash_endpoint(verification_data: dict):
    """Verificar hash de contenido"""
    try:
        blockchain_type = BlockchainType(verification_data["blockchain"])
        content = verification_data["content"]
        algorithm = HashAlgorithm(verification_data.get("algorithm", "sha256"))
        
        verification = blockchain_manager.content_verification[blockchain_type]
        result = verification.verify_content_hash(content, algorithm)
        
        return {
            "message": "Content hash verification completed",
            "verification_result": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid blockchain type or algorithm: {e}")
    except Exception as e:
        logger.error(f"Error verifying content hash: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify content hash: {str(e)}")


@blockchain_router.post("/blockchain/content/proof")
async def create_verification_proof_endpoint(proof_data: dict):
    """Crear prueba de verificación"""
    try:
        blockchain_type = BlockchainType(proof_data["blockchain"])
        content = proof_data["content"]
        tx_type = TransactionType(proof_data["type"])
        additional_data = proof_data.get("additional_data", {})
        
        verification = blockchain_manager.content_verification[blockchain_type]
        tx_id = verification.create_verification_proof(content, tx_type, additional_data)
        
        return {
            "message": "Verification proof created successfully",
            "transaction_id": tx_id,
            "blockchain": blockchain_type.value,
            "type": tx_type.value
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid blockchain type or transaction type: {e}")
    except Exception as e:
        logger.error(f"Error creating verification proof: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create verification proof: {str(e)}")


@blockchain_router.get("/blockchain/content/history/{content_hash}")
async def get_verification_history_endpoint(content_hash: str, blockchain: str):
    """Obtener historial de verificación"""
    try:
        blockchain_type = BlockchainType(blockchain)
        verification = blockchain_manager.content_verification[blockchain_type]
        history = verification.get_verification_history(content_hash)
        
        return {
            "content_hash": content_hash,
            "blockchain": blockchain_type.value,
            "verification_history": history
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid blockchain type: {e}")
    except Exception as e:
        logger.error(f"Error getting verification history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get verification history: {str(e)}")


@blockchain_router.post("/blockchain/contracts/deploy")
async def deploy_smart_contract_endpoint(contract_data: dict):
    """Desplegar contrato inteligente"""
    try:
        name = contract_data["name"]
        blockchain = BlockchainType(contract_data["blockchain"])
        abi = contract_data["abi"]
        bytecode = contract_data["bytecode"]
        creator = contract_data.get("creator", "system")
        
        contract_id = blockchain_manager.smart_contracts.create_contract(
            name, blockchain, abi, bytecode, creator
        )
        
        contract = blockchain_manager.smart_contracts.get_contract(contract_id)
        
        return {
            "message": "Smart contract deployed successfully",
            "contract_id": contract_id,
            "address": contract.address,
            "name": contract.name,
            "blockchain": blockchain.value,
            "creator": creator
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid blockchain type: {e}")
    except Exception as e:
        logger.error(f"Error deploying smart contract: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy smart contract: {str(e)}")


@blockchain_router.get("/blockchain/contracts/{address}")
async def get_smart_contract_endpoint(address: str):
    """Obtener contrato inteligente"""
    try:
        contract = blockchain_manager.smart_contracts.get_contract_by_address(address)
        
        if contract:
            return {
                "address": contract.address,
                "name": contract.name,
                "blockchain": contract.blockchain.value,
                "abi": contract.abi,
                "deployed_at": contract.deployed_at,
                "creator": contract.creator,
                "functions": contract.functions,
                "events": contract.events
            }
        else:
            raise HTTPException(status_code=404, detail="Smart contract not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting smart contract: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get smart contract: {str(e)}")


@blockchain_router.post("/blockchain/contracts/{address}/call")
async def call_contract_function_endpoint(address: str, call_data: dict):
    """Llamar función del contrato"""
    try:
        function_name = call_data["function_name"]
        parameters = call_data.get("parameters", [])
        
        result = blockchain_manager.smart_contracts.call_contract_function(
            address, function_name, parameters
        )
        
        return {
            "message": "Contract function called successfully",
            "result": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error calling contract function: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to call contract function: {str(e)}")


@blockchain_router.get("/blockchain/stats")
async def get_blockchain_stats_endpoint():
    """Obtener estadísticas de blockchain"""
    try:
        stats = await blockchain_manager.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting blockchain stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain stats: {str(e)}")


# Funciones de utilidad para integración
async def start_blockchain():
    """Iniciar blockchain"""
    await blockchain_manager.start()


async def stop_blockchain():
    """Detener blockchain"""
    await blockchain_manager.stop()


def create_blockchain_transaction(blockchain_type: BlockchainType, tx_type: TransactionType, 
                                 content_hash: str, data: Dict[str, Any]) -> str:
    """Crear transacción blockchain"""
    blockchain = blockchain_manager.blockchains[blockchain_type]
    return blockchain.create_transaction(tx_type, content_hash, data)


def verify_content_hash(blockchain_type: BlockchainType, content: str, 
                       algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> Dict[str, Any]:
    """Verificar hash de contenido"""
    verification = blockchain_manager.content_verification[blockchain_type]
    return verification.verify_content_hash(content, algorithm)


async def get_blockchain_stats() -> Dict[str, Any]:
    """Obtener estadísticas de blockchain"""
    return await blockchain_manager.get_system_stats()


logger.info("Blockchain support module loaded successfully")

