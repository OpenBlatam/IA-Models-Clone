"""
Blockchain Engine - Motor de Blockchain avanzado
"""

import asyncio
import logging
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import pickle
import ecdsa
from ecdsa import SigningKey, VerifyingKey
import base58
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Tipos de transacciones."""
    TRANSFER = "transfer"
    CONTRACT = "contract"
    DATA_STORAGE = "data_storage"
    DOCUMENT_HASH = "document_hash"
    SMART_CONTRACT = "smart_contract"
    VOTE = "vote"
    CERTIFICATE = "certificate"


class BlockStatus(Enum):
    """Estados de bloque."""
    PENDING = "pending"
    MINED = "mined"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class Transaction:
    """Transacción blockchain."""
    transaction_id: str
    transaction_type: TransactionType
    sender: str
    recipient: str
    amount: float
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    signature: Optional[str] = None
    fee: float = 0.0
    nonce: int = 0


@dataclass
class Block:
    """Bloque blockchain."""
    block_id: str
    previous_hash: str
    transactions: List[Transaction]
    timestamp: datetime
    nonce: int
    hash: str
    merkle_root: str
    status: BlockStatus = BlockStatus.PENDING
    miner: Optional[str] = None
    difficulty: int = 4


@dataclass
class Wallet:
    """Cartera blockchain."""
    wallet_id: str
    address: str
    public_key: str
    private_key: str
    balance: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


class BlockchainEngine:
    """
    Motor de Blockchain avanzado.
    """
    
    def __init__(self, blockchain_directory: str = "blockchain_data"):
        """Inicializar motor de Blockchain."""
        self.blockchain_directory = Path(blockchain_directory)
        self.blockchain_directory.mkdir(exist_ok=True)
        
        # Cadena de bloques
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.wallets: Dict[str, Wallet] = {}
        
        # Configuración
        self.difficulty = 4
        self.mining_reward = 50.0
        self.transaction_fee = 0.1
        self.max_transactions_per_block = 100
        self.block_time_seconds = 10
        
        # Estadísticas
        self.stats = {
            "total_blocks": 0,
            "total_transactions": 0,
            "total_wallets": 0,
            "total_mined": 0.0,
            "start_time": datetime.now()
        }
        
        # Cargar blockchain existente
        self._load_blockchain()
        
        logger.info("BlockchainEngine inicializado")
    
    async def initialize(self):
        """Inicializar el motor de Blockchain."""
        try:
            # Crear bloque génesis si no existe
            if len(self.chain) == 0:
                await self._create_genesis_block()
            
            # Iniciar minería automática
            asyncio.create_task(self._mining_loop())
            
            logger.info("BlockchainEngine inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar BlockchainEngine: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el motor de Blockchain."""
        try:
            # Guardar blockchain
            await self._save_blockchain()
            
            logger.info("BlockchainEngine cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar BlockchainEngine: {e}")
    
    def _load_blockchain(self):
        """Cargar blockchain existente."""
        try:
            chain_file = self.blockchain_directory / "blockchain.pkl"
            if chain_file.exists():
                with open(chain_file, 'rb') as f:
                    self.chain = pickle.load(f)
                
                logger.info(f"Blockchain cargada: {len(self.chain)} bloques")
            
            # Cargar carteras
            wallets_file = self.blockchain_directory / "wallets.pkl"
            if wallets_file.exists():
                with open(wallets_file, 'rb') as f:
                    self.wallets = pickle.load(f)
                
                logger.info(f"Carteras cargadas: {len(self.wallets)}")
                
        except Exception as e:
            logger.error(f"Error al cargar blockchain: {e}")
    
    async def _save_blockchain(self):
        """Guardar blockchain."""
        try:
            # Guardar cadena
            chain_file = self.blockchain_directory / "blockchain.pkl"
            with open(chain_file, 'wb') as f:
                pickle.dump(self.chain, f)
            
            # Guardar carteras
            wallets_file = self.blockchain_directory / "wallets.pkl"
            with open(wallets_file, 'wb') as f:
                pickle.dump(self.wallets, f)
                
        except Exception as e:
            logger.error(f"Error al guardar blockchain: {e}")
    
    async def _create_genesis_block(self):
        """Crear bloque génesis."""
        try:
            genesis_transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                transaction_type=TransactionType.TRANSFER,
                sender="genesis",
                recipient="genesis",
                amount=0.0,
                data={"message": "Genesis block"}
            )
            
            genesis_block = Block(
                block_id=str(uuid.uuid4()),
                previous_hash="0",
                transactions=[genesis_transaction],
                timestamp=datetime.now(),
                nonce=0,
                hash="",
                merkle_root=""
            )
            
            # Calcular hash del bloque génesis
            genesis_block.hash = self._calculate_block_hash(genesis_block)
            genesis_block.merkle_root = self._calculate_merkle_root([genesis_transaction])
            genesis_block.status = BlockStatus.CONFIRMED
            
            self.chain.append(genesis_block)
            self.stats["total_blocks"] = 1
            
            logger.info("Bloque génesis creado")
            
        except Exception as e:
            logger.error(f"Error al crear bloque génesis: {e}")
    
    async def create_wallet(self, wallet_name: str = None) -> str:
        """Crear nueva cartera."""
        try:
            wallet_id = str(uuid.uuid4())
            
            # Generar par de claves
            private_key = SigningKey.generate()
            public_key = private_key.get_verifying_key()
            
            # Generar dirección
            address = self._generate_address(public_key)
            
            wallet = Wallet(
                wallet_id=wallet_id,
                address=address,
                public_key=public_key.to_string().hex(),
                private_key=private_key.to_string().hex()
            )
            
            self.wallets[wallet_id] = wallet
            self.stats["total_wallets"] += 1
            
            logger.info(f"Cartera creada: {address}")
            return wallet_id
            
        except Exception as e:
            logger.error(f"Error al crear cartera: {e}")
            raise
    
    def _generate_address(self, public_key: VerifyingKey) -> str:
        """Generar dirección de cartera."""
        try:
            # Hash de la clave pública
            public_key_bytes = public_key.to_string()
            hash_obj = hashlib.sha256(public_key_bytes)
            address_hash = hash_obj.digest()
            
            # Codificar en base58
            address = base58.b58encode(address_hash).decode('utf-8')
            return address[:34]  # Limitar longitud
            
        except Exception as e:
            logger.error(f"Error al generar dirección: {e}")
            return f"addr_{uuid.uuid4().hex()[:16]}"
    
    async def create_transaction(
        self,
        sender_wallet_id: str,
        recipient_address: str,
        amount: float,
        transaction_type: TransactionType = TransactionType.TRANSFER,
        data: Dict[str, Any] = None
    ) -> str:
        """Crear transacción."""
        try:
            if sender_wallet_id not in self.wallets:
                raise ValueError(f"Cartera {sender_wallet_id} no encontrada")
            
            sender_wallet = self.wallets[sender_wallet_id]
            
            # Verificar balance
            if sender_wallet.balance < amount + self.transaction_fee:
                raise ValueError("Balance insuficiente")
            
            # Crear transacción
            transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                transaction_type=transaction_type,
                sender=sender_wallet.address,
                recipient=recipient_address,
                amount=amount,
                data=data or {},
                fee=self.transaction_fee,
                nonce=sender_wallet.nonce if hasattr(sender_wallet, 'nonce') else 0
            )
            
            # Firmar transacción
            transaction.signature = self._sign_transaction(transaction, sender_wallet.private_key)
            
            # Agregar a transacciones pendientes
            self.pending_transactions.append(transaction)
            self.stats["total_transactions"] += 1
            
            logger.info(f"Transacción creada: {transaction.transaction_id}")
            return transaction.transaction_id
            
        except Exception as e:
            logger.error(f"Error al crear transacción: {e}")
            raise
    
    def _sign_transaction(self, transaction: Transaction, private_key_hex: str) -> str:
        """Firmar transacción."""
        try:
            private_key = SigningKey.from_string(bytes.fromhex(private_key_hex))
            
            # Crear mensaje para firmar
            message = f"{transaction.transaction_id}{transaction.sender}{transaction.recipient}{transaction.amount}{transaction.timestamp.isoformat()}"
            message_bytes = message.encode('utf-8')
            
            # Firmar
            signature = private_key.sign(message_bytes)
            return signature.hex()
            
        except Exception as e:
            logger.error(f"Error al firmar transacción: {e}")
            return ""
    
    def _verify_transaction(self, transaction: Transaction) -> bool:
        """Verificar transacción."""
        try:
            if not transaction.signature:
                return False
            
            # Obtener clave pública del remitente
            sender_wallet = None
            for wallet in self.wallets.values():
                if wallet.address == transaction.sender:
                    sender_wallet = wallet
                    break
            
            if not sender_wallet:
                return False
            
            # Verificar firma
            public_key = VerifyingKey.from_string(bytes.fromhex(sender_wallet.public_key))
            
            message = f"{transaction.transaction_id}{transaction.sender}{transaction.recipient}{transaction.amount}{transaction.timestamp.isoformat()}"
            message_bytes = message.encode('utf-8')
            
            try:
                public_key.verify(bytes.fromhex(transaction.signature), message_bytes)
                return True
            except:
                return False
                
        except Exception as e:
            logger.error(f"Error al verificar transacción: {e}")
            return False
    
    async def _mining_loop(self):
        """Bucle de minería automática."""
        while True:
            try:
                if len(self.pending_transactions) > 0:
                    await self._mine_block()
                
                await asyncio.sleep(self.block_time_seconds)
                
            except Exception as e:
                logger.error(f"Error en bucle de minería: {e}")
                await asyncio.sleep(self.block_time_seconds)
    
    async def _mine_block(self):
        """Minar bloque."""
        try:
            # Seleccionar transacciones para el bloque
            block_transactions = self.pending_transactions[:self.max_transactions_per_block]
            
            if not block_transactions:
                return
            
            # Crear bloque
            previous_hash = self.chain[-1].hash if self.chain else "0"
            
            block = Block(
                block_id=str(uuid.uuid4()),
                previous_hash=previous_hash,
                transactions=block_transactions,
                timestamp=datetime.now(),
                nonce=0,
                hash="",
                merkle_root="",
                difficulty=self.difficulty
            )
            
            # Calcular merkle root
            block.merkle_root = self._calculate_merkle_root(block_transactions)
            
            # Minar bloque (Proof of Work)
            block = await self._proof_of_work(block)
            
            # Agregar recompensa de minería
            mining_reward_transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                transaction_type=TransactionType.TRANSFER,
                sender="mining_reward",
                recipient="miner",  # En una implementación real, sería la dirección del minero
                amount=self.mining_reward,
                data={"type": "mining_reward"}
            )
            block.transactions.append(mining_reward_transaction)
            
            # Agregar bloque a la cadena
            block.status = BlockStatus.MINED
            self.chain.append(block)
            
            # Remover transacciones procesadas
            self.pending_transactions = self.pending_transactions[self.max_transactions_per_block:]
            
            # Actualizar balances
            await self._update_balances(block)
            
            self.stats["total_blocks"] += 1
            self.stats["total_mined"] += self.mining_reward
            
            logger.info(f"Bloque minado: {block.block_id}")
            
        except Exception as e:
            logger.error(f"Error al minar bloque: {e}")
    
    async def _proof_of_work(self, block: Block) -> Block:
        """Algoritmo Proof of Work."""
        try:
            target = "0" * self.difficulty
            
            while True:
                block.nonce += 1
                block.hash = self._calculate_block_hash(block)
                
                if block.hash.startswith(target):
                    break
                
                # Pequeña pausa para no sobrecargar CPU
                if block.nonce % 1000 == 0:
                    await asyncio.sleep(0.001)
            
            return block
            
        except Exception as e:
            logger.error(f"Error en Proof of Work: {e}")
            return block
    
    def _calculate_block_hash(self, block: Block) -> str:
        """Calcular hash del bloque."""
        try:
            block_string = f"{block.block_id}{block.previous_hash}{block.merkle_root}{block.timestamp.isoformat()}{block.nonce}"
            return hashlib.sha256(block_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error al calcular hash del bloque: {e}")
            return ""
    
    def _calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """Calcular Merkle root."""
        try:
            if not transactions:
                return ""
            
            # Calcular hashes de transacciones
            transaction_hashes = []
            for tx in transactions:
                tx_string = f"{tx.transaction_id}{tx.sender}{tx.recipient}{tx.amount}{tx.timestamp.isoformat()}"
                tx_hash = hashlib.sha256(tx_string.encode()).hexdigest()
                transaction_hashes.append(tx_hash)
            
            # Calcular Merkle tree
            while len(transaction_hashes) > 1:
                next_level = []
                for i in range(0, len(transaction_hashes), 2):
                    left = transaction_hashes[i]
                    right = transaction_hashes[i + 1] if i + 1 < len(transaction_hashes) else left
                    combined = left + right
                    next_level.append(hashlib.sha256(combined.encode()).hexdigest())
                transaction_hashes = next_level
            
            return transaction_hashes[0] if transaction_hashes else ""
            
        except Exception as e:
            logger.error(f"Error al calcular Merkle root: {e}")
            return ""
    
    async def _update_balances(self, block: Block):
        """Actualizar balances de carteras."""
        try:
            for transaction in block.transactions:
                # Actualizar balance del remitente
                if transaction.sender != "genesis" and transaction.sender != "mining_reward":
                    for wallet in self.wallets.values():
                        if wallet.address == transaction.sender:
                            wallet.balance -= (transaction.amount + transaction.fee)
                            wallet.last_activity = datetime.now()
                            break
                
                # Actualizar balance del destinatario
                if transaction.recipient != "genesis":
                    for wallet in self.wallets.values():
                        if wallet.address == transaction.recipient:
                            wallet.balance += transaction.amount
                            wallet.last_activity = datetime.now()
                            break
                            
        except Exception as e:
            logger.error(f"Error al actualizar balances: {e}")
    
    async def get_wallet_balance(self, wallet_id: str) -> float:
        """Obtener balance de cartera."""
        try:
            if wallet_id not in self.wallets:
                raise ValueError(f"Cartera {wallet_id} no encontrada")
            
            return self.wallets[wallet_id].balance
            
        except Exception as e:
            logger.error(f"Error al obtener balance: {e}")
            raise
    
    async def get_transaction_history(self, wallet_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de transacciones."""
        try:
            if wallet_id not in self.wallets:
                raise ValueError(f"Cartera {wallet_id} no encontrada")
            
            wallet_address = self.wallets[wallet_id].address
            transactions = []
            
            # Buscar transacciones en todos los bloques
            for block in self.chain:
                for transaction in block.transactions:
                    if (transaction.sender == wallet_address or 
                        transaction.recipient == wallet_address):
                        transactions.append({
                            "transaction_id": transaction.transaction_id,
                            "type": transaction.transaction_type.value,
                            "sender": transaction.sender,
                            "recipient": transaction.recipient,
                            "amount": transaction.amount,
                            "fee": transaction.fee,
                            "timestamp": transaction.timestamp.isoformat(),
                            "block_id": block.block_id,
                            "block_hash": block.hash
                        })
            
            # Ordenar por timestamp descendente
            transactions.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return transactions[:limit]
            
        except Exception as e:
            logger.error(f"Error al obtener historial: {e}")
            raise
    
    async def get_blockchain_info(self) -> Dict[str, Any]:
        """Obtener información de la blockchain."""
        try:
            return {
                "chain_length": len(self.chain),
                "pending_transactions": len(self.pending_transactions),
                "total_wallets": len(self.wallets),
                "difficulty": self.difficulty,
                "mining_reward": self.mining_reward,
                "transaction_fee": self.transaction_fee,
                "last_block_hash": self.chain[-1].hash if self.chain else None,
                "last_block_timestamp": self.chain[-1].timestamp.isoformat() if self.chain else None,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener información de blockchain: {e}")
            raise
    
    async def verify_blockchain(self) -> Dict[str, Any]:
        """Verificar integridad de la blockchain."""
        try:
            verification_results = {
                "is_valid": True,
                "errors": [],
                "blocks_verified": 0,
                "transactions_verified": 0
            }
            
            for i, block in enumerate(self.chain):
                # Verificar hash del bloque
                calculated_hash = self._calculate_block_hash(block)
                if calculated_hash != block.hash:
                    verification_results["is_valid"] = False
                    verification_results["errors"].append(f"Hash inválido en bloque {i}")
                
                # Verificar hash anterior
                if i > 0:
                    if block.previous_hash != self.chain[i-1].hash:
                        verification_results["is_valid"] = False
                        verification_results["errors"].append(f"Hash anterior inválido en bloque {i}")
                
                # Verificar Merkle root
                calculated_merkle = self._calculate_merkle_root(block.transactions)
                if calculated_merkle != block.merkle_root:
                    verification_results["is_valid"] = False
                    verification_results["errors"].append(f"Merkle root inválido en bloque {i}")
                
                # Verificar transacciones
                for transaction in block.transactions:
                    if not self._verify_transaction(transaction):
                        verification_results["is_valid"] = False
                        verification_results["errors"].append(f"Transacción inválida: {transaction.transaction_id}")
                    verification_results["transactions_verified"] += 1
                
                verification_results["blocks_verified"] += 1
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Error al verificar blockchain: {e}")
            return {
                "is_valid": False,
                "errors": [str(e)],
                "blocks_verified": 0,
                "transactions_verified": 0
            }
    
    async def get_blockchain_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la blockchain."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "total_wallets": len(self.wallets),
            "difficulty": self.difficulty,
            "mining_reward": self.mining_reward,
            "transaction_fee": self.transaction_fee,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor de Blockchain."""
        try:
            return {
                "status": "healthy",
                "chain_length": len(self.chain),
                "pending_transactions": len(self.pending_transactions),
                "total_wallets": len(self.wallets),
                "mining_active": True,
                "stats": self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de Blockchain: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




