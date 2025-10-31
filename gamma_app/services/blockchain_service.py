"""
Gamma App - Blockchain and NFT Service
Advanced blockchain integration with smart contracts, NFTs, and DeFi features
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import numpy as np
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict
import requests
from web3 import Web3
from eth_account import Account
from eth_typing import Address
import ipfshttpclient
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64
import secrets

logger = logging.getLogger(__name__)

class BlockchainNetwork(Enum):
    """Blockchain networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    AVALANCHE = "avalanche"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"

class NFTStandard(Enum):
    """NFT standards"""
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    ERC4907 = "erc4907"  # Rentable NFTs

class TokenType(Enum):
    """Token types"""
    NATIVE = "native"
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"

class TransactionStatus(Enum):
    """Transaction status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Wallet:
    """Blockchain wallet"""
    wallet_id: str
    address: str
    private_key: str  # Encrypted
    network: BlockchainNetwork
    balance: float = 0.0
    nonce: int = 0
    created_at: datetime = None

@dataclass
class NFT:
    """NFT definition"""
    nft_id: str
    token_id: str
    contract_address: str
    owner_address: str
    metadata: Dict[str, Any]
    ipfs_hash: str
    network: BlockchainNetwork
    standard: NFTStandard
    created_at: datetime = None

@dataclass
class SmartContract:
    """Smart contract definition"""
    contract_id: str
    address: str
    abi: List[Dict[str, Any]]
    bytecode: str
    network: BlockchainNetwork
    deployed_at: datetime = None

@dataclass
class Transaction:
    """Blockchain transaction"""
    tx_id: str
    hash: str
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: float
    status: TransactionStatus
    network: BlockchainNetwork
    block_number: Optional[int] = None
    created_at: datetime = None

class AdvancedBlockchainService:
    """Advanced Blockchain and NFT Service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "blockchain.db")
        self.redis_client = None
        self.wallets = {}
        self.nfts = {}
        self.contracts = {}
        self.transactions = {}
        self.web3_connections = {}
        self.ipfs_client = None
        self.encryption_key = None
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_encryption()
        self._init_web3_connections()
        self._init_ipfs()
    
    def _init_database(self):
        """Initialize blockchain database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create wallets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wallets (
                    wallet_id TEXT PRIMARY KEY,
                    address TEXT NOT NULL,
                    private_key TEXT NOT NULL,
                    network TEXT NOT NULL,
                    balance REAL DEFAULT 0.0,
                    nonce INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create NFTs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nfts (
                    nft_id TEXT PRIMARY KEY,
                    token_id TEXT NOT NULL,
                    contract_address TEXT NOT NULL,
                    owner_address TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    ipfs_hash TEXT NOT NULL,
                    network TEXT NOT NULL,
                    standard TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create smart contracts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS smart_contracts (
                    contract_id TEXT PRIMARY KEY,
                    address TEXT NOT NULL,
                    abi TEXT NOT NULL,
                    bytecode TEXT NOT NULL,
                    network TEXT NOT NULL,
                    deployed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_id TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    from_address TEXT NOT NULL,
                    to_address TEXT NOT NULL,
                    value REAL NOT NULL,
                    gas_used INTEGER NOT NULL,
                    gas_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    network TEXT NOT NULL,
                    block_number INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        logger.info("Blockchain database initialized")
    
    def _init_redis(self):
        """Initialize Redis for caching"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized for blockchain")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _init_encryption(self):
        """Initialize encryption for private keys"""
        try:
            # Generate or load encryption key
            key_file = Path("data/blockchain_encryption.key")
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = secrets.token_bytes(32)
                key_file.parent.mkdir(exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
            
            logger.info("Blockchain encryption initialized")
        except Exception as e:
            logger.error(f"Blockchain encryption initialization failed: {e}")
    
    def _init_web3_connections(self):
        """Initialize Web3 connections for different networks"""
        try:
            network_configs = {
                BlockchainNetwork.ETHEREUM: {
                    "rpc_url": self.config.get("ethereum_rpc", "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"),
                    "chain_id": 1
                },
                BlockchainNetwork.POLYGON: {
                    "rpc_url": self.config.get("polygon_rpc", "https://polygon-rpc.com"),
                    "chain_id": 137
                },
                BlockchainNetwork.BSC: {
                    "rpc_url": self.config.get("bsc_rpc", "https://bsc-dataseed.binance.org"),
                    "chain_id": 56
                },
                BlockchainNetwork.AVALANCHE: {
                    "rpc_url": self.config.get("avalanche_rpc", "https://api.avax.network/ext/bc/C/rpc"),
                    "chain_id": 43114
                }
            }
            
            for network, config in network_configs.items():
                try:
                    w3 = Web3(Web3.HTTPProvider(config["rpc_url"]))
                    if w3.is_connected():
                        self.web3_connections[network] = w3
                        logger.info(f"Connected to {network.value} network")
                    else:
                        logger.warning(f"Failed to connect to {network.value} network")
                except Exception as e:
                    logger.warning(f"Error connecting to {network.value}: {e}")
            
            logger.info("Web3 connections initialized")
        except Exception as e:
            logger.error(f"Web3 connections initialization failed: {e}")
    
    def _init_ipfs(self):
        """Initialize IPFS client"""
        try:
            ipfs_host = self.config.get("ipfs_host", "localhost")
            ipfs_port = self.config.get("ipfs_port", 5001)
            self.ipfs_client = ipfshttpclient.connect(f"/ip4/{ipfs_host}/tcp/{ipfs_port}/http")
            logger.info("IPFS client initialized")
        except Exception as e:
            logger.warning(f"IPFS initialization failed: {e}")
    
    def _encrypt_private_key(self, private_key: str) -> str:
        """Encrypt private key"""
        try:
            from cryptography.fernet import Fernet
            key = base64.urlsafe_b64encode(self.encryption_key)
            cipher = Fernet(key)
            encrypted_key = cipher.encrypt(private_key.encode())
            return base64.b64encode(encrypted_key).decode()
        except Exception as e:
            logger.error(f"Private key encryption failed: {e}")
            return private_key
    
    def _decrypt_private_key(self, encrypted_key: str) -> str:
        """Decrypt private key"""
        try:
            from cryptography.fernet import Fernet
            key = base64.urlsafe_b64encode(self.encryption_key)
            cipher = Fernet(key)
            encrypted_data = base64.b64decode(encrypted_key.encode())
            decrypted_key = cipher.decrypt(encrypted_data)
            return decrypted_key.decode()
        except Exception as e:
            logger.error(f"Private key decryption failed: {e}")
            return encrypted_key
    
    async def create_wallet(
        self,
        network: BlockchainNetwork,
        password: Optional[str] = None
    ) -> Wallet:
        """Create a new blockchain wallet"""
        
        try:
            # Generate new account
            account = Account.create()
            private_key = account.key.hex()
            address = account.address
            
            # Encrypt private key
            encrypted_private_key = self._encrypt_private_key(private_key)
            
            # Create wallet object
            wallet = Wallet(
                wallet_id=str(uuid.uuid4()),
                address=address,
                private_key=encrypted_private_key,
                network=network,
                created_at=datetime.now()
            )
            
            # Get initial balance
            if network in self.web3_connections:
                w3 = self.web3_connections[network]
                balance_wei = w3.eth.get_balance(address)
                wallet.balance = w3.from_wei(balance_wei, 'ether')
                wallet.nonce = w3.eth.get_transaction_count(address)
            
            # Store wallet
            self.wallets[wallet.wallet_id] = wallet
            await self._store_wallet(wallet)
            
            logger.info(f"Wallet created: {address} on {network.value}")
            return wallet
            
        except Exception as e:
            logger.error(f"Wallet creation failed: {e}")
            raise e
    
    async def import_wallet(
        self,
        private_key: str,
        network: BlockchainNetwork,
        password: Optional[str] = None
    ) -> Wallet:
        """Import existing wallet"""
        
        try:
            # Create account from private key
            account = Account.from_key(private_key)
            address = account.address
            
            # Encrypt private key
            encrypted_private_key = self._encrypt_private_key(private_key)
            
            # Create wallet object
            wallet = Wallet(
                wallet_id=str(uuid.uuid4()),
                address=address,
                private_key=encrypted_private_key,
                network=network,
                created_at=datetime.now()
            )
            
            # Get balance and nonce
            if network in self.web3_connections:
                w3 = self.web3_connections[network]
                balance_wei = w3.eth.get_balance(address)
                wallet.balance = w3.from_wei(balance_wei, 'ether')
                wallet.nonce = w3.eth.get_transaction_count(address)
            
            # Store wallet
            self.wallets[wallet.wallet_id] = wallet
            await self._store_wallet(wallet)
            
            logger.info(f"Wallet imported: {address} on {network.value}")
            return wallet
            
        except Exception as e:
            logger.error(f"Wallet import failed: {e}")
            raise e
    
    async def get_wallet_balance(
        self,
        wallet_id: str,
        token_address: Optional[str] = None
    ) -> float:
        """Get wallet balance"""
        
        try:
            wallet = self.wallets.get(wallet_id)
            if not wallet:
                wallet = await self._load_wallet(wallet_id)
            
            if not wallet:
                raise ValueError(f"Wallet not found: {wallet_id}")
            
            if network in self.web3_connections:
                w3 = self.web3_connections[wallet.network]
                
                if token_address:
                    # ERC20 token balance
                    balance = await self._get_erc20_balance(
                        w3, wallet.address, token_address
                    )
                else:
                    # Native token balance
                    balance_wei = w3.eth.get_balance(wallet.address)
                    balance = w3.from_wei(balance_wei, 'ether')
                
                # Update wallet balance
                wallet.balance = balance
                await self._update_wallet(wallet)
                
                return balance
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Balance retrieval failed: {e}")
            raise e
    
    async def _get_erc20_balance(
        self,
        w3: Web3,
        address: str,
        token_address: str
    ) -> float:
        """Get ERC20 token balance"""
        
        try:
            # ERC20 balanceOf function ABI
            balance_abi = [{
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }]
            
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=balance_abi
            )
            
            balance = contract.functions.balanceOf(address).call()
            
            # Get token decimals
            decimals_abi = [{
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }]
            
            decimals_contract = w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=decimals_abi
            )
            
            decimals = decimals_contract.functions.decimals().call()
            
            return balance / (10 ** decimals)
            
        except Exception as e:
            logger.error(f"ERC20 balance retrieval failed: {e}")
            return 0.0
    
    async def send_transaction(
        self,
        wallet_id: str,
        to_address: str,
        value: float,
        gas_limit: Optional[int] = None,
        gas_price: Optional[float] = None,
        data: Optional[str] = None
    ) -> Transaction:
        """Send blockchain transaction"""
        
        try:
            wallet = self.wallets.get(wallet_id)
            if not wallet:
                wallet = await self._load_wallet(wallet_id)
            
            if not wallet:
                raise ValueError(f"Wallet not found: {wallet_id}")
            
            if wallet.network not in self.web3_connections:
                raise ValueError(f"Network not supported: {wallet.network.value}")
            
            w3 = self.web3_connections[wallet.network]
            
            # Decrypt private key
            private_key = self._decrypt_private_key(wallet.private_key)
            
            # Prepare transaction
            transaction = {
                'to': Web3.to_checksum_address(to_address),
                'value': w3.to_wei(value, 'ether'),
                'gas': gas_limit or 21000,
                'gasPrice': w3.to_wei(gas_price or 20, 'gwei'),
                'nonce': wallet.nonce,
            }
            
            if data:
                transaction['data'] = data
            
            # Sign transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Create transaction object
            tx = Transaction(
                tx_id=str(uuid.uuid4()),
                hash=tx_hash.hex(),
                from_address=wallet.address,
                to_address=to_address,
                value=value,
                gas_used=transaction['gas'],
                gas_price=transaction['gasPrice'],
                status=TransactionStatus.PENDING,
                network=wallet.network,
                created_at=datetime.now()
            )
            
            # Store transaction
            self.transactions[tx.tx_id] = tx
            await self._store_transaction(tx)
            
            # Update wallet nonce
            wallet.nonce += 1
            await self._update_wallet(wallet)
            
            logger.info(f"Transaction sent: {tx_hash.hex()}")
            return tx
            
        except Exception as e:
            logger.error(f"Transaction sending failed: {e}")
            raise e
    
    async def create_nft(
        self,
        wallet_id: str,
        name: str,
        description: str,
        image_url: str,
        attributes: List[Dict[str, Any]],
        network: BlockchainNetwork,
        standard: NFTStandard = NFTStandard.ERC721
    ) -> NFT:
        """Create and mint NFT"""
        
        try:
            wallet = self.wallets.get(wallet_id)
            if not wallet:
                wallet = await self._load_wallet(wallet_id)
            
            if not wallet:
                raise ValueError(f"Wallet not found: {wallet_id}")
            
            # Create metadata
            metadata = {
                "name": name,
                "description": description,
                "image": image_url,
                "attributes": attributes,
                "created_at": datetime.now().isoformat()
            }
            
            # Upload metadata to IPFS
            ipfs_hash = await self._upload_to_ipfs(metadata)
            
            # Generate token ID
            token_id = str(uuid.uuid4())
            
            # Create NFT object
            nft = NFT(
                nft_id=str(uuid.uuid4()),
                token_id=token_id,
                contract_address="",  # Will be set after deployment
                owner_address=wallet.address,
                metadata=metadata,
                ipfs_hash=ipfs_hash,
                network=network,
                standard=standard,
                created_at=datetime.now()
            )
            
            # Store NFT
            self.nfts[nft.nft_id] = nft
            await self._store_nft(nft)
            
            logger.info(f"NFT created: {nft.nft_id}")
            return nft
            
        except Exception as e:
            logger.error(f"NFT creation failed: {e}")
            raise e
    
    async def _upload_to_ipfs(self, data: Dict[str, Any]) -> str:
        """Upload data to IPFS"""
        
        try:
            if not self.ipfs_client:
                # Fallback: return mock hash
                return f"Qm{hashlib.sha256(json.dumps(data).encode()).hexdigest()[:44]}"
            
            # Upload to IPFS
            result = self.ipfs_client.add_json(data)
            return result['Hash']
            
        except Exception as e:
            logger.error(f"IPFS upload failed: {e}")
            # Return mock hash as fallback
            return f"Qm{hashlib.sha256(json.dumps(data).encode()).hexdigest()[:44]}"
    
    async def deploy_smart_contract(
        self,
        wallet_id: str,
        contract_name: str,
        abi: List[Dict[str, Any]],
        bytecode: str,
        constructor_args: List[Any] = None
    ) -> SmartContract:
        """Deploy smart contract"""
        
        try:
            wallet = self.wallets.get(wallet_id)
            if not wallet:
                wallet = await self._load_wallet(wallet_id)
            
            if not wallet:
                raise ValueError(f"Wallet not found: {wallet_id}")
            
            if wallet.network not in self.web3_connections:
                raise ValueError(f"Network not supported: {wallet.network.value}")
            
            w3 = self.web3_connections[wallet.network]
            
            # Decrypt private key
            private_key = self._decrypt_private_key(wallet.private_key)
            
            # Create contract
            contract = w3.eth.contract(abi=abi, bytecode=bytecode)
            
            # Build constructor transaction
            constructor = contract.constructor(*(constructor_args or []))
            transaction = constructor.build_transaction({
                'from': wallet.address,
                'gas': 2000000,
                'gasPrice': w3.to_wei(20, 'gwei'),
                'nonce': wallet.nonce,
            })
            
            # Sign and send transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = receipt.contractAddress
            
            # Create smart contract object
            smart_contract = SmartContract(
                contract_id=str(uuid.uuid4()),
                address=contract_address,
                abi=abi,
                bytecode=bytecode,
                network=wallet.network,
                deployed_at=datetime.now()
            )
            
            # Store contract
            self.contracts[smart_contract.contract_id] = smart_contract
            await self._store_smart_contract(smart_contract)
            
            # Update wallet nonce
            wallet.nonce += 1
            await self._update_wallet(wallet)
            
            logger.info(f"Smart contract deployed: {contract_address}")
            return smart_contract
            
        except Exception as e:
            logger.error(f"Smart contract deployment failed: {e}")
            raise e
    
    async def get_transaction_status(self, tx_hash: str, network: BlockchainNetwork) -> TransactionStatus:
        """Get transaction status"""
        
        try:
            if network not in self.web3_connections:
                return TransactionStatus.FAILED
            
            w3 = self.web3_connections[network]
            
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                if receipt.status == 1:
                    return TransactionStatus.CONFIRMED
                else:
                    return TransactionStatus.FAILED
            except:
                # Transaction not found or pending
                return TransactionStatus.PENDING
                
        except Exception as e:
            logger.error(f"Transaction status check failed: {e}")
            return TransactionStatus.FAILED
    
    async def get_nft_metadata(self, nft_id: str) -> Dict[str, Any]:
        """Get NFT metadata"""
        
        try:
            nft = self.nfts.get(nft_id)
            if not nft:
                nft = await self._load_nft(nft_id)
            
            if not nft:
                raise ValueError(f"NFT not found: {nft_id}")
            
            # If IPFS hash exists, try to fetch from IPFS
            if nft.ipfs_hash and self.ipfs_client:
                try:
                    metadata = self.ipfs_client.get_json(nft.ipfs_hash)
                    return metadata
                except:
                    pass
            
            # Return stored metadata
            return nft.metadata
            
        except Exception as e:
            logger.error(f"NFT metadata retrieval failed: {e}")
            raise e
    
    async def transfer_nft(
        self,
        nft_id: str,
        from_wallet_id: str,
        to_address: str
    ) -> Transaction:
        """Transfer NFT to another address"""
        
        try:
            nft = self.nfts.get(nft_id)
            if not nft:
                nft = await self._load_nft(nft_id)
            
            if not nft:
                raise ValueError(f"NFT not found: {nft_id}")
            
            from_wallet = self.wallets.get(from_wallet_id)
            if not from_wallet:
                from_wallet = await self._load_wallet(from_wallet_id)
            
            if not from_wallet:
                raise ValueError(f"Wallet not found: {from_wallet_id}")
            
            if nft.network not in self.web3_connections:
                raise ValueError(f"Network not supported: {nft.network.value}")
            
            w3 = self.web3_connections[nft.network]
            
            # ERC721 transfer function ABI
            transfer_abi = [{
                "constant": False,
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_tokenId", "type": "uint256"}
                ],
                "name": "transferFrom",
                "outputs": [],
                "type": "function"
            }]
            
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(nft.contract_address),
                abi=transfer_abi
            )
            
            # Build transaction
            transaction = contract.functions.transferFrom(
                from_wallet.address,
                Web3.to_checksum_address(to_address),
                int(nft.token_id)
            ).build_transaction({
                'from': from_wallet.address,
                'gas': 100000,
                'gasPrice': w3.to_wei(20, 'gwei'),
                'nonce': from_wallet.nonce,
            })
            
            # Decrypt private key
            private_key = self._decrypt_private_key(from_wallet.private_key)
            
            # Sign and send transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Create transaction object
            tx = Transaction(
                tx_id=str(uuid.uuid4()),
                hash=tx_hash.hex(),
                from_address=from_wallet.address,
                to_address=to_address,
                value=0.0,
                gas_used=transaction['gas'],
                gas_price=transaction['gasPrice'],
                status=TransactionStatus.PENDING,
                network=nft.network,
                created_at=datetime.now()
            )
            
            # Store transaction
            self.transactions[tx.tx_id] = tx
            await self._store_transaction(tx)
            
            # Update NFT owner
            nft.owner_address = to_address
            await self._update_nft(nft)
            
            # Update wallet nonce
            from_wallet.nonce += 1
            await self._update_wallet(from_wallet)
            
            logger.info(f"NFT transferred: {nft_id} to {to_address}")
            return tx
            
        except Exception as e:
            logger.error(f"NFT transfer failed: {e}")
            raise e
    
    async def _store_wallet(self, wallet: Wallet):
        """Store wallet in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO wallets
                (wallet_id, address, private_key, network, balance, nonce, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                wallet.wallet_id,
                wallet.address,
                wallet.private_key,
                wallet.network.value,
                wallet.balance,
                wallet.nonce,
                wallet.created_at.isoformat()
            ))
            conn.commit()
    
    async def _load_wallet(self, wallet_id: str) -> Optional[Wallet]:
        """Load wallet from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM wallets WHERE wallet_id = ?
            """, (wallet_id,))
            row = cursor.fetchone()
            
            if row:
                return Wallet(
                    wallet_id=row[0],
                    address=row[1],
                    private_key=row[2],
                    network=BlockchainNetwork(row[3]),
                    balance=row[4],
                    nonce=row[5],
                    created_at=datetime.fromisoformat(row[6])
                )
        
        return None
    
    async def _update_wallet(self, wallet: Wallet):
        """Update wallet in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE wallets
                SET balance = ?, nonce = ?
                WHERE wallet_id = ?
            """, (wallet.balance, wallet.nonce, wallet.wallet_id))
            conn.commit()
    
    async def _store_nft(self, nft: NFT):
        """Store NFT in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO nfts
                (nft_id, token_id, contract_address, owner_address, metadata, ipfs_hash, network, standard, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                nft.nft_id,
                nft.token_id,
                nft.contract_address,
                nft.owner_address,
                json.dumps(nft.metadata),
                nft.ipfs_hash,
                nft.network.value,
                nft.standard.value,
                nft.created_at.isoformat()
            ))
            conn.commit()
    
    async def _load_nft(self, nft_id: str) -> Optional[NFT]:
        """Load NFT from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM nfts WHERE nft_id = ?
            """, (nft_id,))
            row = cursor.fetchone()
            
            if row:
                return NFT(
                    nft_id=row[0],
                    token_id=row[1],
                    contract_address=row[2],
                    owner_address=row[3],
                    metadata=json.loads(row[4]),
                    ipfs_hash=row[5],
                    network=BlockchainNetwork(row[6]),
                    standard=NFTStandard(row[7]),
                    created_at=datetime.fromisoformat(row[8])
                )
        
        return None
    
    async def _update_nft(self, nft: NFT):
        """Update NFT in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE nfts
                SET owner_address = ?, metadata = ?
                WHERE nft_id = ?
            """, (nft.owner_address, json.dumps(nft.metadata), nft.nft_id))
            conn.commit()
    
    async def _store_smart_contract(self, contract: SmartContract):
        """Store smart contract in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO smart_contracts
                (contract_id, address, abi, bytecode, network, deployed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                contract.contract_id,
                contract.address,
                json.dumps(contract.abi),
                contract.bytecode,
                contract.network.value,
                contract.deployed_at.isoformat()
            ))
            conn.commit()
    
    async def _store_transaction(self, transaction: Transaction):
        """Store transaction in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions
                (tx_id, hash, from_address, to_address, value, gas_used, gas_price, status, network, block_number, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction.tx_id,
                transaction.hash,
                transaction.from_address,
                transaction.to_address,
                transaction.value,
                transaction.gas_used,
                transaction.gas_price,
                transaction.status.value,
                transaction.network.value,
                transaction.block_number,
                transaction.created_at.isoformat()
            ))
            conn.commit()
    
    async def get_blockchain_analytics(self) -> Dict[str, Any]:
        """Get blockchain analytics"""
        
        try:
            analytics = {
                "total_wallets": len(self.wallets),
                "total_nfts": len(self.nfts),
                "total_contracts": len(self.contracts),
                "total_transactions": len(self.transactions),
                "networks": {},
                "generated_at": datetime.now().isoformat()
            }
            
            # Network statistics
            for network in BlockchainNetwork:
                network_wallets = len([w for w in self.wallets.values() if w.network == network])
                network_nfts = len([n for n in self.nfts.values() if n.network == network])
                network_txs = len([t for t in self.transactions.values() if t.network == network])
                
                analytics["networks"][network.value] = {
                    "wallets": network_wallets,
                    "nfts": network_nfts,
                    "transactions": network_txs
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Blockchain analytics failed: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.redis_client:
            self.redis_client.close()
        
        if self.ipfs_client:
            self.ipfs_client.close()
        
        logger.info("Blockchain service cleanup completed")

# Global instance
blockchain_service = None

async def get_blockchain_service() -> AdvancedBlockchainService:
    """Get global blockchain service instance"""
    global blockchain_service
    if not blockchain_service:
        config = {
            "database_path": "data/blockchain.db",
            "redis_url": "redis://localhost:6379",
            "ethereum_rpc": "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
            "polygon_rpc": "https://polygon-rpc.com",
            "bsc_rpc": "https://bsc-dataseed.binance.org",
            "avalanche_rpc": "https://api.avax.network/ext/bc/C/rpc",
            "ipfs_host": "localhost",
            "ipfs_port": 5001
        }
        blockchain_service = AdvancedBlockchainService(config)
    return blockchain_service



