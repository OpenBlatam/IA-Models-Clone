"""
Ultimate BUL System - Blockchain & Web3 Integration
Comprehensive blockchain integration with smart contracts, NFTs, and DeFi capabilities
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import redis
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_typing import Address
import hashlib
import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
from prometheus_client import Counter, Histogram, Gauge
import time

logger = logging.getLogger(__name__)

class BlockchainType(str, Enum):
    """Blockchain types"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BINANCE_SMART_CHAIN = "bsc"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"
    SOLANA = "solana"

class ContractType(str, Enum):
    """Smart contract types"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    GOVERNANCE = "governance"
    STAKING = "staking"
    LENDING = "lending"
    DEX = "dex"
    NFT_MARKETPLACE = "nft_marketplace"

class TransactionStatus(str, Enum):
    """Transaction status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    blockchain_type: BlockchainType
    rpc_url: str
    chain_id: int
    gas_price: int
    gas_limit: int
    private_key: Optional[str] = None
    contract_address: Optional[str] = None
    abi: Optional[Dict[str, Any]] = None

@dataclass
class SmartContract:
    """Smart contract definition"""
    id: str
    name: str
    contract_type: ContractType
    address: str
    abi: Dict[str, Any]
    blockchain_type: BlockchainType
    deployed_at: datetime
    owner: str
    version: str = "1.0.0"
    is_verified: bool = False

@dataclass
class NFTMetadata:
    """NFT metadata"""
    name: str
    description: str
    image: str
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    external_url: Optional[str] = None
    animation_url: Optional[str] = None
    background_color: Optional[str] = None

@dataclass
class BlockchainTransaction:
    """Blockchain transaction"""
    id: str
    hash: str
    from_address: str
    to_address: str
    value: int
    gas_used: int
    gas_price: int
    status: TransactionStatus
    block_number: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    confirmed_at: Optional[datetime] = None

class Web3Integration:
    """Web3 and blockchain integration system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blockchain_configs = {}
        self.smart_contracts = {}
        self.transactions = {}
        self.nft_metadata = {}
        
        # Web3 connections
        self.web3_connections = {}
        
        # Redis for caching
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 3)
        )
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Initialize blockchain connections
        self._initialize_blockchain_connections()
        
        # Monitoring active
        self.monitoring_active = False
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_blockchain_connections(self):
        """Initialize blockchain connections"""
        # Ethereum Mainnet
        self.blockchain_configs["ethereum"] = BlockchainConfig(
            blockchain_type=BlockchainType.ETHEREUM,
            rpc_url="https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
            chain_id=1,
            gas_price=20000000000,  # 20 gwei
            gas_limit=21000
        )
        
        # Polygon
        self.blockchain_configs["polygon"] = BlockchainConfig(
            blockchain_type=BlockchainType.POLYGON,
            rpc_url="https://polygon-rpc.com",
            chain_id=137,
            gas_price=30000000000,  # 30 gwei
            gas_limit=21000
        )
        
        # Binance Smart Chain
        self.blockchain_configs["bsc"] = BlockchainConfig(
            blockchain_type=BlockchainType.BINANCE_SMART_CHAIN,
            rpc_url="https://bsc-dataseed.binance.org",
            chain_id=56,
            gas_price=5000000000,  # 5 gwei
            gas_limit=21000
        )
        
        # Initialize Web3 connections
        for blockchain_type, config in self.blockchain_configs.items():
            try:
                w3 = Web3(Web3.HTTPProvider(config.rpc_url))
                
                # Add PoA middleware for some networks
                if blockchain_type in ["bsc", "polygon"]:
                    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
                self.web3_connections[blockchain_type] = w3
                logger.info(f"Connected to {blockchain_type} blockchain")
                
            except Exception as e:
                logger.error(f"Failed to connect to {blockchain_type}: {e}")
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""
        return {
            "blockchain_transactions": Counter(
                "bul_blockchain_transactions_total",
                "Total blockchain transactions",
                ["blockchain", "status"]
            ),
            "blockchain_transaction_duration": Histogram(
                "bul_blockchain_transaction_duration_seconds",
                "Blockchain transaction duration in seconds",
                ["blockchain", "contract_type"]
            ),
            "blockchain_gas_used": Histogram(
                "bul_blockchain_gas_used",
                "Gas used for blockchain transactions",
                ["blockchain", "contract_type"]
            ),
            "nft_minted": Counter(
                "bul_nft_minted_total",
                "Total NFTs minted",
                ["blockchain", "contract"]
            ),
            "smart_contract_calls": Counter(
                "bul_smart_contract_calls_total",
                "Total smart contract calls",
                ["blockchain", "contract", "method"]
            ),
            "active_blockchain_connections": Gauge(
                "bul_active_blockchain_connections",
                "Number of active blockchain connections"
            )
        }
    
    async def start_monitoring(self):
        """Start blockchain monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting blockchain monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_transactions())
        asyncio.create_task(self._monitor_blockchain_health())
        asyncio.create_task(self._update_metrics())
    
    async def stop_monitoring(self):
        """Stop blockchain monitoring"""
        self.monitoring_active = False
        logger.info("Stopping blockchain monitoring")
    
    async def _monitor_transactions(self):
        """Monitor blockchain transactions"""
        while self.monitoring_active:
            try:
                for blockchain_type, w3 in self.web3_connections.items():
                    # Check pending transactions
                    pending_txs = [
                        tx for tx in self.transactions.values()
                        if tx.status == TransactionStatus.PENDING and tx.block_number is None
                    ]
                    
                    for tx in pending_txs:
                        try:
                            # Get transaction receipt
                            receipt = w3.eth.get_transaction_receipt(tx.hash)
                            
                            if receipt:
                                tx.block_number = receipt.blockNumber
                                tx.gas_used = receipt.gasUsed
                                
                                if receipt.status == 1:
                                    tx.status = TransactionStatus.CONFIRMED
                                    tx.confirmed_at = datetime.utcnow()
                                else:
                                    tx.status = TransactionStatus.FAILED
                                
                                # Update Prometheus metrics
                                self.prometheus_metrics["blockchain_transactions"].labels(
                                    blockchain=blockchain_type,
                                    status=tx.status.value
                                ).inc()
                                
                                logger.info(f"Transaction {tx.hash} {tx.status.value}")
                        
                        except Exception as e:
                            # Transaction might still be pending
                            pass
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring transactions: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_blockchain_health(self):
        """Monitor blockchain health"""
        while self.monitoring_active:
            try:
                for blockchain_type, w3 in self.web3_connections.items():
                    try:
                        # Check if blockchain is accessible
                        latest_block = w3.eth.block_number
                        
                        # Update Prometheus metrics
                        self.prometheus_metrics["active_blockchain_connections"].set(
                            len(self.web3_connections)
                        )
                        
                    except Exception as e:
                        logger.warning(f"Blockchain {blockchain_type} health check failed: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring blockchain health: {e}")
                await asyncio.sleep(60)
    
    async def _update_metrics(self):
        """Update Prometheus metrics"""
        while self.monitoring_active:
            try:
                # Update active connections
                self.prometheus_metrics["active_blockchain_connections"].set(
                    len(self.web3_connections)
                )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    def create_wallet(self, blockchain_type: BlockchainType) -> Dict[str, str]:
        """Create a new wallet"""
        try:
            # Generate private key
            private_key = Account.create().key.hex()
            
            # Get account from private key
            account = Account.from_key(private_key)
            
            wallet = {
                "address": account.address,
                "private_key": private_key,
                "blockchain_type": blockchain_type.value
            }
            
            logger.info(f"Created wallet for {blockchain_type.value}: {account.address}")
            return wallet
            
        except Exception as e:
            logger.error(f"Error creating wallet: {e}")
            raise
    
    def get_balance(self, address: str, blockchain_type: BlockchainType) -> int:
        """Get wallet balance"""
        try:
            w3 = self.web3_connections.get(blockchain_type.value)
            if not w3:
                raise ValueError(f"No connection to {blockchain_type.value}")
            
            balance = w3.eth.get_balance(address)
            return balance
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            raise
    
    async def send_transaction(self, from_address: str, to_address: str, 
                             value: int, blockchain_type: BlockchainType,
                             private_key: str, gas_price: Optional[int] = None,
                             gas_limit: Optional[int] = None) -> str:
        """Send a transaction"""
        try:
            w3 = self.web3_connections.get(blockchain_type.value)
            if not w3:
                raise ValueError(f"No connection to {blockchain_type.value}")
            
            # Get account from private key
            account = Account.from_key(private_key)
            
            # Get nonce
            nonce = w3.eth.get_transaction_count(from_address)
            
            # Build transaction
            transaction = {
                'to': to_address,
                'value': value,
                'gas': gas_limit or self.blockchain_configs[blockchain_type.value].gas_limit,
                'gasPrice': gas_price or self.blockchain_configs[blockchain_type.value].gas_price,
                'nonce': nonce,
            }
            
            # Sign transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Create transaction record
            tx_id = f"tx_{int(time.time())}"
            tx = BlockchainTransaction(
                id=tx_id,
                hash=tx_hash.hex(),
                from_address=from_address,
                to_address=to_address,
                value=value,
                gas_used=0,  # Will be updated when confirmed
                gas_price=transaction['gasPrice'],
                status=TransactionStatus.PENDING
            )
            
            self.transactions[tx_id] = tx
            
            # Update Prometheus metrics
            self.prometheus_metrics["blockchain_transactions"].labels(
                blockchain=blockchain_type.value,
                status="pending"
            ).inc()
            
            logger.info(f"Transaction sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            raise
    
    def deploy_contract(self, contract_abi: Dict[str, Any], bytecode: str,
                       blockchain_type: BlockchainType, private_key: str,
                       constructor_args: List[Any] = None) -> str:
        """Deploy a smart contract"""
        try:
            w3 = self.web3_connections.get(blockchain_type.value)
            if not w3:
                raise ValueError(f"No connection to {blockchain_type.value}")
            
            # Get account from private key
            account = Account.from_key(private_key)
            
            # Create contract
            contract = w3.eth.contract(abi=contract_abi, bytecode=bytecode)
            
            # Build constructor transaction
            constructor = contract.constructor(*(constructor_args or []))
            transaction = constructor.build_transaction({
                'from': account.address,
                'gas': 2000000,  # Adjust as needed
                'gasPrice': self.blockchain_configs[blockchain_type.value].gas_price,
                'nonce': w3.eth.get_transaction_count(account.address),
            })
            
            # Sign and send transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = receipt.contractAddress
            
            # Create smart contract record
            contract_id = f"contract_{int(time.time())}"
            smart_contract = SmartContract(
                id=contract_id,
                name=f"Contract_{contract_id}",
                contract_type=ContractType.ERC20,  # Default, should be determined from ABI
                address=contract_address,
                abi=contract_abi,
                blockchain_type=blockchain_type,
                deployed_at=datetime.utcnow(),
                owner=account.address
            )
            
            self.smart_contracts[contract_id] = smart_contract
            
            logger.info(f"Contract deployed at: {contract_address}")
            return contract_address
            
        except Exception as e:
            logger.error(f"Error deploying contract: {e}")
            raise
    
    def call_contract_method(self, contract_address: str, abi: Dict[str, Any],
                           method_name: str, args: List[Any],
                           blockchain_type: BlockchainType,
                           private_key: Optional[str] = None) -> Any:
        """Call a smart contract method"""
        try:
            w3 = self.web3_connections.get(blockchain_type.value)
            if not w3:
                raise ValueError(f"No connection to {blockchain_type.value}")
            
            # Create contract instance
            contract = w3.eth.contract(address=contract_address, abi=abi)
            
            # Get method
            method = getattr(contract.functions, method_name)
            
            if private_key:
                # Transaction call (write)
                account = Account.from_key(private_key)
                transaction = method(*args).build_transaction({
                    'from': account.address,
                    'gas': 200000,  # Adjust as needed
                    'gasPrice': self.blockchain_configs[blockchain_type.value].gas_price,
                    'nonce': w3.eth.get_transaction_count(account.address),
                })
                
                signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
                tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for transaction to be mined
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Update Prometheus metrics
                self.prometheus_metrics["smart_contract_calls"].labels(
                    blockchain=blockchain_type.value,
                    contract=contract_address,
                    method=method_name
                ).inc()
                
                return receipt
            else:
                # Call (read)
                result = method(*args).call()
                
                # Update Prometheus metrics
                self.prometheus_metrics["smart_contract_calls"].labels(
                    blockchain=blockchain_type.value,
                    contract=contract_address,
                    method=method_name
                ).inc()
                
                return result
                
        except Exception as e:
            logger.error(f"Error calling contract method: {e}")
            raise
    
    async def mint_nft(self, contract_address: str, abi: Dict[str, Any],
                      to_address: str, token_uri: str, metadata: NFTMetadata,
                      blockchain_type: BlockchainType, private_key: str) -> str:
        """Mint an NFT"""
        try:
            # Store metadata
            metadata_id = f"metadata_{int(time.time())}"
            self.nft_metadata[metadata_id] = metadata
            
            # Call mint function
            result = self.call_contract_method(
                contract_address=contract_address,
                abi=abi,
                method_name="mint",
                args=[to_address, token_uri],
                blockchain_type=blockchain_type,
                private_key=private_key
            )
            
            # Update Prometheus metrics
            self.prometheus_metrics["nft_minted"].labels(
                blockchain=blockchain_type.value,
                contract=contract_address
            ).inc()
            
            logger.info(f"NFT minted for {to_address}")
            return result.transactionHash.hex()
            
        except Exception as e:
            logger.error(f"Error minting NFT: {e}")
            raise
    
    def get_nft_metadata(self, token_id: int, contract_address: str,
                        abi: Dict[str, Any], blockchain_type: BlockchainType) -> NFTMetadata:
        """Get NFT metadata"""
        try:
            # Get token URI from contract
            token_uri = self.call_contract_method(
                contract_address=contract_address,
                abi=abi,
                method_name="tokenURI",
                args=[token_id],
                blockchain_type=blockchain_type
            )
            
            # Fetch metadata from URI
            # This would typically involve fetching from IPFS or HTTP
            # For now, return a placeholder
            return NFTMetadata(
                name=f"NFT #{token_id}",
                description="NFT generated by BUL system",
                image="https://example.com/nft-image.png",
                attributes=[
                    {"trait_type": "Token ID", "value": token_id},
                    {"trait_type": "Contract", "value": contract_address}
                ]
            )
            
        except Exception as e:
            logger.error(f"Error getting NFT metadata: {e}")
            raise
    
    def get_transaction_status(self, tx_hash: str) -> TransactionStatus:
        """Get transaction status"""
        for tx in self.transactions.values():
            if tx.hash == tx_hash:
                return tx.status
        
        return TransactionStatus.UNKNOWN
    
    def get_transaction_history(self, address: str, blockchain_type: BlockchainType) -> List[BlockchainTransaction]:
        """Get transaction history for an address"""
        return [
            tx for tx in self.transactions.values()
            if (tx.from_address == address or tx.to_address == address) and
            tx.blockchain_type == blockchain_type
        ]
    
    def get_contract_balance(self, contract_address: str, token_address: str,
                           abi: Dict[str, Any], user_address: str,
                           blockchain_type: BlockchainType) -> int:
        """Get ERC20 token balance"""
        try:
            balance = self.call_contract_method(
                contract_address=token_address,
                abi=abi,
                method_name="balanceOf",
                args=[user_address],
                blockchain_type=blockchain_type
            )
            
            return balance
            
        except Exception as e:
            logger.error(f"Error getting contract balance: {e}")
            raise
    
    def transfer_tokens(self, token_address: str, abi: Dict[str, Any],
                       to_address: str, amount: int, blockchain_type: BlockchainType,
                       private_key: str) -> str:
        """Transfer ERC20 tokens"""
        try:
            result = self.call_contract_method(
                contract_address=token_address,
                abi=abi,
                method_name="transfer",
                args=[to_address, amount],
                blockchain_type=blockchain_type,
                private_key=private_key
            )
            
            return result.transactionHash.hex()
            
        except Exception as e:
            logger.error(f"Error transferring tokens: {e}")
            raise
    
    def get_blockchain_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        total_transactions = len(self.transactions)
        confirmed_transactions = len([tx for tx in self.transactions.values() if tx.status == TransactionStatus.CONFIRMED])
        failed_transactions = len([tx for tx in self.transactions.values() if tx.status == TransactionStatus.FAILED])
        pending_transactions = len([tx for tx in self.transactions.values() if tx.status == TransactionStatus.PENDING])
        
        # Count by blockchain
        blockchain_counts = {}
        for tx in self.transactions.values():
            blockchain = tx.blockchain_type.value
            blockchain_counts[blockchain] = blockchain_counts.get(blockchain, 0) + 1
        
        # Count by contract type
        contract_type_counts = {}
        for contract in self.smart_contracts.values():
            contract_type = contract.contract_type.value
            contract_type_counts[contract_type] = contract_type_counts.get(contract_type, 0) + 1
        
        return {
            "total_transactions": total_transactions,
            "confirmed_transactions": confirmed_transactions,
            "failed_transactions": failed_transactions,
            "pending_transactions": pending_transactions,
            "success_rate": (confirmed_transactions / total_transactions * 100) if total_transactions > 0 else 0,
            "blockchain_counts": blockchain_counts,
            "contract_type_counts": contract_type_counts,
            "total_contracts": len(self.smart_contracts),
            "total_nft_metadata": len(self.nft_metadata),
            "active_blockchain_connections": len(self.web3_connections)
        }
    
    def export_blockchain_data(self) -> Dict[str, Any]:
        """Export blockchain data for analysis"""
        return {
            "transactions": [
                {
                    "id": tx.id,
                    "hash": tx.hash,
                    "from_address": tx.from_address,
                    "to_address": tx.to_address,
                    "value": tx.value,
                    "gas_used": tx.gas_used,
                    "gas_price": tx.gas_price,
                    "status": tx.status.value,
                    "block_number": tx.block_number,
                    "created_at": tx.created_at.isoformat(),
                    "confirmed_at": tx.confirmed_at.isoformat() if tx.confirmed_at else None
                }
                for tx in self.transactions.values()
            ],
            "smart_contracts": [
                {
                    "id": contract.id,
                    "name": contract.name,
                    "contract_type": contract.contract_type.value,
                    "address": contract.address,
                    "blockchain_type": contract.blockchain_type.value,
                    "deployed_at": contract.deployed_at.isoformat(),
                    "owner": contract.owner,
                    "version": contract.version,
                    "is_verified": contract.is_verified
                }
                for contract in self.smart_contracts.values()
            ],
            "nft_metadata": [
                {
                    "id": metadata_id,
                    "name": metadata.name,
                    "description": metadata.description,
                    "image": metadata.image,
                    "attributes": metadata.attributes,
                    "external_url": metadata.external_url,
                    "animation_url": metadata.animation_url,
                    "background_color": metadata.background_color
                }
                for metadata_id, metadata in self.nft_metadata.items()
            ],
            "statistics": self.get_blockchain_statistics(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global Web3 integration instance
web3_integration = None

def get_web3_integration() -> Web3Integration:
    """Get the global Web3 integration instance"""
    global web3_integration
    if web3_integration is None:
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 3
        }
        web3_integration = Web3Integration(config)
    return web3_integration

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 3
        }
        
        web3 = Web3Integration(config)
        
        # Create a wallet
        wallet = web3.create_wallet(BlockchainType.ETHEREUM)
        print(f"Created wallet: {wallet['address']}")
        
        # Get balance
        try:
            balance = web3.get_balance(wallet['address'], BlockchainType.ETHEREUM)
            print(f"Balance: {balance} wei")
        except Exception as e:
            print(f"Error getting balance: {e}")
        
        # Get blockchain statistics
        stats = web3.get_blockchain_statistics()
        print("Blockchain Statistics:")
        print(json.dumps(stats, indent=2))
        
        await web3.stop_monitoring()
    
    asyncio.run(main())













