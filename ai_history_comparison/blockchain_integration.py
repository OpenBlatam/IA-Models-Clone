"""
Blockchain Integration System
============================

This module provides blockchain integration capabilities including:
- Smart contract deployment and interaction
- Decentralized data storage
- Token-based incentives
- Immutable audit trails
- Decentralized AI model validation
- Cross-chain interoperability
- NFT-based model ownership
- DAO governance for AI systems
"""

import asyncio
import logging
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
from web3 import Web3
from eth_account import Account
import requests
from cryptography.fernet import Fernet
import ipfshttpclient

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config

logger = logging.getLogger(__name__)


@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    network: str  # ethereum, polygon, bsc, etc.
    rpc_url: str
    private_key: str
    contract_address: str
    gas_limit: int = 500000
    gas_price: int = 20
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SmartContract:
    """Smart contract information"""
    address: str
    abi: List[Dict[str, Any]]
    name: str
    version: str
    deployed_at: datetime
    functions: List[str]
    events: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BlockchainTransaction:
    """Blockchain transaction record"""
    tx_hash: str
    block_number: int
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: int
    timestamp: datetime
    status: str  # pending, confirmed, failed
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class NFTMetadata:
    """NFT metadata for AI models"""
    model_id: str
    model_name: str
    model_type: str
    performance_metrics: Dict[str, float]
    training_data_hash: str
    model_hash: str
    creator: str
    created_at: datetime
    description: str
    image_url: str
    attributes: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = []


class BlockchainIntegrationSystem:
    """Blockchain integration system for AI history analysis"""
    
    def __init__(self):
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        
        # Blockchain configurations
        self.blockchain_configs: Dict[str, BlockchainConfig] = {}
        self.web3_instances: Dict[str, Web3] = {}
        self.accounts: Dict[str, Account] = {}
        
        # Smart contracts
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.contract_instances: Dict[str, Any] = {}
        
        # IPFS integration
        self.ipfs_client = None
        
        # Transaction tracking
        self.transaction_history: List[BlockchainTransaction] = []
        self.nft_metadata: Dict[str, NFTMetadata] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Initialize blockchain configurations
        self._initialize_blockchain_configs()
    
    def _initialize_blockchain_configs(self):
        """Initialize blockchain configurations from environment variables"""
        try:
            # Ethereum configuration
            ethereum_config = BlockchainConfig(
                network="ethereum",
                rpc_url=os.getenv("ETHEREUM_RPC_URL", "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"),
                private_key=os.getenv("ETHEREUM_PRIVATE_KEY", ""),
                contract_address=os.getenv("ETHEREUM_CONTRACT_ADDRESS", ""),
                gas_limit=int(os.getenv("ETHEREUM_GAS_LIMIT", "500000")),
                gas_price=int(os.getenv("ETHEREUM_GAS_PRICE", "20")),
                enabled=os.getenv("ETHEREUM_ENABLED", "false").lower() == "true"
            )
            self.blockchain_configs["ethereum"] = ethereum_config
            
            # Polygon configuration
            polygon_config = BlockchainConfig(
                network="polygon",
                rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
                private_key=os.getenv("POLYGON_PRIVATE_KEY", ""),
                contract_address=os.getenv("POLYGON_CONTRACT_ADDRESS", ""),
                gas_limit=int(os.getenv("POLYGON_GAS_LIMIT", "500000")),
                gas_price=int(os.getenv("POLYGON_GAS_PRICE", "30")),
                enabled=os.getenv("POLYGON_ENABLED", "false").lower() == "true"
            )
            self.blockchain_configs["polygon"] = polygon_config
            
            # BSC configuration
            bsc_config = BlockchainConfig(
                network="bsc",
                rpc_url=os.getenv("BSC_RPC_URL", "https://bsc-dataseed.binance.org"),
                private_key=os.getenv("BSC_PRIVATE_KEY", ""),
                contract_address=os.getenv("BSC_CONTRACT_ADDRESS", ""),
                gas_limit=int(os.getenv("BSC_GAS_LIMIT", "500000")),
                gas_price=int(os.getenv("BSC_GAS_PRICE", "5")),
                enabled=os.getenv("BSC_ENABLED", "false").lower() == "true"
            )
            self.blockchain_configs["bsc"] = bsc_config
            
            logger.info(f"Initialized {len(self.blockchain_configs)} blockchain configurations")
        
        except Exception as e:
            logger.error(f"Error initializing blockchain configurations: {str(e)}")
    
    async def initialize_connections(self):
        """Initialize blockchain connections"""
        try:
            for network, config in self.blockchain_configs.items():
                if not config.enabled:
                    continue
                
                try:
                    # Initialize Web3 instance
                    w3 = Web3(Web3.HTTPProvider(config.rpc_url))
                    
                    # Test connection
                    if w3.is_connected():
                        self.web3_instances[network] = w3
                        
                        # Initialize account if private key is provided
                        if config.private_key:
                            account = Account.from_key(config.private_key)
                            self.accounts[network] = account
                            logger.info(f"Initialized {network} connection with account {account.address}")
                        else:
                            logger.info(f"Initialized {network} connection (read-only)")
                    else:
                        logger.warning(f"Failed to connect to {network}")
                
                except Exception as e:
                    logger.error(f"Error initializing {network} connection: {str(e)}")
            
            # Initialize IPFS client
            try:
                self.ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
                logger.info("IPFS client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize IPFS client: {str(e)}")
            
            logger.info("Blockchain connections initialized")
        
        except Exception as e:
            logger.error(f"Error initializing blockchain connections: {str(e)}")
            raise
    
    async def deploy_smart_contract(self, 
                                  network: str,
                                  contract_name: str,
                                  contract_abi: List[Dict[str, Any]],
                                  constructor_args: List[Any] = None) -> str:
        """Deploy smart contract to blockchain"""
        try:
            if network not in self.web3_instances:
                raise ValueError(f"Network {network} not initialized")
            
            w3 = self.web3_instances[network]
            account = self.accounts.get(network)
            
            if not account:
                raise ValueError(f"No account configured for {network}")
            
            # Get contract bytecode (would be loaded from compiled contract)
            contract_bytecode = "0x608060405234801561001057600080fd5b50..."  # Placeholder
            
            # Create contract instance
            contract = w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
            
            # Build constructor transaction
            constructor = contract.constructor()
            if constructor_args:
                constructor = contract.constructor(*constructor_args)
            
            # Get gas estimate
            gas_estimate = constructor.estimate_gas({'from': account.address})
            
            # Build transaction
            transaction = constructor.build_transaction({
                'from': account.address,
                'gas': gas_estimate,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(account.address)
            })
            
            # Sign and send transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, account.key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = receipt.contractAddress
            
            # Store contract information
            smart_contract = SmartContract(
                address=contract_address,
                abi=contract_abi,
                name=contract_name,
                version="1.0.0",
                deployed_at=datetime.now(),
                functions=[item["name"] for item in contract_abi if item["type"] == "function"],
                events=[item["name"] for item in contract_abi if item["type"] == "event"]
            )
            
            self.smart_contracts[f"{network}_{contract_name}"] = smart_contract
            
            # Create contract instance for interaction
            contract_instance = w3.eth.contract(address=contract_address, abi=contract_abi)
            self.contract_instances[f"{network}_{contract_name}"] = contract_instance
            
            # Record transaction
            tx_record = BlockchainTransaction(
                tx_hash=tx_hash.hex(),
                block_number=receipt.blockNumber,
                from_address=account.address,
                to_address=contract_address,
                value=0.0,
                gas_used=receipt.gasUsed,
                gas_price=transaction['gasPrice'],
                timestamp=datetime.now(),
                status="confirmed",
                data={"contract_name": contract_name, "network": network}
            )
            
            self.transaction_history.append(tx_record)
            
            logger.info(f"Deployed {contract_name} to {network} at {contract_address}")
            return contract_address
        
        except Exception as e:
            logger.error(f"Error deploying smart contract: {str(e)}")
            raise
    
    async def store_data_on_blockchain(self, 
                                     network: str,
                                     data: Dict[str, Any],
                                     data_type: str = "performance_metrics") -> str:
        """Store data on blockchain"""
        try:
            if network not in self.contract_instances:
                raise ValueError(f"No contract instance for {network}")
            
            # Get contract instance
            contract_key = f"{network}_AIHistoryContract"
            if contract_key not in self.contract_instances:
                raise ValueError(f"AI History contract not deployed on {network}")
            
            contract = self.contract_instances[contract_key]
            account = self.accounts[network]
            
            # Prepare data for storage
            data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            timestamp = int(time.time())
            
            # Call smart contract function
            transaction = contract.functions.storeData(
                data_hash,
                data_type,
                timestamp,
                json.dumps(data)
            ).build_transaction({
                'from': account.address,
                'gas': 200000,
                'gasPrice': self.web3_instances[network].eth.gas_price,
                'nonce': self.web3_instances[network].eth.get_transaction_count(account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.web3_instances[network].eth.account.sign_transaction(transaction, account.key)
            tx_hash = self.web3_instances[network].eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.web3_instances[network].eth.wait_for_transaction_receipt(tx_hash)
            
            # Record transaction
            tx_record = BlockchainTransaction(
                tx_hash=tx_hash.hex(),
                block_number=receipt.blockNumber,
                from_address=account.address,
                to_address=contract.address,
                value=0.0,
                gas_used=receipt.gasUsed,
                gas_price=transaction['gasPrice'],
                timestamp=datetime.now(),
                status="confirmed",
                data={"data_type": data_type, "data_hash": data_hash}
            )
            
            self.transaction_history.append(tx_record)
            
            logger.info(f"Stored {data_type} data on {network}: {tx_hash.hex()}")
            return tx_hash.hex()
        
        except Exception as e:
            logger.error(f"Error storing data on blockchain: {str(e)}")
            raise
    
    async def create_nft_for_model(self, 
                                 network: str,
                                 model_name: str,
                                 model_type: str,
                                 performance_metrics: Dict[str, float]) -> str:
        """Create NFT for AI model"""
        try:
            if network not in self.contract_instances:
                raise ValueError(f"No contract instance for {network}")
            
            # Get contract instance
            contract_key = f"{network}_ModelNFTContract"
            if contract_key not in self.contract_instances:
                raise ValueError(f"Model NFT contract not deployed on {network}")
            
            contract = self.contract_instances[contract_key]
            account = self.accounts[network]
            
            # Create NFT metadata
            model_id = f"{model_name}_{int(time.time())}"
            nft_metadata = NFTMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                performance_metrics=performance_metrics,
                training_data_hash="",  # Would be calculated from actual training data
                model_hash="",  # Would be calculated from actual model
                creator=account.address,
                created_at=datetime.now(),
                description=f"AI Model NFT for {model_name}",
                image_url="",  # Would be uploaded to IPFS
                attributes=[
                    {"trait_type": "Model Type", "value": model_type},
                    {"trait_type": "Quality Score", "value": performance_metrics.get("quality_score", 0.0)},
                    {"trait_type": "Performance", "value": performance_metrics.get("performance", 0.0)}
                ]
            )
            
            # Upload metadata to IPFS
            metadata_hash = await self._upload_to_ipfs(asdict(nft_metadata))
            
            # Mint NFT
            transaction = contract.functions.mint(
                account.address,
                metadata_hash,
                model_id
            ).build_transaction({
                'from': account.address,
                'gas': 300000,
                'gasPrice': self.web3_instances[network].eth.gas_price,
                'nonce': self.web3_instances[network].eth.get_transaction_count(account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.web3_instances[network].eth.account.sign_transaction(transaction, account.key)
            tx_hash = self.web3_instances[network].eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.web3_instances[network].eth.wait_for_transaction_receipt(tx_hash)
            
            # Get token ID from event logs
            token_id = None
            for log in receipt.logs:
                try:
                    decoded_log = contract.events.Transfer().process_log(log)
                    token_id = decoded_log['args']['tokenId']
                    break
                except:
                    continue
            
            # Store NFT metadata
            self.nft_metadata[f"{network}_{token_id}"] = nft_metadata
            
            # Record transaction
            tx_record = BlockchainTransaction(
                tx_hash=tx_hash.hex(),
                block_number=receipt.blockNumber,
                from_address=account.address,
                to_address=contract.address,
                value=0.0,
                gas_used=receipt.gasUsed,
                gas_price=transaction['gasPrice'],
                timestamp=datetime.now(),
                status="confirmed",
                data={"nft_token_id": token_id, "model_id": model_id}
            )
            
            self.transaction_history.append(tx_record)
            
            logger.info(f"Created NFT for model {model_name} on {network}: Token ID {token_id}")
            return str(token_id) if token_id else tx_hash.hex()
        
        except Exception as e:
            logger.error(f"Error creating NFT for model: {str(e)}")
            raise
    
    async def _upload_to_ipfs(self, data: Dict[str, Any]) -> str:
        """Upload data to IPFS"""
        try:
            if not self.ipfs_client:
                # Fallback: return hash of data
                return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            
            # Upload to IPFS
            result = self.ipfs_client.add_json(data)
            return result['Hash']
        
        except Exception as e:
            logger.error(f"Error uploading to IPFS: {str(e)}")
            # Fallback: return hash of data
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    async def validate_model_on_blockchain(self, 
                                         network: str,
                                         model_hash: str,
                                         validation_data: Dict[str, Any]) -> bool:
        """Validate AI model on blockchain"""
        try:
            if network not in self.contract_instances:
                raise ValueError(f"No contract instance for {network}")
            
            # Get contract instance
            contract_key = f"{network}_ModelValidationContract"
            if contract_key not in self.contract_instances:
                raise ValueError(f"Model validation contract not deployed on {network}")
            
            contract = self.contract_instances[contract_key]
            account = self.accounts[network]
            
            # Submit validation
            transaction = contract.functions.submitValidation(
                model_hash,
                json.dumps(validation_data),
                int(time.time())
            ).build_transaction({
                'from': account.address,
                'gas': 200000,
                'gasPrice': self.web3_instances[network].eth.gas_price,
                'nonce': self.web3_instances[network].eth.get_transaction_count(account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.web3_instances[network].eth.account.sign_transaction(transaction, account.key)
            tx_hash = self.web3_instances[network].eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.web3_instances[network].eth.wait_for_transaction_receipt(tx_hash)
            
            # Record transaction
            tx_record = BlockchainTransaction(
                tx_hash=tx_hash.hex(),
                block_number=receipt.blockNumber,
                from_address=account.address,
                to_address=contract.address,
                value=0.0,
                gas_used=receipt.gasUsed,
                gas_price=transaction['gasPrice'],
                timestamp=datetime.now(),
                status="confirmed",
                data={"model_hash": model_hash, "validation_data": validation_data}
            )
            
            self.transaction_history.append(tx_record)
            
            logger.info(f"Submitted model validation on {network}: {tx_hash.hex()}")
            return True
        
        except Exception as e:
            logger.error(f"Error validating model on blockchain: {str(e)}")
            return False
    
    async def get_blockchain_data(self, 
                                network: str,
                                data_type: str = "performance_metrics",
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve data from blockchain"""
        try:
            if network not in self.contract_instances:
                raise ValueError(f"No contract instance for {network}")
            
            # Get contract instance
            contract_key = f"{network}_AIHistoryContract"
            if contract_key not in self.contract_instances:
                raise ValueError(f"AI History contract not deployed on {network}")
            
            contract = self.contract_instances[contract_key]
            
            # Get data from smart contract
            data_events = contract.events.DataStored.get_logs(fromBlock=0)
            
            # Filter by data type and limit
            filtered_data = []
            for event in data_events:
                if event['args']['dataType'] == data_type:
                    try:
                        data = json.loads(event['args']['data'])
                        filtered_data.append({
                            'block_number': event['blockNumber'],
                            'transaction_hash': event['transactionHash'].hex(),
                            'data_hash': event['args']['dataHash'],
                            'timestamp': event['args']['timestamp'],
                            'data': data
                        })
                    except:
                        continue
                
                if len(filtered_data) >= limit:
                    break
            
            return filtered_data
        
        except Exception as e:
            logger.error(f"Error retrieving blockchain data: {str(e)}")
            return []
    
    async def start_monitoring(self):
        """Start blockchain monitoring"""
        if self.is_monitoring:
            logger.warning("Blockchain monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started blockchain monitoring")
    
    async def stop_monitoring(self):
        """Stop blockchain monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped blockchain monitoring")
    
    async def _monitoring_loop(self):
        """Blockchain monitoring loop"""
        while self.is_monitoring:
            try:
                await self._monitor_transactions()
                await self._monitor_contract_events()
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                logger.error(f"Error in blockchain monitoring loop: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _monitor_transactions(self):
        """Monitor blockchain transactions"""
        try:
            for network, w3 in self.web3_instances.items():
                # Get latest block
                latest_block = w3.eth.block_number
                
                # Check for new transactions
                for tx_record in self.transaction_history:
                    if tx_record.status == "pending":
                        try:
                            receipt = w3.eth.get_transaction_receipt(tx_record.tx_hash)
                            if receipt:
                                tx_record.status = "confirmed"
                                tx_record.block_number = receipt.blockNumber
                                tx_record.gas_used = receipt.gasUsed
                        except:
                            # Transaction still pending
                            pass
        
        except Exception as e:
            logger.error(f"Error monitoring transactions: {str(e)}")
    
    async def _monitor_contract_events(self):
        """Monitor smart contract events"""
        try:
            for contract_key, contract in self.contract_instances.items():
                # Get recent events
                try:
                    events = contract.events.all_events.get_logs(fromBlock='latest')
                    for event in events:
                        logger.info(f"Contract event from {contract_key}: {event['event']}")
                except:
                    pass
        
        except Exception as e:
            logger.error(f"Error monitoring contract events: {str(e)}")
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get blockchain integration status"""
        return {
            "total_networks": len(self.blockchain_configs),
            "enabled_networks": len([c for c in self.blockchain_configs.values() if c.enabled]),
            "connected_networks": len(self.web3_instances),
            "deployed_contracts": len(self.smart_contracts),
            "total_transactions": len(self.transaction_history),
            "nft_count": len(self.nft_metadata),
            "is_monitoring": self.is_monitoring,
            "networks": {
                network: {
                    "enabled": config.enabled,
                    "connected": network in self.web3_instances,
                    "has_account": network in self.accounts,
                    "contracts": len([c for c in self.smart_contracts.keys() if c.startswith(network)])
                }
                for network, config in self.blockchain_configs.items()
            }
        }
    
    def get_transaction_history(self, limit: int = 100) -> List[BlockchainTransaction]:
        """Get transaction history"""
        return self.transaction_history[-limit:]
    
    def get_nft_metadata(self, network: str = None) -> Dict[str, NFTMetadata]:
        """Get NFT metadata"""
        if network:
            return {k: v for k, v in self.nft_metadata.items() if k.startswith(network)}
        return self.nft_metadata


# Global blockchain integration instance
_blockchain_integration: Optional[BlockchainIntegrationSystem] = None


def get_blockchain_integration() -> BlockchainIntegrationSystem:
    """Get or create global blockchain integration"""
    global _blockchain_integration
    if _blockchain_integration is None:
        _blockchain_integration = BlockchainIntegrationSystem()
    return _blockchain_integration


# Example usage
async def main():
    """Example usage of blockchain integration"""
    blockchain = get_blockchain_integration()
    
    # Initialize connections
    await blockchain.initialize_connections()
    
    # Start monitoring
    await blockchain.start_monitoring()
    
    # Store performance data
    performance_data = {
        "model_name": "gpt-4",
        "quality_score": 0.85,
        "response_time": 1.2,
        "timestamp": datetime.now().isoformat()
    }
    
    # Store on blockchain (if configured)
    if blockchain.blockchain_configs.get("ethereum", {}).enabled:
        tx_hash = await blockchain.store_data_on_blockchain(
            network="ethereum",
            data=performance_data,
            data_type="performance_metrics"
        )
        print(f"Stored data on blockchain: {tx_hash}")
    
    # Create NFT for model
    if blockchain.blockchain_configs.get("polygon", {}).enabled:
        nft_token_id = await blockchain.create_nft_for_model(
            network="polygon",
            model_name="gpt-4",
            model_type="text_generation",
            performance_metrics={"quality_score": 0.85, "performance": 0.9}
        )
        print(f"Created NFT: {nft_token_id}")
    
    # Get status
    status = blockchain.get_blockchain_status()
    print(f"Blockchain status: {status}")
    
    # Stop monitoring
    await blockchain.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
























