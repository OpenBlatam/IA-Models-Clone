"""
Advanced Blockchain and Web3 Integration for Microservices
Features: Smart contracts, DeFi protocols, NFT management, DAO governance, cross-chain bridges
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
from abc import ABC, abstractmethod

# Blockchain imports
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    from web3.contract import Contract
    from eth_account import Account
    from eth_typing import Address, HexStr
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

try:
    import brownie
    from brownie import network, accounts, Contract as BrownieContract
    BROWNIE_AVAILABLE = True
except ImportError:
    BROWNIE_AVAILABLE = False

try:
    import solana
    from solana.rpc.api import Client
    from solana.keypair import Keypair
    from solana.publickey import PublicKey
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False

try:
    import substrateinterface
    from substrateinterface import SubstrateInterface
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False

logger = logging.getLogger(__name__)

class BlockchainType(Enum):
    """Blockchain types"""
    ETHEREUM = "ethereum"
    BINANCE_SMART_CHAIN = "bsc"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"
    SOLANA = "solana"
    POLKADOT = "polkadot"
    CARDANO = "cardano"
    COSMOS = "cosmos"

class ContractType(Enum):
    """Smart contract types"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    DEFI_PROTOCOL = "defi_protocol"
    DAO_GOVERNANCE = "dao_governance"
    CROSS_CHAIN_BRIDGE = "cross_chain_bridge"
    ORACLE = "oracle"
    STAKING = "staking"

class TransactionStatus(Enum):
    """Transaction status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVERTED = "reverted"

@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    blockchain_type: BlockchainType
    rpc_url: str
    chain_id: int
    gas_price: int = 20000000000  # 20 gwei
    gas_limit: int = 21000
    timeout: int = 300
    retry_attempts: int = 3
    private_key: Optional[str] = None
    contract_addresses: Dict[str, str] = field(default_factory=dict)

@dataclass
class SmartContract:
    """Smart contract definition"""
    contract_id: str
    contract_type: ContractType
    address: str
    abi: List[Dict[str, Any]]
    blockchain: BlockchainType
    deployed_at: Optional[int] = None
    owner: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Transaction:
    """Blockchain transaction"""
    tx_hash: str
    from_address: str
    to_address: str
    value: int
    gas_used: int
    gas_price: int
    status: TransactionStatus
    block_number: Optional[int] = None
    timestamp: Optional[int] = None
    data: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Token:
    """Token definition"""
    token_id: str
    contract_address: str
    symbol: str
    name: str
    decimals: int
    total_supply: int
    blockchain: BlockchainType
    metadata: Dict[str, Any] = field(default_factory=dict)

class BlockchainConnector:
    """
    Blockchain connection manager
    """
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.web3: Optional[Web3] = None
        self.account: Optional[Account] = None
        self.connected = False
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize blockchain connection"""
        try:
            if not WEB3_AVAILABLE:
                raise ImportError("Web3 not available")
            
            # Connect to blockchain
            self.web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            
            # Add PoA middleware for some networks
            if self.config.blockchain_type in [BlockchainType.BINANCE_SMART_CHAIN, BlockchainType.POLYGON]:
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Check connection
            if self.web3.is_connected():
                self.connected = True
                
                # Initialize account if private key provided
                if self.config.private_key:
                    self.account = Account.from_key(self.config.private_key)
                
                logger.info(f"Connected to {self.config.blockchain_type.value}")
            else:
                raise ConnectionError("Failed to connect to blockchain")
                
        except Exception as e:
            logger.error(f"Blockchain connection failed: {e}")
            self.connected = False
    
    async def get_balance(self, address: str) -> int:
        """Get account balance"""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to blockchain")
            
            balance = self.web3.eth.get_balance(address)
            return balance
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    async def send_transaction(
        self, 
        to_address: str, 
        value: int, 
        data: str = None,
        gas_limit: int = None
    ) -> str:
        """Send transaction"""
        try:
            if not self.connected or not self.account:
                raise RuntimeError("Not connected or no account available")
            
            # Build transaction
            transaction = {
                'to': to_address,
                'value': value,
                'gas': gas_limit or self.config.gas_limit,
                'gasPrice': self.config.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'chainId': self.config.chain_id
            }
            
            if data:
                transaction['data'] = data
            
            # Sign transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Send transaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise
    
    async def get_transaction_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get transaction receipt"""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to blockchain")
            
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            return dict(receipt)
            
        except Exception as e:
            logger.error(f"Failed to get transaction receipt: {e}")
            return None
    
    async def wait_for_transaction(self, tx_hash: str, timeout: int = None) -> TransactionStatus:
        """Wait for transaction confirmation"""
        try:
            timeout = timeout or self.config.timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                receipt = await self.get_transaction_receipt(tx_hash)
                
                if receipt:
                    if receipt['status'] == 1:
                        return TransactionStatus.CONFIRMED
                    else:
                        return TransactionStatus.REVERTED
                
                await asyncio.sleep(1)
            
            return TransactionStatus.PENDING
            
        except Exception as e:
            logger.error(f"Failed to wait for transaction: {e}")
            return TransactionStatus.FAILED

class SmartContractManager:
    """
    Smart contract management
    """
    
    def __init__(self, blockchain_connector: BlockchainConnector):
        self.connector = blockchain_connector
        self.contracts: Dict[str, SmartContract] = {}
        self.contract_instances: Dict[str, Contract] = {}
    
    def deploy_contract(
        self, 
        contract_id: str, 
        contract_type: ContractType,
        bytecode: str, 
        abi: List[Dict[str, Any]],
        constructor_args: List[Any] = None
    ) -> str:
        """Deploy smart contract"""
        try:
            if not self.connector.connected or not self.connector.account:
                raise RuntimeError("Not connected or no account available")
            
            # Create contract
            contract = self.connector.web3.eth.contract(
                abi=abi, 
                bytecode=bytecode
            )
            
            # Build constructor transaction
            constructor = contract.constructor(*(constructor_args or []))
            transaction = constructor.build_transaction({
                'from': self.connector.account.address,
                'gas': 2000000,  # Higher gas limit for deployment
                'gasPrice': self.connector.config.gas_price,
                'nonce': self.connector.web3.eth.get_transaction_count(self.connector.account.address),
                'chainId': self.connector.config.chain_id
            })
            
            # Sign and send transaction
            signed_txn = self.connector.web3.eth.account.sign_transaction(
                transaction, 
                self.connector.account.key
            )
            tx_hash = self.connector.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for deployment
            receipt = self.connector.web3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = receipt['contractAddress']
            
            # Store contract
            smart_contract = SmartContract(
                contract_id=contract_id,
                contract_type=contract_type,
                address=contract_address,
                abi=abi,
                blockchain=self.connector.config.blockchain_type,
                deployed_at=int(time.time()),
                owner=self.connector.account.address
            )
            
            self.contracts[contract_id] = smart_contract
            self.contract_instances[contract_id] = self.connector.web3.eth.contract(
                address=contract_address,
                abi=abi
            )
            
            logger.info(f"Contract {contract_id} deployed at {contract_address}")
            return contract_address
            
        except Exception as e:
            logger.error(f"Contract deployment failed: {e}")
            raise
    
    def load_contract(self, contract_id: str, address: str, abi: List[Dict[str, Any]]) -> SmartContract:
        """Load existing contract"""
        try:
            smart_contract = SmartContract(
                contract_id=contract_id,
                contract_type=ContractType.ERC20,  # Default type
                address=address,
                abi=abi,
                blockchain=self.connector.config.blockchain_type
            )
            
            self.contracts[contract_id] = smart_contract
            self.contract_instances[contract_id] = self.connector.web3.eth.contract(
                address=address,
                abi=abi
            )
            
            logger.info(f"Contract {contract_id} loaded from {address}")
            return smart_contract
            
        except Exception as e:
            logger.error(f"Contract loading failed: {e}")
            raise
    
    async def call_contract_function(
        self, 
        contract_id: str, 
        function_name: str, 
        args: List[Any] = None,
        value: int = 0
    ) -> Any:
        """Call contract function"""
        try:
            if contract_id not in self.contract_instances:
                raise ValueError(f"Contract {contract_id} not found")
            
            contract = self.contract_instances[contract_id]
            function = getattr(contract.functions, function_name)
            
            if value > 0:
                # State-changing function
                transaction = function(*(args or [])).build_transaction({
                    'from': self.connector.account.address,
                    'value': value,
                    'gas': 200000,
                    'gasPrice': self.connector.config.gas_price,
                    'nonce': self.connector.web3.eth.get_transaction_count(self.connector.account.address),
                    'chainId': self.connector.config.chain_id
                })
                
                signed_txn = self.connector.web3.eth.account.sign_transaction(
                    transaction, 
                    self.connector.account.key
                )
                tx_hash = self.connector.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                return tx_hash.hex()
            else:
                # Read-only function
                result = function(*(args or [])).call()
                return result
                
        except Exception as e:
            logger.error(f"Contract function call failed: {e}")
            raise

class DeFiProtocol:
    """
    DeFi protocol integration
    """
    
    def __init__(self, contract_manager: SmartContractManager):
        self.contract_manager = contract_manager
        self.protocols: Dict[str, Dict[str, Any]] = {}
    
    async def add_liquidity(
        self, 
        protocol: str, 
        token_a: str, 
        token_b: str, 
        amount_a: int, 
        amount_b: int
    ) -> str:
        """Add liquidity to DEX"""
        try:
            # This would interact with actual DeFi protocols like Uniswap, PancakeSwap, etc.
            # For demo purposes, we'll simulate the transaction
            
            tx_hash = await self.contract_manager.call_contract_function(
                contract_id=f"{protocol}_router",
                function_name="addLiquidity",
                args=[token_a, token_b, amount_a, amount_b, 0, 0, self.contract_manager.connector.account.address, int(time.time()) + 300]
            )
            
            logger.info(f"Added liquidity to {protocol}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Add liquidity failed: {e}")
            raise
    
    async def swap_tokens(
        self, 
        protocol: str, 
        token_in: str, 
        token_out: str, 
        amount_in: int,
        min_amount_out: int = 0
    ) -> str:
        """Swap tokens on DEX"""
        try:
            tx_hash = await self.contract_manager.call_contract_function(
                contract_id=f"{protocol}_router",
                function_name="swapExactTokensForTokens",
                args=[amount_in, min_amount_out, [token_in, token_out], self.contract_manager.connector.account.address, int(time.time()) + 300]
            )
            
            logger.info(f"Swapped tokens on {protocol}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Token swap failed: {e}")
            raise
    
    async def stake_tokens(
        self, 
        protocol: str, 
        token: str, 
        amount: int
    ) -> str:
        """Stake tokens in DeFi protocol"""
        try:
            tx_hash = await self.contract_manager.call_contract_function(
                contract_id=f"{protocol}_staking",
                function_name="stake",
                args=[amount]
            )
            
            logger.info(f"Staked {amount} {token} in {protocol}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Staking failed: {e}")
            raise

class NFTManager:
    """
    NFT management system
    """
    
    def __init__(self, contract_manager: SmartContractManager):
        self.contract_manager = contract_manager
        self.nft_contracts: Dict[str, str] = {}
        self.nft_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def mint_nft(
        self, 
        contract_id: str, 
        to_address: str, 
        token_uri: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Mint new NFT"""
        try:
            tx_hash = await self.contract_manager.call_contract_function(
                contract_id=contract_id,
                function_name="mint",
                args=[to_address, token_uri]
            )
            
            # Store metadata
            if metadata:
                self.nft_metadata[token_uri] = metadata
            
            logger.info(f"Minted NFT to {to_address}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"NFT minting failed: {e}")
            raise
    
    async def transfer_nft(
        self, 
        contract_id: str, 
        from_address: str, 
        to_address: str, 
        token_id: int
    ) -> str:
        """Transfer NFT"""
        try:
            tx_hash = await self.contract_manager.call_contract_function(
                contract_id=contract_id,
                function_name="transferFrom",
                args=[from_address, to_address, token_id]
            )
            
            logger.info(f"Transferred NFT {token_id} from {from_address} to {to_address}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"NFT transfer failed: {e}")
            raise
    
    async def get_nft_owner(self, contract_id: str, token_id: int) -> str:
        """Get NFT owner"""
        try:
            owner = await self.contract_manager.call_contract_function(
                contract_id=contract_id,
                function_name="ownerOf",
                args=[token_id]
            )
            
            return owner
            
        except Exception as e:
            logger.error(f"Failed to get NFT owner: {e}")
            return ""

class DAOGovernance:
    """
    DAO governance system
    """
    
    def __init__(self, contract_manager: SmartContractManager):
        self.contract_manager = contract_manager
        self.proposals: Dict[str, Dict[str, Any]] = {}
    
    async def create_proposal(
        self, 
        contract_id: str, 
        description: str, 
        targets: List[str], 
        values: List[int], 
        calldatas: List[str]
    ) -> str:
        """Create governance proposal"""
        try:
            tx_hash = await self.contract_manager.call_contract_function(
                contract_id=contract_id,
                function_name="propose",
                args=[targets, values, calldatas, description]
            )
            
            # Store proposal info
            proposal_id = hashlib.sha256(tx_hash.encode()).hexdigest()[:16]
            self.proposals[proposal_id] = {
                "description": description,
                "targets": targets,
                "values": values,
                "calldatas": calldatas,
                "tx_hash": tx_hash,
                "created_at": time.time()
            }
            
            logger.info(f"Created proposal {proposal_id}: {tx_hash}")
            return proposal_id
            
        except Exception as e:
            logger.error(f"Proposal creation failed: {e}")
            raise
    
    async def vote_on_proposal(
        self, 
        contract_id: str, 
        proposal_id: str, 
        support: int  # 0 = against, 1 = for, 2 = abstain
    ) -> str:
        """Vote on governance proposal"""
        try:
            tx_hash = await self.contract_manager.call_contract_function(
                contract_id=contract_id,
                function_name="castVote",
                args=[proposal_id, support]
            )
            
            logger.info(f"Voted on proposal {proposal_id}: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Voting failed: {e}")
            raise

class CrossChainBridge:
    """
    Cross-chain bridge functionality
    """
    
    def __init__(self, blockchain_configs: Dict[BlockchainType, BlockchainConfig]):
        self.configs = blockchain_configs
        self.connectors: Dict[BlockchainType, BlockchainConnector] = {}
        self.bridge_contracts: Dict[BlockchainType, str] = {}
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize blockchain connectors"""
        for blockchain_type, config in self.configs.items():
            try:
                connector = BlockchainConnector(config)
                if connector.connected:
                    self.connectors[blockchain_type] = connector
                    logger.info(f"Connected to {blockchain_type.value}")
            except Exception as e:
                logger.error(f"Failed to connect to {blockchain_type.value}: {e}")
    
    async def bridge_tokens(
        self, 
        from_chain: BlockchainType, 
        to_chain: BlockchainType, 
        token: str, 
        amount: int,
        recipient: str
    ) -> Dict[str, str]:
        """Bridge tokens between chains"""
        try:
            if from_chain not in self.connectors or to_chain not in self.connectors:
                raise ValueError("Chain not supported")
            
            from_connector = self.connectors[from_chain]
            to_connector = self.connectors[to_chain]
            
            # Lock tokens on source chain
            lock_tx = await from_connector.send_transaction(
                to_address=self.bridge_contracts.get(from_chain, ""),
                value=0,
                data=f"0x{token}{amount:064x}{recipient}"
            )
            
            # Wait for confirmation
            status = await from_connector.wait_for_transaction(lock_tx)
            
            if status == TransactionStatus.CONFIRMED:
                # Mint tokens on destination chain
                mint_tx = await to_connector.send_transaction(
                    to_address=self.bridge_contracts.get(to_chain, ""),
                    value=0,
                    data=f"0x{token}{amount:064x}{recipient}"
                )
                
                return {
                    "lock_tx": lock_tx,
                    "mint_tx": mint_tx,
                    "status": "success"
                }
            else:
                return {
                    "lock_tx": lock_tx,
                    "mint_tx": "",
                    "status": "failed"
                }
                
        except Exception as e:
            logger.error(f"Cross-chain bridge failed: {e}")
            raise

class BlockchainManager:
    """
    Main blockchain management system
    """
    
    def __init__(self, configs: Dict[BlockchainType, BlockchainConfig]):
        self.configs = configs
        self.connectors: Dict[BlockchainType, BlockchainConnector] = {}
        self.contract_managers: Dict[BlockchainType, SmartContractManager] = {}
        self.defi_protocols: Dict[BlockchainType, DeFiProtocol] = {}
        self.nft_managers: Dict[BlockchainType, NFTManager] = {}
        self.dao_governance: Dict[BlockchainType, DAOGovernance] = {}
        self.cross_chain_bridge: Optional[CrossChainBridge] = None
        self.transaction_history: deque = deque(maxlen=10000)
        
        self._initialize_managers()
    
    def _initialize_managers(self):
        """Initialize blockchain managers"""
        for blockchain_type, config in self.configs.items():
            try:
                # Initialize connector
                connector = BlockchainConnector(config)
                if connector.connected:
                    self.connectors[blockchain_type] = connector
                    
                    # Initialize managers
                    contract_manager = SmartContractManager(connector)
                    self.contract_managers[blockchain_type] = contract_manager
                    
                    defi_protocol = DeFiProtocol(contract_manager)
                    self.defi_protocols[blockchain_type] = defi_protocol
                    
                    nft_manager = NFTManager(contract_manager)
                    self.nft_managers[blockchain_type] = nft_manager
                    
                    dao_gov = DAOGovernance(contract_manager)
                    self.dao_governance[blockchain_type] = dao_gov
                    
                    logger.info(f"Initialized managers for {blockchain_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {blockchain_type.value}: {e}")
        
        # Initialize cross-chain bridge
        if len(self.connectors) > 1:
            self.cross_chain_bridge = CrossChainBridge(self.configs)
    
    async def get_balance(self, blockchain: BlockchainType, address: str) -> int:
        """Get balance on specific blockchain"""
        if blockchain not in self.connectors:
            raise ValueError(f"Blockchain {blockchain.value} not supported")
        
        return await self.connectors[blockchain].get_balance(address)
    
    async def send_transaction(
        self, 
        blockchain: BlockchainType, 
        to_address: str, 
        value: int,
        data: str = None
    ) -> str:
        """Send transaction on specific blockchain"""
        if blockchain not in self.connectors:
            raise ValueError(f"Blockchain {blockchain.value} not supported")
        
        tx_hash = await self.connectors[blockchain].send_transaction(to_address, value, data)
        
        # Store transaction
        self.transaction_history.append({
            "blockchain": blockchain.value,
            "tx_hash": tx_hash,
            "to_address": to_address,
            "value": value,
            "timestamp": time.time()
        })
        
        return tx_hash
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        return {
            "supported_blockchains": [chain.value for chain in self.connectors.keys()],
            "total_transactions": len(self.transaction_history),
            "recent_transactions": len(list(self.transaction_history)[-10:]),
            "cross_chain_bridge": self.cross_chain_bridge is not None,
            "blockchain_status": {
                chain.value: {
                    "connected": True,
                    "chain_id": self.configs[chain].chain_id,
                    "rpc_url": self.configs[chain].rpc_url
                }
                for chain in self.connectors.keys()
            }
        }

# Global blockchain manager
blockchain_manager: Optional[BlockchainManager] = None

def initialize_blockchain(configs: Dict[BlockchainType, BlockchainConfig] = None):
    """Initialize blockchain manager"""
    global blockchain_manager
    
    if configs is None:
        # Default configurations for popular blockchains
        configs = {
            BlockchainType.ETHEREUM: BlockchainConfig(
                blockchain_type=BlockchainType.ETHEREUM,
                rpc_url="https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
                chain_id=1
            ),
            BlockchainType.BINANCE_SMART_CHAIN: BlockchainConfig(
                blockchain_type=BlockchainType.BINANCE_SMART_CHAIN,
                rpc_url="https://bsc-dataseed.binance.org/",
                chain_id=56
            ),
            BlockchainType.POLYGON: BlockchainConfig(
                blockchain_type=BlockchainType.POLYGON,
                rpc_url="https://polygon-rpc.com/",
                chain_id=137
            )
        }
    
    blockchain_manager = BlockchainManager(configs)
    logger.info("Blockchain manager initialized")

# Decorator for blockchain operations
def blockchain_operation(blockchain: BlockchainType):
    """Decorator for blockchain operations"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            if not blockchain_manager:
                initialize_blockchain()
            
            if blockchain not in blockchain_manager.connectors:
                raise ValueError(f"Blockchain {blockchain.value} not available")
            
            return await func(*args, **kwargs)
        
        return async_wrapper
    return decorator

# Initialize blockchain on import
initialize_blockchain()






























