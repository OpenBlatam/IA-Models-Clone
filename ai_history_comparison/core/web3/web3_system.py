"""
Web3 and DeFi Technology System - Advanced Decentralized Applications

This module provides comprehensive Web3 and DeFi capabilities following FastAPI best practices:
- Decentralized applications (dApps)
- Smart contracts and DeFi protocols
- Decentralized exchanges (DEX)
- Yield farming and liquidity mining
- NFT marketplaces and collections
- Decentralized identity (DID)
- Cross-chain bridges and interoperability
- Governance tokens and DAOs
- Decentralized storage (IPFS)
- Web3 wallets and authentication
"""

import asyncio
import json
import uuid
import time
import secrets
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import hashlib
import base64

logger = logging.getLogger(__name__)

class ProtocolType(Enum):
    """DeFi protocol types"""
    LENDING = "lending"
    BORROWING = "borrowing"
    STAKING = "staking"
    YIELD_FARMING = "yield_farming"
    LIQUIDITY_MINING = "liquidity_mining"
    DEX = "dex"
    AMM = "amm"
    OPTIONS = "options"
    DERIVATIVES = "derivatives"
    INSURANCE = "insurance"

class TokenType(Enum):
    """Token types"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    GOVERNANCE = "governance"
    UTILITY = "utility"
    STABLE_COIN = "stable_coin"
    WRAPPED = "wrapped"
    LIQUIDITY = "liquidity"

class NetworkType(Enum):
    """Blockchain network types"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    AVALANCHE = "avalanche"
    SOLANA = "solana"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    FANTOM = "fantom"

@dataclass
class SmartContract:
    """Smart contract data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    address: str = ""
    network: NetworkType = NetworkType.ETHEREUM
    abi: Dict[str, Any] = field(default_factory=dict)
    bytecode: str = ""
    deployed_at: Optional[datetime] = None
    gas_used: int = 0
    gas_price: int = 0
    creator: str = ""
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Token:
    """Token data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    symbol: str = ""
    token_type: TokenType = TokenType.ERC20
    contract_address: str = ""
    network: NetworkType = NetworkType.ETHEREUM
    total_supply: int = 0
    decimals: int = 18
    price_usd: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeFiProtocol:
    """DeFi protocol data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    protocol_type: ProtocolType = ProtocolType.LENDING
    description: str = ""
    website: str = ""
    total_value_locked: float = 0.0
    apy: float = 0.0
    risk_score: float = 0.0
    contracts: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LiquidityPool:
    """Liquidity pool data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    token_a: str = ""
    token_b: str = ""
    reserve_a: float = 0.0
    reserve_b: float = 0.0
    total_liquidity: float = 0.0
    fee_rate: float = 0.003  # 0.3%
    volume_24h: float = 0.0
    apy: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NFT:
    """NFT data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    image_url: str = ""
    token_id: str = ""
    contract_address: str = ""
    network: NetworkType = NetworkType.ETHEREUM
    owner: str = ""
    creator: str = ""
    price: float = 0.0
    last_sale_price: float = 0.0
    rarity_score: float = 0.0
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Base service classes
class BaseWeb3Service(ABC):
    """Base Web3 service class"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.is_initialized = False
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize service"""
        pass
    
    @abstractmethod
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process service request"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service"""
        pass

class SmartContractService(BaseWeb3Service):
    """Smart contract management service"""
    
    def __init__(self):
        super().__init__("SmartContract")
        self.contracts: Dict[str, SmartContract] = {}
        self.deployments: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize smart contract service"""
        try:
            await asyncio.sleep(0.1)
            self.is_initialized = True
            logger.info("Smart contract service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize smart contract service: {e}")
            return False
    
    async def deploy_contract(self, 
                            name: str,
                            bytecode: str,
                            abi: Dict[str, Any],
                            network: NetworkType,
                            creator: str) -> SmartContract:
        """Deploy smart contract"""
        
        contract = SmartContract(
            name=name,
            address=f"0x{secrets.token_hex(20)}",  # Simulated address
            network=network,
            abi=abi,
            bytecode=bytecode,
            deployed_at=datetime.utcnow(),
            gas_used=secrets.randbelow(1000000) + 100000,
            gas_price=secrets.randbelow(100) + 20,
            creator=creator
        )
        
        async with self._lock:
            self.contracts[contract.id] = contract
            self.deployments[contract.address] = {
                "contract_id": contract.id,
                "deployed_at": contract.deployed_at,
                "gas_used": contract.gas_used,
                "status": "deployed"
            }
        
        logger.info(f"Deployed smart contract: {name} at {contract.address}")
        return contract
    
    async def call_contract_function(self, 
                                   contract_address: str,
                                   function_name: str,
                                   parameters: List[Any],
                                   caller: str) -> Dict[str, Any]:
        """Call smart contract function"""
        async with self._lock:
            if contract_address not in self.deployments:
                return {"success": False, "error": "Contract not found"}
            
            # Simulate contract function call
            await asyncio.sleep(0.1)
            
            result = {
                "contract_address": contract_address,
                "function_name": function_name,
                "parameters": parameters,
                "caller": caller,
                "result": f"Function {function_name} executed successfully",
                "gas_used": secrets.randbelow(10000) + 1000,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.debug(f"Called function {function_name} on contract {contract_address}")
            return result
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process smart contract request"""
        if not self.is_initialized:
            return {"success": False, "error": "Service not initialized"}
        
        operation = request_data.get("operation", "deploy_contract")
        
        if operation == "deploy_contract":
            contract = await self.deploy_contract(
                name=request_data.get("name", "Contract"),
                bytecode=request_data.get("bytecode", "0x"),
                abi=request_data.get("abi", {}),
                network=NetworkType(request_data.get("network", "ethereum")),
                creator=request_data.get("creator", "")
            )
            return {"success": True, "result": contract.__dict__, "service": "smart_contract"}
        
        elif operation == "call_function":
            result = await self.call_contract_function(
                contract_address=request_data.get("contract_address", ""),
                function_name=request_data.get("function_name", ""),
                parameters=request_data.get("parameters", []),
                caller=request_data.get("caller", "")
            )
            return {"success": True, "result": result, "service": "smart_contract"}
        
        else:
            return {"success": False, "error": "Unknown operation"}
    
    async def cleanup(self) -> None:
        """Cleanup smart contract service"""
        self.contracts.clear()
        self.deployments.clear()
        self.is_initialized = False
        logger.info("Smart contract service cleaned up")

class DeFiProtocolService(BaseWeb3Service):
    """DeFi protocol management service"""
    
    def __init__(self):
        super().__init__("DeFiProtocol")
        self.protocols: Dict[str, DeFiProtocol] = {}
        self.user_positions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def initialize(self) -> bool:
        """Initialize DeFi protocol service"""
        try:
            await asyncio.sleep(0.1)
            self.is_initialized = True
            logger.info("DeFi protocol service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DeFi protocol service: {e}")
            return False
    
    async def create_protocol(self, 
                            name: str,
                            protocol_type: ProtocolType,
                            description: str = "",
                            website: str = "") -> DeFiProtocol:
        """Create DeFi protocol"""
        
        protocol = DeFiProtocol(
            name=name,
            protocol_type=protocol_type,
            description=description,
            website=website,
            total_value_locked=secrets.randbelow(1000000000) + 1000000,
            apy=secrets.randbelow(50) + 5.0,
            risk_score=secrets.randbelow(100) / 100.0
        )
        
        async with self._lock:
            self.protocols[protocol.id] = protocol
        
        logger.info(f"Created DeFi protocol: {name} ({protocol_type.value})")
        return protocol
    
    async def deposit_liquidity(self, 
                              protocol_id: str,
                              user_id: str,
                              token_address: str,
                              amount: float) -> Dict[str, Any]:
        """Deposit liquidity to protocol"""
        async with self._lock:
            if protocol_id not in self.protocols:
                return {"success": False, "error": "Protocol not found"}
            
            protocol = self.protocols[protocol_id]
            
            # Simulate liquidity deposit
            await asyncio.sleep(0.1)
            
            position = {
                "id": str(uuid.uuid4()),
                "protocol_id": protocol_id,
                "user_id": user_id,
                "token_address": token_address,
                "amount": amount,
                "deposited_at": datetime.utcnow(),
                "apy": protocol.apy
            }
            
            self.user_positions[user_id].append(position)
            
            result = {
                "position_id": position["id"],
                "protocol_name": protocol.name,
                "amount": amount,
                "apy": protocol.apy,
                "estimated_yearly_return": amount * (protocol.apy / 100),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"User {user_id} deposited {amount} to protocol {protocol.name}")
            return result
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process DeFi protocol request"""
        if not self.is_initialized:
            return {"success": False, "error": "Service not initialized"}
        
        operation = request_data.get("operation", "create_protocol")
        
        if operation == "create_protocol":
            protocol = await self.create_protocol(
                name=request_data.get("name", "Protocol"),
                protocol_type=ProtocolType(request_data.get("protocol_type", "lending")),
                description=request_data.get("description", ""),
                website=request_data.get("website", "")
            )
            return {"success": True, "result": protocol.__dict__, "service": "defi_protocol"}
        
        elif operation == "deposit_liquidity":
            result = await self.deposit_liquidity(
                protocol_id=request_data.get("protocol_id", ""),
                user_id=request_data.get("user_id", ""),
                token_address=request_data.get("token_address", ""),
                amount=request_data.get("amount", 0.0)
            )
            return {"success": True, "result": result, "service": "defi_protocol"}
        
        else:
            return {"success": False, "error": "Unknown operation"}
    
    async def cleanup(self) -> None:
        """Cleanup DeFi protocol service"""
        self.protocols.clear()
        self.user_positions.clear()
        self.is_initialized = False
        logger.info("DeFi protocol service cleaned up")

class DecentralizedExchangeService(BaseWeb3Service):
    """Decentralized exchange service"""
    
    def __init__(self):
        super().__init__("DecentralizedExchange")
        self.pools: Dict[str, LiquidityPool] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.trades: deque = deque(maxlen=10000)
    
    async def initialize(self) -> bool:
        """Initialize DEX service"""
        try:
            await asyncio.sleep(0.1)
            self.is_initialized = True
            logger.info("DEX service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DEX service: {e}")
            return False
    
    async def create_liquidity_pool(self, 
                                  name: str,
                                  token_a: str,
                                  token_b: str,
                                  initial_liquidity_a: float,
                                  initial_liquidity_b: float) -> LiquidityPool:
        """Create liquidity pool"""
        
        pool = LiquidityPool(
            name=name,
            token_a=token_a,
            token_b=token_b,
            reserve_a=initial_liquidity_a,
            reserve_b=initial_liquidity_b,
            total_liquidity=initial_liquidity_a + initial_liquidity_b,
            apy=secrets.randbelow(30) + 5.0
        )
        
        async with self._lock:
            self.pools[pool.id] = pool
        
        logger.info(f"Created liquidity pool: {name}")
        return pool
    
    async def swap_tokens(self, 
                        pool_id: str,
                        token_in: str,
                        amount_in: float,
                        user_id: str) -> Dict[str, Any]:
        """Execute token swap"""
        async with self._lock:
            if pool_id not in self.pools:
                return {"success": False, "error": "Pool not found"}
            
            pool = self.pools[pool_id]
            
            # Calculate swap using constant product formula (x * y = k)
            if token_in == pool.token_a:
                amount_out = (pool.reserve_b * amount_in) / (pool.reserve_a + amount_in)
                pool.reserve_a += amount_in
                pool.reserve_b -= amount_out
            else:
                amount_out = (pool.reserve_a * amount_in) / (pool.reserve_b + amount_in)
                pool.reserve_b += amount_in
                pool.reserve_a -= amount_out
            
            # Record trade
            trade = {
                "id": str(uuid.uuid4()),
                "pool_id": pool_id,
                "user_id": user_id,
                "token_in": token_in,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "timestamp": datetime.utcnow()
            }
            self.trades.append(trade)
            
            result = {
                "trade_id": trade["id"],
                "pool_name": pool.name,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "price_impact": abs(amount_out - amount_in) / amount_in * 100,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Executed swap in pool {pool.name}: {amount_in} -> {amount_out}")
            return result
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process DEX request"""
        if not self.is_initialized:
            return {"success": False, "error": "Service not initialized"}
        
        operation = request_data.get("operation", "create_pool")
        
        if operation == "create_pool":
            pool = await self.create_liquidity_pool(
                name=request_data.get("name", "Pool"),
                token_a=request_data.get("token_a", ""),
                token_b=request_data.get("token_b", ""),
                initial_liquidity_a=request_data.get("initial_liquidity_a", 0.0),
                initial_liquidity_b=request_data.get("initial_liquidity_b", 0.0)
            )
            return {"success": True, "result": pool.__dict__, "service": "dex"}
        
        elif operation == "swap_tokens":
            result = await self.swap_tokens(
                pool_id=request_data.get("pool_id", ""),
                token_in=request_data.get("token_in", ""),
                amount_in=request_data.get("amount_in", 0.0),
                user_id=request_data.get("user_id", "")
            )
            return {"success": True, "result": result, "service": "dex"}
        
        else:
            return {"success": False, "error": "Unknown operation"}
    
    async def cleanup(self) -> None:
        """Cleanup DEX service"""
        self.pools.clear()
        self.orders.clear()
        self.trades.clear()
        self.is_initialized = False
        logger.info("DEX service cleaned up")

class NFTMarketplaceService(BaseWeb3Service):
    """NFT marketplace service"""
    
    def __init__(self):
        super().__init__("NFTMarketplace")
        self.nfts: Dict[str, NFT] = {}
        self.collections: Dict[str, Dict[str, Any]] = {}
        self.listings: Dict[str, Dict[str, Any]] = {}
        self.sales: deque = deque(maxlen=10000)
    
    async def initialize(self) -> bool:
        """Initialize NFT marketplace service"""
        try:
            await asyncio.sleep(0.1)
            self.is_initialized = True
            logger.info("NFT marketplace service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NFT marketplace service: {e}")
            return False
    
    async def mint_nft(self, 
                      name: str,
                      description: str,
                      image_url: str,
                      creator: str,
                      contract_address: str,
                      network: NetworkType) -> NFT:
        """Mint NFT"""
        
        nft = NFT(
            name=name,
            description=description,
            image_url=image_url,
            token_id=str(secrets.randbelow(1000000)),
            contract_address=contract_address,
            network=network,
            owner=creator,
            creator=creator,
            rarity_score=secrets.randbelow(100) / 100.0,
            attributes=[
                {"trait_type": "Rarity", "value": "Common"},
                {"trait_type": "Color", "value": "Blue"},
                {"trait_type": "Power", "value": secrets.randbelow(100)}
            ]
        )
        
        async with self._lock:
            self.nfts[nft.id] = nft
        
        logger.info(f"Minted NFT: {name} by {creator}")
        return nft
    
    async def list_nft_for_sale(self, 
                              nft_id: str,
                              price: float,
                              seller: str) -> Dict[str, Any]:
        """List NFT for sale"""
        async with self._lock:
            if nft_id not in self.nfts:
                return {"success": False, "error": "NFT not found"}
            
            nft = self.nfts[nft_id]
            
            if nft.owner != seller:
                return {"success": False, "error": "Not the owner"}
            
            listing_id = str(uuid.uuid4())
            self.listings[listing_id] = {
                "nft_id": nft_id,
                "seller": seller,
                "price": price,
                "listed_at": datetime.utcnow(),
                "status": "active"
            }
            
            result = {
                "listing_id": listing_id,
                "nft_name": nft.name,
                "price": price,
                "seller": seller,
                "listed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Listed NFT {nft.name} for sale at {price}")
            return result
    
    async def buy_nft(self, 
                     listing_id: str,
                     buyer: str,
                     price: float) -> Dict[str, Any]:
        """Buy NFT"""
        async with self._lock:
            if listing_id not in self.listings:
                return {"success": False, "error": "Listing not found"}
            
            listing = self.listings[listing_id]
            nft_id = listing["nft_id"]
            
            if nft_id not in self.nfts:
                return {"success": False, "error": "NFT not found"}
            
            nft = self.nfts[nft_id]
            
            if listing["price"] != price:
                return {"success": False, "error": "Price mismatch"}
            
            # Transfer ownership
            nft.owner = buyer
            nft.last_sale_price = price
            
            # Record sale
            sale = {
                "id": str(uuid.uuid4()),
                "nft_id": nft_id,
                "seller": listing["seller"],
                "buyer": buyer,
                "price": price,
                "timestamp": datetime.utcnow()
            }
            self.sales.append(sale)
            
            # Remove listing
            del self.listings[listing_id]
            
            result = {
                "sale_id": sale["id"],
                "nft_name": nft.name,
                "seller": listing["seller"],
                "buyer": buyer,
                "price": price,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Sold NFT {nft.name} to {buyer} for {price}")
            return result
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process NFT marketplace request"""
        if not self.is_initialized:
            return {"success": False, "error": "Service not initialized"}
        
        operation = request_data.get("operation", "mint_nft")
        
        if operation == "mint_nft":
            nft = await self.mint_nft(
                name=request_data.get("name", "NFT"),
                description=request_data.get("description", ""),
                image_url=request_data.get("image_url", ""),
                creator=request_data.get("creator", ""),
                contract_address=request_data.get("contract_address", ""),
                network=NetworkType(request_data.get("network", "ethereum"))
            )
            return {"success": True, "result": nft.__dict__, "service": "nft_marketplace"}
        
        elif operation == "list_nft":
            result = await self.list_nft_for_sale(
                nft_id=request_data.get("nft_id", ""),
                price=request_data.get("price", 0.0),
                seller=request_data.get("seller", "")
            )
            return {"success": True, "result": result, "service": "nft_marketplace"}
        
        elif operation == "buy_nft":
            result = await self.buy_nft(
                listing_id=request_data.get("listing_id", ""),
                buyer=request_data.get("buyer", ""),
                price=request_data.get("price", 0.0)
            )
            return {"success": True, "result": result, "service": "nft_marketplace"}
        
        else:
            return {"success": False, "error": "Unknown operation"}
    
    async def cleanup(self) -> None:
        """Cleanup NFT marketplace service"""
        self.nfts.clear()
        self.collections.clear()
        self.listings.clear()
        self.sales.clear()
        self.is_initialized = False
        logger.info("NFT marketplace service cleaned up")

# Advanced Web3 Manager
class Web3Manager:
    """Main Web3 and DeFi management system"""
    
    def __init__(self):
        self.tokens: Dict[str, Token] = {}
        self.wallets: Dict[str, Dict[str, Any]] = {}
        
        # Services
        self.smart_contract_service = SmartContractService()
        self.defi_protocol_service = DeFiProtocolService()
        self.dex_service = DecentralizedExchangeService()
        self.nft_marketplace_service = NFTMarketplaceService()
        
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize Web3 system"""
        if self._initialized:
            return
        
        # Initialize services
        await self.smart_contract_service.initialize()
        await self.defi_protocol_service.initialize()
        await self.dex_service.initialize()
        await self.nft_marketplace_service.initialize()
        
        self._initialized = True
        logger.info("Web3 system initialized")
    
    async def shutdown(self) -> None:
        """Shutdown Web3 system"""
        # Cleanup services
        await self.smart_contract_service.cleanup()
        await self.defi_protocol_service.cleanup()
        await self.dex_service.cleanup()
        await self.nft_marketplace_service.cleanup()
        
        self.tokens.clear()
        self.wallets.clear()
        
        self._initialized = False
        logger.info("Web3 system shut down")
    
    async def create_wallet(self, user_id: str) -> Dict[str, Any]:
        """Create Web3 wallet"""
        wallet_address = f"0x{secrets.token_hex(20)}"
        private_key = f"0x{secrets.token_hex(32)}"
        
        wallet = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "address": wallet_address,
            "private_key": private_key,  # In production, encrypt this
            "created_at": datetime.utcnow(),
            "balance": 0.0,
            "tokens": {}
        }
        
        async with self._lock:
            self.wallets[wallet["id"]] = wallet
        
        logger.info(f"Created wallet for user {user_id}: {wallet_address}")
        return wallet
    
    async def process_web3_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Web3 request"""
        if not self._initialized:
            return {"success": False, "error": "Web3 system not initialized"}
        
        service_type = request_data.get("service_type", "smart_contract")
        
        if service_type == "smart_contract":
            return await self.smart_contract_service.process_request(request_data)
        elif service_type == "defi_protocol":
            return await self.defi_protocol_service.process_request(request_data)
        elif service_type == "dex":
            return await self.dex_service.process_request(request_data)
        elif service_type == "nft_marketplace":
            return await self.nft_marketplace_service.process_request(request_data)
        else:
            return {"success": False, "error": "Unknown service type"}
    
    def get_web3_summary(self) -> Dict[str, Any]:
        """Get Web3 system summary"""
        return {
            "initialized": self._initialized,
            "wallets": len(self.wallets),
            "tokens": len(self.tokens),
            "services": {
                "smart_contract": self.smart_contract_service.is_initialized,
                "defi_protocol": self.defi_protocol_service.is_initialized,
                "dex": self.dex_service.is_initialized,
                "nft_marketplace": self.nft_marketplace_service.is_initialized
            },
            "statistics": {
                "total_contracts": len(self.smart_contract_service.contracts),
                "total_protocols": len(self.defi_protocol_service.protocols),
                "total_pools": len(self.dex_service.pools),
                "total_nfts": len(self.nft_marketplace_service.nfts),
                "total_trades": len(self.dex_service.trades),
                "total_sales": len(self.nft_marketplace_service.sales)
            }
        }

# Global Web3 manager instance
_global_web3_manager: Optional[Web3Manager] = None

def get_web3_manager() -> Web3Manager:
    """Get global Web3 manager instance"""
    global _global_web3_manager
    if _global_web3_manager is None:
        _global_web3_manager = Web3Manager()
    return _global_web3_manager

async def initialize_web3() -> None:
    """Initialize global Web3 system"""
    manager = get_web3_manager()
    await manager.initialize()

async def shutdown_web3() -> None:
    """Shutdown global Web3 system"""
    manager = get_web3_manager()
    await manager.shutdown()

async def create_web3_wallet(user_id: str) -> Dict[str, Any]:
    """Create Web3 wallet using global manager"""
    manager = get_web3_manager()
    return await manager.create_wallet(user_id)





















