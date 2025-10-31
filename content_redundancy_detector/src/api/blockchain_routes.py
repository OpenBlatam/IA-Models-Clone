"""
Blockchain API Routes - Advanced blockchain and cryptocurrency endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.blockchain_engine import (
    get_blockchain_engine, BlockchainConfig, 
    BlockchainTransaction, SmartContract, Wallet, DeFiProtocol
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blockchain", tags=["Blockchain"])


# Request/Response Models
class WalletCreationRequest(BaseModel):
    """Wallet creation request model"""
    network: str = Field(default="ethereum", description="Blockchain network")
    wallet_type: str = Field(default="standard", description="Wallet type (standard, multisig, hardware)")


class TransactionRequest(BaseModel):
    """Transaction request model"""
    from_address: str = Field(..., description="Sender address", min_length=1)
    to_address: str = Field(..., description="Recipient address", min_length=1)
    amount: float = Field(..., description="Amount to send", gt=0)
    private_key: str = Field(..., description="Private key for signing", min_length=1)
    network: str = Field(default="ethereum", description="Blockchain network")
    gas_limit: Optional[int] = Field(default=None, description="Gas limit")
    gas_price_gwei: Optional[int] = Field(default=None, description="Gas price in Gwei")


class SmartContractDeploymentRequest(BaseModel):
    """Smart contract deployment request model"""
    bytecode: str = Field(..., description="Contract bytecode", min_length=1)
    abi: Dict[str, Any] = Field(..., description="Contract ABI")
    deployer_address: str = Field(..., description="Deployer address", min_length=1)
    deployer_private_key: str = Field(..., description="Deployer private key", min_length=1)
    network: str = Field(default="ethereum", description="Blockchain network")
    constructor_args: Optional[List[Any]] = Field(default=None, description="Constructor arguments")


class BalanceRequest(BaseModel):
    """Balance request model"""
    address: str = Field(..., description="Wallet address", min_length=1)
    network: str = Field(default="ethereum", description="Blockchain network")
    token_address: Optional[str] = Field(default=None, description="Token contract address for ERC-20 tokens")


class DeFiYieldRequest(BaseModel):
    """DeFi yield calculation request model"""
    amount: float = Field(..., description="Amount to invest", gt=0)
    protocol_id: str = Field(..., description="DeFi protocol ID", min_length=1)
    duration_days: int = Field(default=365, description="Investment duration in days", gt=0)


class BlockchainConfigRequest(BaseModel):
    """Blockchain configuration request model"""
    enable_ethereum: bool = Field(default=True, description="Enable Ethereum support")
    enable_bitcoin: bool = Field(default=True, description="Enable Bitcoin support")
    enable_polygon: bool = Field(default=True, description="Enable Polygon support")
    enable_bsc: bool = Field(default=True, description="Enable BSC support")
    enable_arbitrum: bool = Field(default=True, description="Enable Arbitrum support")
    enable_optimism: bool = Field(default=True, description="Enable Optimism support")
    enable_avalanche: bool = Field(default=True, description="Enable Avalanche support")
    enable_fantom: bool = Field(default=True, description="Enable Fantom support")
    enable_solana: bool = Field(default=True, description="Enable Solana support")
    enable_cardano: bool = Field(default=True, description="Enable Cardano support")
    enable_polkadot: bool = Field(default=True, description="Enable Polkadot support")
    enable_cosmos: bool = Field(default=True, description="Enable Cosmos support")
    enable_chainlink: bool = Field(default=True, description="Enable Chainlink support")
    enable_defi: bool = Field(default=True, description="Enable DeFi protocols")
    enable_nft: bool = Field(default=True, description="Enable NFT support")
    enable_dao: bool = Field(default=True, description="Enable DAO support")
    enable_smart_contracts: bool = Field(default=True, description="Enable smart contracts")
    enable_dex: bool = Field(default=True, description="Enable DEX support")
    enable_lending: bool = Field(default=True, description="Enable lending protocols")
    enable_staking: bool = Field(default=True, description="Enable staking")
    enable_yield_farming: bool = Field(default=True, description="Enable yield farming")
    enable_liquidity_mining: bool = Field(default=True, description="Enable liquidity mining")
    enable_governance: bool = Field(default=True, description="Enable governance")
    enable_cross_chain: bool = Field(default=True, description="Enable cross-chain")
    enable_layer2: bool = Field(default=True, description="Enable Layer 2")
    enable_sidechains: bool = Field(default=True, description="Enable sidechains")
    enable_testnets: bool = Field(default=True, description="Enable testnets")
    enable_mainnets: bool = Field(default=False, description="Enable mainnets")
    gas_limit: int = Field(default=21000, description="Default gas limit", gt=0)
    gas_price_gwei: int = Field(default=20, description="Default gas price in Gwei", gt=0)
    max_fee_per_gas_gwei: int = Field(default=50, description="Max fee per gas in Gwei", gt=0)
    max_priority_fee_per_gas_gwei: int = Field(default=2, description="Max priority fee per gas in Gwei", gt=0)
    confirmation_blocks: int = Field(default=12, description="Confirmation blocks", gt=0)
    timeout_seconds: int = Field(default=300, description="Transaction timeout in seconds", gt=0)


# Dependency to get blockchain engine
async def get_blockchain_engine_dep():
    """Get blockchain engine dependency"""
    engine = await get_blockchain_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Blockchain Engine not available")
    return engine


# Blockchain Routes
@router.post("/create-wallet", response_model=Dict[str, Any])
async def create_wallet(
    request: WalletCreationRequest,
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Create a new blockchain wallet"""
    try:
        start_time = time.time()
        
        # Create wallet
        wallet = await engine.create_wallet(network=request.network)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "wallet": {
                "wallet_id": wallet.wallet_id,
                "timestamp": wallet.timestamp.isoformat(),
                "address": wallet.address,
                "public_key": wallet.public_key,
                "network": wallet.network,
                "balance": wallet.balance,
                "currency": wallet.currency,
                "nonce": wallet.nonce,
                "transaction_count": len(wallet.transactions)
            },
            "processing_time_ms": processing_time,
            "message": f"Wallet created successfully on {request.network} network"
        }
        
    except Exception as e:
        logger.error(f"Error creating wallet: {e}")
        raise HTTPException(status_code=500, detail=f"Wallet creation failed: {str(e)}")


@router.post("/get-balance", response_model=Dict[str, Any])
async def get_balance(
    request: BalanceRequest,
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Get wallet balance"""
    try:
        start_time = time.time()
        
        # Get balance
        balance = await engine.get_balance(
            address=request.address,
            network=request.network
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "balance_info": {
                "address": request.address,
                "network": request.network,
                "balance": balance,
                "currency": "ETH" if request.network == "ethereum" else "BTC",
                "token_address": request.token_address,
                "timestamp": datetime.now().isoformat()
            },
            "processing_time_ms": processing_time,
            "message": f"Balance retrieved successfully for {request.address}"
        }
        
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        raise HTTPException(status_code=500, detail=f"Balance retrieval failed: {str(e)}")


@router.post("/send-transaction", response_model=Dict[str, Any])
async def send_transaction(
    request: TransactionRequest,
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Send a blockchain transaction"""
    try:
        start_time = time.time()
        
        # Send transaction
        transaction = await engine.send_transaction(
            from_address=request.from_address,
            to_address=request.to_address,
            amount=request.amount,
            private_key=request.private_key,
            network=request.network
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "transaction": {
                "tx_id": transaction.tx_id,
                "timestamp": transaction.timestamp.isoformat(),
                "from_address": transaction.from_address,
                "to_address": transaction.to_address,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "network": transaction.network,
                "gas_used": transaction.gas_used,
                "gas_price": transaction.gas_price,
                "block_number": transaction.block_number,
                "status": transaction.status,
                "hash": transaction.hash,
                "nonce": transaction.nonce
            },
            "processing_time_ms": processing_time,
            "message": f"Transaction sent successfully: {transaction.hash}"
        }
        
    except Exception as e:
        logger.error(f"Error sending transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Transaction failed: {str(e)}")


@router.post("/deploy-contract", response_model=Dict[str, Any])
async def deploy_smart_contract(
    request: SmartContractDeploymentRequest,
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Deploy a smart contract"""
    try:
        start_time = time.time()
        
        # Deploy smart contract
        contract = await engine.deploy_smart_contract(
            bytecode=request.bytecode,
            abi=request.abi,
            deployer_address=request.deployer_address,
            deployer_private_key=request.deployer_private_key,
            network=request.network
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "smart_contract": {
                "contract_id": contract.contract_id,
                "timestamp": contract.timestamp.isoformat(),
                "address": contract.address,
                "network": contract.network,
                "creator": contract.creator,
                "gas_used": contract.gas_used,
                "status": contract.status,
                "function_count": len(contract.functions),
                "event_count": len(contract.events)
            },
            "processing_time_ms": processing_time,
            "message": f"Smart contract deployed successfully at {contract.address}"
        }
        
    except Exception as e:
        logger.error(f"Error deploying smart contract: {e}")
        raise HTTPException(status_code=500, detail=f"Contract deployment failed: {str(e)}")


@router.get("/defi-protocols", response_model=Dict[str, Any])
async def get_defi_protocols(
    protocol_type: Optional[str] = None,
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Get DeFi protocols"""
    try:
        # Get DeFi protocols
        protocols = await engine.get_defi_protocols(protocol_type)
        
        # Format protocols
        formatted_protocols = []
        for protocol in protocols:
            formatted_protocols.append({
                "protocol_id": protocol.protocol_id,
                "timestamp": protocol.timestamp.isoformat(),
                "name": protocol.name,
                "network": protocol.network,
                "type": protocol.type,
                "tvl": protocol.tvl,
                "apy": protocol.apy,
                "tokens": protocol.tokens,
                "contracts": protocol.contracts,
                "risk_score": protocol.risk_score,
                "status": protocol.status
            })
        
        return {
            "success": True,
            "defi_protocols": formatted_protocols,
            "total_count": len(formatted_protocols),
            "filter": {"protocol_type": protocol_type},
            "message": "DeFi protocols retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting DeFi protocols: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get DeFi protocols: {str(e)}")


@router.post("/calculate-yield", response_model=Dict[str, Any])
async def calculate_defi_yield(
    request: DeFiYieldRequest,
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Calculate DeFi yield"""
    try:
        start_time = time.time()
        
        # Calculate yield
        yield_info = await engine.calculate_yield(
            amount=request.amount,
            protocol_id=request.protocol_id,
            duration_days=request.duration_days
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "yield_calculation": yield_info,
            "processing_time_ms": processing_time,
            "message": f"Yield calculated successfully for {yield_info['protocol_name']}"
        }
        
    except Exception as e:
        logger.error(f"Error calculating yield: {e}")
        raise HTTPException(status_code=500, detail=f"Yield calculation failed: {str(e)}")


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_blockchain_capabilities(
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Get blockchain capabilities"""
    try:
        # Get capabilities
        capabilities = await engine.get_blockchain_capabilities()
        
        return {
            "success": True,
            "blockchain_capabilities": capabilities,
            "message": "Blockchain capabilities retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting blockchain capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain capabilities: {str(e)}")


@router.get("/performance-metrics", response_model=Dict[str, Any])
async def get_blockchain_performance_metrics(
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Get blockchain performance metrics"""
    try:
        # Get performance metrics
        metrics = await engine.get_blockchain_performance_metrics()
        
        return {
            "success": True,
            "performance_metrics": metrics,
            "message": "Blockchain performance metrics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting blockchain performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain performance metrics: {str(e)}")


@router.post("/configure", response_model=Dict[str, Any])
async def configure_blockchain(
    request: BlockchainConfigRequest,
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Configure blockchain settings"""
    try:
        # Update configuration
        config = BlockchainConfig(
            enable_ethereum=request.enable_ethereum,
            enable_bitcoin=request.enable_bitcoin,
            enable_polygon=request.enable_polygon,
            enable_bsc=request.enable_bsc,
            enable_arbitrum=request.enable_arbitrum,
            enable_optimism=request.enable_optimism,
            enable_avalanche=request.enable_avalanche,
            enable_fantom=request.enable_fantom,
            enable_solana=request.enable_solana,
            enable_cardano=request.enable_cardano,
            enable_polkadot=request.enable_polkadot,
            enable_cosmos=request.enable_cosmos,
            enable_chainlink=request.enable_chainlink,
            enable_defi=request.enable_defi,
            enable_nft=request.enable_nft,
            enable_dao=request.enable_dao,
            enable_smart_contracts=request.enable_smart_contracts,
            enable_dex=request.enable_dex,
            enable_lending=request.enable_lending,
            enable_staking=request.enable_staking,
            enable_yield_farming=request.enable_yield_farming,
            enable_liquidity_mining=request.enable_liquidity_mining,
            enable_governance=request.enable_governance,
            enable_cross_chain=request.enable_cross_chain,
            enable_layer2=request.enable_layer2,
            enable_sidechains=request.enable_sidechains,
            enable_testnets=request.enable_testnets,
            enable_mainnets=request.enable_mainnets,
            gas_limit=request.gas_limit,
            gas_price_gwei=request.gas_price_gwei,
            max_fee_per_gas_gwei=request.max_fee_per_gas_gwei,
            max_priority_fee_per_gas_gwei=request.max_priority_fee_per_gas_gwei,
            confirmation_blocks=request.confirmation_blocks,
            timeout_seconds=request.timeout_seconds
        )
        
        # Update engine configuration
        engine.config = config
        
        return {
            "success": True,
            "configuration": {
                "enable_ethereum": config.enable_ethereum,
                "enable_bitcoin": config.enable_bitcoin,
                "enable_polygon": config.enable_polygon,
                "enable_bsc": config.enable_bsc,
                "enable_arbitrum": config.enable_arbitrum,
                "enable_optimism": config.enable_optimism,
                "enable_avalanche": config.enable_avalanche,
                "enable_fantom": config.enable_fantom,
                "enable_solana": config.enable_solana,
                "enable_cardano": config.enable_cardano,
                "enable_polkadot": config.enable_polkadot,
                "enable_cosmos": config.enable_cosmos,
                "enable_chainlink": config.enable_chainlink,
                "enable_defi": config.enable_defi,
                "enable_nft": config.enable_nft,
                "enable_dao": config.enable_dao,
                "enable_smart_contracts": config.enable_smart_contracts,
                "enable_dex": config.enable_dex,
                "enable_lending": config.enable_lending,
                "enable_staking": config.enable_staking,
                "enable_yield_farming": config.enable_yield_farming,
                "enable_liquidity_mining": config.enable_liquidity_mining,
                "enable_governance": config.enable_governance,
                "enable_cross_chain": config.enable_cross_chain,
                "enable_layer2": config.enable_layer2,
                "enable_sidechains": config.enable_sidechains,
                "enable_testnets": config.enable_testnets,
                "enable_mainnets": config.enable_mainnets,
                "gas_limit": config.gas_limit,
                "gas_price_gwei": config.gas_price_gwei,
                "max_fee_per_gas_gwei": config.max_fee_per_gas_gwei,
                "max_priority_fee_per_gas_gwei": config.max_priority_fee_per_gas_gwei,
                "confirmation_blocks": config.confirmation_blocks,
                "timeout_seconds": config.timeout_seconds
            },
            "message": "Blockchain configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error configuring blockchain: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/networks", response_model=Dict[str, Any])
async def get_supported_networks(
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Get supported blockchain networks"""
    try:
        networks = {
            "ethereum": {
                "name": "Ethereum",
                "chain_id": 1,
                "currency": "ETH",
                "features": ["smart_contracts", "defi", "nft", "dao", "dex", "lending", "staking"],
                "status": "active"
            },
            "bitcoin": {
                "name": "Bitcoin",
                "chain_id": 0,
                "currency": "BTC",
                "features": ["transactions", "multisig", "lightning_network"],
                "status": "active"
            },
            "polygon": {
                "name": "Polygon",
                "chain_id": 137,
                "currency": "MATIC",
                "features": ["smart_contracts", "defi", "nft", "low_fees", "fast_transactions"],
                "status": "active"
            },
            "bsc": {
                "name": "Binance Smart Chain",
                "chain_id": 56,
                "currency": "BNB",
                "features": ["smart_contracts", "defi", "nft", "low_fees", "fast_transactions"],
                "status": "active"
            },
            "arbitrum": {
                "name": "Arbitrum One",
                "chain_id": 42161,
                "currency": "ETH",
                "features": ["layer2", "smart_contracts", "defi", "low_fees", "fast_transactions"],
                "status": "active"
            },
            "optimism": {
                "name": "Optimism",
                "chain_id": 10,
                "currency": "ETH",
                "features": ["layer2", "smart_contracts", "defi", "low_fees", "fast_transactions"],
                "status": "active"
            },
            "avalanche": {
                "name": "Avalanche",
                "chain_id": 43114,
                "currency": "AVAX",
                "features": ["smart_contracts", "defi", "nft", "subnets", "fast_finality"],
                "status": "active"
            },
            "fantom": {
                "name": "Fantom",
                "chain_id": 250,
                "currency": "FTM",
                "features": ["smart_contracts", "defi", "nft", "low_fees", "fast_transactions"],
                "status": "active"
            }
        }
        
        return {
            "success": True,
            "supported_networks": networks,
            "total_networks": len(networks),
            "message": "Supported networks retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting supported networks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported networks: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    engine: BlockchainEngine = Depends(get_blockchain_engine_dep)
):
    """Blockchain Engine health check"""
    try:
        # Check engine components
        components_status = {
            "ethereum_engine": engine.ethereum_engine is not None,
            "bitcoin_engine": engine.bitcoin_engine is not None,
            "defi_engine": engine.defi_engine is not None
        }
        
        # Get capabilities
        capabilities = await engine.get_blockchain_capabilities()
        
        # Get performance metrics
        metrics = await engine.get_blockchain_performance_metrics()
        
        # Determine overall health
        all_healthy = any(components_status.values())
        
        overall_health = "healthy" if all_healthy else "degraded"
        
        return {
            "status": overall_health,
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "capabilities": capabilities,
            "performance_metrics": metrics,
            "message": "Blockchain Engine is operational" if overall_health == "healthy" else "Some blockchain networks may not be available"
        }
        
    except Exception as e:
        logger.error(f"Error in Blockchain health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Blockchain Engine health check failed"
        }

















