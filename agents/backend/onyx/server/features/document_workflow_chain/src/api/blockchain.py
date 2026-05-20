"""
Blockchain API - Advanced Implementation
======================================

Advanced blockchain API with smart contracts, NFT support, and decentralized workflows.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime

from ..services import blockchain_service, BlockchainType, TransactionType, TransactionStatus

# Create router
router = APIRouter()

logger = logging.getLogger(__name__)


# Request/Response models
class BlockchainCreateRequest(BaseModel):
    """Blockchain create request model"""
    name: str
    blockchain_type: str
    rpc_url: str
    chain_id: int
    network_id: int = 1
    gas_price: int = 20000000000
    gas_limit: int = 21000
    block_time: int = 15
    consensus: str = "proof_of_work"


class WalletCreateRequest(BaseModel):
    """Wallet create request model"""
    blockchain_id: str
    wallet_name: str
    private_key: Optional[str] = None


class SmartContractDeployRequest(BaseModel):
    """Smart contract deploy request model"""
    blockchain_id: str
    wallet_id: str
    contract_name: str
    contract_code: str
    constructor_args: Optional[List[Any]] = None


class NFTMintRequest(BaseModel):
    """NFT mint request model"""
    blockchain_id: str
    wallet_id: str
    contract_id: str
    token_uri: str
    metadata: Dict[str, Any]
    recipient_address: Optional[str] = None


class WorkflowTransactionRequest(BaseModel):
    """Workflow transaction request model"""
    blockchain_id: str
    wallet_id: str
    workflow_id: str
    workflow_data: Dict[str, Any]


class SmartContractFunctionRequest(BaseModel):
    """Smart contract function request model"""
    blockchain_id: str
    wallet_id: str
    contract_id: str
    function_name: str
    function_args: Optional[List[Any]] = None


class TokenTransferRequest(BaseModel):
    """Token transfer request model"""
    blockchain_id: str
    from_wallet_id: str
    to_address: str
    amount: float
    token_contract_id: Optional[str] = None


class BlockCreateRequest(BaseModel):
    """Block create request model"""
    blockchain_id: str
    transactions: List[str]


class BlockchainResponse(BaseModel):
    """Blockchain response model"""
    blockchain_id: str
    name: str
    type: str
    chain_id: int
    message: str


class WalletResponse(BaseModel):
    """Wallet response model"""
    wallet_id: str
    address: str
    public_key: str
    balance: float


class SmartContractResponse(BaseModel):
    """Smart contract response model"""
    contract_id: str
    contract_name: str
    contract_address: str
    deployment_tx_hash: str
    message: str


class NFTResponse(BaseModel):
    """NFT response model"""
    nft_id: str
    token_id: int
    token_uri: str
    owner_address: str
    mint_tx_hash: str
    message: str


class TransactionResponse(BaseModel):
    """Transaction response model"""
    transaction_id: str
    tx_hash: str
    status: str
    message: str


class SmartContractFunctionResponse(BaseModel):
    """Smart contract function response model"""
    result: str
    gas_used: int
    status: str


class TransactionStatusResponse(BaseModel):
    """Transaction status response model"""
    transaction_id: str
    status: str
    block_hash: Optional[str]
    block_number: Optional[int]
    gas_used: Optional[int]
    created_at: str
    confirmed_at: Optional[str]


class BlockchainStatsResponse(BaseModel):
    """Blockchain statistics response model"""
    total_transactions: int
    confirmed_transactions: int
    failed_transactions: int
    total_blocks: int
    total_contracts: int
    total_nfts: int
    transactions_by_type: Dict[str, int]
    blockchains_connected: int
    total_wallets: int


# Blockchain management endpoints
@router.post("/blockchains", response_model=BlockchainResponse)
async def create_blockchain(request: BlockchainCreateRequest):
    """Create a new blockchain connection"""
    try:
        # Validate blockchain type
        try:
            blockchain_type = BlockchainType(request.blockchain_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid blockchain type: {request.blockchain_type}"
            )
        
        blockchain_id = await blockchain_service.create_blockchain(
            name=request.name,
            blockchain_type=blockchain_type,
            rpc_url=request.rpc_url,
            chain_id=request.chain_id,
            network_id=request.network_id,
            gas_price=request.gas_price,
            gas_limit=request.gas_limit,
            block_time=request.block_time,
            consensus=request.consensus
        )
        
        return BlockchainResponse(
            blockchain_id=blockchain_id,
            name=request.name,
            type=request.blockchain_type,
            chain_id=request.chain_id,
            message="Blockchain created successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create blockchain: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create blockchain: {str(e)}"
        )


@router.post("/wallets", response_model=WalletResponse)
async def create_wallet(request: WalletCreateRequest):
    """Create a new wallet"""
    try:
        result = await blockchain_service.create_wallet(
            blockchain_id=request.blockchain_id,
            wallet_name=request.wallet_name,
            private_key=request.private_key
        )
        
        return WalletResponse(
            wallet_id=result["wallet_id"],
            address=result["address"],
            public_key=result["public_key"],
            balance=result["balance"]
        )
    
    except Exception as e:
        logger.error(f"Failed to create wallet: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create wallet: {str(e)}"
        )


@router.post("/smart-contracts", response_model=SmartContractResponse)
async def deploy_smart_contract(request: SmartContractDeployRequest):
    """Deploy a smart contract"""
    try:
        contract_id = await blockchain_service.deploy_smart_contract(
            blockchain_id=request.blockchain_id,
            wallet_id=request.wallet_id,
            contract_name=request.contract_name,
            contract_code=request.contract_code,
            constructor_args=request.constructor_args
        )
        
        # Get contract details
        contract = blockchain_service.smart_contracts.get(contract_id)
        
        return SmartContractResponse(
            contract_id=contract_id,
            contract_name=request.contract_name,
            contract_address=contract["contract_address"],
            deployment_tx_hash=contract["deployment_tx_hash"],
            message="Smart contract deployed successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to deploy smart contract: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy smart contract: {str(e)}"
        )


@router.post("/nfts", response_model=NFTResponse)
async def mint_nft(request: NFTMintRequest):
    """Mint an NFT"""
    try:
        nft_id = await blockchain_service.mint_nft(
            blockchain_id=request.blockchain_id,
            wallet_id=request.wallet_id,
            contract_id=request.contract_id,
            token_uri=request.token_uri,
            metadata=request.metadata,
            recipient_address=request.recipient_address
        )
        
        # Get NFT details
        nft = blockchain_service.nfts.get(nft_id)
        
        return NFTResponse(
            nft_id=nft_id,
            token_id=nft["token_id"],
            token_uri=request.token_uri,
            owner_address=nft["owner_address"],
            mint_tx_hash=nft["mint_tx_hash"],
            message="NFT minted successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to mint NFT: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mint NFT: {str(e)}"
        )


@router.post("/workflow-transactions", response_model=TransactionResponse)
async def create_workflow_transaction(request: WorkflowTransactionRequest):
    """Create a workflow transaction on blockchain"""
    try:
        transaction_id = await blockchain_service.create_workflow_transaction(
            blockchain_id=request.blockchain_id,
            wallet_id=request.wallet_id,
            workflow_id=request.workflow_id,
            workflow_data=request.workflow_data
        )
        
        # Get transaction details
        transaction = blockchain_service.transactions.get(transaction_id)
        
        return TransactionResponse(
            transaction_id=transaction_id,
            tx_hash=transaction["tx_hash"],
            status=transaction["status"],
            message="Workflow transaction created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create workflow transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow transaction: {str(e)}"
        )


@router.post("/smart-contracts/{contract_id}/execute", response_model=SmartContractFunctionResponse)
async def execute_smart_contract_function(
    contract_id: str,
    request: SmartContractFunctionRequest
):
    """Execute a smart contract function"""
    try:
        result = await blockchain_service.execute_smart_contract_function(
            blockchain_id=request.blockchain_id,
            wallet_id=request.wallet_id,
            contract_id=contract_id,
            function_name=request.function_name,
            function_args=request.function_args
        )
        
        return SmartContractFunctionResponse(
            result=result["result"],
            gas_used=result["gas_used"],
            status=result["status"]
        )
    
    except Exception as e:
        logger.error(f"Failed to execute smart contract function: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute smart contract function: {str(e)}"
        )


@router.post("/transfers", response_model=TransactionResponse)
async def transfer_tokens(request: TokenTransferRequest):
    """Transfer tokens between addresses"""
    try:
        transaction_id = await blockchain_service.transfer_tokens(
            blockchain_id=request.blockchain_id,
            from_wallet_id=request.from_wallet_id,
            to_address=request.to_address,
            amount=request.amount,
            token_contract_id=request.token_contract_id
        )
        
        # Get transaction details
        transaction = blockchain_service.transactions.get(transaction_id)
        
        return TransactionResponse(
            transaction_id=transaction_id,
            tx_hash=transaction["tx_hash"],
            status=transaction["status"],
            message="Token transfer created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to transfer tokens: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transfer tokens: {str(e)}"
        )


@router.post("/blocks", response_model=Dict[str, str])
async def create_block(request: BlockCreateRequest):
    """Create a new block"""
    try:
        block_id = await blockchain_service.create_block(
            blockchain_id=request.blockchain_id,
            transactions=request.transactions
        )
        
        return {
            "block_id": block_id,
            "message": "Block created successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to create block: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create block: {str(e)}"
        )


# Query endpoints
@router.get("/transactions/{transaction_id}/status", response_model=TransactionStatusResponse)
async def get_transaction_status(transaction_id: str):
    """Get transaction status"""
    try:
        result = await blockchain_service.get_transaction_status(transaction_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found"
            )
        
        return TransactionStatusResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transaction status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transaction status: {str(e)}"
        )


@router.get("/wallets/{wallet_id}/balance")
async def get_wallet_balance(wallet_id: str):
    """Get wallet balance"""
    try:
        balance = await blockchain_service.get_wallet_balance(wallet_id)
        if balance is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Wallet not found"
            )
        
        return {
            "wallet_id": wallet_id,
            "balance": balance,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get wallet balance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get wallet balance: {str(e)}"
        )


# Statistics endpoint
@router.get("/stats", response_model=BlockchainStatsResponse)
async def get_blockchain_stats():
    """Get blockchain service statistics"""
    try:
        stats = await blockchain_service.get_blockchain_stats()
        return BlockchainStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get blockchain stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get blockchain stats: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def blockchain_health():
    """Blockchain service health check"""
    try:
        stats = await blockchain_service.get_blockchain_stats()
        
        return {
            "service": "blockchain_service",
            "status": "healthy",
            "blockchains_connected": stats["blockchains_connected"],
            "total_transactions": stats["total_transactions"],
            "total_contracts": stats["total_contracts"],
            "total_nfts": stats["total_nfts"],
            "total_wallets": stats["total_wallets"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Blockchain service health check failed: {e}")
        return {
            "service": "blockchain_service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

