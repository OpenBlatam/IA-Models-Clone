"""
Blockchain API routes for Facebook Posts API
Blockchain integration, smart contracts, and decentralized content verification
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, Path, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..core.config import get_settings
from ..api.schemas import ErrorResponse
from ..api.dependencies import get_request_id
from ..services.blockchain_service import (
    get_blockchain_service, BlockchainType, TransactionStatus, SmartContractType,
    BlockchainTransaction, SmartContract, ContentHash
)
from ..services.security_service import get_security_service
from ..infrastructure.monitoring import get_monitor, timed

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/blockchain", tags=["Blockchain"])

# Security scheme
security = HTTPBearer()


# Content Verification Routes

@router.post(
    "/verify-content",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content verification completed successfully"},
        400: {"description": "Invalid content data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Content verification error"}
    },
    summary="Verify content authenticity",
    description="Verify content authenticity using blockchain"
)
@timed("blockchain_verify_content")
async def verify_content(
    content: str = Query(..., description="Content to verify"),
    content_id: str = Query(..., description="Content ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Verify content authenticity using blockchain"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not content or not content_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content and content ID are required"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Verify content
        verified = await blockchain_service.verify_content(content, content_id)
        
        logger.info(
            "Content verification completed",
            content_id=content_id,
            verified=verified,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Content verification completed",
            "content_id": content_id,
            "verified": verified,
            "verification_method": "blockchain",
            "request_id": request_id,
            "verified_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Content verification failed",
            content_id=content_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content verification failed: {str(e)}"
        )


@router.post(
    "/store-content-hash",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Content hash stored successfully"},
        400: {"description": "Invalid content data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Content hash storage error"}
    },
    summary="Store content hash on blockchain",
    description="Store content hash on blockchain for verification"
)
@timed("blockchain_store_content_hash")
async def store_content_hash(
    content: str = Query(..., description="Content to hash"),
    content_id: str = Query(..., description="Content ID"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Store content hash on blockchain"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not content or not content_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content and content ID are required"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Store content hash
        transaction_hash = await blockchain_service.store_content_hash(content, content_id)
        
        # Calculate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        logger.info(
            "Content hash stored on blockchain",
            content_id=content_id,
            content_hash=content_hash,
            transaction_hash=transaction_hash,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Content hash stored on blockchain",
            "content_id": content_id,
            "content_hash": content_hash,
            "transaction_hash": transaction_hash,
            "blockchain_type": "mock",
            "request_id": request_id,
            "stored_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Content hash storage failed",
            content_id=content_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content hash storage failed: {str(e)}"
        )


# Copyright Protection Routes

@router.post(
    "/register-copyright",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Copyright registered successfully"},
        400: {"description": "Invalid copyright data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Copyright registration error"}
    },
    summary="Register copyright for content",
    description="Register copyright for content on blockchain"
)
@timed("blockchain_register_copyright")
async def register_copyright(
    content: str = Query(..., description="Content to register copyright for"),
    owner_address: str = Query(..., description="Owner blockchain address"),
    metadata: Dict[str, Any] = Query(..., description="Copyright metadata"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Register copyright for content"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not content or not owner_address:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content and owner address are required"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Register copyright
        transaction_hash = await blockchain_service.register_copyright(content, owner_address, metadata)
        
        logger.info(
            "Copyright registered",
            owner_address=owner_address,
            transaction_hash=transaction_hash,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Copyright registered successfully",
            "owner_address": owner_address,
            "transaction_hash": transaction_hash,
            "metadata": metadata,
            "blockchain_type": "mock",
            "request_id": request_id,
            "registered_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Copyright registration failed",
            owner_address=owner_address,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Copyright registration failed: {str(e)}"
        )


@router.post(
    "/check-copyright",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Copyright check completed successfully"},
        400: {"description": "Invalid content data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Copyright check error"}
    },
    summary="Check copyright for content",
    description="Check copyright for content on blockchain"
)
@timed("blockchain_check_copyright")
async def check_copyright(
    content: str = Query(..., description="Content to check copyright for"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Check copyright for content"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content is required"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Check copyright
        copyright_info = await blockchain_service.check_copyright(content)
        
        logger.info(
            "Copyright check completed",
            has_copyright=copyright_info is not None,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Copyright check completed",
            "has_copyright": copyright_info is not None,
            "copyright_info": copyright_info,
            "blockchain_type": "mock",
            "request_id": request_id,
            "checked_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Copyright check failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Copyright check failed: {str(e)}"
        )


# Reward Distribution Routes

@router.post(
    "/distribute-rewards",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Rewards distributed successfully"},
        400: {"description": "Invalid reward data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Reward distribution error"}
    },
    summary="Distribute rewards to recipients",
    description="Distribute rewards to recipients using blockchain"
)
@timed("blockchain_distribute_rewards")
async def distribute_rewards(
    recipients: List[str] = Query(..., description="List of recipient addresses"),
    amounts: List[float] = Query(..., description="List of reward amounts"),
    reason: str = Query(..., description="Reason for reward distribution"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Distribute rewards to recipients"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not recipients or not amounts or not reason:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Recipients, amounts, and reason are required"
            )
        
        if len(recipients) != len(amounts):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of recipients must match number of amounts"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Distribute rewards
        transaction_hash = await blockchain_service.distribute_rewards(recipients, amounts, reason)
        
        total_amount = sum(amounts)
        
        logger.info(
            "Rewards distributed",
            recipients_count=len(recipients),
            total_amount=total_amount,
            reason=reason,
            transaction_hash=transaction_hash,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Rewards distributed successfully",
            "recipients": recipients,
            "amounts": amounts,
            "total_amount": total_amount,
            "reason": reason,
            "transaction_hash": transaction_hash,
            "blockchain_type": "mock",
            "request_id": request_id,
            "distributed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Reward distribution failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reward distribution failed: {str(e)}"
        )


# NFT Minting Routes

@router.post(
    "/mint-nft",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "NFT minted successfully"},
        400: {"description": "Invalid NFT data"},
        401: {"description": "Unauthorized"},
        500: {"description": "NFT minting error"}
    },
    summary="Mint NFT for content",
    description="Mint NFT for content on blockchain"
)
@timed("blockchain_mint_nft")
async def mint_nft(
    content_id: str = Query(..., description="Content ID to mint NFT for"),
    owner_address: str = Query(..., description="Owner blockchain address"),
    metadata: Dict[str, Any] = Query(..., description="NFT metadata"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Mint NFT for content"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not content_id or not owner_address:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content ID and owner address are required"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Mint NFT
        transaction_hash = await blockchain_service.mint_nft(content_id, owner_address, metadata)
        
        logger.info(
            "NFT minted",
            content_id=content_id,
            owner_address=owner_address,
            transaction_hash=transaction_hash,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "NFT minted successfully",
            "content_id": content_id,
            "owner_address": owner_address,
            "metadata": metadata,
            "transaction_hash": transaction_hash,
            "blockchain_type": "mock",
            "request_id": request_id,
            "minted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "NFT minting failed",
            content_id=content_id,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NFT minting failed: {str(e)}"
        )


# Transaction Management Routes

@router.get(
    "/transaction/{transaction_hash}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Transaction retrieved successfully"},
        404: {"description": "Transaction not found"},
        401: {"description": "Unauthorized"},
        500: {"description": "Transaction retrieval error"}
    },
    summary="Get transaction by hash",
    description="Get transaction details by hash"
)
@timed("blockchain_get_transaction")
async def get_transaction(
    transaction_hash: str = Path(..., description="Transaction hash"),
    blockchain_type: str = Query("mock", description="Blockchain type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get transaction by hash"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not transaction_hash:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transaction hash is required"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Get transaction
        transaction = await blockchain_service.get_transaction(
            transaction_hash, 
            BlockchainType(blockchain_type)
        )
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found"
            )
        
        logger.info(
            "Transaction retrieved",
            transaction_hash=transaction_hash,
            status=transaction.status.value,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Transaction retrieved successfully",
            "transaction": {
                "id": transaction.id,
                "transaction_hash": transaction.transaction_hash,
                "from_address": transaction.from_address,
                "to_address": transaction.to_address,
                "amount": transaction.amount,
                "gas_used": transaction.gas_used,
                "gas_price": transaction.gas_price,
                "status": transaction.status.value,
                "block_number": transaction.block_number,
                "created_at": transaction.created_at.isoformat(),
                "confirmed_at": transaction.confirmed_at.isoformat() if transaction.confirmed_at else None,
                "metadata": transaction.metadata
            },
            "blockchain_type": blockchain_type,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Transaction retrieval failed",
            transaction_hash=transaction_hash,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transaction retrieval failed: {str(e)}"
        )


@router.get(
    "/balance/{address}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Balance retrieved successfully"},
        400: {"description": "Invalid address"},
        401: {"description": "Unauthorized"},
        500: {"description": "Balance retrieval error"}
    },
    summary="Get balance for address",
    description="Get balance for blockchain address"
)
@timed("blockchain_get_balance")
async def get_balance(
    address: str = Path(..., description="Blockchain address"),
    blockchain_type: str = Query("mock", description="Blockchain type"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get balance for address"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Validate parameters
        if not address:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Address is required"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Get balance
        balance = await blockchain_service.get_balance(address, BlockchainType(blockchain_type))
        
        logger.info(
            "Balance retrieved",
            address=address,
            balance=balance,
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Balance retrieved successfully",
            "address": address,
            "balance": balance,
            "blockchain_type": blockchain_type,
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Balance retrieval failed",
            address=address,
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Balance retrieval failed: {str(e)}"
        )


# Smart Contract Management Routes

@router.get(
    "/smart-contracts",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Smart contracts retrieved successfully"},
        401: {"description": "Unauthorized"},
        500: {"description": "Smart contracts retrieval error"}
    },
    summary="Get deployed smart contracts",
    description="Get list of deployed smart contracts"
)
@timed("blockchain_get_smart_contracts")
async def get_smart_contracts(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get deployed smart contracts"""
    
    try:
        # Verify authentication
        security_service = get_security_service()
        payload = security_service.token_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get blockchain service
        blockchain_service = get_blockchain_service()
        
        # Get smart contracts
        contracts = await blockchain_service.get_smart_contracts()
        
        logger.info(
            "Smart contracts retrieved",
            contracts_count=len(contracts),
            user_id=payload.get("user_id"),
            request_id=request_id
        )
        
        return {
            "success": True,
            "message": "Smart contracts retrieved successfully",
            "contracts": contracts,
            "total_count": len(contracts),
            "request_id": request_id,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Smart contracts retrieval failed",
            error=str(e),
            request_id=request_id,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Smart contracts retrieval failed: {str(e)}"
        )


# Export router
__all__ = ["router"]





























