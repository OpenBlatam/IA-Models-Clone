"""
Blockchain integration API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ....services.blockchain_service import BlockchainService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ExternalServiceError

router = APIRouter()


class ContentHashRequest(BaseModel):
    """Request model for content hashing."""
    post_id: int = Field(..., description="Post ID to hash")
    content: str = Field(..., min_length=1, description="Content to hash")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IPFSStorageRequest(BaseModel):
    """Request model for IPFS storage."""
    post_id: int = Field(..., description="Post ID to store")
    content: str = Field(..., min_length=1, description="Content to store")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Content metadata")


class NFTCreationRequest(BaseModel):
    """Request model for NFT creation."""
    post_id: int = Field(..., description="Post ID to create NFT for")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="NFT metadata")


class NFTTransferRequest(BaseModel):
    """Request model for NFT transfer."""
    post_id: int = Field(..., description="Post ID of NFT to transfer")
    from_address: str = Field(..., description="Current owner address")
    to_address: str = Field(..., description="New owner address")


class ContentVerificationRequest(BaseModel):
    """Request model for content verification."""
    post_id: int = Field(..., description="Post ID to verify")
    content: str = Field(..., min_length=1, description="Content to verify")


async def get_blockchain_service(session: DatabaseSessionDep) -> BlockchainService:
    """Get blockchain service instance."""
    return BlockchainService(session)


@router.post("/content-hash", response_model=Dict[str, Any])
async def create_content_hash(
    request: ContentHashRequest = Depends(),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Create a hash of content for blockchain verification."""
    try:
        result = await blockchain_service.create_content_hash(
            post_id=request.post_id,
            content=request.content,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content hash created successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create content hash"
        )


@router.post("/ipfs/store", response_model=Dict[str, Any])
async def store_content_on_ipfs(
    request: IPFSStorageRequest = Depends(),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Store content on IPFS for decentralized storage."""
    try:
        result = await blockchain_service.store_content_on_ipfs(
            post_id=request.post_id,
            content=request.content,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content stored on IPFS successfully"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store content on IPFS"
        )


@router.post("/nft/create", response_model=Dict[str, Any])
async def create_nft_for_post(
    request: NFTCreationRequest = Depends(),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Create an NFT for a blog post."""
    try:
        result = await blockchain_service.create_nft_for_post(
            post_id=request.post_id,
            author_id=str(current_user.id),
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "NFT created successfully"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create NFT"
        )


@router.post("/content/verify", response_model=Dict[str, Any])
async def verify_content_integrity(
    request: ContentVerificationRequest = Depends(),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Verify content integrity using blockchain hash."""
    try:
        result = await blockchain_service.verify_content_integrity(
            post_id=request.post_id,
            content=request.content
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content verification completed"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify content integrity"
        )


@router.get("/nft/collection", response_model=Dict[str, Any])
async def get_nft_collection(
    author_id: Optional[str] = Query(None, description="Filter by author ID"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of NFTs to return"),
    offset: int = Query(default=0, ge=0, description="Number of NFTs to skip"),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Get NFT collection for posts."""
    try:
        nfts, total = await blockchain_service.get_nft_collection(
            author_id=author_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "data": {
                "nfts": nfts,
                "total": total,
                "limit": limit,
                "offset": offset
            },
            "message": "NFT collection retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get NFT collection"
        )


@router.post("/nft/transfer", response_model=Dict[str, Any])
async def transfer_nft_ownership(
    request: NFTTransferRequest = Depends(),
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Transfer NFT ownership."""
    try:
        result = await blockchain_service.transfer_nft_ownership(
            post_id=request.post_id,
            from_address=request.from_address,
            to_address=request.to_address
        )
        
        return {
            "success": True,
            "data": result,
            "message": "NFT transfer initiated successfully"
        }
        
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External service error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to transfer NFT"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_blockchain_stats(
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Get blockchain integration statistics."""
    try:
        stats = await blockchain_service.get_blockchain_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Blockchain statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get blockchain statistics"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_blockchain_health(
    blockchain_service: BlockchainService = Depends(get_blockchain_service)
):
    """Get blockchain service health status."""
    try:
        health = await blockchain_service.get_blockchain_health()
        
        return {
            "success": True,
            "data": health,
            "message": "Blockchain health status retrieved"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get blockchain health status"
        }


@router.get("/nft/{post_id}", response_model=Dict[str, Any])
async def get_nft_details(
    post_id: int,
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Get NFT details for a specific post."""
    try:
        # Get NFT collection for this specific post
        nfts, total = await blockchain_service.get_nft_collection(
            limit=1,
            offset=0
        )
        
        # Find the NFT for this post
        nft = None
        for n in nfts:
            if n["nft_id"] == post_id:
                nft = n
                break
        
        if not nft:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="NFT not found for this post"
            )
        
        return {
            "success": True,
            "data": nft,
            "message": "NFT details retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get NFT details"
        )


@router.get("/ipfs/{ipfs_hash}", response_model=Dict[str, Any])
async def get_ipfs_content(
    ipfs_hash: str,
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Get content from IPFS by hash."""
    try:
        # This would typically fetch content from IPFS
        # For now, we'll return the IPFS URL
        ipfs_url = f"https://ipfs.io/ipfs/{ipfs_hash}"
        
        return {
            "success": True,
            "data": {
                "ipfs_hash": ipfs_hash,
                "ipfs_url": ipfs_url,
                "content_available": True
            },
            "message": "IPFS content information retrieved"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get IPFS content"
        )


@router.get("/transactions/{tx_hash}", response_model=Dict[str, Any])
async def get_transaction_details(
    tx_hash: str,
    blockchain_service: BlockchainService = Depends(get_blockchain_service),
    current_user: CurrentUserDep = Depends()
):
    """Get blockchain transaction details."""
    try:
        # This would typically fetch transaction details from the blockchain
        # For now, we'll return mock data
        transaction_details = {
            "tx_hash": tx_hash,
            "status": "confirmed",
            "block_number": 12345678,
            "gas_used": 21000,
            "gas_price": "20000000000",
            "timestamp": "2024-01-15T10:00:00Z"
        }
        
        return {
            "success": True,
            "data": transaction_details,
            "message": "Transaction details retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get transaction details"
        )






























