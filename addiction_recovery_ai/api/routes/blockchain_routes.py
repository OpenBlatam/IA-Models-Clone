"""
Blockchain integration routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict

try:
    from services.blockchain_integration_service import BlockchainIntegrationService
except ImportError:
    from ...services.blockchain_integration_service import BlockchainIntegrationService

router = APIRouter()

blockchain = BlockchainIntegrationService()


@router.post("/blockchain/mint-nft")
async def mint_achievement_nft(
    user_id: str = Body(...),
    achievement_id: str = Body(...),
    achievement_data: Dict = Body(...)
):
    """Crea NFT de logro en blockchain"""
    try:
        nft = blockchain.mint_achievement_nft(user_id, achievement_id, achievement_data)
        return JSONResponse(content=nft)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando NFT: {str(e)}")


@router.post("/blockchain/create-certificate")
async def create_certificate_on_blockchain(
    user_id: str = Body(...),
    certificate_id: str = Body(...),
    certificate_data: Dict = Body(...)
):
    """Crea certificado en blockchain"""
    try:
        certificate = blockchain.create_certificate_on_blockchain(
            user_id, certificate_id, certificate_data
        )
        return JSONResponse(content=certificate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando certificado: {str(e)}")



