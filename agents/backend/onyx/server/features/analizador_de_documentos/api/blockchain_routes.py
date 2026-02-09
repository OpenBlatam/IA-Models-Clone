"""
Rutas para Blockchain
=====================

Endpoints para blockchain.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.blockchain import get_blockchain, Blockchain

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/blockchain",
    tags=["Blockchain"]
)


class AddBlockRequest(BaseModel):
    """Request para agregar bloque"""
    data: Dict[str, Any] = Field(..., description="Datos del bloque")


@router.post("/blocks")
async def add_block(
    request: AddBlockRequest,
    blockchain: Blockchain = Depends(get_blockchain)
):
    """Agregar bloque a la blockchain"""
    try:
        block = blockchain.add_block(request.data)
        
        return {
            "status": "added",
            "block_index": block.index,
            "hash": block.hash,
            "timestamp": block.timestamp
        }
    except Exception as e:
        logger.error(f"Error agregando bloque: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_blockchain_info(
    blockchain: Blockchain = Depends(get_blockchain)
):
    """Obtener información de la blockchain"""
    info = blockchain.get_blockchain_info()
    return info


@router.get("/validate")
async def validate_chain(
    blockchain: Blockchain = Depends(get_blockchain)
):
    """Validar cadena de bloques"""
    is_valid = blockchain.is_chain_valid()
    
    return {
        "is_valid": is_valid,
        "message": "Cadena válida" if is_valid else "Cadena inválida"
    }


@router.get("/blocks/{index}")
async def get_block(
    index: int,
    blockchain: Blockchain = Depends(get_blockchain)
):
    """Obtener bloque específico"""
    if index < 0 or index >= len(blockchain.chain):
        raise HTTPException(status_code=404, detail="Bloque no encontrado")
    
    block = blockchain.chain[index]
    
    return {
        "index": block.index,
        "timestamp": block.timestamp,
        "data": block.data,
        "previous_hash": block.previous_hash,
        "hash": block.hash,
        "nonce": block.nonce
    }














