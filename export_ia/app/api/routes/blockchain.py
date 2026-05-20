"""
Blockchain API Routes - Rutas API para sistema de Blockchain
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..blockchain.blockchain_engine import BlockchainEngine, TransactionType, BlockStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/blockchain", tags=["Blockchain"])

# Instancia global del motor de Blockchain
blockchain_engine = BlockchainEngine()


# Modelos Pydantic
class CreateWalletRequest(BaseModel):
    wallet_name: Optional[str] = None


class CreateTransactionRequest(BaseModel):
    sender_wallet_id: str
    recipient_address: str
    amount: float
    transaction_type: str = "transfer"
    data: Dict[str, Any] = Field(default_factory=dict)


class TransferRequest(BaseModel):
    from_wallet_id: str
    to_address: str
    amount: float
    description: Optional[str] = None


class DocumentHashRequest(BaseModel):
    wallet_id: str
    document_hash: str
    document_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Rutas de Carteras
@router.post("/wallets")
async def create_wallet(request: CreateWalletRequest):
    """Crear nueva cartera blockchain."""
    try:
        wallet_id = await blockchain_engine.create_wallet(
            wallet_name=request.wallet_name
        )
        
        wallet = blockchain_engine.wallets[wallet_id]
        
        return {
            "wallet_id": wallet_id,
            "address": wallet.address,
            "public_key": wallet.public_key,
            "balance": wallet.balance,
            "created_at": wallet.created_at.isoformat(),
            "success": True,
            "message": "Cartera creada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al crear cartera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets")
async def get_wallets():
    """Obtener todas las carteras."""
    try:
        wallets = []
        for wallet_id, wallet in blockchain_engine.wallets.items():
            wallets.append({
                "wallet_id": wallet_id,
                "address": wallet.address,
                "public_key": wallet.public_key,
                "balance": wallet.balance,
                "created_at": wallet.created_at.isoformat(),
                "last_activity": wallet.last_activity.isoformat()
            })
        
        return {
            "wallets": wallets,
            "count": len(wallets),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener carteras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{wallet_id}")
async def get_wallet(wallet_id: str):
    """Obtener cartera específica."""
    try:
        if wallet_id not in blockchain_engine.wallets:
            raise HTTPException(status_code=404, detail="Cartera no encontrada")
        
        wallet = blockchain_engine.wallets[wallet_id]
        
        return {
            "wallet": {
                "wallet_id": wallet_id,
                "address": wallet.address,
                "public_key": wallet.public_key,
                "balance": wallet.balance,
                "created_at": wallet.created_at.isoformat(),
                "last_activity": wallet.last_activity.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener cartera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{wallet_id}/balance")
async def get_wallet_balance(wallet_id: str):
    """Obtener balance de cartera."""
    try:
        balance = await blockchain_engine.get_wallet_balance(wallet_id)
        
        return {
            "wallet_id": wallet_id,
            "balance": balance,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{wallet_id}/transactions")
async def get_wallet_transactions(
    wallet_id: str,
    limit: int = Query(100, description="Límite de transacciones")
):
    """Obtener historial de transacciones de cartera."""
    try:
        transactions = await blockchain_engine.get_transaction_history(wallet_id, limit)
        
        return {
            "wallet_id": wallet_id,
            "transactions": transactions,
            "count": len(transactions),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener transacciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Transacciones
@router.post("/transactions")
async def create_transaction(request: CreateTransactionRequest):
    """Crear transacción blockchain."""
    try:
        transaction_type = TransactionType(request.transaction_type)
        
        transaction_id = await blockchain_engine.create_transaction(
            sender_wallet_id=request.sender_wallet_id,
            recipient_address=request.recipient_address,
            amount=request.amount,
            transaction_type=transaction_type,
            data=request.data
        )
        
        return {
            "transaction_id": transaction_id,
            "success": True,
            "message": "Transacción creada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al crear transacción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transfer")
async def transfer_tokens(request: TransferRequest):
    """Transferir tokens entre carteras."""
    try:
        data = {}
        if request.description:
            data["description"] = request.description
        
        transaction_id = await blockchain_engine.create_transaction(
            sender_wallet_id=request.from_wallet_id,
            recipient_address=request.to_address,
            amount=request.amount,
            transaction_type=TransactionType.TRANSFER,
            data=data
        )
        
        return {
            "transaction_id": transaction_id,
            "from_wallet": request.from_wallet_id,
            "to_address": request.to_address,
            "amount": request.amount,
            "success": True,
            "message": "Transferencia creada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al transferir tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document-hash")
async def store_document_hash(request: DocumentHashRequest):
    """Almacenar hash de documento en blockchain."""
    try:
        data = {
            "document_name": request.document_name,
            "document_hash": request.document_hash,
            "metadata": request.metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        transaction_id = await blockchain_engine.create_transaction(
            sender_wallet_id=request.wallet_id,
            recipient_address=request.wallet_id,  # Self-transaction
            amount=0.0,
            transaction_type=TransactionType.DOCUMENT_HASH,
            data=data
        )
        
        return {
            "transaction_id": transaction_id,
            "document_name": request.document_name,
            "document_hash": request.document_hash,
            "success": True,
            "message": "Hash de documento almacenado en blockchain",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al almacenar hash de documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Blockchain
@router.get("/info")
async def get_blockchain_info():
    """Obtener información de la blockchain."""
    try:
        info = await blockchain_engine.get_blockchain_info()
        
        return {
            "blockchain_info": info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener información de blockchain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocks")
async def get_blocks(
    limit: int = Query(10, description="Límite de bloques"),
    offset: int = Query(0, description="Offset de bloques")
):
    """Obtener bloques de la blockchain."""
    try:
        blocks = []
        start_idx = max(0, len(blockchain_engine.chain) - offset - limit)
        end_idx = len(blockchain_engine.chain) - offset
        
        for block in blockchain_engine.chain[start_idx:end_idx]:
            blocks.append({
                "block_id": block.block_id,
                "previous_hash": block.previous_hash,
                "hash": block.hash,
                "merkle_root": block.merkle_root,
                "timestamp": block.timestamp.isoformat(),
                "nonce": block.nonce,
                "difficulty": block.difficulty,
                "status": block.status.value,
                "miner": block.miner,
                "transactions_count": len(block.transactions)
            })
        
        return {
            "blocks": blocks,
            "count": len(blocks),
            "total_blocks": len(blockchain_engine.chain),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener bloques: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocks/{block_id}")
async def get_block(block_id: str):
    """Obtener bloque específico."""
    try:
        block = None
        for b in blockchain_engine.chain:
            if b.block_id == block_id:
                block = b
                break
        
        if not block:
            raise HTTPException(status_code=404, detail="Bloque no encontrado")
        
        transactions = []
        for tx in block.transactions:
            transactions.append({
                "transaction_id": tx.transaction_id,
                "type": tx.transaction_type.value,
                "sender": tx.sender,
                "recipient": tx.recipient,
                "amount": tx.amount,
                "fee": tx.fee,
                "timestamp": tx.timestamp.isoformat(),
                "data": tx.data
            })
        
        return {
            "block": {
                "block_id": block.block_id,
                "previous_hash": block.previous_hash,
                "hash": block.hash,
                "merkle_root": block.merkle_root,
                "timestamp": block.timestamp.isoformat(),
                "nonce": block.nonce,
                "difficulty": block.difficulty,
                "status": block.status.value,
                "miner": block.miner,
                "transactions": transactions
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener bloque: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transactions")
async def get_transactions(
    limit: int = Query(100, description="Límite de transacciones"),
    transaction_type: Optional[str] = Query(None, description="Tipo de transacción")
):
    """Obtener transacciones de la blockchain."""
    try:
        all_transactions = []
        
        for block in blockchain_engine.chain:
            for tx in block.transactions:
                if transaction_type and tx.transaction_type.value != transaction_type:
                    continue
                
                all_transactions.append({
                    "transaction_id": tx.transaction_id,
                    "type": tx.transaction_type.value,
                    "sender": tx.sender,
                    "recipient": tx.recipient,
                    "amount": tx.amount,
                    "fee": tx.fee,
                    "timestamp": tx.timestamp.isoformat(),
                    "block_id": block.block_id,
                    "block_hash": block.hash,
                    "data": tx.data
                })
        
        # Ordenar por timestamp descendente
        all_transactions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "transactions": all_transactions[:limit],
            "count": len(all_transactions[:limit]),
            "total_transactions": len(all_transactions),
            "limit": limit,
            "filter_type": transaction_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener transacciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transactions/{transaction_id}")
async def get_transaction(transaction_id: str):
    """Obtener transacción específica."""
    try:
        transaction = None
        block = None
        
        for b in blockchain_engine.chain:
            for tx in b.transactions:
                if tx.transaction_id == transaction_id:
                    transaction = tx
                    block = b
                    break
            if transaction:
                break
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transacción no encontrada")
        
        return {
            "transaction": {
                "transaction_id": transaction.transaction_id,
                "type": transaction.transaction_type.value,
                "sender": transaction.sender,
                "recipient": transaction.recipient,
                "amount": transaction.amount,
                "fee": transaction.fee,
                "timestamp": transaction.timestamp.isoformat(),
                "signature": transaction.signature,
                "data": transaction.data,
                "block_id": block.block_id,
                "block_hash": block.hash
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener transacción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending-transactions")
async def get_pending_transactions():
    """Obtener transacciones pendientes."""
    try:
        pending_transactions = []
        
        for tx in blockchain_engine.pending_transactions:
            pending_transactions.append({
                "transaction_id": tx.transaction_id,
                "type": tx.transaction_type.value,
                "sender": tx.sender,
                "recipient": tx.recipient,
                "amount": tx.amount,
                "fee": tx.fee,
                "timestamp": tx.timestamp.isoformat(),
                "data": tx.data
            })
        
        return {
            "pending_transactions": pending_transactions,
            "count": len(pending_transactions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener transacciones pendientes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Verificación
@router.post("/verify")
async def verify_blockchain():
    """Verificar integridad de la blockchain."""
    try:
        verification = await blockchain_engine.verify_blockchain()
        
        return {
            "verification": verification,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al verificar blockchain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Estadísticas
@router.get("/stats")
async def get_blockchain_stats():
    """Obtener estadísticas de la blockchain."""
    try:
        stats = await blockchain_engine.get_blockchain_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def blockchain_health_check():
    """Verificar salud del sistema de blockchain."""
    try:
        health = await blockchain_engine.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de blockchain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de utilidad
@router.get("/transaction-types")
async def get_transaction_types():
    """Obtener tipos de transacciones disponibles."""
    return {
        "transaction_types": [
            {
                "value": transaction_type.value,
                "name": transaction_type.name,
                "description": f"Transacción de tipo {transaction_type.value}"
            }
            for transaction_type in TransactionType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/block-statuses")
async def get_block_statuses():
    """Obtener estados de bloques disponibles."""
    return {
        "block_statuses": [
            {
                "value": status.value,
                "name": status.name,
                "description": f"Estado de bloque {status.value}"
            }
            for status in BlockStatus
        ],
        "timestamp": datetime.now().isoformat()
    }


# Rutas de ejemplo
@router.post("/examples/create-wallet-and-transfer")
async def create_wallet_and_transfer_example():
    """Ejemplo: Crear cartera y realizar transferencia."""
    try:
        # Crear cartera 1
        wallet1_id = await blockchain_engine.create_wallet("Wallet 1")
        wallet1 = blockchain_engine.wallets[wallet1_id]
        
        # Crear cartera 2
        wallet2_id = await blockchain_engine.create_wallet("Wallet 2")
        wallet2 = blockchain_engine.wallets[wallet2_id]
        
        # Agregar balance inicial a wallet1 (simulado)
        wallet1.balance = 1000.0
        
        # Realizar transferencia
        transaction_id = await blockchain_engine.create_transaction(
            sender_wallet_id=wallet1_id,
            recipient_address=wallet2.address,
            amount=100.0,
            transaction_type=TransactionType.TRANSFER,
            data={"description": "Transferencia de ejemplo"}
        )
        
        return {
            "wallet1": {
                "wallet_id": wallet1_id,
                "address": wallet1.address,
                "balance": wallet1.balance
            },
            "wallet2": {
                "wallet_id": wallet2_id,
                "address": wallet2.address,
                "balance": wallet2.balance
            },
            "transaction_id": transaction_id,
            "success": True,
            "message": "Ejemplo de cartera y transferencia completado",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en ejemplo de cartera y transferencia: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/examples/store-document-hash")
async def store_document_hash_example(
    document_name: str = "Documento de ejemplo",
    document_content: str = "Contenido del documento de ejemplo"
):
    """Ejemplo: Almacenar hash de documento en blockchain."""
    try:
        import hashlib
        
        # Crear cartera
        wallet_id = await blockchain_engine.create_wallet("Document Wallet")
        wallet = blockchain_engine.wallets[wallet_id]
        
        # Calcular hash del documento
        document_hash = hashlib.sha256(document_content.encode()).hexdigest()
        
        # Almacenar hash en blockchain
        transaction_id = await blockchain_engine.create_transaction(
            sender_wallet_id=wallet_id,
            recipient_address=wallet.address,
            amount=0.0,
            transaction_type=TransactionType.DOCUMENT_HASH,
            data={
                "document_name": document_name,
                "document_hash": document_hash,
                "document_size": len(document_content),
                "content_preview": document_content[:100] + "..." if len(document_content) > 100 else document_content
            }
        )
        
        return {
            "wallet_id": wallet_id,
            "wallet_address": wallet.address,
            "document_name": document_name,
            "document_hash": document_hash,
            "transaction_id": transaction_id,
            "success": True,
            "message": "Hash de documento almacenado en blockchain",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en ejemplo de almacenamiento de hash: {e}")
        raise HTTPException(status_code=500, detail=str(e))




