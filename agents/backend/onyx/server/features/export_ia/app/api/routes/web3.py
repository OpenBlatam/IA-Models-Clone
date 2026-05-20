"""
Web3 API Routes - Rutas API para sistema Web3 y DeFi
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal
import logging

from ..web3.web3_engine import Web3Engine, NetworkType, TransactionStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/web3", tags=["Web3"])

# Instancia global del motor de Web3
web3_engine = Web3Engine()


# Modelos Pydantic
class CreateWalletRequest(BaseModel):
    network: str
    password: Optional[str] = None


class SendTransactionRequest(BaseModel):
    from_wallet_id: str
    to_address: str
    amount: str  # Decimal como string
    gas_price: Optional[int] = None
    gas_limit: Optional[int] = None


class TokenInfoRequest(BaseModel):
    token_address: str
    network: str


# Rutas de Carteras Web3
@router.post("/wallets")
async def create_web3_wallet(request: CreateWalletRequest):
    """Crear nueva cartera Web3."""
    try:
        network = NetworkType(request.network)
        
        wallet_id = await web3_engine.create_wallet(
            network=network,
            password=request.password
        )
        
        wallet = web3_engine.wallets[wallet_id]
        
        return {
            "wallet_id": wallet_id,
            "address": wallet.address,
            "network": wallet.network.value,
            "balance": str(wallet.balance),
            "created_at": wallet.created_at.isoformat(),
            "success": True,
            "message": "Cartera Web3 creada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al crear cartera Web3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets")
async def get_web3_wallets():
    """Obtener todas las carteras Web3."""
    try:
        wallets = []
        for wallet_id, wallet in web3_engine.wallets.items():
            wallets.append({
                "wallet_id": wallet_id,
                "address": wallet.address,
                "network": wallet.network.value,
                "balance": str(wallet.balance),
                "nonce": wallet.nonce,
                "created_at": wallet.created_at.isoformat(),
                "last_activity": wallet.last_activity.isoformat()
            })
        
        return {
            "wallets": wallets,
            "count": len(wallets),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener carteras Web3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{wallet_id}")
async def get_web3_wallet(wallet_id: str):
    """Obtener cartera Web3 específica."""
    try:
        if wallet_id not in web3_engine.wallets:
            raise HTTPException(status_code=404, detail="Cartera no encontrada")
        
        wallet = web3_engine.wallets[wallet_id]
        
        return {
            "wallet": {
                "wallet_id": wallet_id,
                "address": wallet.address,
                "network": wallet.network.value,
                "balance": str(wallet.balance),
                "nonce": wallet.nonce,
                "created_at": wallet.created_at.isoformat(),
                "last_activity": wallet.last_activity.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener cartera Web3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{wallet_id}/balance")
async def get_wallet_balance(wallet_id: str):
    """Obtener balance de cartera Web3."""
    try:
        balance = await web3_engine.get_wallet_balance(wallet_id)
        
        return {
            "wallet_id": wallet_id,
            "balance": str(balance),
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallets/{wallet_id}/tokens")
async def get_wallet_tokens(wallet_id: str):
    """Obtener tokens de cartera Web3."""
    try:
        tokens = await web3_engine.get_wallet_tokens(wallet_id)
        
        return {
            "wallet_id": wallet_id,
            "tokens": tokens,
            "count": len(tokens),
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Transacciones Web3
@router.post("/transactions")
async def send_web3_transaction(request: SendTransactionRequest):
    """Enviar transacción Web3."""
    try:
        amount = Decimal(request.amount)
        
        transaction_id = await web3_engine.send_transaction(
            from_wallet_id=request.from_wallet_id,
            to_address=request.to_address,
            amount=amount,
            gas_price=request.gas_price,
            gas_limit=request.gas_limit
        )
        
        return {
            "transaction_id": transaction_id,
            "from_wallet": request.from_wallet_id,
            "to_address": request.to_address,
            "amount": request.amount,
            "success": True,
            "message": "Transacción Web3 enviada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al enviar transacción Web3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transactions/{transaction_id}")
async def get_transaction_status(transaction_id: str):
    """Obtener estado de transacción Web3."""
    try:
        status = await web3_engine.get_transaction_status(transaction_id)
        
        return {
            "transaction": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener estado de transacción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transactions")
async def get_web3_transactions(
    limit: int = Query(100, description="Límite de transacciones"),
    status: Optional[str] = Query(None, description="Estado de transacción")
):
    """Obtener transacciones Web3."""
    try:
        transactions = []
        
        for tx_id, tx in web3_engine.transactions.items():
            if status and tx.status.value != status:
                continue
            
            transactions.append({
                "transaction_id": tx_id,
                "hash": tx.hash,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value": str(tx.value),
                "gas_price": tx.gas_price,
                "gas_limit": tx.gas_limit,
                "nonce": tx.nonce,
                "status": tx.status.value,
                "network": tx.network.value,
                "created_at": tx.created_at.isoformat(),
                "confirmed_at": tx.confirmed_at.isoformat() if tx.confirmed_at else None,
                "block_number": tx.block_number
            })
        
        # Ordenar por fecha descendente
        transactions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "transactions": transactions[:limit],
            "count": len(transactions[:limit]),
            "total_transactions": len(transactions),
            "limit": limit,
            "filter_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener transacciones Web3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Tokens
@router.post("/tokens/info")
async def get_token_info(request: TokenInfoRequest):
    """Obtener información de token ERC20."""
    try:
        network = NetworkType(request.network)
        
        token_info = await web3_engine.get_token_info(
            token_address=request.token_address,
            network=network
        )
        
        return {
            "token": {
                "address": token_info.address,
                "symbol": token_info.symbol,
                "name": token_info.name,
                "decimals": token_info.decimals,
                "total_supply": str(token_info.total_supply),
                "token_type": token_info.token_type.value,
                "network": token_info.network.value
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener información de token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Redes
@router.get("/networks")
async def get_available_networks():
    """Obtener redes disponibles."""
    try:
        networks = []
        for network_type, config in web3_engine.networks.items():
            is_connected = network_type in web3_engine.web3_connections
            
            networks.append({
                "network": network_type.value,
                "chain_id": config["chain_id"],
                "explorer": config["explorer"],
                "connected": is_connected,
                "rpc_url": config["rpc_url"] if is_connected else None
            })
        
        return {
            "networks": networks,
            "count": len(networks),
            "connected_count": len(web3_engine.web3_connections),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener redes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/networks/{network}/status")
async def get_network_status(network: str):
    """Obtener estado de red específica."""
    try:
        network_type = NetworkType(network)
        
        if network_type not in web3_engine.web3_connections:
            return {
                "network": network,
                "connected": False,
                "error": "Red no disponible",
                "timestamp": datetime.now().isoformat()
            }
        
        w3 = web3_engine.web3_connections[network_type]
        
        try:
            latest_block = w3.eth.block_number
            gas_price = w3.eth.gas_price
            
            return {
                "network": network,
                "connected": True,
                "latest_block": latest_block,
                "gas_price": gas_price,
                "chain_id": web3_engine.networks[network_type]["chain_id"],
                "explorer": web3_engine.networks[network_type]["explorer"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "network": network,
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener estado de red: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Estadísticas
@router.get("/stats")
async def get_web3_stats():
    """Obtener estadísticas de Web3."""
    try:
        stats = await web3_engine.get_web3_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas Web3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def web3_health_check():
    """Verificar salud del sistema Web3."""
    try:
        health = await web3_engine.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de Web3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de utilidad
@router.get("/transaction-statuses")
async def get_transaction_statuses():
    """Obtener estados de transacciones disponibles."""
    return {
        "transaction_statuses": [
            {
                "value": status.value,
                "name": status.name,
                "description": f"Estado de transacción {status.value}"
            }
            for status in TransactionStatus
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/network-types")
async def get_network_types():
    """Obtener tipos de redes disponibles."""
    return {
        "network_types": [
            {
                "value": network_type.value,
                "name": network_type.name,
                "description": f"Red blockchain {network_type.value}"
            }
            for network_type in NetworkType
        ],
        "timestamp": datetime.now().isoformat()
    }


# Rutas de ejemplo
@router.post("/examples/create-wallet-and-transfer")
async def create_wallet_and_transfer_example():
    """Ejemplo: Crear cartera Web3 y realizar transferencia."""
    try:
        # Crear cartera en red local (para testing)
        wallet_id = await web3_engine.create_wallet(NetworkType.LOCAL)
        wallet = web3_engine.wallets[wallet_id]
        
        # Simular transferencia (en red real se necesitaría balance)
        return {
            "wallet": {
                "wallet_id": wallet_id,
                "address": wallet.address,
                "network": wallet.network.value,
                "balance": str(wallet.balance)
            },
            "success": True,
            "message": "Ejemplo de cartera Web3 completado",
            "note": "Para transferencias reales, use una red con balance",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en ejemplo de cartera Web3: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples/popular-tokens")
async def get_popular_tokens_example():
    """Ejemplo: Obtener tokens populares por red."""
    try:
        popular_tokens = {
            "ethereum": [
                {
                    "address": "0xA0b86a33E6441c8C06DDD5e8C4C0B3C4D2B8C2e5",
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "decimals": 6
                },
                {
                    "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                    "symbol": "USDT",
                    "name": "Tether USD",
                    "decimals": 6
                },
                {
                    "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                    "symbol": "DAI",
                    "name": "Dai Stablecoin",
                    "decimals": 18
                }
            ],
            "polygon": [
                {
                    "address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "decimals": 6
                },
                {
                    "address": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
                    "symbol": "USDT",
                    "name": "Tether USD",
                    "decimals": 6
                },
                {
                    "address": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
                    "symbol": "DAI",
                    "name": "Dai Stablecoin",
                    "decimals": 18
                }
            ]
        }
        
        return {
            "popular_tokens": popular_tokens,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en ejemplo de tokens populares: {e}")
        raise HTTPException(status_code=500, detail=str(e))




