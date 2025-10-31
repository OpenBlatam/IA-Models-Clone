"""
Web3 Engine - Motor de Web3 y DeFi avanzado
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import aiohttp
import hashlib
from web3 import Web3
from eth_account import Account
import requests
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class NetworkType(Enum):
    """Tipos de redes blockchain."""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"
    LOCAL = "local"


class TokenType(Enum):
    """Tipos de tokens."""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    NATIVE = "native"


class TransactionStatus(Enum):
    """Estados de transacciones."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Web3Wallet:
    """Cartera Web3."""
    wallet_id: str
    address: str
    private_key: str
    network: NetworkType
    balance: Decimal = Decimal('0')
    nonce: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class TokenInfo:
    """Información de token."""
    address: str
    symbol: str
    name: str
    decimals: int
    total_supply: Decimal
    token_type: TokenType
    network: NetworkType


@dataclass
class Web3Transaction:
    """Transacción Web3."""
    transaction_id: str
    hash: str
    from_address: str
    to_address: str
    value: Decimal
    gas_price: int
    gas_limit: int
    nonce: int
    data: str
    status: TransactionStatus
    network: NetworkType
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    block_number: Optional[int] = None


class Web3Engine:
    """
    Motor de Web3 y DeFi avanzado.
    """
    
    def __init__(self, config_directory: str = "web3_config"):
        """Inicializar motor de Web3."""
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(exist_ok=True)
        
        # Configuración de redes
        self.networks = {
            NetworkType.ETHEREUM: {
                "rpc_url": "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
                "chain_id": 1,
                "explorer": "https://etherscan.io"
            },
            NetworkType.POLYGON: {
                "rpc_url": "https://polygon-rpc.com",
                "chain_id": 137,
                "explorer": "https://polygonscan.com"
            },
            NetworkType.BSC: {
                "rpc_url": "https://bsc-dataseed.binance.org",
                "chain_id": 56,
                "explorer": "https://bscscan.com"
            },
            NetworkType.LOCAL: {
                "rpc_url": "http://localhost:8545",
                "chain_id": 1337,
                "explorer": "http://localhost:3000"
            }
        }
        
        # Conexiones Web3
        self.web3_connections: Dict[NetworkType, Web3] = {}
        self.wallets: Dict[str, Web3Wallet] = {}
        self.transactions: Dict[str, Web3Transaction] = {}
        
        # Configuración
        self.default_gas_limit = 21000
        self.default_gas_price = 20000000000  # 20 gwei
        self.confirmation_blocks = 1
        
        # Estadísticas
        self.stats = {
            "total_wallets": 0,
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "total_volume": Decimal('0'),
            "start_time": datetime.now()
        }
        
        # Inicializar conexiones
        self._initialize_connections()
        
        logger.info("Web3Engine inicializado")
    
    async def initialize(self):
        """Inicializar el motor de Web3."""
        try:
            # Cargar carteras existentes
            self._load_wallets()
            
            # Verificar conexiones
            await self._verify_connections()
            
            logger.info("Web3Engine inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar Web3Engine: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el motor de Web3."""
        try:
            # Guardar carteras
            await self._save_wallets()
            
            logger.info("Web3Engine cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar Web3Engine: {e}")
    
    def _initialize_connections(self):
        """Inicializar conexiones Web3."""
        try:
            for network_type, config in self.networks.items():
                try:
                    w3 = Web3(Web3.HTTPProvider(config["rpc_url"]))
                    if w3.is_connected():
                        self.web3_connections[network_type] = w3
                        logger.info(f"Conexión Web3 establecida: {network_type.value}")
                    else:
                        logger.warning(f"No se pudo conectar a {network_type.value}")
                except Exception as e:
                    logger.warning(f"Error al conectar a {network_type.value}: {e}")
                    
        except Exception as e:
            logger.error(f"Error al inicializar conexiones Web3: {e}")
    
    async def _verify_connections(self):
        """Verificar conexiones Web3."""
        for network_type, w3 in self.web3_connections.items():
            try:
                latest_block = w3.eth.block_number
                logger.info(f"Red {network_type.value}: Bloque más reciente {latest_block}")
            except Exception as e:
                logger.warning(f"Error al verificar {network_type.value}: {e}")
    
    def _load_wallets(self):
        """Cargar carteras existentes."""
        try:
            wallets_file = self.config_directory / "wallets.json"
            if wallets_file.exists():
                with open(wallets_file, 'r') as f:
                    wallets_data = json.load(f)
                
                for wallet_id, data in wallets_data.items():
                    data['network'] = NetworkType(data['network'])
                    data['balance'] = Decimal(str(data['balance']))
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['last_activity'] = datetime.fromisoformat(data['last_activity'])
                    
                    self.wallets[wallet_id] = Web3Wallet(**data)
                
                logger.info(f"Cargadas {len(self.wallets)} carteras Web3")
                
        except Exception as e:
            logger.error(f"Error al cargar carteras: {e}")
    
    async def _save_wallets(self):
        """Guardar carteras."""
        try:
            wallets_file = self.config_directory / "wallets.json"
            
            wallets_data = {}
            for wallet_id, wallet in self.wallets.items():
                data = wallet.__dict__.copy()
                data['network'] = data['network'].value
                data['balance'] = str(data['balance'])
                data['created_at'] = data['created_at'].isoformat()
                data['last_activity'] = data['last_activity'].isoformat()
                wallets_data[wallet_id] = data
            
            with open(wallets_file, 'w') as f:
                json.dump(wallets_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error al guardar carteras: {e}")
    
    async def create_wallet(self, network: NetworkType, password: Optional[str] = None) -> str:
        """Crear nueva cartera Web3."""
        try:
            if network not in self.web3_connections:
                raise ValueError(f"Red {network.value} no está disponible")
            
            # Generar cuenta
            account = Account.create()
            
            wallet_id = str(uuid.uuid4())
            
            wallet = Web3Wallet(
                wallet_id=wallet_id,
                address=account.address,
                private_key=account.key.hex(),
                network=network
            )
            
            # Obtener balance inicial
            await self._update_wallet_balance(wallet)
            
            self.wallets[wallet_id] = wallet
            self.stats["total_wallets"] += 1
            
            logger.info(f"Cartera Web3 creada: {account.address} en {network.value}")
            return wallet_id
            
        except Exception as e:
            logger.error(f"Error al crear cartera Web3: {e}")
            raise
    
    async def _update_wallet_balance(self, wallet: Web3Wallet):
        """Actualizar balance de cartera."""
        try:
            if wallet.network not in self.web3_connections:
                return
            
            w3 = self.web3_connections[wallet.network]
            balance_wei = w3.eth.get_balance(wallet.address)
            balance_eth = w3.from_wei(balance_wei, 'ether')
            wallet.balance = Decimal(str(balance_eth))
            
        except Exception as e:
            logger.error(f"Error al actualizar balance: {e}")
    
    async def get_wallet_balance(self, wallet_id: str) -> Decimal:
        """Obtener balance de cartera."""
        try:
            if wallet_id not in self.wallets:
                raise ValueError(f"Cartera {wallet_id} no encontrada")
            
            wallet = self.wallets[wallet_id]
            await self._update_wallet_balance(wallet)
            
            return wallet.balance
            
        except Exception as e:
            logger.error(f"Error al obtener balance: {e}")
            raise
    
    async def send_transaction(
        self,
        from_wallet_id: str,
        to_address: str,
        amount: Decimal,
        gas_price: Optional[int] = None,
        gas_limit: Optional[int] = None
    ) -> str:
        """Enviar transacción."""
        try:
            if from_wallet_id not in self.wallets:
                raise ValueError(f"Cartera {from_wallet_id} no encontrada")
            
            wallet = self.wallets[from_wallet_id]
            
            if wallet.network not in self.web3_connections:
                raise ValueError(f"Red {wallet.network.value} no está disponible")
            
            w3 = self.web3_connections[wallet.network]
            
            # Verificar balance
            if wallet.balance < amount:
                raise ValueError("Balance insuficiente")
            
            # Configurar gas
            if gas_price is None:
                gas_price = w3.eth.gas_price
            if gas_limit is None:
                gas_limit = self.default_gas_limit
            
            # Crear transacción
            transaction = {
                'to': to_address,
                'value': w3.to_wei(amount, 'ether'),
                'gas': gas_limit,
                'gasPrice': gas_price,
                'nonce': wallet.nonce,
                'chainId': self.networks[wallet.network]['chain_id']
            }
            
            # Firmar transacción
            signed_txn = w3.eth.account.sign_transaction(transaction, wallet.private_key)
            
            # Enviar transacción
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            # Crear registro de transacción
            transaction_id = str(uuid.uuid4())
            web3_transaction = Web3Transaction(
                transaction_id=transaction_id,
                hash=tx_hash_hex,
                from_address=wallet.address,
                to_address=to_address,
                value=amount,
                gas_price=gas_price,
                gas_limit=gas_limit,
                nonce=wallet.nonce,
                data="",
                status=TransactionStatus.PENDING,
                network=wallet.network
            )
            
            self.transactions[transaction_id] = web3_transaction
            wallet.nonce += 1
            wallet.last_activity = datetime.now()
            
            self.stats["total_transactions"] += 1
            
            logger.info(f"Transacción enviada: {tx_hash_hex}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"Error al enviar transacción: {e}")
            raise
    
    async def get_transaction_status(self, transaction_id: str) -> Dict[str, Any]:
        """Obtener estado de transacción."""
        try:
            if transaction_id not in self.transactions:
                raise ValueError(f"Transacción {transaction_id} no encontrada")
            
            web3_transaction = self.transactions[transaction_id]
            
            if web3_transaction.network not in self.web3_connections:
                return {
                    "transaction_id": transaction_id,
                    "status": "network_unavailable",
                    "error": f"Red {web3_transaction.network.value} no disponible"
                }
            
            w3 = self.web3_connections[web3_transaction.network]
            
            try:
                # Obtener información de la transacción
                tx_receipt = w3.eth.get_transaction_receipt(web3_transaction.hash)
                
                if tx_receipt:
                    web3_transaction.status = TransactionStatus.CONFIRMED
                    web3_transaction.confirmed_at = datetime.now()
                    web3_transaction.block_number = tx_receipt.blockNumber
                    
                    if tx_receipt.status == 1:
                        self.stats["successful_transactions"] += 1
                    else:
                        web3_transaction.status = TransactionStatus.FAILED
                        self.stats["failed_transactions"] += 1
                
            except Exception:
                # Transacción aún pendiente
                pass
            
            return {
                "transaction_id": transaction_id,
                "hash": web3_transaction.hash,
                "status": web3_transaction.status.value,
                "from": web3_transaction.from_address,
                "to": web3_transaction.to_address,
                "value": str(web3_transaction.value),
                "gas_price": web3_transaction.gas_price,
                "gas_limit": web3_transaction.gas_limit,
                "nonce": web3_transaction.nonce,
                "network": web3_transaction.network.value,
                "created_at": web3_transaction.created_at.isoformat(),
                "confirmed_at": web3_transaction.confirmed_at.isoformat() if web3_transaction.confirmed_at else None,
                "block_number": web3_transaction.block_number
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estado de transacción: {e}")
            raise
    
    async def get_token_info(self, token_address: str, network: NetworkType) -> TokenInfo:
        """Obtener información de token ERC20."""
        try:
            if network not in self.web3_connections:
                raise ValueError(f"Red {network.value} no está disponible")
            
            w3 = self.web3_connections[network]
            
            # ABI básico para ERC20
            erc20_abi = [
                {
                    "constant": True,
                    "inputs": [],
                    "name": "name",
                    "outputs": [{"name": "", "type": "string"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "symbol",
                    "outputs": [{"name": "", "type": "string"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "totalSupply",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "type": "function"
                }
            ]
            
            contract = w3.eth.contract(address=token_address, abi=erc20_abi)
            
            # Obtener información del token
            name = contract.functions.name().call()
            symbol = contract.functions.symbol().call()
            decimals = contract.functions.decimals().call()
            total_supply = contract.functions.totalSupply().call()
            
            return TokenInfo(
                address=token_address,
                symbol=symbol,
                name=name,
                decimals=decimals,
                total_supply=Decimal(str(total_supply)) / Decimal(10 ** decimals),
                token_type=TokenType.ERC20,
                network=network
            )
            
        except Exception as e:
            logger.error(f"Error al obtener información de token: {e}")
            raise
    
    async def get_wallet_tokens(self, wallet_id: str) -> List[Dict[str, Any]]:
        """Obtener tokens de una cartera."""
        try:
            if wallet_id not in self.wallets:
                raise ValueError(f"Cartera {wallet_id} no encontrada")
            
            wallet = self.wallets[wallet_id]
            
            # Lista de tokens populares por red
            popular_tokens = {
                NetworkType.ETHEREUM: [
                    {"address": "0xA0b86a33E6441c8C06DDD5e8C4C0B3C4D2B8C2e5", "symbol": "USDC"},
                    {"address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", "symbol": "USDT"},
                    {"address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "symbol": "DAI"}
                ],
                NetworkType.POLYGON: [
                    {"address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "symbol": "USDC"},
                    {"address": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F", "symbol": "USDT"},
                    {"address": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063", "symbol": "DAI"}
                ]
            }
            
            tokens = []
            
            # Token nativo (ETH, MATIC, BNB)
            native_balance = await self.get_wallet_balance(wallet_id)
            tokens.append({
                "address": "native",
                "symbol": self._get_native_symbol(wallet.network),
                "name": self._get_native_name(wallet.network),
                "decimals": 18,
                "balance": str(native_balance),
                "type": "native"
            })
            
            # Tokens ERC20
            if wallet.network in popular_tokens:
                for token_info in popular_tokens[wallet.network]:
                    try:
                        token = await self.get_token_info(token_info["address"], wallet.network)
                        # Aquí se podría obtener el balance del token específico
                        tokens.append({
                            "address": token.address,
                            "symbol": token.symbol,
                            "name": token.name,
                            "decimals": token.decimals,
                            "balance": "0",  # Se implementaría la obtención del balance
                            "type": "erc20"
                        })
                    except Exception as e:
                        logger.warning(f"Error al obtener token {token_info['symbol']}: {e}")
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error al obtener tokens de cartera: {e}")
            raise
    
    def _get_native_symbol(self, network: NetworkType) -> str:
        """Obtener símbolo del token nativo."""
        symbols = {
            NetworkType.ETHEREUM: "ETH",
            NetworkType.POLYGON: "MATIC",
            NetworkType.BSC: "BNB",
            NetworkType.ARBITRUM: "ETH",
            NetworkType.OPTIMISM: "ETH",
            NetworkType.AVALANCHE: "AVAX",
            NetworkType.FANTOM: "FTM",
            NetworkType.LOCAL: "ETH"
        }
        return symbols.get(network, "ETH")
    
    def _get_native_name(self, network: NetworkType) -> str:
        """Obtener nombre del token nativo."""
        names = {
            NetworkType.ETHEREUM: "Ethereum",
            NetworkType.POLYGON: "Polygon",
            NetworkType.BSC: "Binance Smart Chain",
            NetworkType.ARBITRUM: "Arbitrum",
            NetworkType.OPTIMISM: "Optimism",
            NetworkType.AVALANCHE: "Avalanche",
            NetworkType.FANTOM: "Fantom",
            NetworkType.LOCAL: "Local Ethereum"
        }
        return names.get(network, "Ethereum")
    
    async def get_web3_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de Web3."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "wallets_count": len(self.wallets),
            "transactions_count": len(self.transactions),
            "connected_networks": len(self.web3_connections),
            "available_networks": [network.value for network in self.web3_connections.keys()],
            "total_volume": str(self.stats["total_volume"]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor de Web3."""
        try:
            network_status = {}
            for network_type, w3 in self.web3_connections.items():
                try:
                    latest_block = w3.eth.block_number
                    network_status[network_type.value] = {
                        "connected": True,
                        "latest_block": latest_block
                    }
                except Exception as e:
                    network_status[network_type.value] = {
                        "connected": False,
                        "error": str(e)
                    }
            
            return {
                "status": "healthy",
                "wallets_count": len(self.wallets),
                "transactions_count": len(self.transactions),
                "connected_networks": len(self.web3_connections),
                "network_status": network_status,
                "stats": self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de Web3: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }