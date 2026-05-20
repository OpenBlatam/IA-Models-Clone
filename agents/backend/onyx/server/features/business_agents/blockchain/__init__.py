"""
Blockchain Integration Package
==============================

Blockchain integration for immutable audit trails and decentralized operations.
"""

from .manager import BlockchainManager, SmartContractManager
from .networks import EthereumNetwork, HyperledgerNetwork, CustomBlockchain
from .contracts import SmartContract, ContractTemplate, ContractExecution
from .types import (
    BlockchainType, TransactionType, Block, Transaction, 
    SmartContractType, ConsensusAlgorithm, NetworkNode
)

__all__ = [
    "BlockchainManager",
    "SmartContractManager",
    "EthereumNetwork",
    "HyperledgerNetwork",
    "CustomBlockchain",
    "SmartContract",
    "ContractTemplate",
    "ContractExecution",
    "BlockchainType",
    "TransactionType",
    "Block",
    "Transaction",
    "SmartContractType",
    "ConsensusAlgorithm",
    "NetworkNode"
]
