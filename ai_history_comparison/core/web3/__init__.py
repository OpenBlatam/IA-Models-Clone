"""
Web3 and DeFi Technology Module

This module provides comprehensive Web3 and DeFi capabilities including:
- Decentralized applications (dApps)
- Smart contracts and DeFi protocols
- Decentralized exchanges (DEX)
- Yield farming and liquidity mining
- NFT marketplaces and collections
- Decentralized identity (DID)
- Cross-chain bridges and interoperability
- Governance tokens and DAOs
- Decentralized storage (IPFS)
- Web3 wallets and authentication
"""

from .web3_system import (
    Web3Manager,
    DeFiProtocol,
    SmartContract,
    DecentralizedExchange,
    YieldFarming,
    NFTMarketplace,
    DecentralizedIdentity,
    CrossChainBridge,
    DAOGovernance,
    IPFSStorage,
    Web3Wallet,
    get_web3_manager,
    initialize_web3,
    shutdown_web3
)

__all__ = [
    "Web3Manager",
    "DeFiProtocol",
    "SmartContract",
    "DecentralizedExchange",
    "YieldFarming",
    "NFTMarketplace",
    "DecentralizedIdentity",
    "CrossChainBridge",
    "DAOGovernance",
    "IPFSStorage",
    "Web3Wallet",
    "get_web3_manager",
    "initialize_web3",
    "shutdown_web3"
]





















