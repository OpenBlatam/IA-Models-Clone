"""
Chain Manager
=============

Blockchain chain management.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """Blockchain block."""
    index: int
    timestamp: datetime
    data: Any
    previous_hash: str
    hash: str
    nonce: int = 0
    
    def calculate_hash(self) -> str:
        """Calculate block hash."""
        content = f"{self.index}{self.timestamp.isoformat()}{self.data}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(content.encode()).hexdigest()


class ChainManager:
    """Blockchain chain manager."""
    
    def __init__(self):
        self._chain: List[Block] = []
        self._difficulty: int = 4
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize blockchain with genesis block."""
        genesis = Block(
            index=0,
            timestamp=datetime.now(),
            data="Genesis Block",
            previous_hash="0"
        )
        genesis.hash = genesis.calculate_hash()
        self._chain.append(genesis)
        logger.info("Initialized blockchain with genesis block")
    
    def add_block(self, data: Any) -> Block:
        """Add block to chain."""
        previous_block = self._chain[-1]
        new_block = Block(
            index=len(self._chain),
            timestamp=datetime.now(),
            data=data,
            previous_hash=previous_block.hash
        )
        
        # Mine block (proof of work)
        new_block = self._mine_block(new_block)
        
        self._chain.append(new_block)
        logger.info(f"Added block {new_block.index} to chain")
        return new_block
    
    def _mine_block(self, block: Block) -> Block:
        """Mine block (proof of work)."""
        target = "0" * self._difficulty
        
        while block.hash[:self._difficulty] != target:
            block.nonce += 1
            block.hash = block.calculate_hash()
        
        return block
    
    def is_chain_valid(self) -> bool:
        """Validate blockchain."""
        for i in range(1, len(self._chain)):
            current = self._chain[i]
            previous = self._chain[i - 1]
            
            # Check hash
            if current.hash != current.calculate_hash():
                return False
            
            # Check previous hash
            if current.previous_hash != previous.hash:
                return False
        
        return True
    
    def get_chain(self) -> List[Block]:
        """Get blockchain."""
        return self._chain.copy()
    
    def get_latest_block(self) -> Block:
        """Get latest block."""
        return self._chain[-1]
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        return {
            "chain_length": len(self._chain),
            "difficulty": self._difficulty,
            "is_valid": self.is_chain_valid(),
            "latest_block_index": self._chain[-1].index if self._chain else 0
        }















