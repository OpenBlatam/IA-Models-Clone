"""
Blockchain Verification - Sistema de blockchain/verificación
============================================================
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """Bloque en la cadena"""
    index: int
    timestamp: datetime
    data: Dict[str, Any]
    previous_hash: str
    hash: str
    nonce: int = 0


class BlockchainVerification:
    """Sistema de verificación blockchain"""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.difficulty = 4  # Número de ceros al inicio del hash
    
    def create_genesis_block(self):
        """Crea bloque génesis"""
        genesis = Block(
            index=0,
            timestamp=datetime.now(),
            data={"message": "Genesis Block"},
            previous_hash="0",
            hash=""
        )
        genesis.hash = self._calculate_hash(genesis)
        self.chain.append(genesis)
        logger.info("Bloque génesis creado")
    
    def _calculate_hash(self, block: Block) -> str:
        """Calcula hash de un bloque"""
        block_string = f"{block.index}{block.timestamp.isoformat()}{block.data}{block.previous_hash}{block.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _mine_block(self, block: Block) -> Block:
        """Mina un bloque (Proof of Work)"""
        while not block.hash.startswith("0" * self.difficulty):
            block.nonce += 1
            block.hash = self._calculate_hash(block)
        return block
    
    def add_block(self, data: Dict[str, Any]) -> Block:
        """Agrega un bloque a la cadena"""
        if not self.chain:
            self.create_genesis_block()
        
        previous_block = self.chain[-1]
        
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now(),
            data=data,
            previous_hash=previous_block.hash,
            hash=""
        )
        
        # Minar bloque
        new_block = self._mine_block(new_block)
        
        self.chain.append(new_block)
        logger.info(f"Bloque agregado: {new_block.index}")
        
        return new_block
    
    def verify_chain(self) -> bool:
        """Verifica integridad de la cadena"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Verificar hash del bloque anterior
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Hash inválido en bloque {i}")
                return False
            
            # Verificar hash del bloque actual
            calculated_hash = self._calculate_hash(current_block)
            if current_block.hash != calculated_hash:
                logger.error(f"Hash inválido en bloque {i}")
                return False
        
        return True
    
    def register_prototype(self, prototype_id: str, prototype_data: Dict[str, Any]) -> str:
        """Registra un prototipo en blockchain"""
        block_data = {
            "type": "prototype_registration",
            "prototype_id": prototype_id,
            "prototype_data": prototype_data,
            "timestamp": datetime.now().isoformat()
        }
        
        block = self.add_block(block_data)
        
        return block.hash
    
    def verify_prototype(self, prototype_id: str) -> Optional[Dict[str, Any]]:
        """Verifica un prototipo en blockchain"""
        if not self.verify_chain():
            return None
        
        for block in self.chain:
            if (block.data.get("type") == "prototype_registration" and
                block.data.get("prototype_id") == prototype_id):
                return {
                    "prototype_id": prototype_id,
                    "block_index": block.index,
                    "block_hash": block.hash,
                    "timestamp": block.timestamp.isoformat(),
                    "verified": True
                }
        
        return None
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Obtiene información de la cadena"""
        return {
            "chain_length": len(self.chain),
            "is_valid": self.verify_chain(),
            "difficulty": self.difficulty,
            "last_block": {
                "index": self.chain[-1].index if self.chain else None,
                "hash": self.chain[-1].hash if self.chain else None,
                "timestamp": self.chain[-1].timestamp.isoformat() if self.chain else None
            }
        }




