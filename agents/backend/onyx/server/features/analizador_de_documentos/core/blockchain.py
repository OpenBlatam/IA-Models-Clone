"""
Sistema de Blockchain
=====================

Sistema para registro inmutable de análisis usando blockchain.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """Bloque en blockchain"""
    index: int
    timestamp: str
    data: Dict[str, Any]
    previous_hash: str
    hash: str
    nonce: int = 0


class Blockchain:
    """
    Blockchain para registro inmutable
    
    Proporciona:
    - Cadena de bloques
    - Proof of Work (simplificado)
    - Validación de cadena
    - Registro inmutable de análisis
    - Verificación de integridad
    """
    
    def __init__(self, difficulty: int = 2):
        """Inicializar blockchain"""
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self._create_genesis_block()
        logger.info("Blockchain inicializado")
    
    def _create_genesis_block(self):
        """Crear bloque génesis"""
        genesis = Block(
            index=0,
            timestamp=datetime.now().isoformat(),
            data={"message": "Genesis Block"},
            previous_hash="0",
            hash=self._calculate_hash(0, datetime.now().isoformat(), {"message": "Genesis Block"}, "0", 0)
        )
        
        self.chain.append(genesis)
    
    def _calculate_hash(
        self,
        index: int,
        timestamp: str,
        data: Dict[str, Any],
        previous_hash: str,
        nonce: int
    ) -> str:
        """Calcular hash del bloque"""
        import json
        block_string = f"{index}{timestamp}{json.dumps(data, sort_keys=True)}{previous_hash}{nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _proof_of_work(self, block: Block) -> Block:
        """Proof of Work (simplificado)"""
        target = "0" * self.difficulty
        
        while block.hash[:self.difficulty] != target:
            block.nonce += 1
            block.hash = self._calculate_hash(
                block.index,
                block.timestamp,
                block.data,
                block.previous_hash,
                block.nonce
            )
        
        return block
    
    def add_block(self, data: Dict[str, Any]) -> Block:
        """
        Agregar bloque a la cadena
        
        Args:
            data: Datos del bloque
        
        Returns:
            Bloque agregado
        """
        previous_block = self.chain[-1]
        
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            data=data,
            previous_hash=previous_block.hash,
            hash=""
        )
        
        # Calcular hash inicial
        new_block.hash = self._calculate_hash(
            new_block.index,
            new_block.timestamp,
            new_block.data,
            new_block.previous_hash,
            new_block.nonce
        )
        
        # Proof of Work
        new_block = self._proof_of_work(new_block)
        
        self.chain.append(new_block)
        logger.info(f"Bloque agregado: {new_block.index}")
        
        return new_block
    
    def is_chain_valid(self) -> bool:
        """Validar cadena de bloques"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            # Verificar hash del bloque actual
            if current.hash != self._calculate_hash(
                current.index,
                current.timestamp,
                current.data,
                current.previous_hash,
                current.nonce
            ):
                return False
            
            # Verificar que el hash anterior coincida
            if current.previous_hash != previous.hash:
                return False
        
        return True
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Obtener información de la blockchain"""
        return {
            "length": len(self.chain),
            "difficulty": self.difficulty,
            "is_valid": self.is_chain_valid(),
            "last_block": {
                "index": self.chain[-1].index,
                "timestamp": self.chain[-1].timestamp,
                "hash": self.chain[-1].hash
            } if self.chain else None
        }


# Instancia global
_blockchain: Optional[Blockchain] = None


def get_blockchain() -> Blockchain:
    """Obtener instancia global de la blockchain"""
    global _blockchain
    if _blockchain is None:
        _blockchain = Blockchain()
    return _blockchain














