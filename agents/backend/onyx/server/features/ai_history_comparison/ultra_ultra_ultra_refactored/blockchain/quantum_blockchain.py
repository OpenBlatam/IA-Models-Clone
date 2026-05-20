"""
Quantum Blockchain - Blockchain Cuántico
=======================================

Sistema de blockchain cuántico que utiliza principios cuánticos
para la inmutabilidad, consenso y seguridad de los datos.
"""

from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
import uuid

from ..quantum_core.quantum_domain.quantum_value_objects import (
    QuantumContentId,
    QuantumState,
    DimensionalVector,
    TemporalCoordinate
)


@dataclass
class QuantumBlock:
    """
    Bloque cuántico que contiene transacciones y metadatos cuánticos.
    """
    
    # Identidad del bloque
    block_id: str
    previous_hash: str
    timestamp: datetime
    
    # Contenido cuántico
    transactions: List[Dict[str, Any]]
    quantum_state: QuantumState
    dimensional_vector: DimensionalVector
    temporal_coordinate: TemporalCoordinate
    
    # Metadatos cuánticos
    quantum_hash: str
    coherence_level: float
    entanglement_pairs: List[str] = field(default_factory=list)
    
    # Prueba de trabajo cuántica
    quantum_nonce: int
    quantum_proof: str
    
    def __post_init__(self):
        """Calcular hash cuántico del bloque."""
        if not self.quantum_hash:
            self.quantum_hash = self._calculate_quantum_hash()
    
    def _calculate_quantum_hash(self) -> str:
        """Calcular hash cuántico del bloque."""
        # Crear representación cuántica del bloque
        block_data = {
            "block_id": self.block_id,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp.isoformat(),
            "transactions": self.transactions,
            "quantum_state": self.quantum_state.to_dict(),
            "dimensional_vector": self.dimensional_vector.to_dict(),
            "temporal_coordinate": self.temporal_coordinate.to_dict(),
            "coherence_level": self.coherence_level,
            "entanglement_pairs": self.entanglement_pairs,
            "quantum_nonce": self.quantum_nonce
        }
        
        # Convertir a JSON y calcular hash cuántico
        block_json = json.dumps(block_data, sort_keys=True)
        
        # Aplicar transformación cuántica al hash
        quantum_hash = self._quantum_hash_transform(block_json)
        
        return quantum_hash
    
    def _quantum_hash_transform(self, data: str) -> str:
        """Aplicar transformación cuántica al hash."""
        # Hash SHA-256 estándar
        sha256_hash = hashlib.sha256(data.encode()).hexdigest()
        
        # Aplicar transformación cuántica
        quantum_hash = self._apply_quantum_transformation(sha256_hash)
        
        return quantum_hash
    
    def _apply_quantum_transformation(self, hash_str: str) -> str:
        """Aplicar transformación cuántica al hash."""
        # Convertir hash a matriz cuántica
        hash_matrix = self._hash_to_quantum_matrix(hash_str)
        
        # Aplicar operaciones cuánticas
        transformed_matrix = self._apply_quantum_operations(hash_matrix)
        
        # Convertir de vuelta a hash
        quantum_hash = self._quantum_matrix_to_hash(transformed_matrix)
        
        return quantum_hash
    
    def _hash_to_quantum_matrix(self, hash_str: str) -> np.ndarray:
        """Convertir hash a matriz cuántica."""
        # Convertir cada carácter a valor numérico
        values = [ord(c) for c in hash_str]
        
        # Crear matriz cuántica
        matrix_size = int(np.ceil(np.sqrt(len(values))))
        quantum_matrix = np.zeros((matrix_size, matrix_size), dtype=complex)
        
        for i, value in enumerate(values):
            row = i // matrix_size
            col = i % matrix_size
            quantum_matrix[row, col] = complex(value, 0)
        
        return quantum_matrix
    
    def _apply_quantum_operations(self, matrix: np.ndarray) -> np.ndarray:
        """Aplicar operaciones cuánticas a la matriz."""
        # Aplicar rotación cuántica
        rotation_angle = np.pi / 4
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        
        # Aplicar rotación si la matriz es 2x2
        if matrix.shape == (2, 2):
            transformed_matrix = rotation_matrix @ matrix
        else:
            # Para matrices más grandes, aplicar transformación similar
            transformed_matrix = matrix * np.exp(1j * rotation_angle)
        
        return transformed_matrix
    
    def _quantum_matrix_to_hash(self, matrix: np.ndarray) -> str:
        """Convertir matriz cuántica de vuelta a hash."""
        # Extraer valores reales e imaginarios
        real_values = np.real(matrix).flatten()
        imag_values = np.imag(matrix).flatten()
        
        # Combinar valores
        combined_values = np.concatenate([real_values, imag_values])
        
        # Convertir a string hexadecimal
        hash_str = ""
        for value in combined_values:
            # Normalizar valor
            normalized_value = int((value + 1) * 127.5) % 256
            hash_str += f"{normalized_value:02x}"
        
        return hash_str[:64]  # Limitar a 64 caracteres


@dataclass
class QuantumTransaction:
    """
    Transacción cuántica que representa una operación en el blockchain.
    """
    
    # Identidad de la transacción
    transaction_id: str
    timestamp: datetime
    
    # Contenido de la transacción
    transaction_type: str
    data: Dict[str, Any]
    
    # Metadatos cuánticos
    quantum_signature: str
    quantum_state: QuantumState
    dimensional_vector: DimensionalVector
    temporal_coordinate: TemporalCoordinate
    
    # Validación cuántica
    is_valid: bool = True
    validation_confidence: float = 1.0
    
    def __post_init__(self):
        """Validar transacción cuántica."""
        if not self.quantum_signature:
            self.quantum_signature = self._generate_quantum_signature()
        
        self._validate_quantum_transaction()
    
    def _generate_quantum_signature(self) -> str:
        """Generar firma cuántica de la transacción."""
        # Crear datos de la transacción
        transaction_data = {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "transaction_type": self.transaction_type,
            "data": self.data,
            "quantum_state": self.quantum_state.to_dict(),
            "dimensional_vector": self.dimensional_vector.to_dict(),
            "temporal_coordinate": self.temporal_coordinate.to_dict()
        }
        
        # Convertir a JSON
        transaction_json = json.dumps(transaction_data, sort_keys=True)
        
        # Generar firma cuántica
        quantum_signature = self._quantum_signature_algorithm(transaction_json)
        
        return quantum_signature
    
    def _quantum_signature_algorithm(self, data: str) -> str:
        """Algoritmo de firma cuántica."""
        # Hash SHA-256
        sha256_hash = hashlib.sha256(data.encode()).hexdigest()
        
        # Aplicar transformación cuántica
        quantum_signature = self._apply_quantum_signature_transform(sha256_hash)
        
        return quantum_signature
    
    def _apply_quantum_signature_transform(self, hash_str: str) -> str:
        """Aplicar transformación cuántica a la firma."""
        # Convertir a valores numéricos
        values = [ord(c) for c in hash_str]
        
        # Aplicar transformación cuántica
        transformed_values = []
        for value in values:
            # Aplicar rotación cuántica
            transformed_value = (value * 7 + 13) % 256
            transformed_values.append(transformed_value)
        
        # Convertir a string hexadecimal
        signature = ""
        for value in transformed_values:
            signature += f"{value:02x}"
        
        return signature
    
    def _validate_quantum_transaction(self) -> None:
        """Validar transacción cuántica."""
        # Verificar coherencia cuántica
        if self.quantum_state.coherence_level < 0.5:
            self.is_valid = False
            self.validation_confidence = 0.0
            return
        
        # Verificar dimensionalidad
        if not self.dimensional_vector.is_valid():
            self.is_valid = False
            self.validation_confidence = 0.0
            return
        
        # Verificar temporalidad
        if not self.temporal_coordinate.is_valid():
            self.is_valid = False
            self.validation_confidence = 0.0
            return
        
        # Calcular confianza de validación
        self.validation_confidence = self._calculate_validation_confidence()
    
    def _calculate_validation_confidence(self) -> float:
        """Calcular confianza de validación."""
        # Factores de validación
        coherence_factor = self.quantum_state.coherence_level
        dimensional_factor = self.dimensional_vector.validity_score
        temporal_factor = self.temporal_coordinate.validity_score
        
        # Calcular confianza promedio
        confidence = (coherence_factor + dimensional_factor + temporal_factor) / 3.0
        
        return confidence


class QuantumBlockchain:
    """
    Blockchain cuántico que mantiene la inmutabilidad y seguridad
    de los datos utilizando principios cuánticos.
    """
    
    def __init__(self, genesis_data: Optional[Dict[str, Any]] = None):
        self.blocks: List[QuantumBlock] = []
        self.pending_transactions: List[QuantumTransaction] = []
        self.quantum_state: QuantumState = QuantumState()
        
        # Parámetros cuánticos
        self.quantum_parameters = {
            "coherence_threshold": 0.7,
            "entanglement_strength": 1.0,
            "quantum_nonce_range": (0, 1000000),
            "block_time": 10.0  # segundos
        }
        
        # Crear bloque génesis
        self._create_genesis_block(genesis_data)
    
    def _create_genesis_block(self, genesis_data: Optional[Dict[str, Any]] = None) -> None:
        """Crear bloque génesis del blockchain cuántico."""
        genesis_transactions = []
        
        if genesis_data:
            genesis_transaction = QuantumTransaction(
                transaction_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                transaction_type="genesis",
                data=genesis_data,
                quantum_signature="",
                quantum_state=QuantumState(),
                dimensional_vector=DimensionalVector(),
                temporal_coordinate=TemporalCoordinate.now()
            )
            genesis_transactions.append(genesis_transaction)
        
        genesis_block = QuantumBlock(
            block_id=str(uuid.uuid4()),
            previous_hash="0" * 64,
            timestamp=datetime.utcnow(),
            transactions=[tx.__dict__ for tx in genesis_transactions],
            quantum_state=QuantumState(),
            dimensional_vector=DimensionalVector(),
            temporal_coordinate=TemporalCoordinate.now(),
            quantum_hash="",
            coherence_level=1.0,
            quantum_nonce=0,
            quantum_proof=""
        )
        
        self.blocks.append(genesis_block)
    
    def add_transaction(self, transaction: QuantumTransaction) -> bool:
        """
        Agregar transacción al blockchain cuántico.
        
        Args:
            transaction: Transacción cuántica
            
        Returns:
            bool: True si se agregó exitosamente
        """
        # Validar transacción
        if not transaction.is_valid:
            return False
        
        # Verificar coherencia cuántica
        if transaction.quantum_state.coherence_level < self.quantum_parameters["coherence_threshold"]:
            return False
        
        # Agregar a transacciones pendientes
        self.pending_transactions.append(transaction)
        
        return True
    
    def mine_block(self) -> Optional[QuantumBlock]:
        """
        Minar un nuevo bloque cuántico.
        
        Returns:
            Optional[QuantumBlock]: Nuevo bloque minado
        """
        if not self.pending_transactions:
            return None
        
        # Obtener bloque anterior
        previous_block = self.blocks[-1]
        
        # Crear nuevo bloque
        new_block = QuantumBlock(
            block_id=str(uuid.uuid4()),
            previous_hash=previous_block.quantum_hash,
            timestamp=datetime.utcnow(),
            transactions=[tx.__dict__ for tx in self.pending_transactions],
            quantum_state=self.quantum_state,
            dimensional_vector=DimensionalVector(),
            temporal_coordinate=TemporalCoordinate.now(),
            quantum_hash="",
            coherence_level=self._calculate_block_coherence(),
            quantum_nonce=0,
            quantum_proof=""
        )
        
        # Minar bloque (encontrar nonce cuántico)
        new_block.quantum_nonce = self._mine_quantum_nonce(new_block)
        new_block.quantum_proof = self._generate_quantum_proof(new_block)
        
        # Agregar bloque al blockchain
        self.blocks.append(new_block)
        
        # Limpiar transacciones pendientes
        self.pending_transactions.clear()
        
        return new_block
    
    def _mine_quantum_nonce(self, block: QuantumBlock) -> int:
        """Minar nonce cuántico para el bloque."""
        target_difficulty = self._calculate_quantum_difficulty()
        
        for nonce in range(*self.quantum_parameters["quantum_nonce_range"]):
            block.quantum_nonce = nonce
            block.quantum_hash = block._calculate_quantum_hash()
            
            if self._is_quantum_hash_valid(block.quantum_hash, target_difficulty):
                return nonce
        
        return 0  # Fallback
    
    def _calculate_quantum_difficulty(self) -> int:
        """Calcular dificultad cuántica del blockchain."""
        # Dificultad basada en el número de bloques
        base_difficulty = 4
        difficulty_increase = len(self.blocks) // 100
        
        return base_difficulty + difficulty_increase
    
    def _is_quantum_hash_valid(self, hash_str: str, difficulty: int) -> bool:
        """Verificar si el hash cuántico es válido."""
        # Verificar que el hash comience con ceros
        required_zeros = "0" * difficulty
        return hash_str.startswith(required_zeros)
    
    def _generate_quantum_proof(self, block: QuantumBlock) -> str:
        """Generar prueba cuántica del bloque."""
        # Crear prueba basada en el estado cuántico
        proof_data = {
            "block_id": block.block_id,
            "quantum_hash": block.quantum_hash,
            "quantum_nonce": block.quantum_nonce,
            "coherence_level": block.coherence_level,
            "timestamp": block.timestamp.isoformat()
        }
        
        proof_json = json.dumps(proof_data, sort_keys=True)
        proof_hash = hashlib.sha256(proof_json.encode()).hexdigest()
        
        return proof_hash
    
    def _calculate_block_coherence(self) -> float:
        """Calcular coherencia cuántica del bloque."""
        if not self.pending_transactions:
            return 1.0
        
        # Calcular coherencia promedio de las transacciones
        coherence_values = [tx.quantum_state.coherence_level for tx in self.pending_transactions]
        average_coherence = np.mean(coherence_values)
        
        return float(average_coherence)
    
    def validate_blockchain(self) -> bool:
        """
        Validar la integridad del blockchain cuántico.
        
        Returns:
            bool: True si el blockchain es válido
        """
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i - 1]
            
            # Verificar hash del bloque anterior
            if current_block.previous_hash != previous_block.quantum_hash:
                return False
            
            # Verificar hash cuántico del bloque actual
            calculated_hash = current_block._calculate_quantum_hash()
            if current_block.quantum_hash != calculated_hash:
                return False
            
            # Verificar coherencia cuántica
            if current_block.coherence_level < self.quantum_parameters["coherence_threshold"]:
                return False
        
        return True
    
    def get_block_by_id(self, block_id: str) -> Optional[QuantumBlock]:
        """Obtener bloque por ID."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None
    
    def get_transaction_by_id(self, transaction_id: str) -> Optional[QuantumTransaction]:
        """Obtener transacción por ID."""
        for block in self.blocks:
            for tx_data in block.transactions:
                if tx_data.get("transaction_id") == transaction_id:
                    return QuantumTransaction(**tx_data)
        return None
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Obtener información del blockchain cuántico."""
        return {
            "total_blocks": len(self.blocks),
            "pending_transactions": len(self.pending_transactions),
            "quantum_state": self.quantum_state.to_dict(),
            "quantum_parameters": self.quantum_parameters,
            "is_valid": self.validate_blockchain(),
            "last_block_hash": self.blocks[-1].quantum_hash if self.blocks else None,
            "genesis_block_hash": self.blocks[0].quantum_hash if self.blocks else None
        }
    
    def create_quantum_entanglement(self, block_id_1: str, block_id_2: str) -> bool:
        """
        Crear entrelazamiento cuántico entre dos bloques.
        
        Args:
            block_id_1: ID del primer bloque
            block_id_2: ID del segundo bloque
            
        Returns:
            bool: True si se creó el entrelazamiento
        """
        block1 = self.get_block_by_id(block_id_1)
        block2 = self.get_block_by_id(block_id_2)
        
        if not block1 or not block2:
            return False
        
        # Crear entrelazamiento
        entanglement_id = str(uuid.uuid4())
        
        block1.entanglement_pairs.append(entanglement_id)
        block2.entanglement_pairs.append(entanglement_id)
        
        return True
    
    def get_entangled_blocks(self, block_id: str) -> List[QuantumBlock]:
        """Obtener bloques entrelazados con un bloque específico."""
        target_block = self.get_block_by_id(block_id)
        if not target_block:
            return []
        
        entangled_blocks = []
        for entanglement_id in target_block.entanglement_pairs:
            for block in self.blocks:
                if entanglement_id in block.entanglement_pairs and block.block_id != block_id:
                    entangled_blocks.append(block)
        
        return entangled_blocks




