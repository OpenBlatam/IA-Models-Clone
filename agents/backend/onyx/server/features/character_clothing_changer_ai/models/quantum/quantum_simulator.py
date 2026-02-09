"""
Quantum Computing Simulator
===========================
Sistema de simulación de computación cuántica para optimización
"""

import time
import random
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class QuantumGate(Enum):
    """Puertas cuánticas"""
    HADAMARD = "H"
    PAULI_X = "X"
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    CNOT = "CNOT"
    PHASE = "PHASE"
    ROTATION = "ROTATION"


@dataclass
class QuantumCircuit:
    """Circuito cuántico"""
    id: str
    qubits: int
    gates: List[Dict[str, Any]]
    created_at: float
    executed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class QuantumState:
    """Estado cuántico"""
    qubits: int
    amplitudes: List[complex]
    probabilities: List[float]


class QuantumSimulator:
    """
    Simulador de computación cuántica
    """
    
    def __init__(self):
        self.circuits: Dict[str, QuantumCircuit] = {}
        self.max_qubits = 20  # Límite práctico para simulación
    
    def create_circuit(self, qubits: int, name: Optional[str] = None) -> QuantumCircuit:
        """
        Crear circuito cuántico
        
        Args:
            qubits: Número de qubits
            name: Nombre del circuito
        """
        if qubits > self.max_qubits:
            raise ValueError(f"Maximum {self.max_qubits} qubits supported")
        
        circuit_id = f"circuit_{int(time.time())}"
        
        circuit = QuantumCircuit(
            id=circuit_id,
            qubits=qubits,
            gates=[],
            created_at=time.time()
        )
        
        self.circuits[circuit_id] = circuit
        return circuit
    
    def add_gate(
        self,
        circuit_id: str,
        gate: QuantumGate,
        target_qubit: int,
        control_qubit: Optional[int] = None,
        angle: Optional[float] = None
    ):
        """
        Agregar puerta al circuito
        
        Args:
            circuit_id: ID del circuito
            gate: Tipo de puerta
            target_qubit: Qubit objetivo
            control_qubit: Qubit de control (para CNOT)
            angle: Ángulo para rotaciones
        """
        if circuit_id not in self.circuits:
            raise ValueError(f"Circuit {circuit_id} not found")
        
        circuit = self.circuits[circuit_id]
        
        if target_qubit >= circuit.qubits:
            raise ValueError(f"Target qubit {target_qubit} out of range")
        
        gate_info = {
            'gate': gate.value,
            'target': target_qubit,
            'timestamp': time.time()
        }
        
        if control_qubit is not None:
            gate_info['control'] = control_qubit
        if angle is not None:
            gate_info['angle'] = angle
        
        circuit.gates.append(gate_info)
    
    def execute_circuit(self, circuit_id: str) -> Dict[str, Any]:
        """
        Ejecutar circuito cuántico
        
        Args:
            circuit_id: ID del circuito
        """
        if circuit_id not in self.circuits:
            raise ValueError(f"Circuit {circuit_id} not found")
        
        circuit = self.circuits[circuit_id]
        circuit.executed_at = time.time()
        
        # Simular ejecución
        # En implementación real, usar qiskit o cirq
        state = self._initialize_state(circuit.qubits)
        
        # Aplicar puertas
        for gate_info in circuit.gates:
            state = self._apply_gate(state, gate_info)
        
        # Medir
        measurement = self._measure(state)
        
        circuit.result = {
            'measurement': measurement,
            'state': self._state_to_dict(state),
            'execution_time': time.time() - circuit.executed_at
        }
        
        return circuit.result
    
    def _initialize_state(self, qubits: int) -> QuantumState:
        """Inicializar estado cuántico"""
        num_states = 2 ** qubits
        amplitudes = [0.0] * num_states
        amplitudes[0] = 1.0  # Estado |00...0>
        
        probabilities = [abs(amp) ** 2 for amp in amplitudes]
        
        return QuantumState(
            qubits=qubits,
            amplitudes=amplitudes,
            probabilities=probabilities
        )
    
    def _apply_gate(self, state: QuantumState, gate_info: Dict[str, Any]) -> QuantumState:
        """Aplicar puerta cuántica (simplificado)"""
        gate_type = gate_info['gate']
        target = gate_info['target']
        
        # En implementación real, aplicar matrices unitarias
        # Por ahora, simular efectos
        
        if gate_type == 'H':
            # Hadamard: crea superposición
            for i in range(len(state.amplitudes)):
                if i & (1 << target):
                    state.amplitudes[i] *= 0.707  # 1/sqrt(2)
                else:
                    state.amplitudes[i] *= 0.707
        
        elif gate_type == 'X':
            # Pauli-X: flip
            new_amplitudes = state.amplitudes.copy()
            for i in range(len(state.amplitudes)):
                flipped = i ^ (1 << target)
                new_amplitudes[flipped] = state.amplitudes[i]
            state.amplitudes = new_amplitudes
        
        # Recalcular probabilidades
        state.probabilities = [abs(amp) ** 2 for amp in state.amplitudes]
        
        return state
    
    def _measure(self, state: QuantumState) -> Dict[str, int]:
        """Medir estado cuántico"""
        # Muestreo basado en probabilidades
        total = sum(state.probabilities)
        if total == 0:
            return {'0' * state.qubits: 1}
        
        # Normalizar
        normalized = [p / total for p in state.probabilities]
        
        # Muestrear
        result = random.choices(
            range(len(normalized)),
            weights=normalized,
            k=100  # 100 mediciones
        )
        
        # Contar resultados
        counts = {}
        for r in result:
            binary = format(r, f'0{state.qubits}b')
            counts[binary] = counts.get(binary, 0) + 1
        
        return counts
    
    def _state_to_dict(self, state: QuantumState) -> Dict[str, float]:
        """Convertir estado a diccionario"""
        return {
            format(i, f'0{state.qubits}b'): abs(amp) ** 2
            for i, amp in enumerate(state.amplitudes)
            if abs(amp) > 1e-10  # Solo estados significativos
        }
    
    def optimize_parameters(
        self,
        objective_function: callable,
        parameter_bounds: List[Tuple[float, float]],
        iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Optimizar parámetros usando algoritmo cuántico
        
        Args:
            objective_function: Función objetivo
            parameter_bounds: Límites de parámetros
            iterations: Número de iteraciones
        """
        # Algoritmo de optimización cuántica simplificado
        # En implementación real, usar VQE o QAOA
        
        best_params = None
        best_value = float('inf')
        
        for i in range(iterations):
            # Generar parámetros aleatorios
            params = [
                random.uniform(low, high)
                for low, high in parameter_bounds
            ]
            
            value = objective_function(params)
            
            if value < best_value:
                best_value = value
                best_params = params
        
        return {
            'best_parameters': best_params,
            'best_value': best_value,
            'iterations': iterations
        }
    
    def get_circuit_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de circuitos"""
        return {
            'total_circuits': len(self.circuits),
            'executed_circuits': len([c for c in self.circuits.values() if c.executed_at]),
            'average_gates': (
                sum(len(c.gates) for c in self.circuits.values()) / len(self.circuits)
                if self.circuits else 0
            ),
            'max_qubits_used': (
                max(c.qubits for c in self.circuits.values())
                if self.circuits else 0
            )
        }


# Instancia global
quantum_simulator = QuantumSimulator()

