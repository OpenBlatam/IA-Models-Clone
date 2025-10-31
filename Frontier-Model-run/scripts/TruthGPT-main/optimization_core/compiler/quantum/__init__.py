"""
Quantum Compiler Module for TruthGPT
Advanced quantum-inspired compilation with quantum computing optimization
"""

from .quantum_compiler import (
    QuantumCompiler, QuantumCompilationConfig, QuantumCompilationResult,
    QuantumCompilationMode, QuantumOptimizationStrategy, QuantumCompilationTarget,
    create_quantum_compiler, quantum_compilation_context
)

__all__ = [
    'QuantumCompiler',
    'QuantumCompilationConfig',
    'QuantumCompilationResult',
    'QuantumCompilationMode',
    'QuantumOptimizationStrategy',
    'QuantumCompilationTarget',
    'create_quantum_compiler',
    'quantum_compilation_context'
]

__version__ = "1.0.0"


