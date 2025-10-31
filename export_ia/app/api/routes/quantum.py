"""
Quantum API Routes - Rutas API para sistema de Computación Cuántica
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal
import logging

from ..quantum.quantum_engine import QuantumEngine, QuantumGate, QuantumAlgorithm, QuantumBackend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/quantum", tags=["Quantum"])

# Instancia global del motor cuántico
quantum_engine = QuantumEngine()


# Modelos Pydantic
class CreateCircuitRequest(BaseModel):
    name: str
    qubits: int
    gates: List[Dict[str, Any]]
    measurements: Optional[List[int]] = None


class ExecuteCircuitRequest(BaseModel):
    circuit_id: str
    backend: str = "simulator"
    shots: int = 1024


class QuantumGateRequest(BaseModel):
    gate_type: str
    qubits: List[int]
    params: Optional[Dict[str, Any]] = None


# Rutas de Circuitos Cuánticos
@router.post("/circuits")
async def create_quantum_circuit(request: CreateCircuitRequest):
    """Crear circuito cuántico."""
    try:
        circuit_id = await quantum_engine.create_circuit(
            name=request.name,
            qubits=request.qubits,
            gates=request.gates,
            measurements=request.measurements
        )
        
        circuit = quantum_engine.circuits[circuit_id]
        
        return {
            "circuit_id": circuit_id,
            "name": circuit.name,
            "qubits": circuit.qubits,
            "gates": circuit.gates,
            "measurements": circuit.measurements,
            "created_at": circuit.created_at.isoformat(),
            "success": True,
            "message": "Circuito cuántico creado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al crear circuito cuántico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuits")
async def get_quantum_circuits():
    """Obtener todos los circuitos cuánticos."""
    try:
        circuits = []
        for circuit_id, circuit in quantum_engine.circuits.items():
            circuits.append({
                "circuit_id": circuit_id,
                "name": circuit.name,
                "qubits": circuit.qubits,
                "gates_count": len(circuit.gates),
                "measurements": circuit.measurements,
                "created_at": circuit.created_at.isoformat(),
                "executed_at": circuit.executed_at.isoformat() if circuit.executed_at else None,
                "has_results": circuit.results is not None
            })
        
        return {
            "circuits": circuits,
            "count": len(circuits),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener circuitos cuánticos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuits/{circuit_id}")
async def get_quantum_circuit(circuit_id: str):
    """Obtener circuito cuántico específico."""
    try:
        if circuit_id not in quantum_engine.circuits:
            raise HTTPException(status_code=404, detail="Circuito no encontrado")
        
        circuit = quantum_engine.circuits[circuit_id]
        
        return {
            "circuit": {
                "circuit_id": circuit_id,
                "name": circuit.name,
                "qubits": circuit.qubits,
                "gates": circuit.gates,
                "measurements": circuit.measurements,
                "created_at": circuit.created_at.isoformat(),
                "executed_at": circuit.executed_at.isoformat() if circuit.executed_at else None,
                "results": circuit.results
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener circuito cuántico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/circuits/{circuit_id}")
async def delete_quantum_circuit(circuit_id: str):
    """Eliminar circuito cuántico."""
    try:
        if circuit_id not in quantum_engine.circuits:
            raise HTTPException(status_code=404, detail="Circuito no encontrado")
        
        circuit = quantum_engine.circuits[circuit_id]
        del quantum_engine.circuits[circuit_id]
        
        # Limpiar trabajos relacionados
        related_jobs = [
            job_id for job_id, job in quantum_engine.jobs.items()
            if job.circuit_id == circuit_id
        ]
        for job_id in related_jobs:
            del quantum_engine.jobs[job_id]
        
        quantum_engine.stats["total_circuits"] -= 1
        
        return {
            "circuit_id": circuit_id,
            "name": circuit.name,
            "success": True,
            "message": "Circuito cuántico eliminado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar circuito cuántico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Ejecución Cuántica
@router.post("/execute")
async def execute_quantum_circuit(request: ExecuteCircuitRequest):
    """Ejecutar circuito cuántico."""
    try:
        backend = QuantumBackend(request.backend)
        
        job_id = await quantum_engine.execute_circuit(
            circuit_id=request.circuit_id,
            backend=backend,
            shots=request.shots
        )
        
        return {
            "job_id": job_id,
            "circuit_id": request.circuit_id,
            "backend": request.backend,
            "shots": request.shots,
            "success": True,
            "message": "Circuito cuántico enviado para ejecución",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error al ejecutar circuito cuántico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_quantum_job_status(job_id: str):
    """Obtener estado de trabajo cuántico."""
    try:
        job_status = await quantum_engine.get_job_status(job_id)
        
        return {
            "job": job_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener estado de trabajo cuántico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def get_quantum_jobs(
    status: Optional[str] = Query(None, description="Estado del trabajo"),
    backend: Optional[str] = Query(None, description="Backend utilizado")
):
    """Obtener trabajos cuánticos."""
    try:
        jobs = []
        
        for job_id, job in quantum_engine.jobs.items():
            # Filtrar por estado
            if status and job.status != status:
                continue
            
            # Filtrar por backend
            if backend and job.backend.value != backend:
                continue
            
            circuit = quantum_engine.circuits[job.circuit_id]
            
            jobs.append({
                "job_id": job_id,
                "circuit_id": job.circuit_id,
                "circuit_name": circuit.name,
                "backend": job.backend.value,
                "shots": job.shots,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "has_results": job.results is not None,
                "error": job.error
            })
        
        # Ordenar por fecha de creación descendente
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "jobs": jobs,
            "count": len(jobs),
            "filters": {
                "status": status,
                "backend": backend
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener trabajos cuánticos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Backends Cuánticos
@router.get("/backends")
async def get_quantum_backends():
    """Obtener backends cuánticos disponibles."""
    try:
        backends = []
        
        for backend_type, config in quantum_engine.backends.items():
            backends.append({
                "backend": backend_type.value,
                "available": config["available"],
                "max_qubits": config["max_qubits"],
                "max_shots": config["max_shots"],
                "description": config["description"]
            })
        
        return {
            "backends": backends,
            "count": len(backends),
            "available_count": sum(1 for b in backends if b["available"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener backends cuánticos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backends/{backend}/status")
async def get_backend_status(backend: str):
    """Obtener estado de backend específico."""
    try:
        backend_type = QuantumBackend(backend)
        
        if backend_type not in quantum_engine.backends:
            raise HTTPException(status_code=404, detail="Backend no encontrado")
        
        config = quantum_engine.backends[backend_type]
        
        return {
            "backend": backend,
            "available": config["available"],
            "max_qubits": config["max_qubits"],
            "max_shots": config["max_shots"],
            "description": config["description"],
            "status": "online" if config["available"] else "offline",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener estado de backend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Estadísticas
@router.get("/stats")
async def get_quantum_stats():
    """Obtener estadísticas cuánticas."""
    try:
        stats = await quantum_engine.get_quantum_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas cuánticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def quantum_health_check():
    """Verificar salud del sistema cuántico."""
    try:
        health = await quantum_engine.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check cuántico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de utilidad
@router.get("/gates")
async def get_quantum_gates():
    """Obtener puertas cuánticas disponibles."""
    return {
        "gates": [
            {
                "value": gate.value,
                "name": gate.name,
                "description": f"Puerta cuántica {gate.value}"
            }
            for gate in QuantumGate
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/algorithms")
async def get_quantum_algorithms():
    """Obtener algoritmos cuánticos disponibles."""
    return {
        "algorithms": [
            {
                "value": algorithm.value,
                "name": algorithm.name,
                "description": f"Algoritmo cuántico {algorithm.value}"
            }
            for algorithm in QuantumAlgorithm
        ],
        "timestamp": datetime.now().isoformat()
    }


# Rutas de ejemplo
@router.post("/examples/bell-state")
async def create_bell_state_circuit():
    """Ejemplo: Crear circuito de estado de Bell."""
    try:
        # Crear circuito de estado de Bell (2 qubits)
        gates = [
            {"type": "hadamard", "qubits": [0]},
            {"type": "cnot", "qubits": [0, 1]}
        ]
        
        circuit_id = await quantum_engine.create_circuit(
            name="Estado de Bell",
            qubits=2,
            gates=gates,
            measurements=[0, 1]
        )
        
        # Ejecutar circuito
        job_id = await quantum_engine.execute_circuit(
            circuit_id=circuit_id,
            backend=QuantumBackend.SIMULATOR,
            shots=1024
        )
        
        circuit = quantum_engine.circuits[circuit_id]
        
        return {
            "circuit": {
                "circuit_id": circuit_id,
                "name": circuit.name,
                "qubits": circuit.qubits,
                "gates": circuit.gates,
                "measurements": circuit.measurements
            },
            "job_id": job_id,
            "success": True,
            "message": "Circuito de estado de Bell creado y ejecutado",
            "description": "Este circuito crea un estado de Bell entrelazado entre dos qubits",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en ejemplo de estado de Bell: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/examples/quantum-teleportation")
async def create_teleportation_circuit():
    """Ejemplo: Crear circuito de teletransportación cuántica."""
    try:
        # Crear circuito de teletransportación cuántica (3 qubits)
        gates = [
            # Preparar estado de Bell entre qubits 1 y 2
            {"type": "hadamard", "qubits": [1]},
            {"type": "cnot", "qubits": [1, 2]},
            # Aplicar puertas de teletransportación
            {"type": "cnot", "qubits": [0, 1]},
            {"type": "hadamard", "qubits": [0]},
            # Medir qubits 0 y 1
            {"type": "cnot", "qubits": [1, 2]},
            {"type": "pauli_z", "qubits": [2]}
        ]
        
        circuit_id = await quantum_engine.create_circuit(
            name="Teletransportación Cuántica",
            qubits=3,
            gates=gates,
            measurements=[0, 1, 2]
        )
        
        # Ejecutar circuito
        job_id = await quantum_engine.execute_circuit(
            circuit_id=circuit_id,
            backend=QuantumBackend.SIMULATOR,
            shots=1024
        )
        
        circuit = quantum_engine.circuits[circuit_id]
        
        return {
            "circuit": {
                "circuit_id": circuit_id,
                "name": circuit.name,
                "qubits": circuit.qubits,
                "gates": circuit.gates,
                "measurements": circuit.measurements
            },
            "job_id": job_id,
            "success": True,
            "message": "Circuito de teletransportación cuántica creado y ejecutado",
            "description": "Este circuito implementa el protocolo de teletransportación cuántica",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en ejemplo de teletransportación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples/quantum-algorithms")
async def get_quantum_algorithms_examples():
    """Ejemplo: Obtener ejemplos de algoritmos cuánticos."""
    try:
        algorithms_examples = {
            "grover": {
                "name": "Algoritmo de Grover",
                "description": "Algoritmo de búsqueda cuántica que encuentra un elemento en una base de datos no ordenada",
                "complexity": "O(√N)",
                "qubits_required": "log₂(N)",
                "use_cases": ["Búsqueda en bases de datos", "Optimización", "Criptoanálisis"]
            },
            "shor": {
                "name": "Algoritmo de Shor",
                "description": "Algoritmo cuántico para factorización de números enteros",
                "complexity": "O((log N)³)",
                "qubits_required": "2log₂(N)",
                "use_cases": ["Criptografía", "Factorización", "Seguridad RSA"]
            },
            "qaoa": {
                "name": "Quantum Approximate Optimization Algorithm",
                "description": "Algoritmo cuántico para optimización combinatoria",
                "complexity": "Dependiente del problema",
                "qubits_required": "Variable",
                "use_cases": ["Optimización", "Machine Learning", "Problemas NP-completos"]
            },
            "vqe": {
                "name": "Variational Quantum Eigensolver",
                "description": "Algoritmo cuántico para encontrar valores propios de matrices",
                "complexity": "Dependiente del problema",
                "qubits_required": "Variable",
                "use_cases": ["Química cuántica", "Materiales", "Física"]
            }
        }
        
        return {
            "algorithms_examples": algorithms_examples,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en ejemplos de algoritmos cuánticos: {e}")
        raise HTTPException(status_code=500, detail=str(e))




