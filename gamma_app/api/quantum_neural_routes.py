"""
Quantum AI & Neural Interface API Routes for Gamma App
======================================================

API endpoints for Quantum AI and Neural Interface services providing
advanced quantum computing and brain-computer interface capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..services.quantum_ai_service import (
    QuantumAIService,
    QuantumCircuit,
    QuantumJob,
    QuantumMLModel,
    QuantumOptimization,
    QuantumBackend,
    QuantumAlgorithm,
    QuantumGate
)

from ..services.neural_interface_service import (
    NeuralInterfaceService,
    NeuralDevice,
    NeuralSession,
    NeuralSignal,
    NeuralCommand,
    NeuralPattern,
    NeuralSignalType,
    NeuralInterfaceType,
    NeuralCommandType
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/quantum-neural", tags=["Quantum AI & Neural Interface"])

# Dependency to get services
def get_quantum_service() -> QuantumAIService:
    """Get Quantum AI service instance."""
    return QuantumAIService()

def get_neural_service() -> NeuralInterfaceService:
    """Get Neural Interface service instance."""
    return NeuralInterfaceService()

@router.get("/")
async def quantum_neural_root():
    """Quantum AI & Neural Interface root endpoint."""
    return {
        "message": "Quantum AI & Neural Interface Services for Gamma App",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": [
            "Quantum AI",
            "Neural Interface",
            "Quantum Machine Learning",
            "Brain-Computer Interface",
            "Quantum Optimization",
            "Neural Pattern Recognition",
            "Quantum Neural Networks",
            "Direct Neural Communication"
        ]
    }

# ==================== QUANTUM AI ENDPOINTS ====================

@router.post("/quantum/circuits/create")
async def create_quantum_circuit(
    circuit_info: Dict[str, Any],
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Create a quantum circuit."""
    try:
        circuit_id = await quantum_service.create_quantum_circuit(circuit_info)
        return {
            "circuit_id": circuit_id,
            "message": "Quantum circuit created successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating quantum circuit: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create quantum circuit: {e}")

@router.post("/quantum/jobs/execute")
async def execute_quantum_job(
    job_info: Dict[str, Any],
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Execute a quantum computing job."""
    try:
        job_id = await quantum_service.execute_quantum_job(job_info)
        return {
            "job_id": job_id,
            "message": "Quantum job created and execution started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error executing quantum job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute quantum job: {e}")

@router.post("/quantum/ml/train")
async def train_quantum_ml_model(
    model_info: Dict[str, Any],
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Train a quantum machine learning model."""
    try:
        model_id = await quantum_service.train_quantum_ml_model(model_info)
        return {
            "model_id": model_id,
            "message": "Quantum ML model training started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error training quantum ML model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train quantum ML model: {e}")

@router.post("/quantum/optimization/solve")
async def solve_quantum_optimization(
    optimization_info: Dict[str, Any],
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Solve a quantum optimization problem."""
    try:
        optimization_id = await quantum_service.solve_quantum_optimization(optimization_info)
        return {
            "optimization_id": optimization_id,
            "message": "Quantum optimization started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error solving quantum optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to solve quantum optimization: {e}")

@router.get("/quantum/jobs/{job_id}/status")
async def get_quantum_job_status(
    job_id: str,
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Get quantum job status."""
    try:
        status = await quantum_service.get_quantum_job_status(job_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Quantum job not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quantum job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quantum job status: {e}")

@router.get("/quantum/ml/{model_id}")
async def get_quantum_model_info(
    model_id: str,
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Get quantum ML model information."""
    try:
        model_info = await quantum_service.get_quantum_model_info(model_id)
        if model_info:
            return model_info
        else:
            raise HTTPException(status_code=404, detail="Quantum ML model not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quantum model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quantum model info: {e}")

@router.get("/quantum/optimization/{optimization_id}")
async def get_quantum_optimization_result(
    optimization_id: str,
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Get quantum optimization result."""
    try:
        result = await quantum_service.get_quantum_optimization_result(optimization_id)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="Quantum optimization not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quantum optimization result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quantum optimization result: {e}")

@router.get("/quantum/backends")
async def get_available_quantum_backends(
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Get available quantum backends."""
    try:
        backends = await quantum_service.get_available_backends()
        return {
            "backends": backends,
            "total": len(backends),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting quantum backends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quantum backends: {e}")

@router.get("/quantum/statistics")
async def get_quantum_statistics(
    quantum_service: QuantumAIService = Depends(get_quantum_service)
):
    """Get quantum AI service statistics."""
    try:
        stats = await quantum_service.get_quantum_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting quantum statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quantum statistics: {e}")

# ==================== NEURAL INTERFACE ENDPOINTS ====================

@router.post("/neural/devices/register")
async def register_neural_device(
    device_info: Dict[str, Any],
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Register a neural interface device."""
    try:
        device_id = await neural_service.register_neural_device(device_info)
        return {
            "device_id": device_id,
            "message": "Neural device registered successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error registering neural device: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register neural device: {e}")

@router.post("/neural/sessions/start")
async def start_neural_session(
    session_info: Dict[str, Any],
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Start a neural interface session."""
    try:
        session_id = await neural_service.start_neural_session(session_info)
        return {
            "session_id": session_id,
            "message": "Neural session started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting neural session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start neural session: {e}")

@router.post("/neural/signals/process")
async def process_neural_signal(
    signal_data: Dict[str, Any],
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Process neural signal data."""
    try:
        signal_id = await neural_service.process_neural_signal(signal_data)
        return {
            "signal_id": signal_id,
            "message": "Neural signal processed successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing neural signal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process neural signal: {e}")

@router.post("/neural/commands/generate")
async def generate_neural_command(
    command_info: Dict[str, Any],
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Generate a neural command from brain signals."""
    try:
        command_id = await neural_service.generate_neural_command(command_info)
        return {
            "command_id": command_id,
            "message": "Neural command generated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating neural command: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate neural command: {e}")

@router.post("/neural/patterns/train")
async def train_neural_pattern(
    pattern_info: Dict[str, Any],
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Train a neural pattern recognition model."""
    try:
        pattern_id = await neural_service.train_neural_pattern(pattern_info)
        return {
            "pattern_id": pattern_id,
            "message": "Neural pattern training started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error training neural pattern: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train neural pattern: {e}")

@router.post("/neural/patterns/recognize")
async def recognize_neural_pattern(
    pattern_data: Dict[str, Any],
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Recognize neural patterns in real-time."""
    try:
        result = await neural_service.recognize_neural_pattern(pattern_data)
        return {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error recognizing neural pattern: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to recognize neural pattern: {e}")

@router.get("/neural/sessions/{session_id}/status")
async def get_neural_session_status(
    session_id: str,
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Get neural session status."""
    try:
        status = await neural_service.get_neural_session_status(session_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Neural session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting neural session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get neural session status: {e}")

@router.get("/neural/statistics")
async def get_neural_statistics(
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Get neural interface service statistics."""
    try:
        stats = await neural_service.get_neural_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting neural statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get neural statistics: {e}")

# ==================== COMBINED ENDPOINTS ====================

@router.get("/health")
async def health_check(
    quantum_service: QuantumAIService = Depends(get_quantum_service),
    neural_service: NeuralInterfaceService = Depends(get_neural_service)
):
    """Health check for both services."""
    try:
        quantum_stats = await quantum_service.get_quantum_statistics()
        neural_stats = await neural_service.get_neural_statistics()
        
        return {
            "status": "healthy",
            "quantum_service": {
                "status": "operational",
                "total_circuits": quantum_stats.get("total_quantum_circuits", 0),
                "total_jobs": quantum_stats.get("total_quantum_jobs", 0),
                "total_models": quantum_stats.get("total_quantum_models", 0)
            },
            "neural_service": {
                "status": "operational",
                "total_devices": neural_stats.get("total_neural_devices", 0),
                "total_sessions": neural_stats.get("total_neural_sessions", 0),
                "total_commands": neural_stats.get("total_neural_commands", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/capabilities")
async def get_capabilities():
    """Get available capabilities of both services."""
    return {
        "quantum_ai": {
            "backends": [backend.value for backend in QuantumBackend],
            "algorithms": [algorithm.value for algorithm in QuantumAlgorithm],
            "gates": [gate.value for gate in QuantumGate],
            "capabilities": [
                "Quantum Circuit Design",
                "Quantum Job Execution",
                "Quantum Machine Learning",
                "Quantum Optimization",
                "Quantum Neural Networks",
                "Variational Quantum Algorithms",
                "Quantum Error Correction",
                "Quantum Simulation"
            ]
        },
        "neural_interface": {
            "signal_types": [signal_type.value for signal_type in NeuralSignalType],
            "interface_types": [interface_type.value for interface_type in NeuralInterfaceType],
            "command_types": [command_type.value for command_type in NeuralCommandType],
            "capabilities": [
                "Brain-Computer Interface",
                "Neural Signal Processing",
                "Real-time Pattern Recognition",
                "Neural Command Generation",
                "Direct Neural Communication",
                "Neural Data Analysis",
                "Artifact Detection",
                "Signal Quality Assessment"
            ]
        },
        "combined_capabilities": [
            "Quantum-Enhanced Neural Processing",
            "Neural-Quantum Hybrid Computing",
            "Brain-Quantum Interface",
            "Quantum Neural Networks",
            "Neural Quantum Optimization",
            "Quantum Brain Simulation",
            "Neural Quantum Machine Learning",
            "Quantum Consciousness Interface"
        ],
        "timestamp": datetime.now().isoformat()
    }


