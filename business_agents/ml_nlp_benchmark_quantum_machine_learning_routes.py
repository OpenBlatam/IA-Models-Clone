"""
ML NLP Benchmark Quantum Machine Learning Routes
API routes for quantum machine learning system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ml_nlp_benchmark_quantum_machine_learning import (
    get_quantum_ml,
    create_quantum_ml_model,
    train_quantum_ml_model,
    predict_quantum_ml_model,
    quantum_classification,
    quantum_regression,
    quantum_clustering,
    quantum_optimization,
    quantum_feature_mapping,
    quantum_kernel_methods,
    quantum_neural_networks,
    quantum_support_vector_machines,
    quantum_principal_component_analysis,
    quantum_linear_algebra,
    get_quantum_ml_summary,
    clear_quantum_ml_data
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/quantum-ml", tags=["Quantum Machine Learning"])

@router.post("/models")
async def create_quantum_ml_model_endpoint(
    name: str,
    model_type: str,
    quantum_circuit: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None
):
    """Create a quantum ML model"""
    try:
        model_id = create_quantum_ml_model(name, model_type, quantum_circuit, parameters)
        return {"model_id": model_id, "message": "Quantum ML model created successfully"}
    except Exception as e:
        logger.error(f"Error creating quantum ML model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/train")
async def train_quantum_ml_model_endpoint(
    model_id: str,
    training_data: List[Dict[str, Any]],
    validation_data: Optional[List[Dict[str, Any]]] = None
):
    """Train a quantum ML model"""
    try:
        result = train_quantum_ml_model(model_id, training_data, validation_data)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error training quantum ML model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/predict")
async def predict_quantum_ml_model_endpoint(
    model_id: str,
    input_data: Any
):
    """Make predictions with quantum ML model"""
    try:
        result = predict_quantum_ml_model(model_id, input_data)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error predicting with quantum ML model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classification")
async def quantum_classification_endpoint(
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    num_classes: int = 2
):
    """Perform quantum classification"""
    try:
        result = quantum_classification(training_data, test_data, num_classes)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/regression")
async def quantum_regression_endpoint(
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]]
):
    """Perform quantum regression"""
    try:
        result = quantum_regression(training_data, test_data)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum regression: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clustering")
async def quantum_clustering_endpoint(
    data: List[Dict[str, Any]],
    num_clusters: int = 3
):
    """Perform quantum clustering"""
    try:
        result = quantum_clustering(data, num_clusters)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimization")
async def quantum_optimization_endpoint(
    problem_data: Dict[str, Any],
    optimization_type: str = "combinatorial"
):
    """Perform quantum optimization"""
    try:
        result = quantum_optimization(problem_data, optimization_type)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feature-mapping")
async def quantum_feature_mapping_endpoint(
    data: List[Dict[str, Any]],
    mapping_type: str = "pauli_feature_map"
):
    """Perform quantum feature mapping"""
    try:
        result = quantum_feature_mapping(data, mapping_type)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum feature mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kernel-methods")
async def quantum_kernel_methods_endpoint(
    data: List[Dict[str, Any]],
    kernel_type: str = "quantum_kernel"
):
    """Perform quantum kernel methods"""
    try:
        result = quantum_kernel_methods(data, kernel_type)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum kernel methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neural-networks")
async def quantum_neural_networks_endpoint(
    training_data: List[Dict[str, Any]],
    network_architecture: Dict[str, Any]
):
    """Perform quantum neural networks"""
    try:
        result = quantum_neural_networks(training_data, network_architecture)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum neural networks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/support-vector-machines")
async def quantum_support_vector_machines_endpoint(
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]]
):
    """Perform quantum support vector machines"""
    try:
        result = quantum_support_vector_machines(training_data, test_data)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum support vector machines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/principal-component-analysis")
async def quantum_principal_component_analysis_endpoint(
    data: List[Dict[str, Any]],
    num_components: int = 2
):
    """Perform quantum principal component analysis"""
    try:
        result = quantum_principal_component_analysis(data, num_components)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum principal component analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/linear-algebra")
async def quantum_linear_algebra_endpoint(
    matrix_data: Dict[str, Any],
    operation: str = "matrix_multiplication"
):
    """Perform quantum linear algebra"""
    try:
        result = quantum_linear_algebra(matrix_data, operation)
        return {
            "result_id": result.result_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "quantum_advantage": result.quantum_advantage,
            "quantum_fidelity": result.quantum_fidelity,
            "quantum_entanglement": result.quantum_entanglement,
            "quantum_ml_results": result.quantum_ml_results,
            "timestamp": result.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Error performing quantum linear algebra: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_quantum_ml_models(
    model_type: Optional[str] = None,
    active_only: bool = False
):
    """List quantum ML models"""
    try:
        quantum_ml_system = get_quantum_ml()
        models = quantum_ml_system.list_quantum_ml_models(model_type, active_only)
        return {
            "quantum_ml_models": [
                {
                    "model_id": model.model_id,
                    "name": model.name,
                    "model_type": model.model_type,
                    "quantum_circuit": model.quantum_circuit,
                    "is_active": model.is_active,
                    "created_at": model.created_at.isoformat(),
                    "last_updated": model.last_updated.isoformat()
                }
                for model in models
            ]
        }
    except Exception as e:
        logger.error(f"Error listing quantum ML models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}")
async def get_quantum_ml_model_endpoint(model_id: str):
    """Get quantum ML model information"""
    try:
        quantum_ml_system = get_quantum_ml()
        model = quantum_ml_system.get_quantum_ml_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Quantum ML model not found")
        
        return {
            "model_id": model.model_id,
            "name": model.name,
            "model_type": model.model_type,
            "quantum_circuit": model.quantum_circuit,
            "parameters": model.parameters,
            "is_active": model.is_active,
            "created_at": model.created_at.isoformat(),
            "last_updated": model.last_updated.isoformat(),
            "metadata": model.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quantum ML model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_quantum_ml_results(model_id: Optional[str] = None):
    """Get quantum ML results"""
    try:
        quantum_ml_system = get_quantum_ml()
        results = quantum_ml_system.get_quantum_ml_results(model_id)
        return {
            "results": [
                {
                    "result_id": result.result_id,
                    "model_id": result.model_id,
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "quantum_advantage": result.quantum_advantage,
                    "quantum_fidelity": result.quantum_fidelity,
                    "quantum_entanglement": result.quantum_entanglement,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error getting quantum ML results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_quantum_ml_summary_endpoint():
    """Get quantum ML system summary"""
    try:
        summary = get_quantum_ml_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting quantum ML summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_quantum_ml_capabilities():
    """Get quantum ML capabilities"""
    try:
        quantum_ml_system = get_quantum_ml()
        return {
            "quantum_ml_capabilities": quantum_ml_system.quantum_ml_capabilities,
            "quantum_ml_algorithms": list(quantum_ml_system.quantum_ml_algorithms.keys()),
            "quantum_feature_mappings": list(quantum_ml_system.quantum_feature_mappings.keys()),
            "quantum_optimizers": list(quantum_ml_system.quantum_optimizers.keys()),
            "quantum_metrics": list(quantum_ml_system.quantum_metrics.keys())
        }
    except Exception as e:
        logger.error(f"Error getting quantum ML capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def quantum_ml_health():
    """Get quantum ML system health"""
    try:
        quantum_ml_system = get_quantum_ml()
        summary = quantum_ml_system.get_quantum_ml_summary()
        
        return {
            "status": "healthy",
            "total_models": summary["total_models"],
            "active_models": summary["active_models"],
            "total_results": summary["total_results"],
            "recent_models": summary["recent_models"],
            "recent_results": summary["recent_results"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting quantum ML health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.delete("/clear")
async def clear_quantum_ml_data_endpoint():
    """Clear all quantum ML data"""
    try:
        clear_quantum_ml_data()
        return {"message": "Quantum ML data cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing quantum ML data: {e}")
        raise HTTPException(status_code=500, detail=str(e))











