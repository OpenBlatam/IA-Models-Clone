"""
ML NLP Benchmark Biological Computing Routes
API routes for biological computing system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Union
import time
import json
import logging
from datetime import datetime
import numpy as np

from ml_nlp_benchmark_biological_computing import (
    get_biological_computing,
    create_biological_system,
    create_biological_process,
    simulate_biological_system,
    run_genetic_algorithm,
    run_particle_swarm_optimization,
    run_ant_colony_optimization,
    run_cellular_automaton,
    get_biological_summary,
    clear_biological_data
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/biological", tags=["Biological Computing"])

# Dependency to get biological computing instance
def get_biological_computing_instance():
    return get_biological_computing()

@router.post("/systems")
async def create_biological_system_endpoint(
    name: str,
    system_type: str,
    components: List[Dict[str, Any]],
    parameters: Optional[Dict[str, Any]] = None,
    biological_computing = Depends(get_biological_computing_instance)
):
    """Create a biological computing system"""
    try:
        system_id = create_biological_system(name, system_type, components, parameters)
        return {
            "success": True,
            "system_id": system_id,
            "message": f"Biological system '{name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating biological system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/processes")
async def create_biological_process_endpoint(
    name: str,
    process_type: str,
    input_molecules: List[str],
    output_molecules: List[str],
    enzymes: List[str],
    rate_constant: float = 1.0,
    biological_computing = Depends(get_biological_computing_instance)
):
    """Create a biological process"""
    try:
        process_id = create_biological_process(
            name, process_type, input_molecules, output_molecules, enzymes, rate_constant
        )
        return {
            "success": True,
            "process_id": process_id,
            "message": f"Biological process '{name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating biological process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/systems/{system_id}/simulate")
async def simulate_biological_system_endpoint(
    system_id: str,
    initial_conditions: Dict[str, float],
    simulation_time: float = 100.0,
    biological_computing = Depends(get_biological_computing_instance)
):
    """Simulate a biological system"""
    try:
        result = simulate_biological_system(system_id, initial_conditions, simulation_time)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "system_id": result.system_id,
                "process_results": result.process_results,
                "molecular_concentrations": result.molecular_concentrations,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error simulating biological system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/genetic-algorithm")
async def run_genetic_algorithm_endpoint(
    population_size: int,
    generations: int,
    fitness_function: str,
    parameters: Dict[str, Any],
    biological_computing = Depends(get_biological_computing_instance)
):
    """Run a genetic algorithm"""
    try:
        result = run_genetic_algorithm(population_size, generations, fitness_function, parameters)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "system_id": result.system_id,
                "process_results": result.process_results,
                "molecular_concentrations": result.molecular_concentrations,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running genetic algorithm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/particle-swarm-optimization")
async def run_particle_swarm_optimization_endpoint(
    swarm_size: int,
    dimensions: int,
    objective_function: str,
    parameters: Dict[str, Any],
    biological_computing = Depends(get_biological_computing_instance)
):
    """Run particle swarm optimization"""
    try:
        result = run_particle_swarm_optimization(swarm_size, dimensions, objective_function, parameters)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "system_id": result.system_id,
                "process_results": result.process_results,
                "molecular_concentrations": result.molecular_concentrations,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running particle swarm optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ant-colony-optimization")
async def run_ant_colony_optimization_endpoint(
    num_ants: int,
    num_nodes: int,
    distance_matrix: List[List[float]],
    parameters: Dict[str, Any],
    biological_computing = Depends(get_biological_computing_instance)
):
    """Run ant colony optimization"""
    try:
        distance_matrix_array = np.array(distance_matrix)
        result = run_ant_colony_optimization(num_ants, num_nodes, distance_matrix_array, parameters)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "system_id": result.system_id,
                "process_results": result.process_results,
                "molecular_concentrations": result.molecular_concentrations,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running ant colony optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cellular-automaton")
async def run_cellular_automaton_endpoint(
    grid_size: int,
    rules: Dict[str, Any],
    initial_state: List[List[int]],
    iterations: int,
    biological_computing = Depends(get_biological_computing_instance)
):
    """Run cellular automaton"""
    try:
        initial_state_array = np.array(initial_state)
        result = run_cellular_automaton(grid_size, rules, initial_state_array, iterations)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "system_id": result.system_id,
                "process_results": result.process_results,
                "molecular_concentrations": result.molecular_concentrations,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running cellular automaton: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/systems")
async def list_biological_systems(
    system_type: Optional[str] = None,
    active_only: bool = False,
    biological_computing = Depends(get_biological_computing_instance)
):
    """List biological systems"""
    try:
        systems = biological_computing.list_biological_systems(system_type=system_type, active_only=active_only)
        return {
            "success": True,
            "systems": [
                {
                    "system_id": system.system_id,
                    "name": system.name,
                    "system_type": system.system_type,
                    "components": system.components,
                    "parameters": system.parameters,
                    "created_at": system.created_at.isoformat(),
                    "is_active": system.is_active
                }
                for system in systems
            ]
        }
    except Exception as e:
        logger.error(f"Error listing biological systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/systems/{system_id}")
async def get_biological_system(
    system_id: str,
    biological_computing = Depends(get_biological_computing_instance)
):
    """Get biological system information"""
    try:
        system = biological_computing.get_biological_system(system_id)
        if not system:
            raise HTTPException(status_code=404, detail="Biological system not found")
        
        return {
            "success": True,
            "system": {
                "system_id": system.system_id,
                "name": system.name,
                "system_type": system.system_type,
                "components": system.components,
                "parameters": system.parameters,
                "created_at": system.created_at.isoformat(),
                "last_updated": system.last_updated.isoformat(),
                "is_active": system.is_active,
                "metadata": system.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting biological system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processes")
async def list_biological_processes(
    process_type: Optional[str] = None,
    active_only: bool = False,
    biological_computing = Depends(get_biological_computing_instance)
):
    """List biological processes"""
    try:
        processes = biological_computing.list_biological_processes(process_type=process_type, active_only=active_only)
        return {
            "success": True,
            "processes": [
                {
                    "process_id": process.process_id,
                    "name": process.name,
                    "process_type": process.process_type,
                    "input_molecules": process.input_molecules,
                    "output_molecules": process.output_molecules,
                    "enzymes": process.enzymes,
                    "rate_constant": process.rate_constant,
                    "created_at": process.created_at.isoformat(),
                    "is_active": process.is_active
                }
                for process in processes
            ]
        }
    except Exception as e:
        logger.error(f"Error listing biological processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processes/{process_id}")
async def get_biological_process(
    process_id: str,
    biological_computing = Depends(get_biological_computing_instance)
):
    """Get biological process information"""
    try:
        process = biological_computing.get_biological_process(process_id)
        if not process:
            raise HTTPException(status_code=404, detail="Biological process not found")
        
        return {
            "success": True,
            "process": {
                "process_id": process.process_id,
                "name": process.name,
                "process_type": process.process_type,
                "input_molecules": process.input_molecules,
                "output_molecules": process.output_molecules,
                "enzymes": process.enzymes,
                "rate_constant": process.rate_constant,
                "created_at": process.created_at.isoformat(),
                "last_updated": process.last_updated.isoformat(),
                "is_active": process.is_active,
                "metadata": process.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting biological process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_biological_results(
    system_id: Optional[str] = None,
    biological_computing = Depends(get_biological_computing_instance)
):
    """Get biological results"""
    try:
        results = biological_computing.get_biological_results(system_id=system_id)
        return {
            "success": True,
            "results": [
                {
                    "result_id": result.result_id,
                    "system_id": result.system_id,
                    "process_results": result.process_results,
                    "molecular_concentrations": result.molecular_concentrations,
                    "processing_time": result.processing_time,
                    "success": result.success,
                    "error_message": result.error_message,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error getting biological results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_biological_summary_endpoint(
    biological_computing = Depends(get_biological_computing_instance)
):
    """Get biological computing system summary"""
    try:
        summary = get_biological_summary()
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error getting biological summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
async def clear_biological_data_endpoint(
    biological_computing = Depends(get_biological_computing_instance)
):
    """Clear all biological computing data"""
    try:
        clear_biological_data()
        return {
            "success": True,
            "message": "All biological computing data cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing biological data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_biological_capabilities(
    biological_computing = Depends(get_biological_computing_instance)
):
    """Get biological computing capabilities"""
    try:
        capabilities = biological_computing.biological_capabilities
        system_types = list(biological_computing.biological_system_types.keys())
        process_types = list(biological_computing.biological_process_types.keys())
        molecular_components = list(biological_computing.molecular_components.keys())
        genetic_operators = list(biological_computing.genetic_operators.keys())
        swarm_algorithms = list(biological_computing.swarm_algorithms.keys())
        
        return {
            "success": True,
            "capabilities": {
                "biological_capabilities": capabilities,
                "system_types": system_types,
                "process_types": process_types,
                "molecular_components": molecular_components,
                "genetic_operators": genetic_operators,
                "swarm_algorithms": swarm_algorithms
            }
        }
    except Exception as e:
        logger.error(f"Error getting biological capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def biological_health_check(
    biological_computing = Depends(get_biological_computing_instance)
):
    """Biological computing system health check"""
    try:
        summary = get_biological_summary()
        health_status = "healthy" if summary["total_systems"] >= 0 else "unhealthy"
        
        return {
            "success": True,
            "health": {
                "status": health_status,
                "total_systems": summary["total_systems"],
                "total_processes": summary["total_processes"],
                "total_results": summary["total_results"],
                "active_systems": summary["active_systems"],
                "active_processes": summary["active_processes"]
            }
        }
    except Exception as e:
        logger.error(f"Error in biological health check: {e}")
        return {
            "success": False,
            "health": {
                "status": "unhealthy",
                "error": str(e)
            }
        }











