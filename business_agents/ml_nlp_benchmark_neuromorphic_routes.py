"""
ML NLP Benchmark Neuromorphic Computing Routes
API routes for neuromorphic computing system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Union
import time
import json
import logging
from datetime import datetime

from ml_nlp_benchmark_neuromorphic_computing import (
    get_neuromorphic_computing,
    create_neuron,
    create_synapse,
    create_network,
    simulate_network,
    train_network,
    pattern_recognition,
    temporal_processing,
    get_neuromorphic_summary,
    clear_neuromorphic_data
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/neuromorphic", tags=["Neuromorphic Computing"])

# Dependency to get neuromorphic computing instance
def get_neuromorphic_computing_instance():
    return get_neuromorphic_computing()

@router.post("/neurons")
async def create_spiking_neuron(
    neuron_id: str,
    neuron_type: str,
    membrane_potential: float = 0.0,
    threshold: float = 1.0,
    reset_potential: float = 0.0,
    refractory_period: int = 0,
    weights: Optional[Dict[str, float]] = None,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Create a spiking neuron"""
    try:
        neuron_id = create_neuron(
            neuron_id, neuron_type, membrane_potential, 
            threshold, reset_potential, refractory_period, weights
        )
        return {
            "success": True,
            "neuron_id": neuron_id,
            "message": f"Spiking neuron '{neuron_id}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating spiking neuron: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synapses")
async def create_neural_synapse(
    synapse_id: str,
    pre_neuron: str,
    post_neuron: str,
    weight: float = 0.1,
    delay: int = 1,
    plasticity_rule: str = "stdp",
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Create a neural synapse"""
    try:
        synapse_id = create_synapse(synapse_id, pre_neuron, post_neuron, weight, delay, plasticity_rule)
        return {
            "success": True,
            "synapse_id": synapse_id,
            "message": f"Neural synapse '{synapse_id}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating neural synapse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/networks")
async def create_neuromorphic_network(
    name: str,
    neurons: List[str],
    synapses: List[str],
    topology: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Create a neuromorphic network"""
    try:
        network_id = create_network(name, neurons, synapses, topology, parameters)
        return {
            "success": True,
            "network_id": network_id,
            "message": f"Neuromorphic network '{name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating neuromorphic network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/networks/{network_id}/simulate")
async def simulate_neuromorphic_network(
    network_id: str,
    input_spikes: List[Dict[str, Any]],
    simulation_time: int = 1000,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Simulate a neuromorphic network"""
    try:
        result = simulate_network(network_id, input_spikes, simulation_time)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "network_id": result.network_id,
                "spikes": result.spikes,
                "firing_rates": result.firing_rates,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error simulating neuromorphic network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/networks/{network_id}/train")
async def train_neuromorphic_network(
    network_id: str,
    training_data: List[Dict[str, Any]],
    epochs: int = 100,
    learning_rate: float = 0.01,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Train a neuromorphic network"""
    try:
        result = train_network(network_id, training_data, epochs, learning_rate)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "network_id": result.network_id,
                "spikes": result.spikes,
                "firing_rates": result.firing_rates,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error training neuromorphic network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/networks/{network_id}/pattern-recognition")
async def run_pattern_recognition(
    network_id: str,
    input_pattern: List[float],
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Run pattern recognition with neuromorphic network"""
    try:
        result = pattern_recognition(network_id, input_pattern)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "network_id": result.network_id,
                "spikes": result.spikes,
                "firing_rates": result.firing_rates,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running pattern recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/networks/{network_id}/temporal-processing")
async def run_temporal_processing(
    network_id: str,
    temporal_sequence: List[float],
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Run temporal processing with neuromorphic network"""
    try:
        result = temporal_processing(network_id, temporal_sequence)
        return {
            "success": True,
            "result": {
                "result_id": result.result_id,
                "network_id": result.network_id,
                "spikes": result.spikes,
                "firing_rates": result.firing_rates,
                "processing_time": result.processing_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error running temporal processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neurons")
async def list_spiking_neurons(
    neuron_type: Optional[str] = None,
    active_only: bool = False,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """List spiking neurons"""
    try:
        neurons = neuromorphic_computing.list_neurons(neuron_type=neuron_type, active_only=active_only)
        return {
            "success": True,
            "neurons": [
                {
                    "neuron_id": neuron.neuron_id,
                    "neuron_type": neuron.neuron_type,
                    "membrane_potential": neuron.membrane_potential,
                    "threshold": neuron.threshold,
                    "reset_potential": neuron.reset_potential,
                    "refractory_period": neuron.refractory_period,
                    "weights": neuron.weights,
                    "created_at": neuron.created_at.isoformat(),
                    "is_active": neuron.is_active
                }
                for neuron in neurons
            ]
        }
    except Exception as e:
        logger.error(f"Error listing spiking neurons: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neurons/{neuron_id}")
async def get_spiking_neuron(
    neuron_id: str,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Get spiking neuron information"""
    try:
        neuron = neuromorphic_computing.get_neuron(neuron_id)
        if not neuron:
            raise HTTPException(status_code=404, detail="Neuron not found")
        
        return {
            "success": True,
            "neuron": {
                "neuron_id": neuron.neuron_id,
                "neuron_type": neuron.neuron_type,
                "membrane_potential": neuron.membrane_potential,
                "threshold": neuron.threshold,
                "reset_potential": neuron.reset_potential,
                "refractory_period": neuron.refractory_period,
                "last_spike_time": neuron.last_spike_time,
                "weights": neuron.weights,
                "created_at": neuron.created_at.isoformat(),
                "last_updated": neuron.last_updated.isoformat(),
                "is_active": neuron.is_active,
                "metadata": neuron.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting spiking neuron: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/synapses")
async def list_neural_synapses(
    plasticity_rule: Optional[str] = None,
    active_only: bool = False,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """List neural synapses"""
    try:
        synapses = neuromorphic_computing.list_synapses(plasticity_rule=plasticity_rule, active_only=active_only)
        return {
            "success": True,
            "synapses": [
                {
                    "synapse_id": synapse.synapse_id,
                    "pre_neuron": synapse.pre_neuron,
                    "post_neuron": synapse.post_neuron,
                    "weight": synapse.weight,
                    "delay": synapse.delay,
                    "plasticity_rule": synapse.plasticity_rule,
                    "created_at": synapse.created_at.isoformat(),
                    "is_active": synapse.is_active
                }
                for synapse in synapses
            ]
        }
    except Exception as e:
        logger.error(f"Error listing neural synapses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/synapses/{synapse_id}")
async def get_neural_synapse(
    synapse_id: str,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Get neural synapse information"""
    try:
        synapse = neuromorphic_computing.get_synapse(synapse_id)
        if not synapse:
            raise HTTPException(status_code=404, detail="Synapse not found")
        
        return {
            "success": True,
            "synapse": {
                "synapse_id": synapse.synapse_id,
                "pre_neuron": synapse.pre_neuron,
                "post_neuron": synapse.post_neuron,
                "weight": synapse.weight,
                "delay": synapse.delay,
                "plasticity_rule": synapse.plasticity_rule,
                "created_at": synapse.created_at.isoformat(),
                "last_updated": synapse.last_updated.isoformat(),
                "is_active": synapse.is_active,
                "metadata": synapse.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting neural synapse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/networks")
async def list_neuromorphic_networks(
    active_only: bool = False,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """List neuromorphic networks"""
    try:
        networks = neuromorphic_computing.list_networks(active_only=active_only)
        return {
            "success": True,
            "networks": [
                {
                    "network_id": network.network_id,
                    "name": network.name,
                    "neurons": network.neurons,
                    "synapses": network.synapses,
                    "topology": network.topology,
                    "parameters": network.parameters,
                    "created_at": network.created_at.isoformat(),
                    "is_active": network.is_active
                }
                for network in networks
            ]
        }
    except Exception as e:
        logger.error(f"Error listing neuromorphic networks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/networks/{network_id}")
async def get_neuromorphic_network(
    network_id: str,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Get neuromorphic network information"""
    try:
        network = neuromorphic_computing.get_network(network_id)
        if not network:
            raise HTTPException(status_code=404, detail="Network not found")
        
        return {
            "success": True,
            "network": {
                "network_id": network.network_id,
                "name": network.name,
                "neurons": network.neurons,
                "synapses": network.synapses,
                "topology": network.topology,
                "parameters": network.parameters,
                "created_at": network.created_at.isoformat(),
                "last_updated": network.last_updated.isoformat(),
                "is_active": network.is_active,
                "metadata": network.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting neuromorphic network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_neuromorphic_results(
    network_id: Optional[str] = None,
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Get neuromorphic results"""
    try:
        results = neuromorphic_computing.get_spiking_results(network_id=network_id)
        return {
            "success": True,
            "results": [
                {
                    "result_id": result.result_id,
                    "network_id": result.network_id,
                    "spikes": result.spikes,
                    "firing_rates": result.firing_rates,
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
        logger.error(f"Error getting neuromorphic results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_neuromorphic_summary_endpoint(
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Get neuromorphic computing system summary"""
    try:
        summary = get_neuromorphic_summary()
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error getting neuromorphic summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
async def clear_neuromorphic_data_endpoint(
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Clear all neuromorphic computing data"""
    try:
        clear_neuromorphic_data()
        return {
            "success": True,
            "message": "All neuromorphic computing data cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing neuromorphic data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_neuromorphic_capabilities(
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Get neuromorphic computing capabilities"""
    try:
        capabilities = neuromorphic_computing.neuromorphic_capabilities
        neuron_types = list(neuromorphic_computing.neuron_types.keys())
        plasticity_rules = list(neuromorphic_computing.plasticity_rules.keys())
        network_topologies = list(neuromorphic_computing.network_topologies.keys())
        neuromorphic_chips = list(neuromorphic_computing.neuromorphic_chips.keys())
        
        return {
            "success": True,
            "capabilities": {
                "neuromorphic_capabilities": capabilities,
                "neuron_types": neuron_types,
                "plasticity_rules": plasticity_rules,
                "network_topologies": network_topologies,
                "neuromorphic_chips": neuromorphic_chips
            }
        }
    except Exception as e:
        logger.error(f"Error getting neuromorphic capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def neuromorphic_health_check(
    neuromorphic_computing = Depends(get_neuromorphic_computing_instance)
):
    """Neuromorphic computing system health check"""
    try:
        summary = get_neuromorphic_summary()
        health_status = "healthy" if summary["total_neurons"] >= 0 else "unhealthy"
        
        return {
            "success": True,
            "health": {
                "status": health_status,
                "total_neurons": summary["total_neurons"],
                "total_synapses": summary["total_synapses"],
                "total_networks": summary["total_networks"],
                "total_results": summary["total_results"],
                "active_neurons": summary["active_neurons"],
                "active_synapses": summary["active_synapses"],
                "active_networks": summary["active_networks"]
            }
        }
    except Exception as e:
        logger.error(f"Error in neuromorphic health check: {e}")
        return {
            "success": False,
            "health": {
                "status": "unhealthy",
                "error": str(e)
            }
        }











