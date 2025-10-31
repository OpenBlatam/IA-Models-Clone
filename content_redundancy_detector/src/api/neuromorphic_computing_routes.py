"""
Neuromorphic Computing API Routes - Advanced neuromorphic and brain-inspired computing endpoints
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.neuromorphic_computing_engine import (
    get_neuromorphic_computing_engine, NeuromorphicConfig, 
    SpikingNeuralNetwork, NeuromorphicProcessor, Neuron, Synapse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/neuromorphic-computing", tags=["Neuromorphic Computing"])


# Request/Response Models
class SpikingNetworkRequest(BaseModel):
    """Spiking neural network creation request model"""
    name: str = Field(..., description="Network name", min_length=1)
    architecture: str = Field(default="feedforward", description="Network architecture")
    num_neurons: int = Field(default=100, description="Number of neurons", gt=0)
    num_synapses: int = Field(default=1000, description="Number of synapses", gt=0)
    input_layer: List[str] = Field(default=[], description="Input layer neuron IDs")
    output_layer: List[str] = Field(default=[], description="Output layer neuron IDs")
    hidden_layers: List[List[str]] = Field(default=[], description="Hidden layer neuron IDs")
    learning_algorithm: str = Field(default="stdp", description="Learning algorithm")
    plasticity_enabled: bool = Field(default=True, description="Enable plasticity")


class NetworkSimulationRequest(BaseModel):
    """Network simulation request model"""
    network_id: str = Field(..., description="Network ID", min_length=1)
    simulation_time: float = Field(..., description="Simulation time in seconds", gt=0)
    input_data: Optional[Dict[str, Any]] = Field(default=None, description="Input data for simulation")
    dt: Optional[float] = Field(default=0.001, description="Time step", gt=0)


class QuantumNeuromorphicRequest(BaseModel):
    """Quantum neuromorphic processor request model"""
    name: str = Field(..., description="Processor name", min_length=1)
    num_qubits: int = Field(default=10, description="Number of qubits", gt=0)
    quantum_gates: List[str] = Field(default=["hadamard", "pauli_x", "pauli_y", "pauli_z"], description="Available quantum gates")
    entanglement_strength: float = Field(default=1.0, description="Entanglement strength", gt=0)
    coherence_time: float = Field(default=100.0, description="Coherence time in microseconds", gt=0)


class NeuronUpdateRequest(BaseModel):
    """Neuron update request model"""
    neuron_id: str = Field(..., description="Neuron ID", min_length=1)
    network_id: str = Field(..., description="Network ID", min_length=1)
    membrane_potential: Optional[float] = Field(default=None, description="Membrane potential")
    threshold_potential: Optional[float] = Field(default=None, description="Threshold potential")
    input_current: Optional[float] = Field(default=None, description="Input current")
    synaptic_weights: Optional[Dict[str, float]] = Field(default=None, description="Synaptic weights")


class SynapseUpdateRequest(BaseModel):
    """Synapse update request model"""
    synapse_id: str = Field(..., description="Synapse ID", min_length=1)
    network_id: str = Field(..., description="Network ID", min_length=1)
    weight: Optional[float] = Field(default=None, description="Synapse weight")
    delay: Optional[float] = Field(default=None, description="Synaptic delay")
    plasticity_type: Optional[str] = Field(default=None, description="Plasticity type")
    learning_rate: Optional[float] = Field(default=None, description="Learning rate")


class NeuromorphicConfigRequest(BaseModel):
    """Neuromorphic computing configuration request model"""
    enable_spiking_neural_networks: bool = Field(default=True, description="Enable spiking neural networks")
    enable_memristive_computing: bool = Field(default=True, description="Enable memristive computing")
    enable_photonic_computing: bool = Field(default=True, description="Enable photonic computing")
    enable_quantum_neuromorphic: bool = Field(default=True, description="Enable quantum neuromorphic")
    enable_brain_computer_interface: bool = Field(default=True, description="Enable brain-computer interface")
    enable_neural_morphology: bool = Field(default=True, description="Enable neural morphology")
    enable_synaptic_plasticity: bool = Field(default=True, description="Enable synaptic plasticity")
    enable_adaptive_learning: bool = Field(default=True, description="Enable adaptive learning")
    enable_event_driven_processing: bool = Field(default=True, description="Enable event-driven processing")
    enable_energy_efficient_computing: bool = Field(default=True, description="Enable energy-efficient computing")
    enable_real_time_processing: bool = Field(default=True, description="Enable real-time processing")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    enable_self_organizing_networks: bool = Field(default=True, description="Enable self-organizing networks")
    enable_evolutionary_algorithms: bool = Field(default=True, description="Enable evolutionary algorithms")
    enable_swarm_intelligence: bool = Field(default=True, description="Enable swarm intelligence")
    max_neurons: int = Field(default=1000000, description="Maximum neurons", gt=0)
    max_synapses: int = Field(default=10000000, description="Maximum synapses", gt=0)
    simulation_time_step: float = Field(default=0.001, description="Simulation time step", gt=0)
    membrane_time_constant: float = Field(default=20.0, description="Membrane time constant", gt=0)
    synaptic_delay: float = Field(default=1.0, description="Synaptic delay", gt=0)
    refractory_period: float = Field(default=2.0, description="Refractory period", gt=0)
    threshold_potential: float = Field(default=-50.0, description="Threshold potential")
    resting_potential: float = Field(default=-70.0, description="Resting potential")
    reset_potential: float = Field(default=-65.0, description="Reset potential")
    learning_rate: float = Field(default=0.01, description="Learning rate", gt=0)
    plasticity_window: float = Field(default=20.0, description="Plasticity window", gt=0)
    enable_stdp: bool = Field(default=True, description="Enable STDP")
    enable_homeostatic_plasticity: bool = Field(default=True, description="Enable homeostatic plasticity")
    enable_structural_plasticity: bool = Field(default=True, description="Enable structural plasticity")
    enable_meta_plasticity: bool = Field(default=True, description="Enable meta plasticity")
    enable_compartmental_models: bool = Field(default=True, description="Enable compartmental models")
    enable_ion_channels: bool = Field(default=True, description="Enable ion channels")
    enable_gap_junctions: bool = Field(default=True, description="Enable gap junctions")
    enable_astrocytes: bool = Field(default=True, description="Enable astrocytes")
    enable_glial_networks: bool = Field(default=True, description="Enable glial networks")


# Dependency to get neuromorphic computing engine
async def get_neuromorphic_engine():
    """Get neuromorphic computing engine dependency"""
    engine = await get_neuromorphic_computing_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Neuromorphic Computing Engine not available")
    return engine


# Neuromorphic Computing Routes
@router.post("/create-spiking-network", response_model=Dict[str, Any])
async def create_spiking_network(
    request: SpikingNetworkRequest,
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Create a spiking neural network"""
    try:
        start_time = time.time()
        
        # Create spiking network
        network = await engine.create_spiking_network({
            "name": request.name,
            "architecture": request.architecture,
            "num_neurons": request.num_neurons,
            "num_synapses": request.num_synapses,
            "input_layer": request.input_layer,
            "output_layer": request.output_layer,
            "hidden_layers": request.hidden_layers,
            "learning_algorithm": request.learning_algorithm,
            "plasticity_enabled": request.plasticity_enabled
        })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "spiking_network": {
                "network_id": network.network_id,
                "timestamp": network.timestamp.isoformat(),
                "name": network.name,
                "architecture": network.architecture,
                "num_neurons": len(network.neurons),
                "num_synapses": len(network.synapses),
                "input_layer": network.input_layer,
                "output_layer": network.output_layer,
                "hidden_layers": network.hidden_layers,
                "learning_algorithm": network.learning_algorithm,
                "plasticity_enabled": network.plasticity_enabled,
                "status": network.status
            },
            "processing_time_ms": processing_time,
            "message": f"Spiking neural network {request.name} created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating spiking network: {e}")
        raise HTTPException(status_code=500, detail=f"Network creation failed: {str(e)}")


@router.post("/simulate-network", response_model=Dict[str, Any])
async def simulate_network(
    request: NetworkSimulationRequest,
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Simulate a spiking neural network"""
    try:
        start_time = time.time()
        
        # Simulate network
        simulation_results = await engine.simulate_network(
            network_id=request.network_id,
            simulation_time=request.simulation_time,
            input_data=request.input_data
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "simulation_results": simulation_results,
            "processing_time_ms": processing_time,
            "message": f"Network simulation completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error simulating network: {e}")
        raise HTTPException(status_code=500, detail=f"Network simulation failed: {str(e)}")


@router.post("/create-quantum-processor", response_model=Dict[str, Any])
async def create_quantum_neuromorphic_processor(
    request: QuantumNeuromorphicRequest,
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Create a quantum neuromorphic processor"""
    try:
        start_time = time.time()
        
        # Create quantum neuromorphic processor
        processor = await engine.create_quantum_neuromorphic_processor({
            "name": request.name,
            "num_qubits": request.num_qubits,
            "quantum_gates": request.quantum_gates,
            "entanglement_strength": request.entanglement_strength,
            "coherence_time": request.coherence_time
        })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "quantum_processor": {
                "processor_id": processor.processor_id,
                "timestamp": datetime.now().isoformat(),
                "name": request.name,
                "num_qubits": request.num_qubits,
                "quantum_gates": request.quantum_gates,
                "entanglement_strength": request.entanglement_strength,
                "coherence_time": request.coherence_time,
                "quantum_volume": processor.quantum_volume,
                "gate_fidelity": processor.gate_fidelity
            },
            "processing_time_ms": processing_time,
            "message": f"Quantum neuromorphic processor {request.name} created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating quantum processor: {e}")
        raise HTTPException(status_code=500, detail=f"Quantum processor creation failed: {str(e)}")


@router.post("/update-neuron", response_model=Dict[str, Any])
async def update_neuron(
    request: NeuronUpdateRequest,
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Update neuron parameters"""
    try:
        start_time = time.time()
        
        # Check if network exists
        if request.network_id not in engine.networks:
            raise ValueError(f"Network {request.network_id} not found")
        
        network = engine.networks[request.network_id]
        
        # Check if neuron exists
        if request.neuron_id not in network.neurons:
            raise ValueError(f"Neuron {request.neuron_id} not found")
        
        neuron = network.neurons[request.neuron_id]
        
        # Update neuron parameters
        if request.membrane_potential is not None:
            neuron.membrane_potential = request.membrane_potential
        if request.threshold_potential is not None:
            neuron.threshold_potential = request.threshold_potential
        if request.input_current is not None:
            neuron.input_current = request.input_current
        if request.synaptic_weights is not None:
            neuron.synaptic_weights.update(request.synaptic_weights)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "neuron_update": {
                "neuron_id": request.neuron_id,
                "network_id": request.network_id,
                "membrane_potential": neuron.membrane_potential,
                "threshold_potential": neuron.threshold_potential,
                "input_current": neuron.input_current,
                "synaptic_weights": neuron.synaptic_weights,
                "timestamp": datetime.now().isoformat()
            },
            "processing_time_ms": processing_time,
            "message": f"Neuron {request.neuron_id} updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating neuron: {e}")
        raise HTTPException(status_code=500, detail=f"Neuron update failed: {str(e)}")


@router.post("/update-synapse", response_model=Dict[str, Any])
async def update_synapse(
    request: SynapseUpdateRequest,
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Update synapse parameters"""
    try:
        start_time = time.time()
        
        # Check if network exists
        if request.network_id not in engine.networks:
            raise ValueError(f"Network {request.network_id} not found")
        
        network = engine.networks[request.network_id]
        
        # Check if synapse exists
        if request.synapse_id not in network.synapses:
            raise ValueError(f"Synapse {request.synapse_id} not found")
        
        synapse = network.synapses[request.synapse_id]
        
        # Update synapse parameters
        if request.weight is not None:
            synapse.weight = request.weight
        if request.delay is not None:
            synapse.delay = request.delay
        if request.plasticity_type is not None:
            synapse.plasticity_type = request.plasticity_type
        if request.learning_rate is not None:
            synapse.learning_rate = request.learning_rate
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "synapse_update": {
                "synapse_id": request.synapse_id,
                "network_id": request.network_id,
                "weight": synapse.weight,
                "delay": synapse.delay,
                "plasticity_type": synapse.plasticity_type,
                "learning_rate": synapse.learning_rate,
                "timestamp": datetime.now().isoformat()
            },
            "processing_time_ms": processing_time,
            "message": f"Synapse {request.synapse_id} updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating synapse: {e}")
        raise HTTPException(status_code=500, detail=f"Synapse update failed: {str(e)}")


@router.get("/networks", response_model=Dict[str, Any])
async def get_spiking_networks(
    status: Optional[str] = None,
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Get spiking neural networks"""
    try:
        # Get networks
        networks = list(engine.networks.values())
        
        # Filter by status if provided
        if status:
            networks = [n for n in networks if n.status == status]
        
        # Format networks
        formatted_networks = []
        for network in networks:
            formatted_networks.append({
                "network_id": network.network_id,
                "timestamp": network.timestamp.isoformat(),
                "name": network.name,
                "architecture": network.architecture,
                "num_neurons": len(network.neurons),
                "num_synapses": len(network.synapses),
                "input_layer": network.input_layer,
                "output_layer": network.output_layer,
                "hidden_layers": network.hidden_layers,
                "simulation_time": network.simulation_time,
                "total_spikes": network.total_spikes,
                "firing_rate": network.firing_rate,
                "energy_consumption": network.energy_consumption,
                "learning_algorithm": network.learning_algorithm,
                "plasticity_enabled": network.plasticity_enabled,
                "status": network.status
            })
        
        return {
            "success": True,
            "spiking_networks": formatted_networks,
            "total_count": len(formatted_networks),
            "filter": {"status": status},
            "message": "Spiking neural networks retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting spiking networks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get spiking networks: {str(e)}")


@router.get("/processors", response_model=Dict[str, Any])
async def get_neuromorphic_processors(
    status: Optional[str] = None,
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Get neuromorphic processors"""
    try:
        # Get processors
        processors = list(engine.processors.values())
        
        # Filter by status if provided
        if status:
            processors = [p for p in processors if p.status == status]
        
        # Format processors
        formatted_processors = []
        for processor in processors:
            formatted_processors.append({
                "processor_id": processor.processor_id,
                "timestamp": processor.timestamp.isoformat(),
                "name": processor.name,
                "processor_type": processor.processor_type,
                "max_neurons": processor.max_neurons,
                "max_synapses": processor.max_synapses,
                "clock_frequency": processor.clock_frequency,
                "power_consumption": processor.power_consumption,
                "energy_per_spike": processor.energy_per_spike,
                "latency": processor.latency,
                "throughput": processor.throughput,
                "memory_bandwidth": processor.memory_bandwidth,
                "precision": processor.precision,
                "temperature": processor.temperature,
                "status": processor.status,
                "capabilities": processor.capabilities
            })
        
        return {
            "success": True,
            "neuromorphic_processors": formatted_processors,
            "total_count": len(formatted_processors),
            "filter": {"status": status},
            "message": "Neuromorphic processors retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting neuromorphic processors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get neuromorphic processors: {str(e)}")


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_neuromorphic_capabilities(
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Get neuromorphic computing capabilities"""
    try:
        # Get capabilities
        capabilities = await engine.get_neuromorphic_capabilities()
        
        return {
            "success": True,
            "neuromorphic_capabilities": capabilities,
            "message": "Neuromorphic computing capabilities retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting neuromorphic capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get neuromorphic capabilities: {str(e)}")


@router.get("/performance-metrics", response_model=Dict[str, Any])
async def get_neuromorphic_performance_metrics(
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Get neuromorphic computing performance metrics"""
    try:
        # Get performance metrics
        metrics = await engine.get_neuromorphic_performance_metrics()
        
        return {
            "success": True,
            "performance_metrics": metrics,
            "message": "Neuromorphic computing performance metrics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting neuromorphic performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get neuromorphic performance metrics: {str(e)}")


@router.post("/configure", response_model=Dict[str, Any])
async def configure_neuromorphic_computing(
    request: NeuromorphicConfigRequest,
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Configure neuromorphic computing settings"""
    try:
        # Update configuration
        config = NeuromorphicConfig(
            enable_spiking_neural_networks=request.enable_spiking_neural_networks,
            enable_memristive_computing=request.enable_memristive_computing,
            enable_photonic_computing=request.enable_photonic_computing,
            enable_quantum_neuromorphic=request.enable_quantum_neuromorphic,
            enable_brain_computer_interface=request.enable_brain_computer_interface,
            enable_neural_morphology=request.enable_neural_morphology,
            enable_synaptic_plasticity=request.enable_synaptic_plasticity,
            enable_adaptive_learning=request.enable_adaptive_learning,
            enable_event_driven_processing=request.enable_event_driven_processing,
            enable_energy_efficient_computing=request.enable_energy_efficient_computing,
            enable_real_time_processing=request.enable_real_time_processing,
            enable_parallel_processing=request.enable_parallel_processing,
            enable_self_organizing_networks=request.enable_self_organizing_networks,
            enable_evolutionary_algorithms=request.enable_evolutionary_algorithms,
            enable_swarm_intelligence=request.enable_swarm_intelligence,
            max_neurons=request.max_neurons,
            max_synapses=request.max_synapses,
            simulation_time_step=request.simulation_time_step,
            membrane_time_constant=request.membrane_time_constant,
            synaptic_delay=request.synaptic_delay,
            refractory_period=request.refractory_period,
            threshold_potential=request.threshold_potential,
            resting_potential=request.resting_potential,
            reset_potential=request.reset_potential,
            learning_rate=request.learning_rate,
            plasticity_window=request.plasticity_window,
            enable_stdp=request.enable_stdp,
            enable_homeostatic_plasticity=request.enable_homeostatic_plasticity,
            enable_structural_plasticity=request.enable_structural_plasticity,
            enable_meta_plasticity=request.enable_meta_plasticity,
            enable_compartmental_models=request.enable_compartmental_models,
            enable_ion_channels=request.enable_ion_channels,
            enable_gap_junctions=request.enable_gap_junctions,
            enable_astrocytes=request.enable_astrocytes,
            enable_glial_networks=request.enable_glial_networks
        )
        
        # Update engine configuration
        engine.config = config
        
        return {
            "success": True,
            "configuration": {
                "enable_spiking_neural_networks": config.enable_spiking_neural_networks,
                "enable_memristive_computing": config.enable_memristive_computing,
                "enable_photonic_computing": config.enable_photonic_computing,
                "enable_quantum_neuromorphic": config.enable_quantum_neuromorphic,
                "enable_brain_computer_interface": config.enable_brain_computer_interface,
                "enable_neural_morphology": config.enable_neural_morphology,
                "enable_synaptic_plasticity": config.enable_synaptic_plasticity,
                "enable_adaptive_learning": config.enable_adaptive_learning,
                "enable_event_driven_processing": config.enable_event_driven_processing,
                "enable_energy_efficient_computing": config.enable_energy_efficient_computing,
                "enable_real_time_processing": config.enable_real_time_processing,
                "enable_parallel_processing": config.enable_parallel_processing,
                "enable_self_organizing_networks": config.enable_self_organizing_networks,
                "enable_evolutionary_algorithms": config.enable_evolutionary_algorithms,
                "enable_swarm_intelligence": config.enable_swarm_intelligence,
                "max_neurons": config.max_neurons,
                "max_synapses": config.max_synapses,
                "simulation_time_step": config.simulation_time_step,
                "membrane_time_constant": config.membrane_time_constant,
                "synaptic_delay": config.synaptic_delay,
                "refractory_period": config.refractory_period,
                "threshold_potential": config.threshold_potential,
                "resting_potential": config.resting_potential,
                "reset_potential": config.reset_potential,
                "learning_rate": config.learning_rate,
                "plasticity_window": config.plasticity_window,
                "enable_stdp": config.enable_stdp,
                "enable_homeostatic_plasticity": config.enable_homeostatic_plasticity,
                "enable_structural_plasticity": config.enable_structural_plasticity,
                "enable_meta_plasticity": config.enable_meta_plasticity,
                "enable_compartmental_models": config.enable_compartmental_models,
                "enable_ion_channels": config.enable_ion_channels,
                "enable_gap_junctions": config.enable_gap_junctions,
                "enable_astrocytes": config.enable_astrocytes,
                "enable_glial_networks": config.enable_glial_networks
            },
            "message": "Neuromorphic computing configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error configuring neuromorphic computing: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/neuron-types", response_model=Dict[str, Any])
async def get_neuron_types():
    """Get available neuron types"""
    try:
        neuron_types = {
            "excitatory": {
                "name": "Excitatory Neuron",
                "description": "Neuron that increases membrane potential of connected neurons",
                "characteristics": ["Positive synaptic weights", "Glutamate neurotransmitter", "Pyramidal cells"],
                "use_cases": ["Information processing", "Learning", "Memory formation"]
            },
            "inhibitory": {
                "name": "Inhibitory Neuron",
                "description": "Neuron that decreases membrane potential of connected neurons",
                "characteristics": ["Negative synaptic weights", "GABA neurotransmitter", "Interneurons"],
                "use_cases": ["Noise reduction", "Oscillation control", "Competitive learning"]
            },
            "modulatory": {
                "name": "Modulatory Neuron",
                "description": "Neuron that modulates activity of other neurons",
                "characteristics": ["Variable synaptic weights", "Multiple neurotransmitters", "Wide projections"],
                "use_cases": ["Attention", "Arousal", "Learning modulation"]
            }
        }
        
        return {
            "success": True,
            "neuron_types": neuron_types,
            "message": "Neuron types retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting neuron types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get neuron types: {str(e)}")


@router.get("/plasticity-types", response_model=Dict[str, Any])
async def get_plasticity_types():
    """Get available plasticity types"""
    try:
        plasticity_types = {
            "stdp": {
                "name": "Spike-Timing Dependent Plasticity",
                "description": "Synaptic strength changes based on spike timing",
                "mechanism": "Pre-before-post increases weight, post-before-pre decreases weight",
                "use_cases": ["Temporal learning", "Sequence learning", "Causal inference"]
            },
            "hebbian": {
                "name": "Hebbian Plasticity",
                "description": "Neurons that fire together, wire together",
                "mechanism": "Simultaneous activity strengthens connections",
                "use_cases": ["Pattern recognition", "Associative learning", "Feature extraction"]
            },
            "anti_hebbian": {
                "name": "Anti-Hebbian Plasticity",
                "description": "Neurons that fire together, unwire together",
                "mechanism": "Simultaneous activity weakens connections",
                "use_cases": ["Competitive learning", "Sparse coding", "Noise reduction"]
            },
            "homeostatic": {
                "name": "Homeostatic Plasticity",
                "description": "Maintains network stability and activity levels",
                "mechanism": "Scales synaptic weights to maintain target activity",
                "use_cases": ["Network stability", "Activity regulation", "Preventing runaway excitation"]
            }
        }
        
        return {
            "success": True,
            "plasticity_types": plasticity_types,
            "message": "Plasticity types retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting plasticity types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get plasticity types: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    engine: NeuromorphicComputingEngine = Depends(get_neuromorphic_engine)
):
    """Neuromorphic Computing Engine health check"""
    try:
        # Check engine components
        components_status = {
            "stdp_plasticity": engine.stdp_plasticity is not None,
            "networks": len(engine.networks) > 0,
            "processors": len(engine.processors) > 0
        }
        
        # Get capabilities
        capabilities = await engine.get_neuromorphic_capabilities()
        
        # Get performance metrics
        metrics = await engine.get_neuromorphic_performance_metrics()
        
        # Determine overall health
        all_healthy = all(components_status.values())
        
        overall_health = "healthy" if all_healthy else "degraded"
        
        return {
            "status": overall_health,
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "capabilities": capabilities,
            "performance_metrics": metrics,
            "message": "Neuromorphic Computing Engine is operational" if overall_health == "healthy" else "Some neuromorphic computing components may not be available"
        }
        
    except Exception as e:
        logger.error(f"Error in Neuromorphic Computing health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Neuromorphic Computing Engine health check failed"
        }

















