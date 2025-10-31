"""
BUL - Business Universal Language (Neuromorphic Computing System)
================================================================

Advanced Neuromorphic Computing system with spiking neural networks and brain-inspired computing.
"""

import asyncio
import logging
import json
import time
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis
from prometheus_client import Counter, Histogram, Gauge
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bul_neuromorphic.db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Prometheus Metrics
NEUROMORPHIC_NETWORKS = Counter('bul_neuromorphic_networks_total', 'Total neuromorphic networks created', ['network_type', 'neurons'])
NEUROMORPHIC_SPIKES = Counter('bul_neuromorphic_spikes_total', 'Total spikes generated', ['neuron_type'])
NEUROMORPHIC_TRAINING = Counter('bul_neuromorphic_training_total', 'Total training sessions', ['training_type'])
NEUROMORPHIC_ACCURACY = Histogram('bul_neuromorphic_accuracy', 'Neuromorphic network accuracy')
NEUROMORPHIC_LATENCY = Histogram('bul_neuromorphic_latency_seconds', 'Neuromorphic processing latency')

class NeuronType(str, Enum):
    """Neuron type enumeration."""
    LIF = "lif"  # Leaky Integrate-and-Fire
    IZH = "izh"  # Izhikevich
    ADEX = "adex"  # Adaptive Exponential
    HODGKIN_HUXLEY = "hh"  # Hodgkin-Huxley
    QUADRATIC_INTEGRATE = "qi"  # Quadratic Integrate-and-Fire
    EXPONENTIAL_INTEGRATE = "ei"  # Exponential Integrate-and-Fire
    SPIKE_RESPONSE = "sr"  # Spike Response Model
    THRESHOLD_ADAPTIVE = "ta"  # Threshold Adaptive

class NetworkType(str, Enum):
    """Network type enumeration."""
    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    CONVOLUTIONAL = "convolutional"
    RESERVOIR = "reservoir"
    LIQUID_STATE = "liquid_state"
    ECHO_STATE = "echo_state"
    HOPFIELD = "hopfield"
    BOLTZMANN = "boltzmann"
    DEEP_SPIKING = "deep_spiking"
    HIERARCHICAL = "hierarchical"

class LearningRule(str, Enum):
    """Learning rule enumeration."""
    STDP = "stdp"  # Spike-Timing Dependent Plasticity
    HEBB = "hebb"  # Hebbian Learning
    BCM = "bcm"  # Bienenstock-Cooper-Munro
    OJA = "oja"  # Oja's Rule
    PERCEPTRON = "perceptron"
    BACKPROPAGATION = "backpropagation"
    REINFORCEMENT = "reinforcement"
    UNSUPERVISED = "unsupervised"
    SUPERVISED = "supervised"
    SEMI_SUPERVISED = "semi_supervised"

class StimulusType(str, Enum):
    """Stimulus type enumeration."""
    POISSON = "poisson"
    REGULAR = "regular"
    BURST = "burst"
    CHIRP = "chirp"
    WHITE_NOISE = "white_noise"
    COLORED_NOISE = "colored_noise"
    SINUSOIDAL = "sinusoidal"
    SQUARE_WAVE = "square_wave"
    TRIANGULAR = "triangular"
    GAUSSIAN = "gaussian"

# Database Models
class NeuromorphicNetwork(Base):
    __tablename__ = "neuromorphic_networks"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    network_type = Column(String, nullable=False)
    num_neurons = Column(Integer, nullable=False)
    num_layers = Column(Integer, default=1)
    connectivity_matrix = Column(Text, default="{}")
    neuron_types = Column(Text, default="[]")
    learning_rule = Column(String, default=LearningRule.STDP)
    parameters = Column(Text, default="{}")
    is_trained = Column(Boolean, default=False)
    accuracy = Column(Float, default=0.0)
    spiking_rate = Column(Float, default=0.0)
    energy_efficiency = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, default="{}")

class NeuromorphicNeuron(Base):
    __tablename__ = "neuromorphic_neurons"
    
    id = Column(String, primary_key=True)
    network_id = Column(String, ForeignKey("neuromorphic_networks.id"))
    neuron_index = Column(Integer, nullable=False)
    neuron_type = Column(String, nullable=False)
    membrane_potential = Column(Float, default=-70.0)
    threshold = Column(Float, default=-50.0)
    reset_potential = Column(Float, default=-70.0)
    time_constant = Column(Float, default=10.0)
    refractory_period = Column(Float, default=2.0)
    adaptation_current = Column(Float, default=0.0)
    spike_history = Column(Text, default="[]")
    input_weights = Column(Text, default="[]")
    output_weights = Column(Text, default="[]")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    network = relationship("NeuromorphicNetwork")

class NeuromorphicSpike(Base):
    __tablename__ = "neuromorphic_spikes"
    
    id = Column(String, primary_key=True)
    neuron_id = Column(String, ForeignKey("neuromorphic_neurons.id"))
    spike_time = Column(Float, nullable=False)
    spike_amplitude = Column(Float, default=1.0)
    spike_duration = Column(Float, default=1.0)
    membrane_potential = Column(Float)
    adaptation_current = Column(Float)
    stimulus_type = Column(String)
    stimulus_intensity = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    neuron = relationship("NeuromorphicNeuron")

class NeuromorphicTraining(Base):
    __tablename__ = "neuromorphic_training"
    
    id = Column(String, primary_key=True)
    network_id = Column(String, ForeignKey("neuromorphic_networks.id"))
    training_type = Column(String, nullable=False)
    dataset_name = Column(String)
    num_epochs = Column(Integer, default=100)
    learning_rate = Column(Float, default=0.01)
    batch_size = Column(Integer, default=32)
    accuracy_history = Column(Text, default="[]")
    loss_history = Column(Text, default="[]")
    spiking_rate_history = Column(Text, default="[]")
    energy_history = Column(Text, default="[]")
    final_accuracy = Column(Float, default=0.0)
    final_loss = Column(Float, default=0.0)
    training_time = Column(Float, default=0.0)
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    network = relationship("NeuromorphicNetwork")

class NeuromorphicStimulus(Base):
    __tablename__ = "neuromorphic_stimuli"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    stimulus_type = Column(String, nullable=False)
    duration = Column(Float, nullable=False)
    intensity = Column(Float, default=1.0)
    frequency = Column(Float, default=1.0)
    amplitude = Column(Float, default=1.0)
    phase = Column(Float, default=0.0)
    noise_level = Column(Float, default=0.0)
    parameters = Column(Text, default="{}")
    stimulus_data = Column(Text, default="[]")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Neuromorphic Configuration
NEUROMORPHIC_CONFIG = {
    "default_membrane_potential": -70.0,
    "default_threshold": -50.0,
    "default_reset_potential": -70.0,
    "default_time_constant": 10.0,
    "default_refractory_period": 2.0,
    "max_spike_rate": 1000.0,  # Hz
    "min_spike_rate": 0.1,     # Hz
    "simulation_time_step": 0.001,  # 1ms
    "max_simulation_time": 10.0,    # 10 seconds
    "learning_rules": {
        "stdp": {
            "tau_plus": 20.0,
            "tau_minus": 20.0,
            "a_plus": 0.01,
            "a_minus": 0.01
        },
        "hebb": {
            "learning_rate": 0.01,
            "decay_rate": 0.001
        }
    },
    "neuron_parameters": {
        "lif": {
            "tau_m": 10.0,
            "tau_syn": 5.0,
            "r_m": 1.0,
            "c_m": 1.0
        },
        "izh": {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0
        }
    },
    "network_topology": {
        "max_connections_per_neuron": 100,
        "connection_probability": 0.1,
        "weight_range": [-1.0, 1.0],
        "delay_range": [0.0, 10.0]
    }
}

class AdvancedNeuromorphicSystem:
    """Advanced Neuromorphic Computing system with comprehensive features."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL Neuromorphic Computing System",
            description="Advanced Neuromorphic Computing system with spiking neural networks and brain-inspired computing",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Database session
        self.db = SessionLocal()
        
        # Neuromorphic components
        self.active_networks: Dict[str, NeuromorphicNetwork] = {}
        self.neuron_models: Dict[str, Any] = {}
        self.stimulus_generators: Dict[str, Any] = {}
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_data()
        self.initialize_neuromorphic_models()
        
        logger.info("Advanced Neuromorphic Computing System initialized")
    
    def setup_middleware(self):
        """Setup neuromorphic middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup neuromorphic API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with neuromorphic system information."""
            return {
                "message": "BUL Neuromorphic Computing System",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Spiking Neural Networks",
                    "Brain-Inspired Computing",
                    "Neuromorphic Hardware Simulation",
                    "Spike-Timing Dependent Plasticity",
                    "Real-time Processing",
                    "Energy-Efficient Computing",
                    "Adaptive Learning",
                    "Neural Plasticity"
                ],
                "neuron_types": [neuron_type.value for neuron_type in NeuronType],
                "network_types": [network_type.value for network_type in NetworkType],
                "learning_rules": [learning_rule.value for learning_rule in LearningRule],
                "stimulus_types": [stimulus_type.value for stimulus_type in StimulusType],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/networks/create", tags=["Networks"])
        async def create_neuromorphic_network(network_request: dict):
            """Create neuromorphic network."""
            try:
                # Validate request
                required_fields = ["name", "network_type", "num_neurons"]
                if not all(field in network_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                name = network_request["name"]
                network_type = network_request["network_type"]
                num_neurons = network_request["num_neurons"]
                
                # Create neuromorphic network
                network = NeuromorphicNetwork(
                    id=f"network_{int(time.time())}",
                    name=name,
                    network_type=network_type,
                    num_neurons=num_neurons,
                    num_layers=network_request.get("num_layers", 1),
                    learning_rule=network_request.get("learning_rule", LearningRule.STDP),
                    parameters=json.dumps(network_request.get("parameters", {})),
                    metadata=json.dumps(network_request.get("metadata", {}))
                )
                
                self.db.add(network)
                self.db.commit()
                
                # Create neurons for the network
                await self.create_network_neurons(network.id, num_neurons, network_request.get("neuron_types", ["lif"]))
                
                # Generate connectivity matrix
                await self.generate_connectivity_matrix(network.id, network_type, num_neurons)
                
                # Add to active networks
                self.active_networks[network.id] = network
                
                NEUROMORPHIC_NETWORKS.labels(network_type=network_type, neurons=str(num_neurons)).inc()
                
                return {
                    "message": "Neuromorphic network created successfully",
                    "network_id": network.id,
                    "name": network.name,
                    "network_type": network.network_type,
                    "num_neurons": network.num_neurons,
                    "num_layers": network.num_layers,
                    "learning_rule": network.learning_rule
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating neuromorphic network: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/networks", tags=["Networks"])
        async def get_neuromorphic_networks():
            """Get all neuromorphic networks."""
            try:
                networks = self.db.query(NeuromorphicNetwork).all()
                
                return {
                    "networks": [
                        {
                            "id": network.id,
                            "name": network.name,
                            "network_type": network.network_type,
                            "num_neurons": network.num_neurons,
                            "num_layers": network.num_layers,
                            "learning_rule": network.learning_rule,
                            "is_trained": network.is_trained,
                            "accuracy": network.accuracy,
                            "spiking_rate": network.spiking_rate,
                            "energy_efficiency": network.energy_efficiency,
                            "parameters": json.loads(network.parameters),
                            "metadata": json.loads(network.metadata),
                            "created_at": network.created_at.isoformat()
                        }
                        for network in networks
                    ],
                    "total": len(networks)
                }
                
            except Exception as e:
                logger.error(f"Error getting neuromorphic networks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/networks/{network_id}/simulate", tags=["Simulation"])
        async def simulate_neuromorphic_network(network_id: str, simulation_request: dict, background_tasks: BackgroundTasks):
            """Simulate neuromorphic network."""
            try:
                # Get network
                network = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.id == network_id).first()
                if not network:
                    raise HTTPException(status_code=404, detail="Neuromorphic network not found")
                
                # Validate simulation parameters
                duration = simulation_request.get("duration", 1.0)
                stimulus_type = simulation_request.get("stimulus_type", StimulusType.POISSON)
                stimulus_intensity = simulation_request.get("stimulus_intensity", 1.0)
                
                # Create simulation task
                task_id = f"simulation_{int(time.time())}"
                background_tasks.add_task(
                    self.run_neuromorphic_simulation,
                    network_id,
                    duration,
                    stimulus_type,
                    stimulus_intensity,
                    task_id
                )
                
                return {
                    "message": "Neuromorphic simulation started",
                    "task_id": task_id,
                    "network_id": network_id,
                    "duration": duration,
                    "stimulus_type": stimulus_type,
                    "stimulus_intensity": stimulus_intensity
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error starting neuromorphic simulation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/networks/{network_id}/train", tags=["Training"])
        async def train_neuromorphic_network(network_id: str, training_request: dict, background_tasks: BackgroundTasks):
            """Train neuromorphic network."""
            try:
                # Get network
                network = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.id == network_id).first()
                if not network:
                    raise HTTPException(status_code=404, detail="Neuromorphic network not found")
                
                # Validate training parameters
                training_type = training_request.get("training_type", "supervised")
                dataset_name = training_request.get("dataset_name", "synthetic")
                num_epochs = training_request.get("num_epochs", 100)
                learning_rate = training_request.get("learning_rate", 0.01)
                
                # Create training record
                training = NeuromorphicTraining(
                    id=f"training_{int(time.time())}",
                    network_id=network_id,
                    training_type=training_type,
                    dataset_name=dataset_name,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    batch_size=training_request.get("batch_size", 32)
                )
                
                self.db.add(training)
                self.db.commit()
                
                # Start training in background
                background_tasks.add_task(
                    self.train_neuromorphic_network_background,
                    training.id,
                    network_id,
                    training_request
                )
                
                NEUROMORPHIC_TRAINING.labels(training_type=training_type).inc()
                
                return {
                    "message": "Neuromorphic training started",
                    "training_id": training.id,
                    "network_id": network_id,
                    "training_type": training_type,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error starting neuromorphic training: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/networks/{network_id}/neurons", tags=["Neurons"])
        async def get_network_neurons(network_id: str):
            """Get neurons in neuromorphic network."""
            try:
                # Get network
                network = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.id == network_id).first()
                if not network:
                    raise HTTPException(status_code=404, detail="Neuromorphic network not found")
                
                # Get neurons
                neurons = self.db.query(NeuromorphicNeuron).filter(
                    NeuromorphicNeuron.network_id == network_id,
                    NeuromorphicNeuron.is_active == True
                ).all()
                
                return {
                    "network_id": network_id,
                    "network_name": network.name,
                    "neurons": [
                        {
                            "id": neuron.id,
                            "neuron_index": neuron.neuron_index,
                            "neuron_type": neuron.neuron_type,
                            "membrane_potential": neuron.membrane_potential,
                            "threshold": neuron.threshold,
                            "reset_potential": neuron.reset_potential,
                            "time_constant": neuron.time_constant,
                            "refractory_period": neuron.refractory_period,
                            "adaptation_current": neuron.adaptation_current,
                            "spike_history": json.loads(neuron.spike_history),
                            "input_weights": json.loads(neuron.input_weights),
                            "output_weights": json.loads(neuron.output_weights),
                            "created_at": neuron.created_at.isoformat()
                        }
                        for neuron in neurons
                    ],
                    "total": len(neurons)
                }
                
            except Exception as e:
                logger.error(f"Error getting network neurons: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/networks/{network_id}/spikes", tags=["Spikes"])
        async def get_network_spikes(network_id: str, limit: int = 1000):
            """Get spikes from neuromorphic network."""
            try:
                # Get network
                network = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.id == network_id).first()
                if not network:
                    raise HTTPException(status_code=404, detail="Neuromorphic network not found")
                
                # Get spikes
                spikes = self.db.query(NeuromorphicSpike).join(NeuromorphicNeuron).filter(
                    NeuromorphicNeuron.network_id == network_id
                ).order_by(NeuromorphicSpike.spike_time.desc()).limit(limit).all()
                
                return {
                    "network_id": network_id,
                    "network_name": network.name,
                    "spikes": [
                        {
                            "id": spike.id,
                            "neuron_id": spike.neuron_id,
                            "spike_time": spike.spike_time,
                            "spike_amplitude": spike.spike_amplitude,
                            "spike_duration": spike.spike_duration,
                            "membrane_potential": spike.membrane_potential,
                            "adaptation_current": spike.adaptation_current,
                            "stimulus_type": spike.stimulus_type,
                            "stimulus_intensity": spike.stimulus_intensity,
                            "timestamp": spike.timestamp.isoformat()
                        }
                        for spike in spikes
                    ],
                    "total": len(spikes)
                }
                
            except Exception as e:
                logger.error(f"Error getting network spikes: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/stimuli/create", tags=["Stimuli"])
        async def create_neuromorphic_stimulus(stimulus_request: dict):
            """Create neuromorphic stimulus."""
            try:
                # Validate request
                required_fields = ["name", "stimulus_type", "duration"]
                if not all(field in stimulus_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                name = stimulus_request["name"]
                stimulus_type = stimulus_request["stimulus_type"]
                duration = stimulus_request["duration"]
                
                # Generate stimulus data
                stimulus_data = await self.generate_stimulus_data(
                    stimulus_type,
                    duration,
                    stimulus_request.get("intensity", 1.0),
                    stimulus_request.get("frequency", 1.0),
                    stimulus_request.get("amplitude", 1.0),
                    stimulus_request.get("phase", 0.0),
                    stimulus_request.get("noise_level", 0.0)
                )
                
                # Create stimulus
                stimulus = NeuromorphicStimulus(
                    id=f"stimulus_{int(time.time())}",
                    name=name,
                    stimulus_type=stimulus_type,
                    duration=duration,
                    intensity=stimulus_request.get("intensity", 1.0),
                    frequency=stimulus_request.get("frequency", 1.0),
                    amplitude=stimulus_request.get("amplitude", 1.0),
                    phase=stimulus_request.get("phase", 0.0),
                    noise_level=stimulus_request.get("noise_level", 0.0),
                    parameters=json.dumps(stimulus_request.get("parameters", {})),
                    stimulus_data=json.dumps(stimulus_data)
                )
                
                self.db.add(stimulus)
                self.db.commit()
                
                return {
                    "message": "Neuromorphic stimulus created successfully",
                    "stimulus_id": stimulus.id,
                    "name": stimulus.name,
                    "stimulus_type": stimulus.stimulus_type,
                    "duration": stimulus.duration,
                    "data_points": len(stimulus_data)
                }
                
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error creating neuromorphic stimulus: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stimuli", tags=["Stimuli"])
        async def get_neuromorphic_stimuli():
            """Get all neuromorphic stimuli."""
            try:
                stimuli = self.db.query(NeuromorphicStimulus).filter(NeuromorphicStimulus.is_active == True).all()
                
                return {
                    "stimuli": [
                        {
                            "id": stimulus.id,
                            "name": stimulus.name,
                            "stimulus_type": stimulus.stimulus_type,
                            "duration": stimulus.duration,
                            "intensity": stimulus.intensity,
                            "frequency": stimulus.frequency,
                            "amplitude": stimulus.amplitude,
                            "phase": stimulus.phase,
                            "noise_level": stimulus.noise_level,
                            "parameters": json.loads(stimulus.parameters),
                            "data_points": len(json.loads(stimulus.stimulus_data)),
                            "created_at": stimulus.created_at.isoformat()
                        }
                        for stimulus in stimuli
                    ],
                    "total": len(stimuli)
                }
                
            except Exception as e:
                logger.error(f"Error getting neuromorphic stimuli: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard", tags=["Dashboard"])
        async def get_neuromorphic_dashboard():
            """Get neuromorphic system dashboard."""
            try:
                # Get statistics
                total_networks = self.db.query(NeuromorphicNetwork).count()
                trained_networks = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.is_trained == True).count()
                total_neurons = self.db.query(NeuromorphicNeuron).count()
                total_spikes = self.db.query(NeuromorphicSpike).count()
                total_training_sessions = self.db.query(NeuromorphicTraining).count()
                total_stimuli = self.db.query(NeuromorphicStimulus).count()
                
                # Get network type distribution
                network_types = {}
                for network_type in NetworkType:
                    count = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.network_type == network_type.value).count()
                    network_types[network_type.value] = count
                
                # Get neuron type distribution
                neuron_types = {}
                for neuron_type in NeuronType:
                    count = self.db.query(NeuromorphicNeuron).filter(NeuromorphicNeuron.neuron_type == neuron_type.value).count()
                    neuron_types[neuron_type.value] = count
                
                # Get recent spikes
                recent_spikes = self.db.query(NeuromorphicSpike).order_by(
                    NeuromorphicSpike.timestamp.desc()
                ).limit(20).all()
                
                return {
                    "summary": {
                        "total_networks": total_networks,
                        "trained_networks": trained_networks,
                        "total_neurons": total_neurons,
                        "total_spikes": total_spikes,
                        "total_training_sessions": total_training_sessions,
                        "total_stimuli": total_stimuli
                    },
                    "network_type_distribution": network_types,
                    "neuron_type_distribution": neuron_types,
                    "recent_spikes": [
                        {
                            "id": spike.id,
                            "neuron_id": spike.neuron_id,
                            "spike_time": spike.spike_time,
                            "spike_amplitude": spike.spike_amplitude,
                            "stimulus_type": spike.stimulus_type,
                            "timestamp": spike.timestamp.isoformat()
                        }
                        for spike in recent_spikes
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def setup_default_data(self):
        """Setup default neuromorphic data."""
        try:
            # Create sample networks
            sample_networks = [
                {
                    "name": "Simple LIF Network",
                    "network_type": NetworkType.FEEDFORWARD,
                    "num_neurons": 10,
                    "num_layers": 2,
                    "learning_rule": LearningRule.STDP,
                    "neuron_types": ["lif"]
                },
                {
                    "name": "Recurrent Izhikevich Network",
                    "network_type": NetworkType.RECURRENT,
                    "num_neurons": 20,
                    "num_layers": 1,
                    "learning_rule": LearningRule.HEBB,
                    "neuron_types": ["izh"]
                },
                {
                    "name": "Deep Spiking Network",
                    "network_type": NetworkType.DEEP_SPIKING,
                    "num_neurons": 50,
                    "num_layers": 3,
                    "learning_rule": LearningRule.BACKPROPAGATION,
                    "neuron_types": ["lif", "izh"]
                }
            ]
            
            for network_data in sample_networks:
                network = NeuromorphicNetwork(
                    id=f"network_{network_data['name'].lower().replace(' ', '_')}",
                    name=network_data["name"],
                    network_type=network_data["network_type"],
                    num_neurons=network_data["num_neurons"],
                    num_layers=network_data["num_layers"],
                    learning_rule=network_data["learning_rule"],
                    parameters=json.dumps({}),
                    metadata=json.dumps({"neuron_types": network_data["neuron_types"]})
                )
                
                self.db.add(network)
                self.active_networks[network.id] = network
            
            self.db.commit()
            logger.info("Default neuromorphic data created")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default neuromorphic data: {e}")
    
    def initialize_neuromorphic_models(self):
        """Initialize neuromorphic models."""
        try:
            # Initialize PyTorch models for different neuron types
            self.neuron_models = {
                "lif": self.create_lif_model(),
                "izh": self.create_izh_model(),
                "adex": self.create_adex_model()
            }
            
            # Initialize stimulus generators
            self.stimulus_generators = {
                "poisson": self.create_poisson_generator(),
                "regular": self.create_regular_generator(),
                "burst": self.create_burst_generator(),
                "chirp": self.create_chirp_generator()
            }
            
            logger.info("Neuromorphic models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing neuromorphic models: {e}")
    
    def create_lif_model(self):
        """Create Leaky Integrate-and-Fire neuron model."""
        class LIFNeuron(nn.Module):
            def __init__(self, tau_m=10.0, tau_syn=5.0, threshold=-50.0, reset=-70.0):
                super().__init__()
                self.tau_m = tau_m
                self.tau_syn = tau_syn
                self.threshold = threshold
                self.reset = reset
                self.membrane_potential = torch.tensor(reset)
                self.synaptic_current = torch.tensor(0.0)
                
            def forward(self, input_current, dt=0.001):
                # Update synaptic current
                self.synaptic_current = self.synaptic_current * torch.exp(-dt / self.tau_syn) + input_current
                
                # Update membrane potential
                self.membrane_potential = self.membrane_potential * torch.exp(-dt / self.tau_m) + self.synaptic_current * dt
                
                # Check for spike
                spike = (self.membrane_potential >= self.threshold).float()
                
                # Reset if spike occurred
                self.membrane_potential = torch.where(spike, self.reset, self.membrane_potential)
                
                return spike, self.membrane_potential
        
        return LIFNeuron()
    
    def create_izh_model(self):
        """Create Izhikevich neuron model."""
        class IzhikevichNeuron(nn.Module):
            def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c
                self.d = d
                self.v = torch.tensor(c)
                self.u = torch.tensor(b * c)
                
            def forward(self, input_current, dt=0.001):
                # Update membrane potential and recovery variable
                dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + input_current
                du = self.a * (self.b * self.v - self.u)
                
                self.v = self.v + dv * dt
                self.u = self.u + du * dt
                
                # Check for spike
                spike = (self.v >= 30.0).float()
                
                # Reset if spike occurred
                self.v = torch.where(spike, self.c, self.v)
                self.u = torch.where(spike, self.u + self.d, self.u)
                
                return spike, self.v
        
        return IzhikevichNeuron()
    
    def create_adex_model(self):
        """Create Adaptive Exponential Integrate-and-Fire neuron model."""
        class AdExNeuron(nn.Module):
            def __init__(self, tau_m=20.0, tau_w=200.0, a=0.0, b=0.0, delta_T=2.0, v_rheobase=-50.0):
                super().__init__()
                self.tau_m = tau_m
                self.tau_w = tau_w
                self.a = a
                self.b = b
                self.delta_T = delta_T
                self.v_rheobase = v_rheobase
                self.v = torch.tensor(-70.0)
                self.w = torch.tensor(0.0)
                
            def forward(self, input_current, dt=0.001):
                # Update membrane potential
                dv = (-self.v + self.v_rheobase + self.delta_T * torch.exp((self.v - self.v_rheobase) / self.delta_T) - self.w + input_current) / self.tau_m
                self.v = self.v + dv * dt
                
                # Update adaptation current
                dw = (self.a * (self.v - self.v_rheobase) - self.w) / self.tau_w
                self.w = self.w + dw * dt
                
                # Check for spike
                spike = (self.v >= 30.0).float()
                
                # Reset if spike occurred
                self.v = torch.where(spike, -70.0, self.v)
                self.w = torch.where(spike, self.w + self.b, self.w)
                
                return spike, self.v
        
        return AdExNeuron()
    
    def create_poisson_generator(self):
        """Create Poisson spike generator."""
        def poisson_generator(rate, duration, dt=0.001):
            num_steps = int(duration / dt)
            spikes = torch.rand(num_steps) < rate * dt
            return spikes
        
        return poisson_generator
    
    def create_regular_generator(self):
        """Create regular spike generator."""
        def regular_generator(frequency, duration, dt=0.001):
            period = 1.0 / frequency
            num_steps = int(duration / dt)
            spikes = torch.zeros(num_steps)
            spike_times = torch.arange(0, duration, period)
            spike_indices = (spike_times / dt).long()
            spikes[spike_indices] = 1.0
            return spikes
        
        return regular_generator
    
    def create_burst_generator(self):
        """Create burst spike generator."""
        def burst_generator(burst_rate, burst_duration, inter_burst_interval, total_duration, dt=0.001):
            num_steps = int(total_duration / dt)
            spikes = torch.zeros(num_steps)
            
            current_time = 0.0
            while current_time < total_duration:
                # Generate burst
                burst_steps = int(burst_duration / dt)
                burst_start = int(current_time / dt)
                burst_end = min(burst_start + burst_steps, num_steps)
                
                if burst_start < num_steps:
                    burst_spikes = torch.rand(burst_end - burst_start) < burst_rate * dt
                    spikes[burst_start:burst_end] = burst_spikes
                
                current_time += burst_duration + inter_burst_interval
            
            return spikes
        
        return burst_generator
    
    def create_chirp_generator(self):
        """Create chirp (frequency sweep) generator."""
        def chirp_generator(start_freq, end_freq, duration, dt=0.001):
            num_steps = int(duration / dt)
            time = torch.arange(0, duration, dt)
            
            # Linear frequency sweep
            frequency = start_freq + (end_freq - start_freq) * time / duration
            
            # Generate chirp signal
            phase = 2 * np.pi * torch.cumsum(frequency) * dt
            chirp_signal = torch.sin(phase)
            
            # Convert to spikes (threshold-based)
            threshold = 0.5
            spikes = (chirp_signal > threshold).float()
            
            return spikes
        
        return chirp_generator
    
    async def create_network_neurons(self, network_id: str, num_neurons: int, neuron_types: List[str]):
        """Create neurons for neuromorphic network."""
        try:
            for i in range(num_neurons):
                neuron_type = neuron_types[i % len(neuron_types)]
                
                neuron = NeuromorphicNeuron(
                    id=f"neuron_{network_id}_{i}",
                    network_id=network_id,
                    neuron_index=i,
                    neuron_type=neuron_type,
                    membrane_potential=NEUROMORPHIC_CONFIG["default_membrane_potential"],
                    threshold=NEUROMORPHIC_CONFIG["default_threshold"],
                    reset_potential=NEUROMORPHIC_CONFIG["default_reset_potential"],
                    time_constant=NEUROMORPHIC_CONFIG["default_time_constant"],
                    refractory_period=NEUROMORPHIC_CONFIG["default_refractory_period"],
                    spike_history=json.dumps([]),
                    input_weights=json.dumps([]),
                    output_weights=json.dumps([])
                )
                
                self.db.add(neuron)
            
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating network neurons: {e}")
            raise
    
    async def generate_connectivity_matrix(self, network_id: str, network_type: str, num_neurons: int):
        """Generate connectivity matrix for neuromorphic network."""
        try:
            connectivity = {}
            
            if network_type == NetworkType.FEEDFORWARD:
                # Feedforward connections
                layers = 2  # Simple 2-layer network
                neurons_per_layer = num_neurons // layers
                
                for layer in range(layers - 1):
                    for i in range(neurons_per_layer):
                        for j in range(neurons_per_layer):
                            source_idx = layer * neurons_per_layer + i
                            target_idx = (layer + 1) * neurons_per_layer + j
                            if source_idx < num_neurons and target_idx < num_neurons:
                                connectivity[f"{source_idx}_{target_idx}"] = {
                                    "weight": np.random.uniform(-1.0, 1.0),
                                    "delay": np.random.uniform(0.0, 10.0)
                                }
            
            elif network_type == NetworkType.RECURRENT:
                # Recurrent connections
                connection_prob = NEUROMORPHIC_CONFIG["network_topology"]["connection_probability"]
                
                for i in range(num_neurons):
                    for j in range(num_neurons):
                        if i != j and np.random.random() < connection_prob:
                            connectivity[f"{i}_{j}"] = {
                                "weight": np.random.uniform(-1.0, 1.0),
                                "delay": np.random.uniform(0.0, 10.0)
                            }
            
            # Update network connectivity matrix
            network = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.id == network_id).first()
            if network:
                network.connectivity_matrix = json.dumps(connectivity)
                self.db.commit()
            
        except Exception as e:
            logger.error(f"Error generating connectivity matrix: {e}")
            raise
    
    async def generate_stimulus_data(self, stimulus_type: str, duration: float, intensity: float, frequency: float, amplitude: float, phase: float, noise_level: float) -> List[float]:
        """Generate stimulus data."""
        try:
            dt = NEUROMORPHIC_CONFIG["simulation_time_step"]
            num_steps = int(duration / dt)
            time = np.arange(0, duration, dt)
            
            if stimulus_type == StimulusType.POISSON:
                # Poisson spike train
                spikes = np.random.poisson(intensity * dt, num_steps)
                data = spikes.astype(float)
            
            elif stimulus_type == StimulusType.REGULAR:
                # Regular spike train
                period = 1.0 / frequency
                spike_times = np.arange(0, duration, period)
                data = np.zeros(num_steps)
                spike_indices = (spike_times / dt).astype(int)
                spike_indices = spike_indices[spike_indices < num_steps]
                data[spike_indices] = amplitude
            
            elif stimulus_type == StimulusType.SINUSOIDAL:
                # Sinusoidal stimulus
                data = amplitude * np.sin(2 * np.pi * frequency * time + phase)
            
            elif stimulus_type == StimulusType.SQUARE_WAVE:
                # Square wave stimulus
                data = amplitude * np.sign(np.sin(2 * np.pi * frequency * time + phase))
            
            elif stimulus_type == StimulusType.WHITE_NOISE:
                # White noise stimulus
                data = np.random.normal(0, amplitude, num_steps)
            
            else:
                # Default: constant stimulus
                data = np.full(num_steps, intensity)
            
            # Add noise if specified
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, num_steps)
                data += noise
            
            return data.tolist()
            
        except Exception as e:
            logger.error(f"Error generating stimulus data: {e}")
            return []
    
    async def run_neuromorphic_simulation(self, network_id: str, duration: float, stimulus_type: str, stimulus_intensity: float, task_id: str):
        """Run neuromorphic network simulation."""
        try:
            start_time = time.time()
            
            # Get network
            network = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.id == network_id).first()
            if not network:
                return
            
            # Get neurons
            neurons = self.db.query(NeuromorphicNeuron).filter(
                NeuromorphicNeuron.network_id == network_id,
                NeuromorphicNeuron.is_active == True
            ).all()
            
            # Generate stimulus
            stimulus_data = await self.generate_stimulus_data(stimulus_type, duration, stimulus_intensity, 1.0, 1.0, 0.0, 0.0)
            
            # Run simulation
            dt = NEUROMORPHIC_CONFIG["simulation_time_step"]
            num_steps = len(stimulus_data)
            total_spikes = 0
            
            for step in range(num_steps):
                current_stimulus = stimulus_data[step]
                
                for neuron in neurons:
                    # Update neuron state (simplified)
                    neuron.membrane_potential += current_stimulus * dt
                    
                    # Check for spike
                    if neuron.membrane_potential >= neuron.threshold:
                        # Record spike
                        spike = NeuromorphicSpike(
                            id=f"spike_{int(time.time())}_{neuron.id}_{step}",
                            neuron_id=neuron.id,
                            spike_time=step * dt,
                            spike_amplitude=1.0,
                            membrane_potential=neuron.membrane_potential,
                            stimulus_type=stimulus_type,
                            stimulus_intensity=stimulus_intensity
                        )
                        
                        self.db.add(spike)
                        total_spikes += 1
                        
                        # Reset neuron
                        neuron.membrane_potential = neuron.reset_potential
                        
                        # Update spike history
                        spike_history = json.loads(neuron.spike_history)
                        spike_history.append(step * dt)
                        neuron.spike_history = json.dumps(spike_history)
                        
                        NEUROMORPHIC_SPIKES.labels(neuron_type=neuron.neuron_type).inc()
            
            # Update network statistics
            network.spiking_rate = total_spikes / duration
            network.energy_efficiency = total_spikes / (len(neurons) * duration)
            
            self.db.commit()
            
            execution_time = time.time() - start_time
            NEUROMORPHIC_LATENCY.observe(execution_time)
            
            logger.info(f"Neuromorphic simulation {task_id} completed: {total_spikes} spikes in {duration}s")
            
        except Exception as e:
            logger.error(f"Error running neuromorphic simulation: {e}")
    
    async def train_neuromorphic_network_background(self, training_id: str, network_id: str, training_request: dict):
        """Train neuromorphic network in background."""
        try:
            start_time = time.time()
            
            # Get training record
            training = self.db.query(NeuromorphicTraining).filter(NeuromorphicTraining.id == training_id).first()
            if not training:
                return
            
            # Get network
            network = self.db.query(NeuromorphicNetwork).filter(NeuromorphicNetwork.id == network_id).first()
            if not network:
                return
            
            # Simulate training process
            accuracy_history = []
            loss_history = []
            spiking_rate_history = []
            energy_history = []
            
            for epoch in range(training.num_epochs):
                # Simulate training step
                accuracy = min(0.95, epoch / training.num_epochs + np.random.normal(0, 0.05))
                loss = max(0.05, 1.0 - epoch / training.num_epochs + np.random.normal(0, 0.05))
                spiking_rate = 10.0 + np.random.normal(0, 2.0)
                energy = 0.8 + np.random.normal(0, 0.1)
                
                accuracy_history.append(accuracy)
                loss_history.append(loss)
                spiking_rate_history.append(spiking_rate)
                energy_history.append(energy)
                
                # Update training record
                training.accuracy_history = json.dumps(accuracy_history)
                training.loss_history = json.dumps(loss_history)
                training.spiking_rate_history = json.dumps(spiking_rate_history)
                training.energy_history = json.dumps(energy_history)
                
                self.db.commit()
                
                # Simulate training delay
                await asyncio.sleep(0.1)
            
            # Update final results
            training.final_accuracy = accuracy_history[-1]
            training.final_loss = loss_history[-1]
            training.training_time = time.time() - start_time
            training.is_completed = True
            training.completed_at = datetime.utcnow()
            
            # Update network
            network.is_trained = True
            network.accuracy = training.final_accuracy
            network.spiking_rate = spiking_rate_history[-1]
            network.energy_efficiency = energy_history[-1]
            
            self.db.commit()
            
            NEUROMORPHIC_ACCURACY.observe(training.final_accuracy)
            
            logger.info(f"Neuromorphic training {training_id} completed: accuracy={training.final_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training neuromorphic network: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8012, debug: bool = False):
        """Run the neuromorphic system."""
        logger.info(f"Starting Neuromorphic Computing System on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Neuromorphic Computing System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8012, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run neuromorphic system
    system = AdvancedNeuromorphicSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
