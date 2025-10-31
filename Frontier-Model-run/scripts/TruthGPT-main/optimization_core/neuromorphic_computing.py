"""
Advanced Neural Network Neuromorphic Computing System for TruthGPT Optimization Core
Complete neuromorphic computing with spiking neural networks and event-driven processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class NeuronModel(Enum):
    """Neuron models"""
    LEAKY_INTEGRATE_AND_FIRE = "leaky_integrate_and_fire"
    INTEGRATE_AND_FIRE = "integrate_and_fire"
    HODGKIN_HUXLEY = "hodgkin_huxley"
    IZHIKEVICH = "izhikevich"
    ADAPTIVE_EXPONENTIAL = "adaptive_exponential"
    QUADRATIC_INTEGRATE_AND_FIRE = "quadratic_integrate_and_fire"

class SynapseModel(Enum):
    """Synapse models"""
    DELTA_SYNAPSE = "delta_synapse"
    ALPHA_SYNAPSE = "alpha_synapse"
    EXPONENTIAL_SYNAPSE = "exponential_synapse"
    STDP_SYNAPSE = "stdp_synapse"
    PLASTIC_SYNAPSE = "plastic_synapse"
    ADAPTIVE_SYNAPSE = "adaptive_synapse"

class NeuromorphicConfig:
    """Configuration for neuromorphic computing system"""
    # Basic settings
    neuron_model: NeuronModel = NeuronModel.LEAKY_INTEGRATE_AND_FIRE
    synapse_model: SynapseModel = SynapseModel.STDP_SYNAPSE
    
    # Network settings
    num_neurons: int = 1000
    num_synapses: int = 10000
    simulation_time: float = 1000.0  # ms
    time_step: float = 0.1  # ms
    
    # Neuron parameters
    membrane_potential_threshold: float = -50.0  # mV
    membrane_potential_reset: float = -70.0  # mV
    membrane_time_constant: float = 20.0  # ms
    membrane_resistance: float = 1.0  # MOhm
    
    # Synapse parameters
    synaptic_delay: float = 1.0  # ms
    synaptic_weight_range: Tuple[float, float] = (0.0, 1.0)
    stdp_learning_rate: float = 0.01
    stdp_tau_plus: float = 20.0  # ms
    stdp_tau_minus: float = 20.0  # ms
    
    # Event-driven settings
    enable_event_driven: bool = True
    enable_spike_timing_dependent_plasticity: bool = True
    enable_adaptive_thresholds: bool = True
    enable_synaptic_scaling: bool = True
    
    # Advanced features
    enable_plasticity: bool = True
    enable_homeostasis: bool = True
    enable_noise: bool = True
    noise_strength: float = 0.1
    
    def __post_init__(self):
        """Validate neuromorphic configuration"""
        if self.num_neurons <= 0:
            raise ValueError("Number of neurons must be positive")
        if self.num_synapses <= 0:
            raise ValueError("Number of synapses must be positive")
        if self.simulation_time <= 0:
            raise ValueError("Simulation time must be positive")
        if self.time_step <= 0:
            raise ValueError("Time step must be positive")
        if self.membrane_time_constant <= 0:
            raise ValueError("Membrane time constant must be positive")
        if self.membrane_resistance <= 0:
            raise ValueError("Membrane resistance must be positive")
        if self.synaptic_delay <= 0:
            raise ValueError("Synaptic delay must be positive")
        if not (0 <= self.stdp_learning_rate <= 1):
            raise ValueError("STDP learning rate must be between 0 and 1")
        if self.stdp_tau_plus <= 0:
            raise ValueError("STDP tau plus must be positive")
        if self.stdp_tau_minus <= 0:
            raise ValueError("STDP tau minus must be positive")
        if self.noise_strength < 0:
            raise ValueError("Noise strength must be non-negative")

class SpikingNeuron:
    """Spiking neuron implementation"""
    
    def __init__(self, neuron_id: int, config: NeuromorphicConfig):
        self.neuron_id = neuron_id
        self.config = config
        
        # Neuron state
        self.membrane_potential = config.membrane_potential_reset
        self.threshold = config.membrane_potential_threshold
        self.spike_times = []
        self.last_spike_time = -1.0
        
        # Adaptive parameters
        self.adaptive_threshold = config.membrane_potential_threshold
        self.threshold_adaptation = 0.0
        
        # Input currents
        self.external_current = 0.0
        self.synaptic_current = 0.0
        
        logger.info(f"✅ Spiking Neuron {neuron_id} initialized")
    
    def update(self, time: float, dt: float):
        """Update neuron state"""
        # Calculate total current
        total_current = self.external_current + self.synaptic_current
        
        # Add noise if enabled
        if self.config.enable_noise:
            noise = np.random.normal(0, self.config.noise_strength)
            total_current += noise
        
        # Update membrane potential based on neuron model
        if self.config.neuron_model == NeuronModel.LEAKY_INTEGRATE_AND_FIRE:
            self._update_lif(total_current, dt)
        elif self.config.neuron_model == NeuronModel.INTEGRATE_AND_FIRE:
            self._update_if(total_current, dt)
        elif self.config.neuron_model == NeuronModel.IZHIKEVICH:
            self._update_izhikevich(total_current, dt)
        
        # Check for spike
        if self.membrane_potential >= self.adaptive_threshold:
            self._fire_spike(time)
        
        # Update adaptive threshold
        if self.config.enable_adaptive_thresholds:
            self._update_adaptive_threshold(time, dt)
    
    def _update_lif(self, current: float, dt: float):
        """Update LIF neuron"""
        # LIF equation: dV/dt = (I - V + V_rest) / tau
        dv_dt = (current - self.membrane_potential + self.config.membrane_potential_reset) / self.config.membrane_time_constant
        self.membrane_potential += dv_dt * dt
    
    def _update_if(self, current: float, dt: float):
        """Update IF neuron"""
        # IF equation: dV/dt = I
        dv_dt = current
        self.membrane_potential += dv_dt * dt
    
    def _update_izhikevich(self, current: float, dt: float):
        """Update Izhikevich neuron"""
        # Izhikevich model parameters
        a = 0.02
        b = 0.2
        c = -65.0
        d = 8.0
        
        # Izhikevich equations
        dv_dt = 0.04 * self.membrane_potential**2 + 5 * self.membrane_potential + 140 - self.threshold_adaptation + current
        du_dt = a * (b * self.membrane_potential - self.threshold_adaptation)
        
        self.membrane_potential += dv_dt * dt
        self.threshold_adaptation += du_dt * dt
    
    def _fire_spike(self, time: float):
        """Fire spike"""
        self.spike_times.append(time)
        self.last_spike_time = time
        self.membrane_potential = self.config.membrane_potential_reset
        
        # Reset adaptation variable for Izhikevich model
        if self.config.neuron_model == NeuronModel.IZHIKEVICH:
            self.threshold_adaptation += 8.0
    
    def _update_adaptive_threshold(self, time: float, dt: float):
        """Update adaptive threshold"""
        if self.last_spike_time > 0:
            time_since_spike = time - self.last_spike_time
            # Exponential decay of threshold adaptation
            self.adaptive_threshold = self.config.membrane_potential_threshold + self.threshold_adaptation * np.exp(-time_since_spike / 10.0)
        else:
            self.adaptive_threshold = self.config.membrane_potential_threshold

class Synapse:
    """Synapse implementation"""
    
    def __init__(self, synapse_id: int, pre_neuron_id: int, post_neuron_id: int, 
                 weight: float, config: NeuromorphicConfig):
        self.synapse_id = synapse_id
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = weight
        self.config = config
        
        # STDP variables
        self.last_pre_spike_time = -1.0
        self.last_post_spike_time = -1.0
        self.trace_pre = 0.0
        self.trace_post = 0.0
        
        # Synaptic state
        self.delay_buffer = []
        self.delay_steps = int(self.config.synaptic_delay / self.config.time_step)
        
        logger.info(f"✅ Synapse {synapse_id} initialized: {pre_neuron_id} -> {post_neuron_id}")
    
    def update(self, time: float, dt: float, pre_spiked: bool, post_spiked: bool):
        """Update synapse state"""
        # Update traces
        self.trace_pre *= np.exp(-dt / self.config.stdp_tau_plus)
        self.trace_post *= np.exp(-dt / self.config.stdp_tau_minus)
        
        # Update traces based on spikes
        if pre_spiked:
            self.trace_pre += 1.0
            self.last_pre_spike_time = time
        
        if post_spiked:
            self.trace_post += 1.0
            self.last_post_spike_time = time
        
        # Apply STDP if enabled
        if self.config.enable_spike_timing_dependent_plasticity:
            self._apply_stdp(time, dt)
    
    def _apply_stdp(self, time: float, dt: float):
        """Apply spike-timing dependent plasticity"""
        if self.last_pre_spike_time > 0 and self.last_post_spike_time > 0:
            time_diff = self.last_pre_spike_time - self.last_post_spike_time
            
            if time_diff > 0:  # Pre before post (LTP)
                weight_change = self.config.stdp_learning_rate * self.trace_post
            else:  # Post before pre (LTD)
                weight_change = -self.config.stdp_learning_rate * self.trace_pre
            
            # Update weight
            self.weight += weight_change
            
            # Apply weight bounds
            self.weight = np.clip(self.weight, 
                                self.config.synaptic_weight_range[0],
                                self.config.synaptic_weight_range[1])
    
    def propagate_spike(self, time: float) -> bool:
        """Propagate spike with delay"""
        # Add spike to delay buffer
        self.delay_buffer.append(time)
        
        # Check if delayed spike should be delivered
        if len(self.delay_buffer) > self.delay_steps:
            delayed_spike_time = self.delay_buffer.pop(0)
            return True
        
        return False

class SpikingNeuralNetwork:
    """Spiking Neural Network implementation"""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.neurons = []
        self.synapses = []
        self.simulation_time = 0.0
        self.spike_history = []
        logger.info("✅ Spiking Neural Network initialized")
    
    def initialize_network(self):
        """Initialize network"""
        # Create neurons
        for i in range(self.config.num_neurons):
            neuron = SpikingNeuron(i, self.config)
            self.neurons.append(neuron)
        
        # Create synapses
        for i in range(self.config.num_synapses):
            pre_neuron_id = np.random.randint(0, self.config.num_neurons)
            post_neuron_id = np.random.randint(0, self.config.num_neurons)
            
            # Avoid self-connections
            while post_neuron_id == pre_neuron_id:
                post_neuron_id = np.random.randint(0, self.config.num_neurons)
            
            weight = np.random.uniform(*self.config.synaptic_weight_range)
            synapse = Synapse(i, pre_neuron_id, post_neuron_id, weight, self.config)
            self.synapses.append(synapse)
        
        logger.info(f"🧬 Initialized network with {len(self.neurons)} neurons and {len(self.synapses)} synapses")
    
    def simulate(self, input_spikes: List[Tuple[int, float]] = None) -> Dict[str, Any]:
        """Simulate network"""
        logger.info(f"🚀 Starting simulation for {self.config.simulation_time} ms")
        
        # Initialize network
        self.initialize_network()
        
        # Simulation loop
        current_time = 0.0
        dt = self.config.time_step
        
        while current_time < self.config.simulation_time:
            # Apply external input
            if input_spikes:
                self._apply_external_input(current_time, input_spikes)
            
            # Update neurons
            spikes_this_step = []
            for neuron in self.neurons:
                neuron.update(current_time, dt)
                
                # Check for spikes
                if neuron.last_spike_time == current_time:
                    spikes_this_step.append(neuron.neuron_id)
            
            # Update synapses
            for synapse in self.synapses:
                pre_neuron = self.neurons[synapse.pre_neuron_id]
                post_neuron = self.neurons[synapse.post_neuron_id]
                
                pre_spiked = synapse.pre_neuron_id in spikes_this_step
                post_spiked = synapse.post_neuron_id in spikes_this_step
                
                synapse.update(current_time, dt, pre_spiked, post_spiked)
                
                # Propagate spikes
                if synapse.propagate_spike(current_time):
                    post_neuron.synaptic_current += synapse.weight
            
            # Record spike history
            if spikes_this_step:
                self.spike_history.append((current_time, spikes_this_step))
            
            # Reset synaptic currents
            for neuron in self.neurons:
                neuron.synaptic_current = 0.0
            
            current_time += dt
        
        simulation_result = {
            'simulation_time': self.config.simulation_time,
            'time_step': self.config.time_step,
            'num_neurons': len(self.neurons),
            'num_synapses': len(self.synapses),
            'total_spikes': sum(len(spikes) for _, spikes in self.spike_history),
            'spike_history': self.spike_history,
            'status': 'success'
        }
        
        return simulation_result
    
    def _apply_external_input(self, current_time: float, input_spikes: List[Tuple[int, float]]):
        """Apply external input spikes"""
        for neuron_id, spike_time in input_spikes:
            if abs(current_time - spike_time) < self.config.time_step / 2:
                if neuron_id < len(self.neurons):
                    self.neurons[neuron_id].external_current = 10.0  # External current

class EventDrivenProcessor:
    """Event-driven processor for neuromorphic computing"""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.event_queue = []
        self.processed_events = []
        logger.info("✅ Event-Driven Processor initialized")
    
    def add_event(self, event_type: str, neuron_id: int, time: float, data: Any = None):
        """Add event to queue"""
        event = {
            'type': event_type,
            'neuron_id': neuron_id,
            'time': time,
            'data': data
        }
        self.event_queue.append(event)
    
    def process_events(self, max_time: float) -> Dict[str, Any]:
        """Process events"""
        logger.info(f"⚡ Processing events up to time {max_time}")
        
        # Sort events by time
        self.event_queue.sort(key=lambda x: x['time'])
        
        processed_count = 0
        for event in self.event_queue:
            if event['time'] <= max_time:
                self._process_event(event)
                self.processed_events.append(event)
                processed_count += 1
            else:
                break
        
        # Remove processed events
        self.event_queue = self.event_queue[processed_count:]
        
        processing_result = {
            'processed_events': processed_count,
            'remaining_events': len(self.event_queue),
            'max_time': max_time,
            'status': 'success'
        }
        
        return processing_result
    
    def _process_event(self, event: Dict[str, Any]):
        """Process individual event"""
        event_type = event['type']
        neuron_id = event['neuron_id']
        time = event['time']
        data = event['data']
        
        if event_type == 'spike':
            logger.debug(f"Processing spike event: neuron {neuron_id} at time {time}")
        elif event_type == 'synaptic_update':
            logger.debug(f"Processing synaptic update: neuron {neuron_id} at time {time}")
        elif event_type == 'plasticity_update':
            logger.debug(f"Processing plasticity update: neuron {neuron_id} at time {time}")

class NeuromorphicChip:
    """Neuromorphic chip simulator"""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.chip_neurons = []
        self.chip_synapses = []
        self.chip_state = {}
        logger.info("✅ Neuromorphic Chip initialized")
    
    def initialize_chip(self):
        """Initialize neuromorphic chip"""
        # Simulate chip initialization
        self.chip_neurons = list(range(self.config.num_neurons))
        self.chip_synapses = list(range(self.config.num_synapses))
        
        self.chip_state = {
            'power_consumption': 0.0,
            'temperature': 25.0,
            'voltage': 1.0,
            'current': 0.0
        }
        
        logger.info(f"🔧 Initialized chip with {len(self.chip_neurons)} neurons and {len(self.chip_synapses)} synapses")
    
    def run_chip_simulation(self, duration: float) -> Dict[str, Any]:
        """Run chip simulation"""
        logger.info(f"🔧 Running chip simulation for {duration} ms")
        
        self.initialize_chip()
        
        # Simulate chip operation
        power_consumption = 0.0
        for _ in range(int(duration / self.config.time_step)):
            # Simulate power consumption
            power_consumption += np.random.uniform(0.1, 0.5)
            
            # Update chip state
            self.chip_state['power_consumption'] = power_consumption
            self.chip_state['temperature'] = 25.0 + power_consumption * 0.1
            self.chip_state['voltage'] = 1.0 - power_consumption * 0.01
            self.chip_state['current'] = power_consumption
        
        chip_result = {
            'duration': duration,
            'power_consumption': power_consumption,
            'chip_state': self.chip_state,
            'status': 'success'
        }
        
        return chip_result

class NeuromorphicTrainer:
    """Neuromorphic network trainer"""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.training_history = []
        logger.info("✅ Neuromorphic Trainer initialized")
    
    def train_network(self, network: SpikingNeuralNetwork, 
                     training_data: List[Tuple[List[Tuple[int, float]], int]]) -> Dict[str, Any]:
        """Train neuromorphic network"""
        logger.info("🏋️ Training neuromorphic network")
        
        training_epochs = 10
        learning_rate = 0.01
        
        for epoch in range(training_epochs):
            epoch_loss = 0.0
            
            for input_spikes, target in training_data:
                # Run network simulation
                result = network.simulate(input_spikes)
                
                # Calculate output (simplified)
                output_spikes = len(result['spike_history'])
                
                # Calculate loss
                loss = abs(output_spikes - target)
                epoch_loss += loss
                
                # Update weights (simplified)
                self._update_weights(network, loss, learning_rate)
            
            avg_loss = epoch_loss / len(training_data)
            
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_loss
            })
            
            logger.info(f"   Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        training_result = {
            'epochs': training_epochs,
            'final_loss': self.training_history[-1]['loss'],
            'training_history': self.training_history,
            'status': 'success'
        }
        
        return training_result
    
    def _update_weights(self, network: SpikingNeuralNetwork, loss: float, learning_rate: float):
        """Update network weights"""
        # Simplified weight update
        for synapse in network.synapses:
            weight_change = learning_rate * loss * np.random.normal(0, 0.1)
            synapse.weight += weight_change
            
            # Apply weight bounds
            synapse.weight = np.clip(synapse.weight,
                                   self.config.synaptic_weight_range[0],
                                   self.config.synaptic_weight_range[1])

class NeuromorphicAccelerator:
    """Main neuromorphic accelerator system"""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        
        # Components
        self.network = SpikingNeuralNetwork(config)
        self.event_processor = EventDrivenProcessor(config)
        self.chip = NeuromorphicChip(config)
        self.trainer = NeuromorphicTrainer(config)
        
        # Neuromorphic state
        self.neuromorphic_history = []
        
        logger.info("✅ Neuromorphic Accelerator initialized")
    
    def run_neuromorphic_computing(self, input_data: List[Tuple[int, float]] = None) -> Dict[str, Any]:
        """Run neuromorphic computing"""
        logger.info("🚀 Starting neuromorphic computing")
        
        neuromorphic_results = {
            'start_time': time.time(),
            'config': self.config,
            'stages': {}
        }
        
        # Stage 1: Network Simulation
        logger.info("🧠 Stage 1: Network Simulation")
        simulation_result = self.network.simulate(input_data)
        
        neuromorphic_results['stages']['network_simulation'] = simulation_result
        
        # Stage 2: Event-Driven Processing
        if self.config.enable_event_driven:
            logger.info("⚡ Stage 2: Event-Driven Processing")
            
            # Add events from simulation
            for spike_time, neuron_ids in simulation_result['spike_history']:
                for neuron_id in neuron_ids:
                    self.event_processor.add_event('spike', neuron_id, spike_time)
            
            # Process events
            event_result = self.event_processor.process_events(self.config.simulation_time)
            
            neuromorphic_results['stages']['event_processing'] = event_result
        
        # Stage 3: Chip Simulation
        logger.info("🔧 Stage 3: Chip Simulation")
        chip_result = self.chip.run_chip_simulation(self.config.simulation_time)
        
        neuromorphic_results['stages']['chip_simulation'] = chip_result
        
        # Stage 4: Network Training
        logger.info("🏋️ Stage 4: Network Training")
        
        # Create dummy training data
        training_data = []
        for i in range(10):
            input_spikes = [(np.random.randint(0, self.config.num_neurons), 
                           np.random.uniform(0, self.config.simulation_time)) 
                          for _ in range(5)]
            target = np.random.randint(1, 10)
            training_data.append((input_spikes, target))
        
        training_result = self.trainer.train_network(self.network, training_data)
        
        neuromorphic_results['stages']['network_training'] = training_result
        
        # Final evaluation
        neuromorphic_results['end_time'] = time.time()
        neuromorphic_results['total_duration'] = neuromorphic_results['end_time'] - neuromorphic_results['start_time']
        
        # Store results
        self.neuromorphic_history.append(neuromorphic_results)
        
        logger.info("✅ Neuromorphic computing completed")
        return neuromorphic_results
    
    def generate_neuromorphic_report(self, results: Dict[str, Any]) -> str:
        """Generate neuromorphic computing report"""
        report = []
        report.append("=" * 50)
        report.append("NEUROMORPHIC COMPUTING REPORT")
        report.append("=" * 50)
        
        # Configuration
        report.append("\nNEUROMORPHIC CONFIGURATION:")
        report.append("-" * 30)
        report.append(f"Neuron Model: {self.config.neuron_model.value}")
        report.append(f"Synapse Model: {self.config.synapse_model.value}")
        report.append(f"Number of Neurons: {self.config.num_neurons}")
        report.append(f"Number of Synapses: {self.config.num_synapses}")
        report.append(f"Simulation Time: {self.config.simulation_time} ms")
        report.append(f"Time Step: {self.config.time_step} ms")
        report.append(f"Membrane Potential Threshold: {self.config.membrane_potential_threshold} mV")
        report.append(f"Membrane Potential Reset: {self.config.membrane_potential_reset} mV")
        report.append(f"Membrane Time Constant: {self.config.membrane_time_constant} ms")
        report.append(f"Membrane Resistance: {self.config.membrane_resistance} MOhm")
        report.append(f"Synaptic Delay: {self.config.synaptic_delay} ms")
        report.append(f"Synaptic Weight Range: {self.config.synaptic_weight_range}")
        report.append(f"STDP Learning Rate: {self.config.stdp_learning_rate}")
        report.append(f"STDP Tau Plus: {self.config.stdp_tau_plus} ms")
        report.append(f"STDP Tau Minus: {self.config.stdp_tau_minus} ms")
        report.append(f"Event-Driven: {'Enabled' if self.config.enable_event_driven else 'Disabled'}")
        report.append(f"STDP: {'Enabled' if self.config.enable_spike_timing_dependent_plasticity else 'Disabled'}")
        report.append(f"Adaptive Thresholds: {'Enabled' if self.config.enable_adaptive_thresholds else 'Disabled'}")
        report.append(f"Synaptic Scaling: {'Enabled' if self.config.enable_synaptic_scaling else 'Disabled'}")
        report.append(f"Plasticity: {'Enabled' if self.config.enable_plasticity else 'Disabled'}")
        report.append(f"Homeostasis: {'Enabled' if self.config.enable_homeostasis else 'Disabled'}")
        report.append(f"Noise: {'Enabled' if self.config.enable_noise else 'Disabled'}")
        report.append(f"Noise Strength: {self.config.noise_strength}")
        
        # Results
        report.append("\nNEUROMORPHIC COMPUTING RESULTS:")
        report.append("-" * 35)
        report.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        report.append(f"Start Time: {results.get('start_time', 'Unknown')}")
        report.append(f"End Time: {results.get('end_time', 'Unknown')}")
        
        # Stage results
        if 'stages' in results:
            for stage_name, stage_data in results['stages'].items():
                report.append(f"\n{stage_name.upper()}:")
                report.append("-" * len(stage_name))
                
                if isinstance(stage_data, dict):
                    for key, value in stage_data.items():
                        if key != 'spike_history':  # Skip large spike history
                            report.append(f"  {key}: {value}")
        
        return "\n".join(report)
    
    def visualize_neuromorphic_results(self, save_path: str = None):
        """Visualize neuromorphic computing results"""
        if not self.neuromorphic_history:
            logger.warning("No neuromorphic computing history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Spike raster plot
        if self.network.spike_history:
            spike_times = []
            neuron_ids = []
            
            for time, neuron_list in self.network.spike_history:
                for neuron_id in neuron_list:
                    spike_times.append(time)
                    neuron_ids.append(neuron_id)
            
            if spike_times and neuron_ids:
                axes[0, 0].scatter(spike_times, neuron_ids, s=1, alpha=0.6)
                axes[0, 0].set_xlabel('Time (ms)')
                axes[0, 0].set_ylabel('Neuron ID')
                axes[0, 0].set_title('Spike Raster Plot')
                axes[0, 0].grid(True)
        
        # Plot 2: Training progress
        if self.trainer.training_history:
            epochs = [h['epoch'] for h in self.trainer.training_history]
            losses = [h['loss'] for h in self.trainer.training_history]
            
            axes[0, 1].plot(epochs, losses, 'b-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Training Progress')
            axes[0, 1].grid(True)
        
        # Plot 3: Chip power consumption
        if self.chip.chip_state:
            power_values = [self.chip.chip_state['power_consumption']]
            axes[1, 0].bar(['Power Consumption'], power_values, color='red')
            axes[1, 0].set_ylabel('Power (W)')
            axes[1, 0].set_title('Chip Power Consumption')
        
        # Plot 4: Neuromorphic configuration
        config_values = [
            self.config.num_neurons,
            self.config.num_synapses,
            self.config.simulation_time,
            len(self.network.spike_history)
        ]
        config_labels = ['Neurons', 'Synapses', 'Sim Time (ms)', 'Spike Events']
        
        axes[1, 1].bar(config_labels, config_values, color=['blue', 'green', 'orange', 'red'])
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Neuromorphic Configuration')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

# Factory functions
def create_neuromorphic_config(**kwargs) -> NeuromorphicConfig:
    """Create neuromorphic configuration"""
    return NeuromorphicConfig(**kwargs)

def create_spiking_neuron(neuron_id: int, config: NeuromorphicConfig) -> SpikingNeuron:
    """Create spiking neuron"""
    return SpikingNeuron(neuron_id, config)

def create_synapse(synapse_id: int, pre_neuron_id: int, post_neuron_id: int, 
                  weight: float, config: NeuromorphicConfig) -> Synapse:
    """Create synapse"""
    return Synapse(synapse_id, pre_neuron_id, post_neuron_id, weight, config)

def create_spiking_neural_network(config: NeuromorphicConfig) -> SpikingNeuralNetwork:
    """Create spiking neural network"""
    return SpikingNeuralNetwork(config)

def create_event_driven_processor(config: NeuromorphicConfig) -> EventDrivenProcessor:
    """Create event-driven processor"""
    return EventDrivenProcessor(config)

def create_neuromorphic_chip(config: NeuromorphicConfig) -> NeuromorphicChip:
    """Create neuromorphic chip"""
    return NeuromorphicChip(config)

def create_neuromorphic_trainer(config: NeuromorphicConfig) -> NeuromorphicTrainer:
    """Create neuromorphic trainer"""
    return NeuromorphicTrainer(config)

def create_neuromorphic_accelerator(config: NeuromorphicConfig) -> NeuromorphicAccelerator:
    """Create neuromorphic accelerator"""
    return NeuromorphicAccelerator(config)

# Example usage
def example_neuromorphic_computing():
    """Example of neuromorphic computing system"""
    # Create configuration
    config = create_neuromorphic_config(
        neuron_model=NeuronModel.LEAKY_INTEGRATE_AND_FIRE,
        synapse_model=SynapseModel.STDP_SYNAPSE,
        num_neurons=100,
        num_synapses=1000,
        simulation_time=100.0,
        time_step=0.1,
        membrane_potential_threshold=-50.0,
        membrane_potential_reset=-70.0,
        membrane_time_constant=20.0,
        membrane_resistance=1.0,
        synaptic_delay=1.0,
        synaptic_weight_range=(0.0, 1.0),
        stdp_learning_rate=0.01,
        stdp_tau_plus=20.0,
        stdp_tau_minus=20.0,
        enable_event_driven=True,
        enable_spike_timing_dependent_plasticity=True,
        enable_adaptive_thresholds=True,
        enable_synaptic_scaling=True,
        enable_plasticity=True,
        enable_homeostasis=True,
        enable_noise=True,
        noise_strength=0.1
    )
    
    # Create neuromorphic accelerator
    neuromorphic_accelerator = create_neuromorphic_accelerator(config)
    
    # Create dummy input data
    input_data = [(np.random.randint(0, config.num_neurons), 
                  np.random.uniform(0, config.simulation_time)) 
                 for _ in range(10)]
    
    # Run neuromorphic computing
    neuromorphic_results = neuromorphic_accelerator.run_neuromorphic_computing(input_data)
    
    # Generate report
    neuromorphic_report = neuromorphic_accelerator.generate_neuromorphic_report(neuromorphic_results)
    
    print(f"✅ Neuromorphic Computing Example Complete!")
    print(f"🚀 Neuromorphic Computing Statistics:")
    print(f"   Neuron Model: {config.neuron_model.value}")
    print(f"   Synapse Model: {config.synapse_model.value}")
    print(f"   Number of Neurons: {config.num_neurons}")
    print(f"   Number of Synapses: {config.num_synapses}")
    print(f"   Simulation Time: {config.simulation_time} ms")
    print(f"   Time Step: {config.time_step} ms")
    print(f"   Membrane Potential Threshold: {config.membrane_potential_threshold} mV")
    print(f"   Membrane Potential Reset: {config.membrane_potential_reset} mV")
    print(f"   Membrane Time Constant: {config.membrane_time_constant} ms")
    print(f"   Membrane Resistance: {config.membrane_resistance} MOhm")
    print(f"   Synaptic Delay: {config.synaptic_delay} ms")
    print(f"   Synaptic Weight Range: {config.synaptic_weight_range}")
    print(f"   STDP Learning Rate: {config.stdp_learning_rate}")
    print(f"   STDP Tau Plus: {config.stdp_tau_plus} ms")
    print(f"   STDP Tau Minus: {config.stdp_tau_minus} ms")
    print(f"   Event-Driven: {'Enabled' if config.enable_event_driven else 'Disabled'}")
    print(f"   STDP: {'Enabled' if config.enable_spike_timing_dependent_plasticity else 'Disabled'}")
    print(f"   Adaptive Thresholds: {'Enabled' if config.enable_adaptive_thresholds else 'Disabled'}")
    print(f"   Synaptic Scaling: {'Enabled' if config.enable_synaptic_scaling else 'Disabled'}")
    print(f"   Plasticity: {'Enabled' if config.enable_plasticity else 'Disabled'}")
    print(f"   Homeostasis: {'Enabled' if config.enable_homeostasis else 'Disabled'}")
    print(f"   Noise: {'Enabled' if config.enable_noise else 'Disabled'}")
    print(f"   Noise Strength: {config.noise_strength}")
    
    print(f"\n📊 Neuromorphic Computing Results:")
    print(f"   Neuromorphic History Length: {len(neuromorphic_accelerator.neuromorphic_history)}")
    print(f"   Total Duration: {neuromorphic_results.get('total_duration', 0):.2f} seconds")
    
    # Show stage results summary
    if 'stages' in neuromorphic_results:
        for stage_name, stage_data in neuromorphic_results['stages'].items():
            print(f"   {stage_name}: {len(stage_data) if isinstance(stage_data, dict) else 'N/A'} results")
    
    print(f"\n📋 Neuromorphic Computing Report:")
    print(neuromorphic_report)
    
    return neuromorphic_accelerator

# Export utilities
__all__ = [
    'NeuronModel',
    'SynapseModel',
    'NeuromorphicConfig',
    'SpikingNeuron',
    'Synapse',
    'SpikingNeuralNetwork',
    'EventDrivenProcessor',
    'NeuromorphicChip',
    'NeuromorphicTrainer',
    'NeuromorphicAccelerator',
    'create_neuromorphic_config',
    'create_spiking_neuron',
    'create_synapse',
    'create_spiking_neural_network',
    'create_event_driven_processor',
    'create_neuromorphic_chip',
    'create_neuromorphic_trainer',
    'create_neuromorphic_accelerator',
    'example_neuromorphic_computing'
]

if __name__ == "__main__":
    example_neuromorphic_computing()
    print("✅ Neuromorphic computing example completed successfully!")