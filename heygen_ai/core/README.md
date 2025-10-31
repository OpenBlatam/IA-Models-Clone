# Enhanced Transformer Models - Refactored

This is a refactored version of the enhanced transformer models, organized into a clean, modular structure for better maintainability and usability.

## 📁 Project Structure

```
core/
├── __init__.py                 # Main package exports and factory functions
├── main.py                     # Demonstration and usage examples
├── README.md                   # This file
├── transformer_config.py       # Configuration classes and dataclasses
├── transformer_core.py         # Core transformer components
├── attention_mechanisms.py     # Advanced attention mechanisms
└── advanced_architectures.py   # Advanced transformer architectures
```

## 🚀 Quick Start

```python
from core import TransformerConfig, create_transformer_model

# Create configuration
config = TransformerConfig(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=1024,
    dropout=0.1,
    enable_lora=True,
    lora_rank=16
)

# Create model
model = create_transformer_model(config, "standard")

# Use model
input_ids = torch.randint(0, config.vocab_size, (2, 10))
outputs = model(input_ids)
```

## 📋 Features

### Core Components
- **TransformerConfig**: Comprehensive configuration management
- **CustomTransformerModel**: Base transformer implementation
- **MultiHeadAttention**: Standard multi-head attention
- **TransformerBlock**: Complete transformer block
- **LoRALayer**: Low-Rank Adaptation for efficient fine-tuning
- **Positional Encodings**: Rotary, Relative, and Standard positional encodings

### Attention Mechanisms
- **SparseAttention**: Efficient sparse attention patterns
- **LinearAttention**: O(n) complexity linear attention
- **MemoryEfficientAttention**: Memory-optimized attention with chunking
- **AdaptiveAttention**: Learns optimal attention patterns
- **CausalAttention**: Causal attention for autoregressive models
- **SymbolicAttention**: Symbolic reasoning attention
- **QuantumAttention**: Quantum-inspired attention with entanglement
- **BiologicalAttention**: Biologically-inspired attention with plasticity
- **EventDrivenAttention**: Event-driven attention for neuromorphic computing
- **HyperdimensionalAttention**: Hyperdimensional attention with binding
- **SwarmAttention**: Swarm intelligence-based attention

### Advanced Architectures
- **MixtureOfExperts**: MoE for efficient scaling
- **SwitchTransformerBlock**: Switch transformer with routing
- **SparseTransformerBlock**: Sparse attention transformer
- **AdaptiveTransformerBlock**: Dynamic scaling transformer
- **DynamicLayerScaling**: Adaptive depth scaling
- **NeuralArchitectureSearch**: NAS for optimal design
- **ModelEnsemble**: Ensemble methods for improved performance

### Quantum Computing Features
- **QuantumGate**: Base quantum gate implementation
- **HadamardGate**: Quantum superposition gate
- **PauliX/Y/ZGate**: Quantum rotation gates
- **CNOTGate**: Quantum entanglement gate
- **QuantumEntanglement**: Quantum entanglement mechanism
- **QuantumSuperposition**: Quantum superposition mechanism
- **QuantumMeasurement**: Quantum measurement and collapse
- **QuantumNeuralNetwork**: Quantum neural network layer
- **QuantumAttention**: Quantum-inspired attention
- **QuantumTransformerBlock**: Quantum-enhanced transformer block
- **QuantumOptimization**: Quantum-inspired optimization

### Biological Neural Network Features
- **NeuralPlasticity**: Activity-dependent plasticity
- **SynapticScaling**: Synaptic scaling for stability
- **HomeostaticMechanism**: Homeostatic regulation
- **AdaptiveThreshold**: Adaptive threshold mechanism
- **MemoryConsolidation**: Memory consolidation system
- **BiologicalAttention**: Biologically-inspired attention
- **BiologicalTransformerBlock**: Biological transformer block

### Neuromorphic Computing Features
- **SpikeEncoder**: Spike encoding mechanism
- **TemporalProcessor**: Temporal processing for spikes
- **EventDrivenAttention**: Event-driven attention
- **EnergyEfficientProcessing**: Energy-efficient processing
- **NeuromorphicMemory**: Event-driven memory system
- **NeuromorphicTransformerBlock**: Neuromorphic transformer block

### Hyperdimensional Computing Features
- **HyperdimensionalEncoder**: High-dimensional encoding
- **HyperdimensionalBinding**: Information binding operations
- **HyperdimensionalBundling**: Information bundling operations
- **HyperdimensionalSimilarity**: Similarity computation
- **HyperdimensionalAttention**: Hyperdimensional attention
- **HyperdimensionalMemory**: Hyperdimensional memory system
- **HyperdimensionalReasoning**: Symbolic reasoning operations
- **HyperdimensionalTransformerBlock**: Hyperdimensional transformer block

### Swarm Intelligence Features
- **ParticleSwarmOptimization**: PSO algorithm
- **AntColonyOptimization**: ACO algorithm
- **BeeAlgorithm**: Bee optimization algorithm
- **FireflyAlgorithm**: Firefly optimization algorithm
- **SwarmCoordination**: Multi-swarm coordination
- **SwarmAttention**: Swarm-based attention
- **SwarmTransformerBlock**: Swarm transformer block

## 🔧 Usage Examples

### Basic Usage

```python
from core import TransformerConfig, create_transformer_model

# Create configuration
config = TransformerConfig(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12
)

# Create different model types
standard_model = create_transformer_model(config, "standard")
sparse_model = create_transformer_model(config, "sparse")
switch_model = create_transformer_model(config, "switch")
adaptive_model = create_transformer_model(config, "adaptive")
quantum_model = create_transformer_model(config, "quantum")
biological_model = create_transformer_model(config, "biological")
neuromorphic_model = create_transformer_model(config, "neuromorphic")
hyperdimensional_model = create_transformer_model(config, "hyperdimensional")
swarm_model = create_transformer_model(config, "swarm")
```

### Advanced Features Usage

```python
# Quantum Computing Features
from core import QuantumNeuralNetwork, QuantumAttention, QuantumTransformerBlock

# Create quantum neural network
qnn = QuantumNeuralNetwork(768, 768, num_qubits=8, num_layers=3)
quantum_output = qnn(input_tensor)

# Create quantum attention
quantum_attention = QuantumAttention(768, 12, num_qubits=8)
attn_output, attn_weights = quantum_attention(query, key, value)

# Biological Features
from core import NeuralPlasticity, BiologicalAttention, BiologicalTransformerBlock

# Create neural plasticity mechanism
plasticity = NeuralPlasticity(768, plasticity_rate=0.01)
plasticity_output = plasticity(input_tensor, timestep=0)

# Create biological attention
bio_attention = BiologicalAttention(768, 12, plasticity_rate=0.01)
bio_output, bio_weights = bio_attention(query, key, value)

# Neuromorphic Features
from core import SpikeEncoder, EventDrivenAttention, NeuromorphicTransformerBlock

# Create spike encoder
spike_encoder = SpikeEncoder(768, 768, spike_threshold=1.0)
spikes = spike_encoder(input_tensor)

# Create event-driven attention
event_attention = EventDrivenAttention(768, 12)
event_output, event_weights = event_attention(query, key, value)

# Hyperdimensional Features
from core import HyperdimensionalEncoder, HyperdimensionalAttention, HyperdimensionalTransformerBlock

# Create hyperdimensional encoder
hd_encoder = HyperdimensionalEncoder(768, hyperdim_size=10000)
hd_output = hd_encoder(input_tensor)

# Create hyperdimensional attention
hd_attention = HyperdimensionalAttention(768, 12, hyperdim_size=10000)
hd_attn_output, hd_attn_weights = hd_attention(query, key, value)

# Swarm Intelligence Features
from core import ParticleSwarmOptimization, SwarmAttention, SwarmTransformerBlock

# Create particle swarm optimization
pso = ParticleSwarmOptimization(num_particles=50, num_dimensions=768)
pso_output = pso(input_tensor)

# Create swarm attention
swarm_attention = SwarmAttention(768, 12, num_swarms=4)
swarm_output, swarm_weights = swarm_attention(query, key, value)
```

### Advanced Attention

```python
from core import create_attention_mechanism

# Create different attention mechanisms
sparse_attention = create_attention_mechanism("sparse", config)
linear_attention = create_attention_mechanism("linear", config)
adaptive_attention = create_attention_mechanism("adaptive", config)
```

### Model Information

```python
from core import get_model_info

# Get model information
info = get_model_info(model)
print(f"Parameters: {info['total_parameters']:,}")
print(f"Size: {info['model_size_mb']:.2f} MB")
```

### Manager Class

```python
from core.main import TransformerManager

# Create manager
manager = TransformerManager(config)

# Create and compare models
comparison = manager.compare_models(["standard", "sparse", "switch"])
for model_type, info in comparison.items():
    print(f"{model_type}: {info['total_parameters']:,} params")
```

## 📊 Configuration Options

### Basic Parameters
- `vocab_size`: Vocabulary size
- `hidden_size`: Hidden dimension
- `num_layers`: Number of transformer layers
- `num_attention_heads`: Number of attention heads
- `intermediate_size`: Feed-forward intermediate size
- `max_position_embeddings`: Maximum sequence length
- `dropout`: Dropout rate

### LoRA Parameters
- `enable_lora`: Enable LoRA adaptation
- `lora_rank`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `lora_dropout`: LoRA dropout rate

### Performance Parameters
- `enable_ultra_performance`: Enable performance optimizations
- `performance_mode`: Performance mode ("balanced", "speed", "memory", "maximum")
- `enable_torch_compile`: Enable torch.compile optimization
- `enable_flash_attention`: Enable Flash Attention
- `enable_memory_optimization`: Enable memory optimizations
- `mixed_precision`: Enable mixed precision training

### Advanced Features
- `enable_sparse_attention`: Enable sparse attention
- `enable_linear_attention`: Enable linear attention
- `enable_memory_efficient_attention`: Enable memory-efficient attention
- `enable_adaptive_attention`: Enable adaptive attention

## 🧪 Testing

Run the demonstration:

```python
python main.py
```

This will show:
- Model comparison across different types
- Attention mechanism capabilities
- Advanced architecture features
- Memory usage and parameter counts

## 📈 Performance

The refactored models maintain the same performance as the original monolithic version while providing:

- **Better Organization**: Clean modular structure
- **Easier Maintenance**: Separated concerns
- **Improved Usability**: Simple factory functions
- **Better Documentation**: Clear interfaces
- **Type Safety**: Proper type hints
- **Memory Efficiency**: Optimized implementations

## 🔄 Migration from Original

The refactored version is designed to be a drop-in replacement for the original monolithic file:

```python
# Old way
from enhanced_transformer_models import TransformerConfig, CustomTransformerModel

# New way
from core import TransformerConfig, create_transformer_model
```

## 🤝 Contributing

When adding new features:

1. **Core Components**: Add to `transformer_core.py`
2. **Attention Mechanisms**: Add to `attention_mechanisms.py`
3. **Advanced Architectures**: Add to `advanced_architectures.py`
4. **Configuration**: Update `transformer_config.py`
5. **Exports**: Update `__init__.py`
6. **Documentation**: Update this README

## 📝 License

This project is part of the Enhanced Transformer Models package.

## 🆘 Support

For questions or issues, please refer to the main project documentation or create an issue in the project repository.
