from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import structlog
from pytorch_comprehensive_manager import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive PyTorch Management Demo

This demo showcases the complete PyTorch management system with:

- Device management and optimization
- Memory monitoring and optimization
- Model compilation and optimization
- Training pipeline creation
- Performance profiling
- Security validation
- Debugging capabilities
- Real-world usage examples
"""



    ComprehensivePyTorchManager, PyTorchConfig, DeviceType, OptimizationLevel,
    setup_pytorch_environment, get_optimal_config
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class SampleModel(nn.Module):
    """Sample model for demonstration."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 512, num_classes: int = 10):
        
    """__init__ function."""
super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x) -> Any:
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TransformerModel(nn.Module):
    """Sample transformer model for demonstration."""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        
    """__init__ function."""
super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x) -> Any:
        x = self.embedding(x) + self.pos_encoding[:x.size(1)]
        x = self.transformer(x)
        x = self.fc(x)
        return x


class PyTorchComprehensiveDemo:
    """Comprehensive PyTorch management demo."""
    
    def __init__(self) -> Any:
        self.results = {}
        self.manager = None
        
    async def run_device_management_demo(self) -> Dict:
        """Demonstrate device management capabilities."""
        logger.info("Starting Device Management Demo")
        
        # Test different device configurations
        device_configs = [
            DeviceType.AUTO,
            DeviceType.CPU
        ]
        
        if torch.cuda.is_available():
            device_configs.append(DeviceType.CUDA)
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_configs.append(DeviceType.MPS)
        
        device_results = {}
        
        for device_type in device_configs:
            logger.info(f"Testing device type: {device_type.value}")
            
            try:
                config = get_optimal_config(device_type)
                manager = setup_pytorch_environment(config)
                
                system_info = manager.get_system_info()
                device_results[device_type.value] = {
                    'success': True,
                    'system_info': system_info,
                    'device_info': system_info['device_info'],
                    'memory_stats': system_info['memory_stats']
                }
                
            except Exception as e:
                logger.error(f"Failed to setup {device_type.value}: {e}")
                device_results[device_type.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'device_configs': device_configs,
            'device_results': device_results,
            'best_device': self._select_best_device(device_results)
        }
    
    def _select_best_device(self, device_results: Dict) -> str:
        """Select the best available device."""
        priority_order = ['cuda', 'mps', 'cpu']
        
        for device in priority_order:
            if device in device_results and device_results[device]['success']:
                return device
        
        return 'cpu'
    
    async def run_model_optimization_demo(self) -> Dict:
        """Demonstrate model optimization capabilities."""
        logger.info("Starting Model Optimization Demo")
        
        # Create sample models
        models = {
            'simple_nn': SampleModel(),
            'transformer': TransformerModel()
        }
        
        optimization_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Optimizing model: {model_name}")
            
            model_results = {}
            
            for opt_level in OptimizationLevel:
                logger.info(f"Testing optimization level: {opt_level.value}")
                
                try:
                    # Create manager with optimal config
                    config = get_optimal_config()
                    manager = setup_pytorch_environment(config)
                    
                    # Optimize model
                    optimized_model = manager.optimize_model(model, opt_level)
                    
                    # Profile model
                    input_shape = (32, 784) if model_name == 'simple_nn' else (32, 100)
                    profile_results = manager.profile_model(optimized_model, input_shape)
                    
                    model_results[opt_level.value] = {
                        'success': True,
                        'profile_results': profile_results,
                        'model_size': sum(p.numel() for p in optimized_model.parameters())
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to optimize {model_name} with {opt_level.value}: {e}")
                    model_results[opt_level.value] = {
                        'success': False,
                        'error': str(e)
                    }
            
            optimization_results[model_name] = model_results
        
        return optimization_results
    
    async def run_training_pipeline_demo(self) -> Dict:
        """Demonstrate training pipeline creation."""
        logger.info("Starting Training Pipeline Demo")
        
        # Create sample data
        batch_size = 32
        input_size = 784
        num_classes = 10
        
        # Generate dummy data
        X_train = torch.randn(1000, input_size)
        y_train = torch.randint(0, num_classes, (1000,))
        X_val = torch.randn(200, input_size)
        y_val = torch.randint(0, num_classes, (200,))
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model and training pipeline
        model = SampleModel(input_size, 512, num_classes)
        
        config = get_optimal_config()
        manager = setup_pytorch_environment(config)
        
        pipeline = manager.create_training_pipeline(
            model, lr=1e-3, optimizer_type="adamw", scheduler_type="cosine"
        )
        
        # Training loop
        num_epochs = 3
        training_results = {
            'train_losses': [],
            'val_losses': [],
            'epoch_times': []
        }
        
        model = pipeline['model']
        optimizer = pipeline['optimizer']
        scheduler = pipeline['scheduler']
        trainer = pipeline['trainer']
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            logger.info(f"Training epoch {epoch + 1}/{num_epochs}")
            
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                batch = {'input': data, 'labels': target}
                step_result = trainer.train_step(model, optimizer, batch, criterion)
                train_loss += step_result['loss']
                
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}, Loss: {step_result['loss']:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    batch = {'input': data, 'labels': target}
                    step_result = trainer.validate_step(model, batch, criterion)
                    val_loss += step_result['loss']
            
            epoch_time = time.time() - epoch_start
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            training_results['train_losses'].append(avg_train_loss)
            training_results['val_losses'].append(avg_val_loss)
            training_results['epoch_times'].append(epoch_time)
            
            scheduler.step()
            
            logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        return {
            'training_results': training_results,
            'final_model_state': model.state_dict(),
            'memory_stats': manager.memory_manager.get_memory_stats()
        }
    
    async def run_memory_optimization_demo(self) -> Dict:
        """Demonstrate memory optimization capabilities."""
        logger.info("Starting Memory Optimization Demo")
        
        config = get_optimal_config()
        manager = setup_pytorch_environment(config)
        
        memory_results = {
            'initial_stats': manager.memory_manager.get_memory_stats(),
            'memory_tracking': [],
            'optimization_results': {}
        }
        
        # Test memory tracking
        with manager.memory_manager.memory_tracking("memory_test"):
            # Create large tensors
            large_tensor = torch.randn(1000, 1000, 1000)
            large_tensor = large_tensor.to(manager.device_manager.device)
            
            # Perform operations
            result = torch.matmul(large_tensor, large_tensor)
            result = torch.sum(result)
            
            memory_results['memory_tracking'].append({
                'operation': 'large_matrix_multiplication',
                'result': result.item()
            })
        
        # Test cache clearing
        manager.memory_manager.clear_cache()
        memory_results['after_cache_clear'] = manager.memory_manager.get_memory_stats()
        
        # Test different optimization levels
        model = SampleModel()
        
        for opt_level in OptimizationLevel:
            with manager.memory_manager.memory_tracking(f"optimization_{opt_level.value}"):
                optimized_model = manager.optimize_model(model, opt_level)
                
                # Profile memory usage
                input_tensor = torch.randn(32, 784)
                input_tensor = input_tensor.to(manager.device_manager.device)
                
                with torch.no_grad():
                    output = optimized_model(input_tensor)
                
                memory_results['optimization_results'][opt_level.value] = {
                    'model_size': sum(p.numel() for p in optimized_model.parameters()),
                    'output_shape': output.shape,
                    'memory_stats': manager.memory_manager.get_memory_stats()
                }
        
        return memory_results
    
    async def run_security_validation_demo(self) -> Dict:
        """Demonstrate security validation capabilities."""
        logger.info("Starting Security Validation Demo")
        
        config = get_optimal_config()
        manager = setup_pytorch_environment(config)
        
        security_results = {
            'input_validation': {},
            'model_security': {},
            'output_sanitization': {}
        }
        
        # Test input validation
        valid_inputs = {
            'normal': torch.randn(32, 784),
            'large_values': torch.randn(32, 784) * 1e8,
            'nan_values': torch.tensor([float('nan')] * 784).unsqueeze(0).repeat(32, 1),
            'inf_values': torch.tensor([float('inf')] * 784).unsqueeze(0).repeat(32, 1)
        }
        
        for input_name, input_tensor in valid_inputs.items():
            is_valid = manager.security_manager.validate_inputs({'input': input_tensor})
            security_results['input_validation'][input_name] = {
                'is_valid': is_valid,
                'has_nan': torch.isnan(input_tensor).any().item(),
                'has_inf': torch.isinf(input_tensor).any().item(),
                'max_value': torch.abs(input_tensor).max().item()
            }
        
        # Test model security
        model = SampleModel()
        model_security = manager.security_manager.check_model_security(model)
        security_results['model_security'] = model_security
        
        # Test output sanitization
        model = model.to(manager.device_manager.device)
        input_tensor = torch.randn(32, 784).to(manager.device_manager.device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
            # Introduce NaN and Inf values
            output[0, 0] = float('nan')
            output[0, 1] = float('inf')
            
            original_output = output.clone()
            sanitized_output = manager.security_manager.sanitize_outputs(output)
            
            security_results['output_sanitization'] = {
                'original_has_nan': torch.isnan(original_output).any().item(),
                'original_has_inf': torch.isinf(original_output).any().item(),
                'sanitized_has_nan': torch.isnan(sanitized_output).any().item(),
                'sanitized_has_inf': torch.isinf(sanitized_output).any().item()
            }
        
        return security_results
    
    async def run_performance_profiling_demo(self) -> Dict:
        """Demonstrate performance profiling capabilities."""
        logger.info("Starting Performance Profiling Demo")
        
        config = get_optimal_config()
        manager = setup_pytorch_environment(config)
        
        profiling_results = {
            'model_profiling': {},
            'training_profiling': {},
            'memory_profiling': {}
        }
        
        # Test different models
        models = {
            'simple_nn': SampleModel(),
            'transformer': TransformerModel()
        }
        
        for model_name, model in models.items():
            logger.info(f"Profiling model: {model_name}")
            
            # Profile inference
            input_shape = (32, 784) if model_name == 'simple_nn' else (32, 100)
            profile_results = manager.profile_model(model, input_shape)
            
            profiling_results['model_profiling'][model_name] = profile_results
        
        # Profile training
        model = SampleModel()
        pipeline = manager.create_training_pipeline(model)
        
        with manager.trainer.profiling_context("training_profiling"):
            # Simulate training step
            optimizer = pipeline['optimizer']
            criterion = nn.CrossEntropyLoss()
            
            data = torch.randn(32, 784).to(manager.device_manager.device)
            target = torch.randint(0, 10, (32,)).to(manager.device_manager.device)
            
            batch = {'input': data, 'labels': target}
            step_result = manager.trainer.train_step(model, optimizer, batch, criterion)
            
            profiling_results['training_profiling'] = {
                'step_result': step_result,
                'memory_stats': manager.memory_manager.get_memory_stats()
            }
        
        # Memory profiling
        profiling_results['memory_profiling'] = {
            'initial_stats': manager.memory_manager.get_memory_stats(),
            'peak_stats': manager.memory_manager.get_memory_stats()
        }
        
        return profiling_results
    
    async def run_comprehensive_demo(self) -> Dict:
        """Run comprehensive PyTorch management demo."""
        logger.info("Starting Comprehensive PyTorch Management Demo")
        
        results = {}
        
        try:
            # Run individual demos
            results['device_management'] = await self.run_device_management_demo()
            results['model_optimization'] = await self.run_model_optimization_demo()
            results['training_pipeline'] = await self.run_training_pipeline_demo()
            results['memory_optimization'] = await self.run_memory_optimization_demo()
            results['security_validation'] = await self.run_security_validation_demo()
            results['performance_profiling'] = await self.run_performance_profiling_demo()
            
            # Generate comprehensive report
            comprehensive_report = self._generate_comprehensive_report(results)
            results['comprehensive_report'] = comprehensive_report
            
            # Save results
            self._save_results(results)
            
            # Generate visualizations
            self.plot_results(results)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_comprehensive_report(self, results: Dict) -> Dict:
        """Generate comprehensive report from all demo results."""
        report = {
            'summary': {},
            'recommendations': [],
            'performance_metrics': {},
            'system_analysis': {}
        }
        
        # Device analysis
        device_results = results['device_management']
        best_device = device_results['best_device']
        report['summary']['best_device'] = best_device
        report['summary']['available_devices'] = list(device_results['device_results'].keys())
        
        # Performance analysis
        model_opt_results = results['model_optimization']
        best_optimization = self._find_best_optimization(model_opt_results)
        report['performance_metrics']['best_optimization'] = best_optimization
        
        # Training analysis
        training_results = results['training_pipeline']['training_results']
        report['performance_metrics']['training'] = {
            'final_train_loss': training_results['train_losses'][-1],
            'final_val_loss': training_results['val_losses'][-1],
            'avg_epoch_time': np.mean(training_results['epoch_times']),
            'total_training_time': sum(training_results['epoch_times'])
        }
        
        # Memory analysis
        memory_results = results['memory_optimization']
        report['system_analysis']['memory'] = {
            'initial_memory': memory_results['initial_stats'],
            'optimization_impact': memory_results['optimization_results']
        }
        
        # Security analysis
        security_results = results['security_validation']
        report['system_analysis']['security'] = {
            'input_validation_passed': all(
                result['is_valid'] for result in security_results['input_validation'].values()
                if 'normal' in result.get('input_name', '')
            ),
            'model_security_passed': security_results['model_security']['is_valid'],
            'output_sanitization_working': not security_results['output_sanitization']['sanitized_has_nan']
        }
        
        # Generate recommendations
        if best_device != 'cuda':
            report['recommendations'].append("Consider using CUDA for better performance")
        
        if best_optimization != 'maximum':
            report['recommendations'].append("Consider using maximum optimization level")
        
        if training_results['val_losses'][-1] > training_results['train_losses'][-1]:
            report['recommendations'].append("Model may be overfitting - consider regularization")
        
        return report
    
    def _find_best_optimization(self, model_opt_results: Dict) -> str:
        """Find the best optimization level based on performance."""
        best_level = 'none'
        best_time = float('inf')
        
        for model_name, results in model_opt_results.items():
            for opt_level, result in results.items():
                if result['success'] and 'profile_results' in result:
                    inference_time = result['profile_results']['inference_time']
                    if inference_time < best_time:
                        best_time = inference_time
                        best_level = opt_level
        
        return best_level
    
    def plot_results(self, results: Dict, save_path: str = "pytorch_comprehensive_results.png"):
        """Plot comprehensive results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Device Performance
        device_results = results['device_management']['device_results']
        devices = list(device_results.keys())
        success_rates = [1 if result['success'] else 0 for result in device_results.values()]
        
        axes[0, 0].bar(devices, success_rates)
        axes[0, 0].set_title('Device Setup Success Rate')
        axes[0, 0].set_ylabel('Success Rate')
        
        # Plot 2: Model Optimization Performance
        model_opt_results = results['model_optimization']
        opt_levels = list(OptimizationLevel)
        
        for model_name in model_opt_results.keys():
            times = []
            for opt_level in opt_levels:
                result = model_opt_results[model_name].get(opt_level.value, {})
                if result.get('success') and 'profile_results' in result:
                    times.append(result['profile_results']['inference_time'])
                else:
                    times.append(float('inf'))
            
            axes[0, 1].plot([level.value for level in opt_levels], times, marker='o', label=model_name)
        
        axes[0, 1].set_title('Model Optimization Performance')
        axes[0, 1].set_xlabel('Optimization Level')
        axes[0, 1].set_ylabel('Inference Time (s)')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Plot 3: Training Progress
        training_results = results['training_pipeline']['training_results']
        epochs = range(1, len(training_results['train_losses']) + 1)
        
        axes[0, 2].plot(epochs, training_results['train_losses'], label='Train Loss', marker='o')
        axes[0, 2].plot(epochs, training_results['val_losses'], label='Val Loss', marker='s')
        axes[0, 2].set_title('Training Progress')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        
        # Plot 4: Memory Usage
        memory_results = results['memory_optimization']
        opt_levels = list(memory_results['optimization_results'].keys())
        
        if 'gpu_memory' in memory_results['initial_stats']:
            initial_memory = memory_results['initial_stats']['gpu_memory']['allocated'] / 1e9
            memory_usage = [initial_memory]
            
            for opt_level in opt_levels:
                if 'gpu_memory' in memory_results['optimization_results'][opt_level]['memory_stats']:
                    memory = memory_results['optimization_results'][opt_level]['memory_stats']['gpu_memory']['allocated'] / 1e9
                    memory_usage.append(memory)
                else:
                    memory_usage.append(0)
            
            axes[1, 0].bar(['Initial'] + opt_levels, memory_usage)
            axes[1, 0].set_title('GPU Memory Usage by Optimization Level')
            axes[1, 0].set_ylabel('Memory (GB)')
        
        # Plot 5: Security Validation
        security_results = results['security_validation']
        input_validation = security_results['input_validation']
        
        input_names = list(input_validation.keys())
        validation_scores = [result['is_valid'] for result in input_validation.values()]
        
        axes[1, 1].bar(input_names, validation_scores)
        axes[1, 1].set_title('Input Validation Results')
        axes[1, 1].set_ylabel('Validation Passed')
        
        # Plot 6: Performance Profiling
        profiling_results = results['performance_profiling']['model_profiling']
        model_names = list(profiling_results.keys())
        inference_times = [result['inference_time'] for result in profiling_results.values()]
        
        axes[1, 2].bar(model_names, inference_times)
        axes[1, 2].set_title('Model Inference Performance')
        axes[1, 2].set_ylabel('Inference Time (s)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results plot saved to {save_path}")
    
    def _save_results(self, results: Dict):
        """Save demo results to file."""
        output_path = Path("pytorch_comprehensive_results.json")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj) -> Any:
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            else:
                return str(obj)
        
        serializable_results = convert_numpy(results)
        
        with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Demo results saved to {output_path}")


async def main():
    """Main demo function."""
    logger.info("PyTorch Comprehensive Management Demo")
    
    # Create demo instance
    demo = PyTorchComprehensiveDemo()
    
    # Run comprehensive demo
    results = await demo.run_comprehensive_demo()
    
    # Print summary
    logger.info("Demo completed successfully!")
    
    if 'comprehensive_report' in results:
        report = results['comprehensive_report']
        logger.info("PyTorch System Summary:")
        logger.info(f"  Best Device: {report['summary']['best_device']}")
        logger.info(f"  Best Optimization: {report['performance_metrics']['best_optimization']}")
        logger.info(f"  Final Training Loss: {report['performance_metrics']['training']['final_train_loss']:.4f}")
        logger.info(f"  Final Validation Loss: {report['performance_metrics']['training']['final_val_loss']:.4f}")
        
        if report['recommendations']:
            logger.info("Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 