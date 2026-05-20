#!/usr/bin/env python3
"""
Launcher Script for Advanced Noise Scheduler and Sampling System

This script provides a command-line interface to:
- Create and manage diffusion pipelines
- Compare different schedulers and samplers
- Run performance benchmarks
- Interactive configuration management
- Export/import configurations
"""

import argparse
import sys
import os
import time
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from noise_scheduler_sampling_system import (
    AdvancedDiffusionSystem,
    NoiseSchedulerConfig,
    BetaSchedule,
    SamplingMethod
)

def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 Advanced Noise Scheduler & Sampling System 🚀         ║
║                                                                              ║
║  Features:                                                                   ║
║  • Multiple Beta Schedules (Linear, Cosine, Quadratic, Sigmoid, Exponential) ║
║  • Advanced Sampling Methods (DDPM, DDIM, Ancestral, Euler, Heun)          ║
║  • Pipeline Comparison & Benchmarking                                       ║
║  • YAML/JSON Configuration Management                                       ║
║  • Performance Optimization & GPU Support                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_help():
    """Print help information."""
    help_text = """
📖 Available Commands:

🔧 Pipeline Management:
  create <name> <scheduler> <sampler> [options]  - Create new pipeline
  list                                           - List all pipelines
  info <name>                                    - Show pipeline details
  delete <name>                                  - Delete pipeline

⚡ Sampling & Generation:
  sample <name> <shape> [options]                - Generate samples
  compare <shape> [options]                      - Compare all pipelines

📊 Performance & Analysis:
  benchmark <name> [options]                     - Run performance benchmark
  profile <name> [options]                       - Profile pipeline performance

⚙️ Configuration:
  config save <file>                             - Save current configuration
  config load <file>                             - Load configuration from file
  config export <file>                           - Export configuration
  config import <file>                           - Import configuration

📈 Examples:
  create my_pipeline cosine ddim --timesteps 1000
  sample my_pipeline "1,3,64,64" --steps 50
  compare "1,3,64,64" --steps 20
  benchmark my_pipeline --iterations 10

💡 Tips:
  • Use quotes for shape parameters: "1,3,64,64"
  • Available schedulers: linear, cosine, quadratic, sigmoid, exponential
  • Available samplers: ddpm, ddim, ancestral, euler, heun
  • Use --help for detailed option information
"""
    print(help_text)

def parse_shape(shape_str: str) -> Tuple[int, ...]:
    """Parse shape string into tuple."""
    try:
        return tuple(int(x.strip()) for x in shape_str.split(','))
    except ValueError:
        raise ValueError(f"Invalid shape format: {shape_str}. Use format: '1,3,64,64'")

def create_pipeline(args):
    """Create a new pipeline."""
    try:
        # Parse beta schedule
        beta_schedule_map = {
            'linear': BetaSchedule.LINEAR,
            'cosine': BetaSchedule.COSINE,
            'quadratic': BetaSchedule.QUADRATIC,
            'sigmoid': BetaSchedule.SIGMOID,
            'exponential': BetaSchedule.EXPONENTIAL
        }
        
        if args.scheduler not in beta_schedule_map:
            print(f"❌ Error: Unknown scheduler '{args.scheduler}'")
            print(f"   Available schedulers: {', '.join(beta_schedule_map.keys())}")
            return False
        
        # Parse sampling method
        sampling_method_map = {
            'ddpm': SamplingMethod.DDPM,
            'ddim': SamplingMethod.DDIM,
            'ancestral': SamplingMethod.ANCESTRAL,
            'euler': SamplingMethod.EULER,
            'heun': SamplingMethod.HEUN
        }
        
        if args.sampler not in sampling_method_map:
            print(f"❌ Error: Unknown sampler '{args.sampler}'")
            print(f"   Available samplers: {', '.join(sampling_method_map.keys())}")
            return False
        
        # Create configuration
        config = NoiseSchedulerConfig(
            num_train_timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=beta_schedule_map[args.scheduler],
            clip_sample=args.clip_sample,
            prediction_type=args.prediction_type
        )
        
        # Create pipeline
        pipeline = args.system.create_pipeline(
            args.name,
            config,
            sampling_method_map[args.sampler]
        )
        
        print(f"✅ Successfully created pipeline '{args.name}'")
        print(f"   Scheduler: {args.scheduler}")
        print(f"   Sampler: {args.sampler}")
        print(f"   Timesteps: {args.timesteps}")
        print(f"   Beta range: {args.beta_start} → {args.beta_end}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating pipeline: {e}")
        return False

def list_pipelines(args):
    """List all pipelines."""
    pipelines = args.system.list_pipelines()
    
    if not pipelines:
        print("📭 No pipelines found.")
        return
    
    print(f"📋 Found {len(pipelines)} pipeline(s):")
    print("-" * 60)
    
    for name in pipelines:
        config = args.system.configs[name]
        pipeline = args.system.pipelines[name]
        
        print(f"🔧 {name}")
        print(f"   Scheduler: {config.beta_schedule.value}")
        print(f"   Sampler: {pipeline.sampler.__class__.__name__}")
        print(f"   Timesteps: {config.num_train_timesteps}")
        print(f"   Beta: {config.beta_start:.4f} → {config.beta_end:.4f}")
        print()

def show_pipeline_info(args):
    """Show detailed pipeline information."""
    try:
        pipeline = args.system.get_pipeline(args.name)
        config = args.system.configs[args.name]
        
        print(f"🔍 Pipeline: {args.name}")
        print("=" * 50)
        print(f"Scheduler: {config.beta_schedule.value}")
        print(f"Sampler: {pipeline.sampler.__class__.__name__}")
        print(f"Timesteps: {config.num_train_timesteps}")
        print(f"Beta Start: {config.beta_start}")
        print(f"Beta End: {config.beta_end}")
        print(f"Clip Sample: {config.clip_sample}")
        print(f"Prediction Type: {config.prediction_type}")
        print(f"Device: {pipeline.device}")
        
        # Show beta schedule visualization
        print(f"\n📊 Beta Schedule Preview:")
        betas = pipeline.scheduler.betas
        if len(betas) > 10:
            print(f"   First 5: {betas[:5].tolist()}")
            print(f"   Middle: {betas[len(betas)//2-2:len(betas)//2+3].tolist()}")
            print(f"   Last 5: {betas[-5:].tolist()}")
        else:
            print(f"   All: {betas.tolist()}")
            
    except ValueError as e:
        print(f"❌ Error: {e}")

def delete_pipeline(args):
    """Delete a pipeline."""
    try:
        if args.name in args.system.pipelines:
            del args.system.pipelines[args.name]
            del args.system.configs[args.name]
            print(f"✅ Successfully deleted pipeline '{args.name}'")
        else:
            print(f"❌ Pipeline '{args.name}' not found")
    except Exception as e:
        print(f"❌ Error deleting pipeline: {e}")

def sample_pipeline(args):
    """Generate samples using a pipeline."""
    try:
        pipeline = args.system.get_pipeline(args.name)
        shape = parse_shape(args.shape)
        
        print(f"🎨 Generating samples with pipeline '{args.name}'...")
        print(f"   Shape: {shape}")
        print(f"   Steps: {args.steps}")
        print(f"   Guidance: {args.guidance_scale}")
        print(f"   Classifier-free: {args.classifier_free}")
        
        # Create dummy model for testing
        import torch.nn as nn
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, t, **kwargs):
                return torch.randn_like(x)
        
        model = DummyModel()
        
        # Generate samples
        start_time = time.time()
        samples = pipeline.sample(
            model=model,
            shape=shape,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            classifier_free_guidance=args.classifier_free
        )
        end_time = time.time()
        
        print(f"✅ Generation completed in {end_time - start_time:.2f}s")
        print(f"   Output shape: {samples.shape}")
        print(f"   Output range: {samples.min().item():.3f} to {samples.max().item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during sampling: {e}")
        return False

def compare_pipelines(args):
    """Compare all pipelines."""
    pipelines = args.system.list_pipelines()
    
    if not pipelines:
        print("📭 No pipelines to compare.")
        return
    
    shape = parse_shape(args.shape)
    print(f"🔍 Comparing {len(pipelines)} pipeline(s) with shape {shape}...")
    print(f"   Steps: {args.steps}")
    print("-" * 60)
    
    results = args.system.compare_schedulers(
        shape=shape,
        num_inference_steps=args.steps
    )
    
    print("\n📊 Comparison Results:")
    print("-" * 60)
    
    for name, result in results.items():
        if result is not None:
            print(f"✅ {name}: Success (shape: {result.shape})")
        else:
            print(f"❌ {name}: Failed")

def benchmark_pipeline(args):
    """Run performance benchmark on a pipeline."""
    try:
        pipeline = args.system.get_pipeline(args.name)
        shape = parse_shape(args.shape)
        
        print(f"⚡ Running benchmark on pipeline '{args.name}'...")
        print(f"   Shape: {shape}")
        print(f"   Steps: {args.steps}")
        print(f"   Iterations: {args.iterations}")
        print("-" * 60)
        
        # Create dummy model
        import torch.nn as nn
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, t, **kwargs):
                return torch.randn_like(x)
        
        model = DummyModel()
        
        # Run benchmark
        times = []
        memory_usage = []
        
        for i in range(args.iterations):
            print(f"   Iteration {i+1}/{args.iterations}...", end=" ")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Get initial memory
            if hasattr(torch.cuda, 'memory_allocated'):
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            else:
                initial_memory = 0
            
            # Time the operation
            start_time = time.time()
            samples = pipeline.sample(
                model=model,
                shape=shape,
                num_inference_steps=args.steps,
                classifier_free_guidance=False
            )
            end_time = time.time()
            
            # Get final memory
            if hasattr(torch.cuda, 'memory_allocated'):
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                memory_usage.append(final_memory - initial_memory)
            
            iteration_time = end_time - start_time
            times.append(iteration_time)
            
            print(f"✓ ({iteration_time:.3f}s)")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print("\n📊 Benchmark Results:")
        print("-" * 60)
        print(f"Average Time: {avg_time:.3f}s")
        print(f"Min Time: {min_time:.3f}s")
        print(f"Max Time: {max_time:.3f}s")
        print(f"Total Time: {sum(times):.3f}s")
        
        if memory_usage:
            avg_memory = sum(memory_usage) / len(memory_usage)
            print(f"Average Memory Increase: {avg_memory / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        return False

def save_configuration(args):
    """Save current configuration to file."""
    try:
        args.system.save_config(args.file)
        print(f"✅ Configuration saved to '{args.file}'")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def load_configuration(args):
    """Load configuration from file."""
    try:
        new_system = AdvancedDiffusionSystem(args.file)
        print(f"✅ Configuration loaded from '{args.file}'")
        print(f"   Loaded {len(new_system.pipelines)} pipeline(s)")
        
        # Replace current system
        args.system.pipelines = new_system.pipelines
        args.system.configs = new_system.configs
        args.system.config_path = new_system.config_path
        
        return True
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def export_configuration(args):
    """Export configuration in different formats."""
    try:
        # Determine format from file extension
        if args.file.endswith('.json'):
            # Export as JSON
            config_data = {}
            for name, config in args.system.configs.items():
                config_data[name] = {
                    'num_train_timesteps': config.num_train_timesteps,
                    'beta_start': config.beta_start,
                    'beta_end': config.beta_end,
                    'beta_schedule': config.beta_schedule.value,
                    'clip_sample': config.clip_sample,
                    'prediction_type': config.prediction_type
                }
            
            with open(args.file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration exported to JSON: '{args.file}'")
            
        elif args.file.endswith('.yaml') or args.file.endswith('.yml'):
            # Export as YAML
            config_data = {}
            for name, config in args.system.configs.items():
                config_data[name] = {
                    'num_train_timesteps': config.num_train_timesteps,
                    'beta_start': config.beta_start,
                    'beta_end': config.beta_end,
                    'beta_schedule': config.beta_schedule.value,
                    'clip_sample': config.clip_sample,
                    'prediction_type': config.prediction_type
                }
            
            with open(args.file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            print(f"✅ Configuration exported to YAML: '{args.file}'")
            
        else:
            print(f"❌ Unsupported file format. Use .json, .yaml, or .yml")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting configuration: {e}")
        return False

def interactive_mode(args):
    """Run interactive mode."""
    print("🎮 Entering interactive mode...")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 60)
    
    while True:
        try:
            command = input("🚀 > ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if command.lower() == 'help':
                print_help()
                continue
            
            # Parse command
            parts = command.split()
            if not parts:
                continue
            
            cmd = parts[0].lower()
            
            if cmd == 'create' and len(parts) >= 4:
                # create name scheduler sampler [options]
                name, scheduler, sampler = parts[1:4]
                
                # Parse options
                timesteps = 1000
                beta_start = 0.0001
                beta_end = 0.02
                
                for i, part in enumerate(parts[4:], 4):
                    if part == '--timesteps' and i + 1 < len(parts):
                        timesteps = int(parts[i + 1])
                    elif part == '--beta-start' and i + 1 < len(parts):
                        beta_start = float(parts[i + 1])
                    elif part == '--beta-end' and i + 1 < len(parts):
                        beta_end = float(parts[i + 1])
                
                # Create pipeline
                args.name = name
                args.scheduler = scheduler
                args.sampler = sampler
                args.timesteps = timesteps
                args.beta_start = beta_start
                args.beta_end = beta_end
                create_pipeline(args)
                
            elif cmd == 'list':
                list_pipelines(args)
                
            elif cmd == 'info' and len(parts) >= 2:
                args.name = parts[1]
                show_pipeline_info(args)
                
            elif cmd == 'delete' and len(parts) >= 2:
                args.name = parts[1]
                delete_pipeline(args)
                
            elif cmd == 'sample' and len(parts) >= 3:
                name, shape = parts[1:3]
                
                # Parse options
                steps = 50
                guidance_scale = 7.5
                classifier_free = True
                
                for i, part in enumerate(parts[3:], 3):
                    if part == '--steps' and i + 1 < len(parts):
                        steps = int(parts[i + 1])
                    elif part == '--guidance' and i + 1 < len(parts):
                        guidance_scale = float(parts[i + 1])
                    elif part == '--no-guidance':
                        classifier_free = False
                
                args.name = name
                args.shape = shape
                args.steps = steps
                args.guidance_scale = guidance_scale
                args.classifier_free = classifier_free
                sample_pipeline(args)
                
            elif cmd == 'compare' and len(parts) >= 2:
                shape = parts[1]
                
                # Parse options
                steps = 20
                
                for i, part in enumerate(parts[2:], 2):
                    if part == '--steps' and i + 1 < len(parts):
                        steps = int(parts[i + 1])
                
                args.shape = shape
                args.steps = steps
                compare_pipelines(args)
                
            elif cmd == 'benchmark' and len(parts) >= 2:
                name = parts[1]
                
                # Parse options
                shape = "1,3,64,64"
                steps = 20
                iterations = 5
                
                for i, part in enumerate(parts[2:], 2):
                    if part == '--shape' and i + 1 < len(parts):
                        shape = parts[i + 1]
                    elif part == '--steps' and i + 1 < len(parts):
                        steps = int(parts[i + 1])
                    elif part == '--iterations' and i + 1 < len(parts):
                        iterations = int(parts[i + 1])
                
                args.name = name
                args.shape = shape
                args.steps = steps
                args.iterations = iterations
                benchmark_pipeline(args)
                
            elif cmd == 'config' and len(parts) >= 3:
                subcmd = parts[1]
                file_path = parts[2]
                
                if subcmd == 'save':
                    args.file = file_path
                    save_configuration(args)
                elif subcmd == 'load':
                    args.file = file_path
                    load_configuration(args)
                elif subcmd == 'export':
                    args.file = file_path
                    export_configuration(args)
                else:
                    print(f"❌ Unknown config command: {subcmd}")
                    
            else:
                print(f"❌ Unknown command: {cmd}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Advanced Noise Scheduler and Sampling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create my_pipeline cosine ddim --timesteps 1000
  %(prog)s sample my_pipeline "1,3,64,64" --steps 50
  %(prog)s compare "1,3,64,64" --steps 20
  %(prog)s benchmark my_pipeline --iterations 10
  %(prog)s --interactive
        """
    )
    
    # Global options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file to load'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create pipeline command
    create_parser = subparsers.add_parser('create', help='Create a new pipeline')
    create_parser.add_argument('name', help='Pipeline name')
    create_parser.add_argument('scheduler', help='Beta schedule type')
    create_parser.add_argument('sampler', help='Sampling method')
    create_parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps')
    create_parser.add_argument('--beta-start', type=float, default=0.0001, help='Beta start value')
    create_parser.add_argument('--beta-end', type=float, default=0.02, help='Beta end value')
    create_parser.add_argument('--clip-sample', action='store_true', default=True, help='Clip samples')
    create_parser.add_argument('--prediction-type', default='epsilon', help='Prediction type')
    
    # List pipelines command
    subparsers.add_parser('list', help='List all pipelines')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show pipeline information')
    info_parser.add_argument('name', help='Pipeline name')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a pipeline')
    delete_parser.add_argument('name', help='Pipeline name')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Generate samples')
    sample_parser.add_argument('name', help='Pipeline name')
    sample_parser.add_argument('shape', help='Output shape (e.g., "1,3,64,64")')
    sample_parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    sample_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale')
    sample_parser.add_argument('--no-guidance', action='store_true', help='Disable classifier-free guidance')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare all pipelines')
    compare_parser.add_argument('shape', help='Output shape (e.g., "1,3,64,64")')
    compare_parser.add_argument('--steps', type=int, default=20, help='Number of inference steps')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('name', help='Pipeline name')
    benchmark_parser.add_argument('--shape', default='1,3,64,64', help='Output shape')
    benchmark_parser.add_argument('--steps', type=int, default=20, help='Number of inference steps')
    benchmark_parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
    
    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_cmd', help='Config subcommands')
    
    config_save_parser = config_subparsers.add_parser('save', help='Save configuration')
    config_save_parser.add_argument('file', help='Output file path')
    
    config_load_parser = config_subparsers.add_parser('load', help='Load configuration')
    config_load_parser.add_argument('file', help='Input file path')
    
    config_export_parser = config_subparsers.add_parser('export', help='Export configuration')
    config_export_parser.add_argument('file', help='Output file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize system
    try:
        args.system = AdvancedDiffusionSystem(args.config)
        if args.config:
            print(f"📁 Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return 1
    
    # Handle commands
    if args.interactive:
        interactive_mode(args)
    elif args.command == 'create':
        return 0 if create_pipeline(args) else 1
    elif args.command == 'list':
        list_pipelines(args)
    elif args.command == 'info':
        show_pipeline_info(args)
    elif args.command == 'delete':
        delete_pipeline(args)
    elif args.command == 'sample':
        return 0 if sample_pipeline(args) else 1
    elif args.command == 'compare':
        compare_pipelines(args)
    elif args.command == 'benchmark':
        return 0 if benchmark_pipeline(args) else 1
    elif args.command == 'config':
        if args.config_cmd == 'save':
            return 0 if save_configuration(args) else 1
        elif args.config_cmd == 'load':
            return 0 if load_configuration(args) else 1
        elif args.config_cmd == 'export':
            return 0 if export_configuration(args) else 1
        else:
            print("❌ No config subcommand specified")
            return 1
    else:
        print("❌ No command specified")
        print("Use --help for available commands or --interactive for interactive mode")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


