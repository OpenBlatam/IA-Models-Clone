from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import sys
import argparse
from pathlib import Path
from examples.multi_gpu_training_demo import MultiGPUTrainingDemo, main
            from pathlib import Path
            import json
            import traceback
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Multi-GPU Training Demo Launcher

Launcher script for the comprehensive multi-GPU training demonstration.
"""


# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))



def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Multi-GPU Training System Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive demo
  python run_multi_gpu_training_demo.py

  # Run specific demo
  python run_multi_gpu_training_demo.py --demo gpu_monitoring

  # Run with custom configuration
  python run_multi_gpu_training_demo.py --demo data_parallel --epochs 5 --batch-size 128
        """
    )
    
    parser.add_argument(
        "--demo",
        type=str,
        choices=["comprehensive", "data_parallel", "distributed", "auto_detection", "gpu_monitoring", "performance_comparison"],
        default="comprehensive",
        help="Specific demo to run (default: comprehensive)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of training samples (default: 2000)"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="Sequence length for email data (default: 50)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/multi_gpu_demo",
        help="Log directory (default: logs/multi_gpu_demo)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to file"
    )
    
    return parser.parse_args()


async def run_specific_demo(demo_name: str, args):
    """Run a specific demo"""
    
    demo = MultiGPUTrainingDemo(f"{demo_name}_demo")
    
    if demo_name == "gpu_monitoring":
        return demo.demo_gpu_monitoring()
    
    elif demo_name == "performance_comparison":
        return demo.demo_performance_comparison()
    
    elif demo_name == "auto_detection":
        return await demo.demo_auto_detection()
    
    elif demo_name == "data_parallel":
        return await demo.demo_data_parallel_training()
    
    elif demo_name == "distributed":
        return await demo.demo_distributed_training()
    
    else:
        raise ValueError(f"Unknown demo: {demo_name}")


async def run_comprehensive_demo(args) -> Any:
    """Run comprehensive demo with custom parameters"""
    
    demo = MultiGPUTrainingDemo("comprehensive_multi_gpu_demo")
    
    # Update demo parameters based on command line arguments
    # This would require modifying the demo methods to accept parameters
    
    return await demo.run_comprehensive_demo()


def main_launcher():
    """Main launcher function"""
    
    args = parse_arguments()
    
    print("Multi-GPU Training System Demonstration")
    print("="*50)
    print(f"Demo: {args.demo}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Samples: {args.samples}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Log Directory: {args.log_dir}")
    print(f"Debug Mode: {args.debug}")
    print("="*50)
    
    try:
        if args.demo == "comprehensive":
            results = asyncio.run(run_comprehensive_demo(args))
        else:
            results = asyncio.run(run_specific_demo(args.demo, args))
        
        # Print results
        if "error" in results:
            print(f"❌ Demo failed: {results['error']}")
            return 1
        
        print("\n✅ Demo completed successfully!")
        
        # Save results if requested
        if args.save_results:
            
            results_dir = Path(args.log_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = results_dir / f"{args.demo}_results.json"
            with open(results_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(results, f, indent=2)
            
            print(f"📁 Results saved to: {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main_launcher()
    sys.exit(exit_code) 