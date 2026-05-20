from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import sys
import argparse
from pathlib import Path
from examples.gradient_accumulation_demo import GradientAccumulationDemo, main
            from pathlib import Path
            import json
            import traceback
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Gradient Accumulation Demo Launcher

Launcher script for the comprehensive gradient accumulation demonstration.
"""


# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))



def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Gradient Accumulation System Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive demo
  python run_gradient_accumulation_demo.py

  # Run specific demo
  python run_gradient_accumulation_demo.py --demo basic_accumulation

  # Run with custom configuration
  python run_gradient_accumulation_demo.py --demo optimized_training --accumulation-steps 8 --effective-batch-size 256
        """
    )
    
    parser.add_argument(
        "--demo",
        type=str,
        choices=["comprehensive", "basic_accumulation", "optimized_training", "accumulation_steps_calculation", "memory_efficient", "accumulation_monitoring"],
        default="comprehensive",
        help="Specific demo to run (default: comprehensive)"
    )
    
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
        help="Number of accumulation steps (default: 4)"
    )
    
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=128,
        help="Target effective batch size (default: 128)"
    )
    
    parser.add_argument(
        "--current-batch-size",
        type=int,
        default=32,
        help="Current batch size (default: 32)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
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
        default="logs/gradient_accumulation_demo",
        help="Log directory (default: logs/gradient_accumulation_demo)"
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
    
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Use memory-efficient accumulation"
    )
    
    parser.add_argument(
        "--scale-loss",
        action="store_true",
        default=True,
        help="Scale loss by accumulation steps"
    )
    
    parser.add_argument(
        "--scale-gradients",
        action="store_true",
        default=True,
        help="Scale gradients by accumulation steps"
    )
    
    return parser.parse_args()


async def run_specific_demo(demo_name: str, args):
    """Run a specific demo"""
    
    demo = GradientAccumulationDemo(f"{demo_name}_demo")
    
    if demo_name == "accumulation_steps_calculation":
        return demo.demo_accumulation_steps_calculation()
    
    elif demo_name == "memory_efficient":
        return demo.demo_memory_efficient_accumulation()
    
    elif demo_name == "accumulation_monitoring":
        return demo.demo_accumulation_monitoring()
    
    elif demo_name == "basic_accumulation":
        return await demo.demo_basic_accumulation()
    
    elif demo_name == "optimized_training":
        return await demo.demo_optimized_training_with_accumulation()
    
    else:
        raise ValueError(f"Unknown demo: {demo_name}")


async def run_comprehensive_demo(args) -> Any:
    """Run comprehensive demo with custom parameters"""
    
    demo = GradientAccumulationDemo("comprehensive_gradient_accumulation_demo")
    
    # Update demo parameters based on command line arguments
    # This would require modifying the demo methods to accept parameters
    
    return await demo.run_comprehensive_demo()


def main_launcher():
    """Main launcher function"""
    
    args = parse_arguments()
    
    print("Gradient Accumulation System Demonstration")
    print("="*50)
    print(f"Demo: {args.demo}")
    print(f"Accumulation Steps: {args.accumulation_steps}")
    print(f"Effective Batch Size: {args.effective_batch_size}")
    print(f"Current Batch Size: {args.current_batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Samples: {args.samples}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Log Directory: {args.log_dir}")
    print(f"Debug Mode: {args.debug}")
    print(f"Memory Efficient: {args.memory_efficient}")
    print(f"Scale Loss: {args.scale_loss}")
    print(f"Scale Gradients: {args.scale_gradients}")
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