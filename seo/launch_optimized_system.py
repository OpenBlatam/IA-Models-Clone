#!/usr/bin/env python3
"""
Optimized SEO System Launcher
Advanced launcher with multiple options and system optimization
"""

import argparse
import sys
import os
import time
import logging
import psutil
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local imports
from core_config import SEOConfig, get_config, get_container
from optimized_seo_engine import OptimizedSEOEngine, create_optimized_seo_engine
from optimized_gradio_interface import OptimizedGradioInterface, create_optimized_interface
from advanced_monitoring import MonitoringSystem, start_monitoring

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('seo_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class OptimizedSystemLauncher:
    """Advanced launcher for the optimized SEO system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[SEOConfig] = None
        self.engine: Optional[OptimizedSEOEngine] = None
        self.interface: Optional[OptimizedGradioInterface] = None
        self.monitoring: Optional[MonitoringSystem] = None
        
        # Performance tracking
        self.startup_time = 0
        self.system_stats = {}
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the system components."""
        try:
            # Load configuration
            if self.config_path and os.path.exists(self.config_path):
                self.config = SEOConfig.load_from_file(self.config_path)
                logging.info(f"Configuration loaded from: {self.config_path}")
            else:
                self.config = SEOConfig()
                logging.info("Using default configuration")
            
            # Validate configuration
            self.config.validate()
            logging.info("Configuration validated successfully")
            
            # Initialize dependency container
            container = get_container()
            container.register('launcher', self)
            
            logging.info("System initialization completed")
            
        except Exception as e:
            logging.error(f"System initialization failed: {e}")
            raise
    
    def launch_engine_only(self, enable_monitoring: bool = True) -> OptimizedSEOEngine:
        """Launch only the SEO engine."""
        try:
            logging.info("🚀 Launching SEO Engine...")
            start_time = time.time()
            
            # Create and initialize engine
            self.engine = create_optimized_seo_engine(self.config_path)
            
            # Start monitoring if enabled
            if enable_monitoring:
                self.monitoring = self.engine.monitoring
                logging.info("📊 Monitoring system started")
            
            # Record startup time
            self.startup_time = time.time() - start_time
            logging.info(f"✅ SEO Engine launched in {self.startup_time:.2f}s")
            
            return self.engine
            
        except Exception as e:
            logging.error(f"❌ Failed to launch SEO Engine: {e}")
            raise
    
    def launch_interface(self, enable_monitoring: bool = True, 
                        server_port: int = 7860, server_host: str = '0.0.0.0') -> None:
        """Launch the complete system with Gradio interface."""
        try:
            logging.info("🚀 Launching Complete SEO System...")
            start_time = time.time()
            
            # Launch engine first
            self.launch_engine_only(enable_monitoring)
            
            # Create and launch interface
            self.interface = create_optimized_interface(self.config_path)
            
            # Record total startup time
            total_startup_time = time.time() - start_time
            logging.info(f"✅ Complete system launched in {total_startup_time:.2f}s")
            
            # Display system information
            self._display_system_info()
            
            # Launch interface
            self.interface.launch(
                server_name=server_host,
                server_port=server_port,
                share=False,
                debug=False,
                show_error=True
            )
            
        except Exception as e:
            logging.error(f"❌ Failed to launch interface: {e}")
            raise
    
    def launch_cli_mode(self, enable_monitoring: bool = True) -> None:
        """Launch in CLI mode for batch processing."""
        try:
            logging.info("🚀 Launching CLI Mode...")
            
            # Launch engine
            self.launch_engine_only(enable_monitoring)
            
            # Display CLI interface
            self._display_cli_interface()
            
            # Start CLI loop
            self._cli_loop()
            
        except Exception as e:
            logging.error(f"❌ Failed to launch CLI mode: {e}")
            raise
    
    def launch_api_mode(self, enable_monitoring: bool = True, 
                       server_port: int = 8000, server_host: str = '0.0.0.0') -> None:
        """Launch in API mode for programmatic access."""
        try:
            logging.info("🚀 Launching API Mode...")
            
            # Launch engine
            self.launch_engine_only(enable_monitoring)
            
            # Start API server (placeholder for now)
            logging.info(f"🌐 API server would start on {server_host}:{server_port}")
            logging.info("API mode not yet implemented - use engine directly")
            
            # Keep system running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("🛑 API mode stopped by user")
            
        except Exception as e:
            logging.error(f"❌ Failed to launch API mode: {e}")
            raise
    
    def _display_system_info(self) -> None:
        """Display system information."""
        print("\n" + "="*60)
        print("🚀 OPTIMIZED SEO SYSTEM - SYSTEM INFORMATION")
        print("="*60)
        
        # Configuration info
        print(f"📋 Configuration:")
        print(f"   System Workers: {self.config.system.max_workers}")
        print(f"   Cache Enabled: {self.config.models.cache_enabled}")
        print(f"   Batch Size: {self.config.performance.batch_size}")
        print(f"   Monitoring: {self.config.monitoring.metrics_enabled}")
        
        # Performance info
        print(f"\n⚡ Performance:")
        print(f"   Startup Time: {self.startup_time:.2f}s")
        print(f"   Memory Usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"   CPU Usage: {psutil.cpu_percent():.1f}%")
        
        # Interface info
        print(f"\n🌐 Interface:")
        print(f"   Gradio Interface: Ready")
        print(f"   Real-time Monitoring: Active")
        print(f"   Performance Optimization: Enabled")
        
        print("="*60)
        print("🎯 System ready for SEO analysis!")
        print("="*60 + "\n")
    
    def _display_cli_interface(self) -> None:
        """Display CLI interface information."""
        print("\n" + "="*60)
        print("💻 CLI MODE - SEO ANALYSIS SYSTEM")
        print("="*60)
        print("Available commands:")
        print("  analyze <text>     - Analyze single text")
        print("  batch <file>       - Analyze text file")
        print("  metrics            - Show system metrics")
        print("  optimize           - Optimize performance")
        print("  health             - System health check")
        print("  quit               - Exit system")
        print("="*60 + "\n")
    
    def _cli_loop(self) -> None:
        """Main CLI loop."""
        try:
            while True:
                try:
                    command = input("SEO> ").strip().split()
                    if not command:
                        continue
                    
                    cmd = command[0].lower()
                    
                    if cmd == 'quit':
                        break
                    elif cmd == 'analyze' and len(command) > 1:
                        text = ' '.join(command[1:])
                        self._analyze_text_cli(text)
                    elif cmd == 'batch' and len(command) > 1:
                        file_path = command[1]
                        self._analyze_file_cli(file_path)
                    elif cmd == 'metrics':
                        self._show_metrics_cli()
                    elif cmd == 'optimize':
                        self._optimize_cli()
                    elif cmd == 'health':
                        self._health_check_cli()
                    else:
                        print("❌ Unknown command. Type 'help' for available commands.")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.error(f"CLI command failed: {e}")
                    print(f"❌ Error: {e}")
        
        except Exception as e:
            logging.error(f"CLI loop failed: {e}")
        finally:
            print("\n👋 Goodbye!")
    
    def _analyze_text_cli(self, text: str) -> None:
        """Analyze text in CLI mode."""
        if not self.engine:
            print("❌ Engine not available")
            return
        
        try:
            print(f"🔍 Analyzing text ({len(text)} characters)...")
            start_time = time.time()
            
            result = self.engine.analyze_text(text)
            
            analysis_time = time.time() - start_time
            print(f"✅ Analysis completed in {analysis_time:.2f}s")
            print(f"🎯 SEO Score: {result['seo_score']:.1f}/100")
            
            if result.get('recommendations'):
                print("\n💡 Recommendations:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"   {i}. {rec}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    def _analyze_file_cli(self, file_path: str) -> None:
        """Analyze file in CLI mode."""
        if not self.engine:
            print("❌ Engine not available")
            return
        
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return
            
            print(f"📁 Analyzing file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self._analyze_text_cli(content)
            
        except Exception as e:
            print(f"❌ File analysis failed: {e}")
    
    def _show_metrics_cli(self) -> None:
        """Show system metrics in CLI mode."""
        if not self.engine:
            print("❌ Engine not available")
            return
        
        try:
            metrics = self.engine.get_system_metrics()
            
            print("\n📊 System Metrics:")
            print(f"   System Health: {metrics['system_health']['status']}")
            print(f"   Cache Items: {metrics['cache_stats']['total_items']}")
            print(f"   Models Loaded: {metrics['model_info']['total_models']}")
            
            # Performance stats
            perf_stats = metrics.get('performance_stats', {})
            if 'analysis_time' in perf_stats:
                avg_time = perf_stats['analysis_time'].get('mean', 0)
                print(f"   Avg Analysis Time: {avg_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Failed to get metrics: {e}")
    
    def _optimize_cli(self) -> None:
        """Optimize performance in CLI mode."""
        if not self.engine:
            print("❌ Engine not available")
            return
        
        try:
            print("⚡ Optimizing performance...")
            optimizations = self.engine.optimize_performance()
            
            if optimizations:
                print("✅ Optimizations applied:")
                for key, value in optimizations.items():
                    print(f"   {key}: {value}")
            else:
                print("✅ No optimizations needed")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
    
    def _health_check_cli(self) -> None:
        """Perform system health check in CLI mode."""
        if not self.engine:
            print("❌ Engine not available")
            return
        
        try:
            print("🏥 Performing system health check...")
            metrics = self.engine.get_system_metrics()
            health = metrics['system_health']
            
            print(f"   Status: {health['status']}")
            print(f"   CPU Usage: {psutil.cpu_percent():.1f}%")
            print(f"   Memory Usage: {psutil.virtual_memory().percent:.1f}%")
            
            if health['alerts']:
                print(f"   Alerts: {len(health['alerts'])} active")
            else:
                print("   Alerts: None")
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
    
    def run_benchmark(self, num_texts: int = 10) -> Dict[str, Any]:
        """Run performance benchmark."""
        if not self.engine:
            raise RuntimeError("Engine not available")
        
        try:
            logging.info(f"🚀 Running benchmark with {num_texts} texts...")
            
            # Create sample texts
            sample_texts = [
                f"This is sample text number {i} for benchmarking purposes. " * 20
                for i in range(num_texts)
            ]
            
            # Benchmark single analysis
            start_time = time.time()
            result = self.engine.analyze_text(sample_texts[0])
            single_time = time.time() - start_time
            
            # Benchmark batch analysis
            start_time = time.time()
            results = self.engine.analyze_texts(sample_texts)
            batch_time = time.time() - start_time
            
            # Calculate metrics
            benchmark_results = {
                'single_analysis_time': single_time,
                'batch_analysis_time': batch_time,
                'total_texts': len(sample_texts),
                'average_time_per_text': batch_time / len(sample_texts),
                'throughput': len(sample_texts) / batch_time,
                'seo_score_sample': result.get('seo_score', 0)
            }
            
            logging.info("✅ Benchmark completed successfully")
            return benchmark_results
            
        except Exception as e:
            logging.error(f"❌ Benchmark failed: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up system resources."""
        try:
            logging.info("🧹 Cleaning up system resources...")
            
            if self.interface:
                # Interface cleanup is handled by Gradio
                pass
            
            if self.engine:
                self.engine.cleanup()
            
            if self.monitoring:
                self.monitoring.stop()
            
            logging.info("✅ Cleanup completed")
            
        except Exception as e:
            logging.error(f"❌ Cleanup failed: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🚀 Optimized SEO System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with Gradio interface
  python launch_optimized_system.py --interface --port 7860
  
  # Launch in CLI mode
  python launch_optimized_system.py --cli
  
  # Launch engine only
  python launch_optimized_system.py --engine-only
  
  # Run benchmark
  python launch_optimized_system.py --benchmark --texts 20
  
  # Custom configuration
  python launch_optimized_system.py --config config.yaml --interface
        """
    )
    
    # Launch mode options
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--interface', action='store_true', 
                           help='Launch with Gradio interface')
    mode_group.add_argument('--cli', action='store_true', 
                           help='Launch in CLI mode')
    mode_group.add_argument('--engine-only', action='store_true', 
                           help='Launch only the SEO engine')
    mode_group.add_argument('--api', action='store_true', 
                           help='Launch in API mode')
    mode_group.add_argument('--benchmark', action='store_true', 
                           help='Run performance benchmark')
    
    # Configuration options
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file')
    parser.add_argument('--port', type=int, default=7860,
                       help='Server port (default: 7860)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host (default: 0.0.0.0)')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable monitoring system')
    parser.add_argument('--texts', type=int, default=10,
                       help='Number of texts for benchmark (default: 10)')
    
    args = parser.parse_args()
    
    try:
        # Create launcher
        launcher = OptimizedSystemLauncher(args.config)
        
        # Launch based on mode
        if args.interface:
            launcher.launch_interface(
                enable_monitoring=not args.no_monitoring,
                server_port=args.port,
                server_host=args.host
            )
        elif args.cli:
            launcher.launch_cli_mode(enable_monitoring=not args.no_monitoring)
        elif args.engine_only:
            engine = launcher.launch_engine_only(enable_monitoring=not args.no_monitoring)
            print("✅ SEO Engine launched successfully")
            print("Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping engine...")
        elif args.api:
            launcher.launch_api_mode(
                enable_monitoring=not args.no_monitoring,
                server_port=args.port,
                server_host=args.host
            )
        elif args.benchmark:
            engine = launcher.launch_engine_only(enable_monitoring=not args.no_monitoring)
            results = launcher.run_benchmark(args.texts)
            
            print("\n" + "="*60)
            print("🚀 BENCHMARK RESULTS")
            print("="*60)
            print(f"Single Analysis Time: {results['single_analysis_time']:.3f}s")
            print(f"Batch Analysis Time: {results['batch_analysis_time']:.3f}s")
            print(f"Total Texts: {results['total_texts']}")
            print(f"Average Time per Text: {results['average_time_per_text']:.3f}s")
            print(f"Throughput: {results['throughput']:.1f} texts/second")
            print(f"Sample SEO Score: {results['seo_score_sample']:.1f}")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        logging.error(f"❌ System launch failed: {e}")
        print(f"❌ Failed to launch system: {e}")
        sys.exit(1)
    finally:
        if 'launcher' in locals():
            launcher.cleanup()

if __name__ == "__main__":
    main()


