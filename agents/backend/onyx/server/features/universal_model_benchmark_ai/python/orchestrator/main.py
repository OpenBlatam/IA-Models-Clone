"""
Benchmark Orchestrator - Coordinates benchmark execution across all models.

Refactored modular version using:
- types: ExecutionResult
- executor: BenchmarkExecutor
- registry: BenchmarkRegistry
- results: ResultsManager
- progress: ProgressTracker
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from rich.console import Console
from rich.panel import Panel

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import SystemConfig, load_config, format_duration
from .types import ExecutionResult
from .executor import BenchmarkExecutor
from .registry import BenchmarkRegistry
from .results import ResultsManager
from .progress import ProgressTracker
from .report_generator import OrchestratorReportGenerator

console = Console()
logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """
    Orquestador principal de benchmarks (versión modular).
    
    Coordinates execution of benchmarks across multiple models with:
    - Parallel execution support
    - Progress tracking
    - Error handling
    - Result aggregation
    - Rich console output
    
    Examples:
        >>> orchestrator = BenchmarkOrchestrator(config)
        >>> orchestrator.run_all()
        
        >>> # Run specific benchmark
        >>> orchestrator.run_benchmark(model_config, benchmark_config)
    """
    
    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        max_workers: int = 1,
        parallel: bool = False,
    ):
        """
        Inicializa el orquestador.
        
        Args:
            config: Configuración del sistema (opcional)
            max_workers: Maximum number of parallel workers
            parallel: Enable parallel execution
        """
        self.config = config or load_config()
        self.max_workers = max_workers
        self.parallel = parallel
        
        # Initialize modular components
        self.registry = BenchmarkRegistry()
        self.executor = BenchmarkExecutor(
            benchmark_registry=self.registry._benchmarks,
            max_workers=max_workers
        )
        self.results_manager = ResultsManager()
        self.progress_tracker = ProgressTracker(show_progress=True)
        self.report_generator = OrchestratorReportGenerator()
    
    def get_available_benchmarks(self) -> List[str]:
        """
        Obtiene lista de benchmarks disponibles.
        
        Returns:
            List of benchmark names
        """
        return self.registry.list()
    
    def get_available_models(self) -> List[Any]:
        """
        Obtiene lista de modelos configurados.
        
        Returns:
            List of ModelConfig objects
        """
        return self.config.models
    
    def register_benchmark(self, name: str, benchmark_class: type) -> None:
        """
        Register a new benchmark class.
        
        Args:
            name: Benchmark name
            benchmark_class: Benchmark class (subclass of BaseBenchmark)
        """
        self.registry.register(name, benchmark_class)
        # Update executor registry
        self.executor.benchmark_registry = self.registry._benchmarks
    
    def run_all(
        self,
        progress_callback: Optional[Callable] = None,
        save_results: bool = True,
    ) -> List[ExecutionResult]:
        """
        Ejecuta todos los benchmarks en todos los modelos.
        
        Args:
            progress_callback: Optional callback for progress updates
            save_results: Whether to save results to disk
        
        Returns:
            List of execution results
        """
        models = self.get_available_models()
        benchmarks = self.config.benchmarks
        
        if not models:
            console.print("[yellow]No models configured[/yellow]")
            return []
        
        if not benchmarks:
            console.print("[yellow]No benchmarks configured[/yellow]")
            return []
        
        total_tasks = len(models) * len(benchmarks)
        
        console.print(
            Panel(
                f"[bold green]Running {len(benchmarks)} benchmarks on {len(models)} models[/bold green]\n"
                f"Total tasks: {total_tasks}\n"
                f"Parallel execution: {'Enabled' if self.parallel else 'Disabled'}",
                title="Benchmark Orchestrator",
                border_style="green"
            )
        )
        
        # Execute benchmarks
        if self.parallel and total_tasks > 1:
            results = self.executor.run_parallel(
                models,
                benchmarks,
                progress_callback
            )
        else:
            results = self.executor.run_sequential(
                models,
                benchmarks,
                progress_callback
            )
        
        # Store results
        self.results_manager.add_all(results)
        
        return results
    
    def run_benchmark(
        self,
        model_config: Any,
        benchmark_config: Any,
    ) -> ExecutionResult:
        """
        Ejecuta un benchmark específico en un modelo.
        
        Args:
            model_config: Model configuration
            benchmark_config: Benchmark configuration
        
        Returns:
            ExecutionResult
        """
        console.print(
            f"\n[bold cyan]Running {benchmark_config.name} on {model_config.name}[/bold cyan]"
        )
        
        result = self.executor.execute_benchmark(model_config, benchmark_config)
        self.results_manager.add(result)
        
        if result.success and result.result:
            console.print(
                f"[green]✓ Completed: Accuracy = {result.result.accuracy:.2%}[/green]"
            )
        else:
            console.print(f"[red]✗ Failed: {result.error}[/red]")
        
        return result
    
    def save_results(
        self,
        output_dir: str = "results",
        generate_reports: bool = True,
        report_format: str = "json",
    ) -> Path:
        """
        Guarda los resultados.
        
        Args:
            output_dir: Output directory
            generate_reports: Whether to generate reports
            report_format: Report format
        
        Returns:
            Path to saved results file
        """
        json_path, summary_path = self.results_manager.save(output_dir)
        
        # Generate enhanced reports if requested
        if generate_reports:
            try:
                report_path = self.report_generator.generate_execution_report(
                    self.results_manager.results,
                    output_format=report_format,
                )
                console.print(f"[green]Reports generated in {report_path}[/green]")
            except Exception as e:
                logger.warning(f"Failed to generate reports: {e}")
        
        return json_path
    
    def print_summary(self) -> None:
        """Imprime un resumen de los resultados."""
        self.results_manager.print_summary()
    
    def get_results_by_model(self, model_name: str) -> List[ExecutionResult]:
        """Get all results for a specific model."""
        return self.results_manager.get_by_model(model_name)
    
    def get_results_by_benchmark(self, benchmark_name: str) -> List[ExecutionResult]:
        """Get all results for a specific benchmark."""
        return self.results_manager.get_by_benchmark(benchmark_name)
    
    def get_best_model(self, benchmark_name: str) -> Optional[ExecutionResult]:
        """Get best performing model for a benchmark."""
        return self.results_manager.get_best_model(benchmark_name)
    
    @property
    def results(self) -> List[ExecutionResult]:
        """Get all results."""
        return self.results_manager.results


# ════════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Universal Model Benchmark AI - Comprehensive benchmark orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks on all models
  python orchestrator/main.py --all
  
  # Run specific benchmark on specific model
  python orchestrator/main.py --benchmark mmlu --model llama2-7b
  
  # Run with parallel execution
  python orchestrator/main.py --all --parallel --workers 4
  
  # List available benchmarks and models
  python orchestrator/main.py --list-benchmarks
  python orchestrator/main.py --list-models
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (YAML)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks on all models"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Run specific benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Run on specific model"
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(
        config=config,
        max_workers=args.workers,
        parallel=args.parallel
    )
    
    # Handle list commands
    if args.list_benchmarks:
        console.print("[bold]Available benchmarks:[/bold]")
        for bench in orchestrator.get_available_benchmarks():
            console.print(f"  - {bench}")
        return
    
    if args.list_models:
        console.print("[bold]Available models:[/bold]")
        for model in orchestrator.get_available_models():
            console.print(f"  - {model.name}")
        return
    
    # Execute benchmarks
    try:
        if args.all:
            orchestrator.run_all()
            orchestrator.save_results(args.output_dir)
            orchestrator.print_summary()
        elif args.benchmark and args.model:
            # Run specific benchmark on specific model
            models = orchestrator.get_available_models()
            model_config = next((m for m in models if m.name == args.model), None)
            if not model_config:
                console.print(f"[red]Model not found: {args.model}[/red]")
                return
            
            benchmark_config = config.get_benchmark_config(args.benchmark)
            if not benchmark_config:
                console.print(f"[red]Benchmark not found: {args.benchmark}[/red]")
                return
            
            orchestrator.run_benchmark(model_config, benchmark_config)
            orchestrator.save_results(args.output_dir)
            orchestrator.print_summary()
        else:
            parser.print_help()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Fatal error")
        raise


if __name__ == "__main__":
    main()
