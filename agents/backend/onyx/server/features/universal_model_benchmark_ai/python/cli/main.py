"""
CLI Module - Enhanced Command Line Interface for benchmark system.

Provides:
- Comprehensive CLI commands for all operations
- Interactive mode
- Batch operations
- Configuration management
- Better error handling and validation
- Rich output formatting
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from orchestrator.results import ResultsManager
from orchestrator.types import ExecutionResult
from orchestrator.main import main as run_benchmark

console = Console()


@click.group()
@click.version_option(version="1.0.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """
    Universal Model Benchmark AI - CLI Tool.
    
    A comprehensive command-line interface for running benchmarks,
    managing results, and analyzing model performance.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


@cli.command()
@click.option("--model", "-m", required=True, help="Model name or path")
@click.option("--benchmark", "-b", required=True, help="Benchmark name")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file (JSON)")
@click.option("--output", "-o", type=click.Path(), default="results", help="Output directory")
@click.option("--parallel", "-p", is_flag=True, help="Run in parallel mode")
@click.option("--max-samples", type=int, help="Maximum number of samples")
@click.option("--shots", type=int, default=0, help="Number of few-shot examples")
@click.option("--timeout", type=float, help="Timeout in seconds")
@click.pass_context
def run(
    ctx,
    model: str,
    benchmark: str,
    config: Optional[str],
    output: Optional[str],
    parallel: bool,
    max_samples: Optional[int],
    shots: int,
    timeout: Optional[float],
):
    """
    Run a benchmark on a model.
    
    Examples:
        \b
        # Run MMLU benchmark
        $ benchmark run -m llama-2-7b -b mmlu
        
        # Run with custom config
        $ benchmark run -m llama-2-7b -b mmlu -c config.json
        
        # Run in parallel with timeout
        $ benchmark run -m llama-2-7b -b mmlu -p --timeout 3600
    """
    console.print(f"[cyan]Running benchmark:[/cyan] {benchmark} on [cyan]model:[/cyan] {model}")
    
    # Parse config if provided
    config_dict = {}
    if config:
        try:
            with open(config, 'r') as f:
                config_dict = json.load(f)
            if ctx.obj.get('verbose'):
                console.print(f"[dim]Loaded config from {config}[/dim]")
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing config file:[/red] {e}", err=True)
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error reading config file:[/red] {e}", err=True)
            sys.exit(1)
    
    # Add CLI options to config
    if max_samples:
        config_dict['max_samples'] = max_samples
    if shots:
        config_dict['shots'] = shots
    if timeout:
        config_dict['timeout'] = timeout
    
    # Run benchmark
    try:
        with console.status("[bold green]Running benchmark..."):
            result = run_benchmark(
                model_name=model,
                benchmark_name=benchmark,
                config=config_dict,
                output_dir=output,
                parallel=parallel,
            )
        
        console.print(f"[green]✓[/green] Benchmark completed successfully!")
        if isinstance(result, dict) and 'accuracy' in result:
            console.print(f"  Accuracy: {result['accuracy']:.2%}")
            console.print(f"  Throughput: {result.get('throughput', 0):.2f} tokens/s")
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.group()
def results():
    """Manage benchmark results."""
    pass


@results.command("list")
@click.option("--model", "-m", help="Filter by model name")
@click.option("--benchmark", "-b", help="Filter by benchmark name")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "csv"]))
@click.option("--min-accuracy", type=float, help="Minimum accuracy threshold")
@click.pass_context
def list_results(
    ctx,
    model: Optional[str],
    benchmark: Optional[str],
    format: str,
    min_accuracy: Optional[float],
):
    """
    List benchmark results.
    
    Examples:
        \b
        # List all results
        $ benchmark results list
        
        # Filter by model
        $ benchmark results list -m llama-2-7b
        
        # Filter by accuracy
        $ benchmark results list --min-accuracy 0.8
    """
    manager = ResultsManager()
    
    # Load results if available
    results_dir = Path("results")
    if results_dir.exists():
        for result_file in results_dir.glob("results_*.json"):
            try:
                manager.load(result_file)
            except Exception as e:
                if ctx.obj.get('verbose'):
                    console.print(f"[yellow]Warning:[/yellow] Failed to load {result_file}: {e}")
    
    # Filter results
    filtered = manager.filter(
        model_name=model,
        benchmark_name=benchmark,
        min_accuracy=min_accuracy,
    )
    
    if not filtered:
        console.print("[yellow]No results found[/yellow]")
        return
    
    if format == "json":
        import json
        console.print(json.dumps([r.to_dict() for r in filtered], indent=2))
    elif format == "csv":
        import csv
        import io
        output = io.StringIO()
        if filtered:
            # Flatten results for CSV
            rows = []
            for r in filtered:
                row = {
                    "model_name": r.model_name,
                    "benchmark_name": r.benchmark_name,
                    "success": r.success,
                    "status": r.status.value,
                    "execution_time": r.execution_time,
                }
                if r.result:
                    row.update({
                        "accuracy": r.result.accuracy,
                        "throughput": r.result.throughput,
                        "latency_p50": r.result.latency_p50,
                        "total_samples": r.result.total_samples,
                    })
                rows.append(row)
            
            if rows:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        console.print(output.getvalue())
    else:
        # Table format
        table = Table(box=box.ROUNDED, show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Benchmark", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Accuracy", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Time", justify="right")
        
        for r in filtered:
            status_icon = "✓" if r.is_successful() else "✗"
            accuracy = f"{r.result.accuracy:.2%}" if r.result else "N/A"
            throughput = f"{r.result.throughput:.2f}" if r.result else "N/A"
            time_str = f"{r.execution_time:.1f}s"
            
            table.add_row(
                r.model_name,
                r.benchmark_name,
                f"{status_icon} {r.status.value}",
                accuracy,
                throughput,
                time_str,
            )
        
        console.print(table)


@results.command("compare")
@click.argument("benchmark_name")
@click.option("--models", "-m", multiple=True, help="Model names to compare")
@click.option("--metric", default="accuracy", type=click.Choice(["accuracy", "throughput", "latency_p50"]))
def compare_results(benchmark_name: str, models: tuple, metric: str):
    """
    Compare results for a benchmark.
    
    Examples:
        \b
        # Compare all models for MMLU
        $ benchmark results compare mmlu
        
        # Compare specific models
        $ benchmark results compare mmlu -m llama-2-7b -m llama-2-13b
    """
    manager = ResultsManager()
    
    # Load results
    results_dir = Path("results")
    if results_dir.exists():
        for result_file in results_dir.glob("results_*.json"):
            try:
                manager.load(result_file)
            except Exception:
                pass
    
    model_list = list(models) if models else None
    
    if model_list:
        comparison = manager.compare_models(model_list, benchmark_name)
    else:
        # Get ranking
        ranking = manager.get_ranking(benchmark_name, metric)
        if not ranking:
            console.print(f"[yellow]No results found for benchmark: {benchmark_name}[/yellow]")
            return
        
        table = Table(title=f"Ranking: {benchmark_name} (by {metric})", box=box.ROUNDED)
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Accuracy", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Latency P50", justify="right")
        
        for i, result in enumerate(ranking, 1):
            table.add_row(
                str(i),
                result.model_name,
                f"{result.result.accuracy:.2%}",
                f"{result.result.throughput:.2f}",
                f"{result.result.latency_p50:.3f}s",
            )
        
        console.print(table)
        return
    
    # Show comparison
    if benchmark_name in comparison.get("benchmarks", {}):
        bench_data = comparison["benchmarks"][benchmark_name]
        
        table = Table(title=f"Comparison: {benchmark_name}", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Latency P50", justify="right")
        table.add_column("Time", justify="right")
        
        for data in bench_data:
            table.add_row(
                data["model"],
                f"{data['accuracy']:.2%}",
                f"{data['throughput']:.2f}",
                f"{data['latency_p50']:.3f}s",
                f"{data['execution_time']:.1f}s",
            )
        
        console.print(table)
    else:
        console.print(f"[yellow]No comparison data found for {benchmark_name}[/yellow]")


@results.command("export")
@click.argument("output_path", type=click.Path())
@click.option("--format", "-f", default="json", type=click.Choice(["json", "csv"]))
@click.option("--model", "-m", help="Filter by model")
@click.option("--benchmark", "-b", help="Filter by benchmark")
def export_results(
    output_path: str,
    format: str,
    model: Optional[str],
    benchmark: Optional[str],
):
    """
    Export results to file.
    
    Examples:
        \b
        # Export to JSON
        $ benchmark results export results.json
        
        # Export to CSV
        $ benchmark results export results.csv -f csv
        
        # Export filtered results
        $ benchmark results export results.json -m llama-2-7b
    """
    manager = ResultsManager()
    
    # Load results
    results_dir = Path("results")
    if results_dir.exists():
        for result_file in results_dir.glob("results_*.json"):
            try:
                manager.load(result_file)
            except Exception:
                pass
    
    try:
        results_file, summary_file = manager.save(
            output_dir=Path(output_path).parent,
            format=format
        )
        console.print(f"[green]✓[/green] Exported results to {results_file}")
        console.print(f"[green]✓[/green] Exported summary to {summary_file}")
    except Exception as e:
        console.print(f"[red]Error exporting results:[/red] {e}", err=True)
        sys.exit(1)


@results.command("summary")
def results_summary():
    """Show summary of all results."""
    manager = ResultsManager()
    
    # Load results
    results_dir = Path("results")
    if results_dir.exists():
        for result_file in results_dir.glob("results_*.json"):
            try:
                manager.load(result_file)
            except Exception:
                pass
    
    manager.print_summary()


@cli.command()
def stats():
    """Show system statistics."""
    manager = ResultsManager()
    
    # Load results
    results_dir = Path("results")
    if results_dir.exists():
        for result_file in results_dir.glob("results_*.json"):
            try:
                manager.load(result_file)
            except Exception:
                pass
    
    summary = manager.get_summary()
    
    panel = Panel.fit(
        f"[bold]Total Runs:[/bold] {summary['total']}\n"
        f"[bold]Successful:[/bold] {summary['successful']} ({summary['success_rate']:.1%})\n"
        f"[bold]Failed:[/bold] {summary['failed']}\n"
        f"[bold]Models:[/bold] {summary['models']}\n"
        f"[bold]Benchmarks:[/bold] {summary['benchmarks']}\n"
        f"[bold]Average Accuracy:[/bold] {summary.get('average_accuracy', 0):.2%}",
        title="[bold cyan]System Statistics[/bold cyan]",
        border_style="cyan"
    )
    
    console.print(panel)


if __name__ == "__main__":
    cli()
