"""
Main CLI application for Export IA.
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import ExportIAEngine
from core.models import ExportConfig, ExportFormat, DocumentType, QualityLevel


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """Export IA - Advanced AI Document Processing System"""
    pass


@cli.command()
@click.option('--input', '-i', required=True, help='Input content file (JSON)')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice([f.value for f in ExportFormat]), 
              default='pdf', help='Export format')
@click.option('--document-type', '-t', type=click.Choice([d.value for d in DocumentType]), 
              default='report', help='Document type')
@click.option('--quality', '-q', type=click.Choice([q.value for q in QualityLevel]), 
              default='professional', help='Quality level')
@click.option('--wait', '-w', is_flag=True, help='Wait for completion')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def export(input: str, output: Optional[str], format: str, document_type: str, 
           quality: str, wait: bool, verbose: bool):
    """Export a document to the specified format."""
    
    async def _export():
        try:
            # Load input content
            with open(input, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Create export configuration
            config = ExportConfig(
                format=ExportFormat(format),
                document_type=DocumentType(document_type),
                quality_level=QualityLevel(quality)
            )
            
            if verbose:
                click.echo(f"Exporting to {format} format...")
                click.echo(f"Document type: {document_type}")
                click.echo(f"Quality level: {quality}")
            
            # Export document
            async with ExportIAEngine() as engine:
                task_id = await engine.export_document(content, config, output)
                
                if verbose:
                    click.echo(f"Task created: {task_id}")
                
                if wait:
                    # Wait for completion
                    while True:
                        status = await engine.get_task_status(task_id)
                        
                        if verbose:
                            progress = status.get('progress', 0)
                            click.echo(f"Status: {status['status']}, Progress: {progress:.1%}")
                        
                        if status['status'] == 'completed':
                            click.echo(f"Export completed: {status['file_path']}")
                            click.echo(f"Quality score: {status['quality_score']:.2f}")
                            break
                        elif status['status'] == 'failed':
                            click.echo(f"Export failed: {status['error']}", err=True)
                            sys.exit(1)
                        
                        await asyncio.sleep(1)
                else:
                    click.echo(f"Export task created: {task_id}")
                    click.echo("Use 'export-ia status <task-id>' to check progress")
        
        except FileNotFoundError:
            click.echo(f"Input file not found: {input}", err=True)
            sys.exit(1)
        except json.JSONDecodeError:
            click.echo(f"Invalid JSON in input file: {input}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Export failed: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_export())


@cli.command()
@click.argument('task_id')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def status(task_id: str, verbose: bool):
    """Get the status of an export task."""
    
    async def _status():
        try:
            async with ExportIAEngine() as engine:
                status_data = await engine.get_task_status(task_id)
                
                if status_data is None:
                    click.echo(f"Task not found: {task_id}", err=True)
                    sys.exit(1)
                
                if verbose:
                    click.echo(json.dumps(status_data, indent=2, default=str))
                else:
                    click.echo(f"Status: {status_data['status']}")
                    if status_data['status'] == 'completed':
                        click.echo(f"File: {status_data['file_path']}")
                        click.echo(f"Quality: {status_data['quality_score']:.2f}")
                    elif status_data['status'] == 'failed':
                        click.echo(f"Error: {status_data['error']}")
        
        except Exception as e:
            click.echo(f"Error getting status: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_status())


@cli.command()
@click.argument('task_id')
def cancel(task_id: str):
    """Cancel an export task."""
    
    async def _cancel():
        try:
            async with ExportIAEngine() as engine:
                success = await engine.cancel_task(task_id)
                
                if success:
                    click.echo(f"Task cancelled: {task_id}")
                else:
                    click.echo(f"Task not found or cannot be cancelled: {task_id}", err=True)
                    sys.exit(1)
        
        except Exception as e:
            click.echo(f"Error cancelling task: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_cancel())


@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'table']), 
              default='table', help='Output format')
def formats(format: str):
    """List supported export formats."""
    
    async def _formats():
        try:
            async with ExportIAEngine() as engine:
                formats_data = engine.list_supported_formats()
                
                if format == 'json':
                    click.echo(json.dumps(formats_data, indent=2))
                else:
                    # Table format
                    click.echo("Supported Export Formats:")
                    click.echo("=" * 50)
                    
                    for fmt in formats_data:
                        click.echo(f"\n{fmt['name']} ({fmt['format']})")
                        click.echo(f"  Description: {fmt['description']}")
                        click.echo(f"  Features: {', '.join(fmt['professional_features'])}")
        
        except Exception as e:
            click.echo(f"Error listing formats: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_formats())


@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'table']), 
              default='table', help='Output format')
def stats(format: str):
    """Get system statistics."""
    
    async def _stats():
        try:
            async with ExportIAEngine() as engine:
                stats_data = engine.get_export_statistics()
                
                if format == 'json':
                    click.echo(json.dumps({
                        'total_tasks': stats_data.total_tasks,
                        'active_tasks': stats_data.active_tasks,
                        'completed_tasks': stats_data.completed_tasks,
                        'failed_tasks': stats_data.failed_tasks,
                        'format_distribution': stats_data.format_distribution,
                        'quality_distribution': stats_data.quality_distribution,
                        'average_quality_score': stats_data.average_quality_score,
                        'average_processing_time': stats_data.average_processing_time,
                        'total_processing_time': stats_data.total_processing_time
                    }, indent=2))
                else:
                    # Table format
                    click.echo("Export IA Statistics:")
                    click.echo("=" * 30)
                    click.echo(f"Total Tasks: {stats_data.total_tasks}")
                    click.echo(f"Active Tasks: {stats_data.active_tasks}")
                    click.echo(f"Completed Tasks: {stats_data.completed_tasks}")
                    click.echo(f"Failed Tasks: {stats_data.failed_tasks}")
                    click.echo(f"Average Quality Score: {stats_data.average_quality_score:.2f}")
                    click.echo(f"Average Processing Time: {stats_data.average_processing_time:.2f}s")
                    
                    if stats_data.format_distribution:
                        click.echo("\nFormat Distribution:")
                        for fmt, count in stats_data.format_distribution.items():
                            click.echo(f"  {fmt}: {count}")
                    
                    if stats_data.quality_distribution:
                        click.echo("\nQuality Distribution:")
                        for quality, count in stats_data.quality_distribution.items():
                            click.echo(f"  {quality}: {count}")
        
        except Exception as e:
            click.echo(f"Error getting statistics: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_stats())


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', default=1, help='Number of worker processes')
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the Export IA API server."""
    
    import uvicorn
    from api.fastapi_app import create_app
    
    click.echo(f"Starting Export IA API server on {host}:{port}")
    click.echo(f"Workers: {workers}, Reload: {reload}")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1
    )


@cli.command()
@click.argument('input_file')
@click.option('--format', '-f', type=click.Choice([f.value for f in ExportFormat]), 
              default='pdf', help='Export format')
@click.option('--output', '-o', help='Output file path')
def validate(input_file: str, format: str, output: Optional[str]):
    """Validate content and get quality metrics."""
    
    async def _validate():
        try:
            # Load input content
            with open(input_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Create export configuration
            config = ExportConfig(
                format=ExportFormat(format),
                document_type=DocumentType.REPORT,
                quality_level=QualityLevel.PROFESSIONAL
            )
            
            async with ExportIAEngine() as engine:
                metrics = engine.validate_content(content, config)
                
                if output:
                    with open(output, 'w', encoding='utf-8') as f:
                        json.dump({
                            'overall_score': metrics.overall_score,
                            'formatting_score': metrics.formatting_score,
                            'content_score': metrics.content_score,
                            'accessibility_score': metrics.accessibility_score,
                            'professional_score': metrics.professional_score,
                            'issues': metrics.issues,
                            'suggestions': metrics.suggestions
                        }, f, indent=2)
                    click.echo(f"Validation results saved to: {output}")
                else:
                    click.echo("Content Validation Results:")
                    click.echo("=" * 30)
                    click.echo(f"Overall Score: {metrics.overall_score:.2f}")
                    click.echo(f"Formatting Score: {metrics.formatting_score:.2f}")
                    click.echo(f"Content Score: {metrics.content_score:.2f}")
                    click.echo(f"Accessibility Score: {metrics.accessibility_score:.2f}")
                    click.echo(f"Professional Score: {metrics.professional_score:.2f}")
                    
                    if metrics.issues:
                        click.echo("\nIssues Found:")
                        for issue in metrics.issues:
                            click.echo(f"  - {issue}")
                    
                    if metrics.suggestions:
                        click.echo("\nSuggestions:")
                        for suggestion in metrics.suggestions:
                            click.echo(f"  - {suggestion}")
        
        except FileNotFoundError:
            click.echo(f"Input file not found: {input_file}", err=True)
            sys.exit(1)
        except json.JSONDecodeError:
            click.echo(f"Invalid JSON in input file: {input_file}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Validation failed: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_validate())


if __name__ == '__main__':
    cli()




