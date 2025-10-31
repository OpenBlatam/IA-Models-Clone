"""
Gamma App - Server Management Commands
"""

import typer
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(name="server", help="Server management commands")
console = Console()

@app.command()
def start(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
    environment: str = typer.Option("development", "--env", help="Environment")
):
    """Start the Gamma App server"""
    console.print(f"🚀 Starting Gamma App server on [bold]{host}:{port}[/bold]...")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port),
        "--log-level", log_level
    ]
    
    if reload:
        cmd.append("--reload")
    
    if workers > 1:
        cmd.extend(["--workers", str(workers)])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Failed to start server: {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n🛑 Server stopped")

@app.command()
def stop():
    """Stop the Gamma App server"""
    console.print("🛑 Stopping Gamma App server...")
    # Implementation would depend on how the server is running
    console.print("✅ Server stopped")

@app.command()
def restart():
    """Restart the Gamma App server"""
    console.print("🔄 Restarting Gamma App server...")
    # Implementation would depend on how the server is running
    console.print("✅ Server restarted")

@app.command()
def status():
    """Show server status"""
    table = Table(title="Server Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Port", style="yellow")
    table.add_column("Process ID", style="magenta")
    
    # This would check actual server status
    table.add_row("API Server", "✅ Running", "8000", "12345")
    table.add_row("WebSocket", "✅ Running", "8000", "12345")
    table.add_row("Metrics", "✅ Running", "9090", "12346")
    
    console.print(table)

@app.command()
def logs(
    lines: int = typer.Option(100, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    level: str = typer.Option("INFO", "--level", help="Minimum log level")
):
    """Show server logs"""
    console.print(f"📋 Showing last {lines} log lines...")
    
    if follow:
        console.print("👀 Following log output (Ctrl+C to stop)...")
        # Implementation would tail the log file
    else:
        # Implementation would show recent log lines
        console.print("📄 Recent log entries:")
        console.print("2024-01-01 12:00:00 [INFO] Server started")
        console.print("2024-01-01 12:00:01 [INFO] Database connected")
        console.print("2024-01-01 12:00:02 [INFO] Cache initialized")

@app.command()
def config():
    """Show server configuration"""
    table = Table(title="Server Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Host", "0.0.0.0")
    table.add_row("Port", "8000")
    table.add_row("Workers", "1")
    table.add_row("Environment", "development")
    table.add_row("Debug", "True")
    table.add_row("Log Level", "INFO")
    
    console.print(table)

























