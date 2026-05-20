from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import sys
from pathlib import Path
import uvicorn
import click
    import os
    import subprocess
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Run script for improved AI Video API
===================================

Launch the improved FastAPI application with optimal settings.
"""



def setup_path():
    """Setup Python path for imports."""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--workers", default=1, help="Number of worker processes")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--env", default="development", help="Environment (development/production)")
def run_api(host: str, port: int, workers: int, reload: bool, env: str):
    """Run the improved AI Video API."""
    
    setup_path()
    
    # Set environment
    os.environ["ENVIRONMENT"] = env
    
    print(f"🚀 Starting AI Video API (Improved)")
    print(f"📍 Environment: {env}")
    print(f"🌐 URL: http://{host}:{port}")
    print(f"👥 Workers: {workers}")
    print(f"🔄 Reload: {'✅' if reload else '❌'}")
    print("=" * 50)
    
    # Configuration based on environment
    config = {
        "app": "agents.backend.onyx.server.features.ai_video.api.improved_main:app",
        "host": host,
        "port": port,
        "reload": reload,
        "access_log": env == "development",
    }
    
    if env == "production":
        config.update({
            "workers": workers,
            "loop": "uvloop",
            "http": "httptools",
            "reload": False,
            "log_level": "info",
        })
    else:
        config.update({
            "workers": 1,
            "log_level": "debug",
        })
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n🛑 API shutdown requested")
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        sys.exit(1)


@click.command()
def install_deps():
    """Install dependencies for the improved API."""
    
    print("📦 Installing improved API dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements_improved.txt"
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)


@click.group()
def cli():
    """AI Video API Improved - Management CLI"""
    pass


cli.add_command(run_api, name="run")
cli.add_command(install_deps, name="install")


match __name__:
    case "__main__":
    cli() 