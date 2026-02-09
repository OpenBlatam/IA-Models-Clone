"""
Development Optimizations

Optimizations for:
- Hot reload
- Development server
- Code quality
- Linting
- Formatting
- Type checking
"""

import logging
import subprocess
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DevelopmentServer:
    """Optimized development server."""
    
    @staticmethod
    def run_dev_server(
        host: str = "0.0.0.0",
        port: int = 8020,
        reload: bool = True,
        workers: int = 1
    ) -> None:
        """
        Run optimized development server.
        
        Args:
            host: Host address
            port: Port number
            reload: Enable hot reload
            workers: Number of workers
        """
        import uvicorn
        
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level="info",
            access_log=True
        )
    
    @staticmethod
    def get_dev_config() -> Dict[str, Any]:
        """Get development configuration."""
        return {
            "debug": True,
            "reload": True,
            "workers": 1,
            "log_level": "DEBUG",
            "enable_cache": False,
            "enable_compression": False
        }


class CodeQuality:
    """Code quality optimization."""
    
    @staticmethod
    def run_linter(path: str = ".", fix: bool = False) -> int:
        """
        Run linter with optimizations.
        
        Args:
            path: Path to lint
            fix: Auto-fix issues
            
        Returns:
            Exit code
        """
        cmd = ["ruff", "check", path]
        if fix:
            cmd.append("--fix")
        
        result = subprocess.run(cmd)
        return result.returncode
    
    @staticmethod
    def run_formatter(path: str = ".", check: bool = False) -> int:
        """
        Run code formatter.
        
        Args:
            path: Path to format
            check: Check only (don't format)
            
        Returns:
            Exit code
        """
        cmd = ["black", path]
        if check:
            cmd.append("--check")
        
        result = subprocess.run(cmd)
        return result.returncode
    
    @staticmethod
    def run_type_checker(path: str = ".") -> int:
        """
        Run type checker.
        
        Args:
            path: Path to check
            
        Returns:
            Exit code
        """
        cmd = ["mypy", path, "--ignore-missing-imports"]
        result = subprocess.run(cmd)
        return result.returncode
    
    @staticmethod
    def run_all_checks(path: str = ".") -> Dict[str, int]:
        """
        Run all code quality checks.
        
        Args:
            path: Path to check
            
        Returns:
            Dictionary of exit codes
        """
        return {
            "linter": CodeQuality.run_linter(path),
            "formatter": CodeQuality.run_formatter(path, check=True),
            "type_checker": CodeQuality.run_type_checker(path)
        }


class HotReloadOptimizer:
    """Hot reload optimization."""
    
    @staticmethod
    def setup_watchdog(paths: List[str], callback: Callable) -> None:
        """
        Setup file watching for hot reload.
        
        Args:
            paths: Paths to watch
            callback: Callback function
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class Handler(FileSystemEventHandler):
                def on_modified(self, event):
                    if event.src_path.endswith('.py'):
                        callback(event.src_path)
            
            observer = Observer()
            for path in paths:
                observer.schedule(Handler(), path, recursive=True)
            
            observer.start()
            logger.info("File watching started")
            
        except ImportError:
            logger.warning("watchdog not installed, hot reload disabled")


class PreCommitHooks:
    """Pre-commit hooks optimization."""
    
    @staticmethod
    def create_pre_commit_hook() -> str:
        """Create optimized pre-commit hook."""
        return """#!/bin/bash
# Pre-commit hook

# Run linter
ruff check --fix .

# Run formatter
black .

# Run type checker
mypy . --ignore-missing-imports

# Run tests
pytest tests/ -v --tb=short

exit 0
"""








