#!/usr/bin/env python3
"""
Development Helper Script
Quick commands for common development tasks
"""

import sys
import subprocess
import argparse
from pathlib import Path


def get_venv_command(cmd: str) -> str:
    """Get command path in virtual environment"""
    if sys.platform == 'win32':
        return f".venv\\Scripts\\{cmd}.exe"
    else:
        return f".venv/bin/{cmd}"


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Run development server"""
    reload_flag = "--reload" if reload else ""
    cmd = f"uvicorn app:app --host {host} --port {port} {reload_flag}"
    subprocess.run(cmd.split())


def run_tests(verbose: bool = False, coverage: bool = False):
    """Run tests"""
    cmd = ["pytest"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    cmd.append("tests/")
    subprocess.run(cmd)


def run_lint():
    """Run linting"""
    try:
        # Try ruff first (faster)
        subprocess.run(["ruff", "check", "."])
    except FileNotFoundError:
        # Fallback to flake8
        try:
            subprocess.run(["flake8", "."])
        except FileNotFoundError:
            print("⚠️  No linter found. Install ruff or flake8")


def run_format():
    """Format code"""
    try:
        # Try ruff first
        subprocess.run(["ruff", "format", "."])
    except FileNotFoundError:
        # Fallback to black
        try:
            subprocess.run(["black", "."])
        except FileNotFoundError:
            print("⚠️  No formatter found. Install ruff or black")


def run_type_check():
    """Run type checking"""
    try:
        subprocess.run(["mypy", "."])
    except FileNotFoundError:
        print("⚠️  mypy not found. Install with: pip install mypy")


def run_security_check():
    """Run security audit"""
    subprocess.run(["pip-audit", "--desc"])


def generate_migration(message: str):
    """Generate database migration"""
    subprocess.run(["alembic", "revision", "--autogenerate", "-m", message])


def apply_migrations():
    """Apply database migrations"""
    subprocess.run(["alembic", "upgrade", "head"])


def open_docs():
    """Open API documentation in browser"""
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000/docs")


def main():
    parser = argparse.ArgumentParser(description="Development helper script")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server
    server_parser = subparsers.add_parser("server", help="Run development server")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    server_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    # Tests
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("-v", "--verbose", action="store_true")
    test_parser.add_argument("-c", "--coverage", action="store_true")
    
    # Lint
    subparsers.add_parser("lint", help="Run linter")
    
    # Format
    subparsers.add_parser("format", help="Format code")
    
    # Type check
    subparsers.add_parser("typecheck", help="Run type checker")
    
    # Security
    subparsers.add_parser("security", help="Run security audit")
    
    # Migrations
    migration_parser = subparsers.add_parser("migrate", help="Generate migration")
    migration_parser.add_argument("message", help="Migration message")
    
    subparsers.add_parser("migrate-up", help="Apply migrations")
    
    # Docs
    subparsers.add_parser("docs", help="Open API docs")
    
    args = parser.parse_args()
    
    # Change to project root
    script_dir = Path(__file__).parent.parent
    import os
    os.chdir(script_dir)
    
    if args.command == "server":
        run_server(args.host, args.port, not args.no_reload)
    elif args.command == "test":
        run_tests(args.verbose, args.coverage)
    elif args.command == "lint":
        run_lint()
    elif args.command == "format":
        run_format()
    elif args.command == "typecheck":
        run_type_check()
    elif args.command == "security":
        run_security_check()
    elif args.command == "migrate":
        generate_migration(args.message)
    elif args.command == "migrate-up":
        apply_migrations()
    elif args.command == "docs":
        open_docs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


