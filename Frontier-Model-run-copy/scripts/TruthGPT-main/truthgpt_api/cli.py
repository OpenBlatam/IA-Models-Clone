"""
TruthGPT API Command Line Interface
==================================

CLI for TruthGPT API operations.
"""

import argparse
import sys
import os
from typing import List, Optional


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TruthGPT API - TensorFlow-like interface for TruthGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  truthgpt-api --version
  truthgpt-api --help
  truthgpt-api run examples/basic_example.py
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='TruthGPT API 1.0.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a Python script')
    run_parser.add_argument('script', help='Python script to run')
    run_parser.add_argument('args', nargs='*', help='Script arguments')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--coverage', action='store_true', help='Run with coverage')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install TruthGPT API')
    install_parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("TruthGPT API CLI - Verbose mode enabled")
    
    if args.command == 'run':
        run_script(args.script, args.args)
    elif args.command == 'test':
        run_tests(args.coverage)
    elif args.command == 'install':
        install_package(args.dev)
    else:
        parser.print_help()


def run_script(script: str, script_args: List[str]):
    """Run a Python script."""
    if not os.path.exists(script):
        print(f"Error: Script '{script}' not found")
        sys.exit(1)
    
    print(f"Running script: {script}")
    print(f"Arguments: {script_args}")
    
    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())
    
    # Run the script
    try:
        import subprocess
        result = subprocess.run([sys.executable, script] + script_args, check=True)
        print("Script completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error running script: {e}")
        sys.exit(1)


def run_tests(coverage: bool = False):
    """Run tests."""
    print("Running TruthGPT API tests...")
    
    try:
        import subprocess
        cmd = [sys.executable, '-m', 'pytest']
        if coverage:
            cmd.extend(['--cov=truthgpt_api', '--cov-report=html'])
        
        result = subprocess.run(cmd, check=True)
        print("Tests completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


def install_package(dev: bool = False):
    """Install TruthGPT API package."""
    print("Installing TruthGPT API...")
    
    try:
        import subprocess
        cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.']
        if dev:
            cmd.append('--dev')
        
        result = subprocess.run(cmd, check=True)
        print("Installation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error during installation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


