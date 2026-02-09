"""
Script para ejecutar tests con opciones avanzadas
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(
    marker: str = None,
    path: str = None,
    coverage: bool = False,
    verbose: bool = True,
    parallel: bool = False,
    slow: bool = True
):
    """Ejecuta tests con opciones personalizadas"""
    cmd = ["pytest"]
    
    if path:
        cmd.append(path)
    else:
        cmd.append("tests/")
    
    if marker:
        cmd.extend(["-m", marker])
    
    if not slow:
        cmd.extend(["-m", "not slow"])
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    cmd.extend(["--tb=short"])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for Lovable Community")
    
    parser.add_argument(
        "-m", "--marker",
        help="Run tests with specific marker (unit, integration, api, security, etc.)"
    )
    
    parser.add_argument(
        "-p", "--path",
        help="Path to specific test file or directory"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--no-slow",
        action="store_true",
        help="Skip slow tests"
    )
    
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run only security tests"
    )
    
    parser.add_argument(
        "--load",
        action="store_true",
        help="Run only load tests"
    )
    
    args = parser.parse_args()
    
    # Determinar marker
    marker = args.marker
    if args.unit:
        marker = "unit"
    elif args.integration:
        marker = "integration"
    elif args.security:
        marker = "security"
    elif args.load:
        marker = "load"
    
    return run_tests(
        marker=marker,
        path=args.path,
        coverage=args.coverage,
        verbose=not args.quiet,
        parallel=args.parallel,
        slow=not args.no_slow
    )


if __name__ == "__main__":
    sys.exit(main())

