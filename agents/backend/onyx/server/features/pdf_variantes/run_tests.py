#!/usr/bin/env python
"""
Test Runner for PDF Variantes
==============================
Script para ejecutar los tests del módulo PDF Variantes.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(
    verbose: bool = True,
    coverage: bool = False,
    markers: str = None,
    specific_file: str = None,
    parallel: bool = False
):
    """Ejecutar tests con diferentes opciones."""
    
    # Base command
    cmd = ["pytest"]
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Add markers
    if markers:
        cmd.extend(["-m", markers])
    
    # Add specific file
    if specific_file:
        cmd.append(specific_file)
    else:
        cmd.append("tests/")
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Run tests
    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejecutar tests de PDF Variantes")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Ejecutar en modo verbose (default: True)"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Ejecutar con cobertura de código"
    )
    parser.add_argument(
        "-m", "--markers",
        type=str,
        help="Ejecutar tests con marcadores específicos (ej: 'unit', 'integration')"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Ejecutar tests de un archivo específico"
    )
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Ejecutar tests en paralelo (requiere pytest-xdist)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Ejecutar solo tests rápidos (excluye 'slow')"
    )
    
    args = parser.parse_args()
    
    # Handle quick flag
    if args.quick:
        if args.markers:
            args.markers = f"{args.markers} and not slow"
        else:
            args.markers = "not slow"
    
    # Run tests
    exit_code = run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        markers=args.markers,
        specific_file=args.file,
        parallel=args.parallel
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()



