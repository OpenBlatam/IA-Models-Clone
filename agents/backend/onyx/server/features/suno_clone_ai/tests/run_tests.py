#!/usr/bin/env python3
"""
Script para ejecutar tests con opciones

Uso:
    python run_tests.py                    # Todos los tests
    python run_tests.py --api              # Solo tests de API
    python run_tests.py --helpers          # Solo tests de helpers
    python run_tests.py --coverage         # Con cobertura
    python run_tests.py --generate         # Generar tests
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Ejecutar tests del proyecto")
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Ejecutar solo tests de API"
    )
    parser.add_argument(
        "--helpers",
        action="store_true",
        help="Ejecutar solo tests de helpers"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Ejecutar con cobertura"
    )
    parser.add_argument(
        "--markers",
        nargs="+",
        help="Ejecutar tests con marcadores específicos"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generar tests usando el generador"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Modo verbose"
    )
    parser.add_argument(
        "--file",
        help="Ejecutar un archivo de test específico"
    )
    
    args = parser.parse_args()
    
    # Determinar comando pytest
    cmd_parts = ["pytest"]
    
    if args.verbose:
        cmd_parts.append("-v")
    else:
        cmd_parts.append("-q")
    
    if args.api:
        cmd_parts.append("tests/test_api/")
    elif args.helpers:
        cmd_parts.append("tests/test_helpers/")
    elif args.file:
        cmd_parts.append(args.file)
    else:
        cmd_parts.append("tests/")
    
    if args.markers:
        for marker in args.markers:
            cmd_parts.extend(["-m", marker])
    
    if args.coverage:
        cmd_parts.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term"
        ])
    
    cmd = " ".join(cmd_parts)
    
    # Ejecutar comando
    if args.generate:
        print("\nGenerando tests...")
        print("Ejecuta: python example_generate_tests.py")
        return run_command(
            "python tests/example_generate_tests.py",
            "Generador de Tests"
        )
    else:
        return run_command(cmd, "Ejecutando Tests")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

