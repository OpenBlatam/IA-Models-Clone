#!/usr/bin/env python3
"""
Database Optimization Script
=============================

Script para optimizar la base de datos.
"""

import sys
from pathlib import Path

# Agregar ruta del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.optimizer import DatabaseOptimizer
import argparse


def main():
    """Ejecutar optimización."""
    parser = argparse.ArgumentParser(description="Database optimization tool")
    parser.add_argument("--db-path", default="artist_manager.db", help="Database path")
    parser.add_argument("--action", choices=["analyze", "vacuum", "reindex", "all"], default="all", help="Action to perform")
    
    args = parser.parse_args()
    
    optimizer = DatabaseOptimizer(args.db_path)
    
    if args.action == "analyze":
        analysis = optimizer.analyze()
        print("Database Analysis:")
        print(f"  Total size: {analysis['total_size'] / 1024 / 1024:.2f} MB")
        print(f"  Tables: {len(analysis['tables'])}")
        for table, info in analysis['tables'].items():
            print(f"    - {table}: {info['row_count']} rows")
        
        if analysis['recommendations']:
            print("\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")
    
    elif args.action == "vacuum":
        if optimizer.vacuum():
            print("✓ VACUUM completed successfully")
        else:
            print("✗ VACUUM failed")
    
    elif args.action == "reindex":
        if optimizer.reindex():
            print("✓ REINDEX completed successfully")
        else:
            print("✗ REINDEX failed")
    
    elif args.action == "all":
        print("Running full optimization...")
        result = optimizer.optimize()
        
        if result['vacuum']:
            print("✓ VACUUM completed")
        if result['reindex']:
            print("✓ REINDEX completed")
        
        print("\nAnalysis:")
        analysis = result['analysis']
        print(f"  Total size: {analysis['total_size'] / 1024 / 1024:.2f} MB")
        print(f"  Tables: {len(analysis['tables'])}")


if __name__ == "__main__":
    main()




