#!/usr/bin/env python3
"""
Test Runner Unificado
Ejecutor unificado que combina todas las funcionalidades
"""

import sys
import argparse
from pathlib import Path
from typing import Optional


def main():
    """Función principal unificada"""
    parser = argparse.ArgumentParser(
        description='Test Runner Unificado - TruthGPT Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s test unit              # Ejecutar tests unitarios
  %(prog)s analyze                # Analizar estructura
  %(prog)s monitor --duration 3600 # Monitorear 1 hora
  %(prog)s report                 # Generar reporte
  %(prog)s health                 # Health check
  %(prog)s backup                 # Crear backup
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Test commands
    test_parser = subparsers.add_parser('test', help='Ejecutar tests')
    test_parser.add_argument('category', nargs='?', default='all',
                            help='Categoría de tests')
    test_parser.add_argument('-v', '--verbose', action='store_true')
    test_parser.add_argument('--coverage', action='store_true')
    
    # Analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analizar tests')
    analyze_parser.add_argument('--output', type=Path)
    
    # Monitor
    monitor_parser = subparsers.add_parser('monitor', help='Monitorear tests')
    monitor_parser.add_argument('--duration', type=int)
    monitor_parser.add_argument('--interval', type=int, default=60)
    
    # Report
    report_parser = subparsers.add_parser('report', help='Generar reporte')
    report_parser.add_argument('--output', type=Path)
    
    # Health
    health_parser = subparsers.add_parser('health', help='Health check')
    
    # Backup
    backup_parser = subparsers.add_parser('backup', help='Crear backup')
    backup_parser.add_argument('--type', choices=['all', 'results', 'config'])
    
    # Stats
    stats_parser = subparsers.add_parser('stats', help='Estadísticas')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    base_path = Path.cwd()
    
    # Ejecutar comando correspondiente
    if args.command == 'test':
        from run_tests import run_tests
        extra_args = []
        if args.verbose:
            extra_args.append('-v')
        if args.coverage:
            extra_args.append('--coverage')
        return run_tests(args.category, extra_args)
    
    elif args.command == 'analyze':
        from analyze_tests import main as analyze_main
        sys.argv = ['analyze_tests.py']
        if args.output:
            sys.argv.extend(['--output', str(args.output)])
        return analyze_main()
    
    elif args.command == 'monitor':
        from monitor_tests import main as monitor_main
        sys.argv = ['monitor_tests.py']
        if args.duration:
            sys.argv.extend(['--duration', str(args.duration)])
        if args.interval:
            sys.argv.extend(['--interval', str(args.interval)])
        return monitor_main()
    
    elif args.command == 'report':
        from generate_report import main as report_main
        sys.argv = ['generate_report.py']
        if args.output:
            sys.argv.extend(['--output', str(args.output)])
        return report_main()
    
    elif args.command == 'health':
        from health_check import main as health_main
        return health_main()
    
    elif args.command == 'backup':
        from backup_tests import main as backup_main
        sys.argv = ['backup_tests.py']
        if args.type == 'all':
            sys.argv.append('--all')
        elif args.type == 'results':
            sys.argv.append('--results')
        elif args.type == 'config':
            sys.argv.append('--config')
        return backup_main()
    
    elif args.command == 'stats':
        from stats_dashboard import main as stats_main
        return stats_main()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

