"""
CLI para utilidades del módulo de pipelines
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def cmd_check(args):
    """Comando para verificar compatibilidad"""
    from .pipelines import check_compatibility, get_import_statistics
    
    compatibility = check_compatibility()
    statistics = get_import_statistics()
    
    if args.format == "json":
        output = json.dumps({
            "compatibility": compatibility,
            "statistics": statistics
        }, indent=2)
        print(output)
    else:
        print("\n" + "=" * 60)
        print("Pipeline Compatibility Check")
        print("=" * 60)
        print(f"Status: {compatibility['status'].upper()}")
        print(f"Health Score: {compatibility['health_score']}/1.0")
        print(f"Coverage: {statistics['coverage_percentage']}%")
        print(f"Available Imports: {statistics['available_imports']}/{statistics['total_imports']}")
        
        if compatibility.get("recommendations"):
            print("\nRecommendations:")
            for rec in compatibility["recommendations"]:
                print(f"  {rec}")
        
        print("=" * 60 + "\n")


def cmd_monitor(args):
    """Comando para monitoreo"""
    from .pipelines_monitor import get_monitor, quick_health_check
    
    if args.quick:
        result = quick_health_check()
        print(json.dumps(result, indent=2))
        return
    
    monitor = get_monitor(
        check_interval=args.interval,
        alert_threshold=args.threshold,
        auto_start=args.start
    )
    
    if args.start:
        monitor.start_monitoring()
        print("Monitoring started. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\nMonitoring stopped.")
    else:
        summary = monitor.get_summary()
        print(json.dumps(summary, indent=2, default=str))


def cmd_export(args):
    """Comando para exportar reportes"""
    from .pipelines_utils import export_compatibility_report
    
    output_path = Path(args.output) if args.output else None
    report = export_compatibility_report(output_path, format=args.format)
    
    if not args.output:
        print(report)


def cmd_migration(args):
    """Comando para análisis de migración"""
    from .pipelines_migration_helper import generate_migration_report
    
    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    output_file = Path(args.output) if args.output else None
    
    print(f"Analyzing project: {project_root}")
    report = generate_migration_report(project_root, output_file)
    
    if not args.output:
        print(report)


def cmd_stats(args):
    """Comando para estadísticas"""
    from .pipelines import get_import_statistics
    
    stats = get_import_statistics()
    
    if args.format == "json":
        print(json.dumps(stats, indent=2))
    else:
        print("\n" + "=" * 60)
        print("Import Statistics")
        print("=" * 60)
        print(f"Total Imports: {stats['total_imports']}")
        print(f"Available: {stats['available_imports']}")
        print(f"Missing: {stats['missing_imports']}")
        print(f"Coverage: {stats['coverage_percentage']}%")
        
        if args.detailed:
            print("\nBy Category:")
            for category, data in stats['categories'].items():
                print(f"  {category}:")
                print(f"    Total: {data['total']}")
                print(f"    Available: {data['available']}")
                print(f"    Missing: {data['missing']}")
        
        print("=" * 60 + "\n")


def cmd_system(args):
    """Comando para sistema completo"""
    from .pipelines_integration import get_pipeline_system, quick_system_check
    
    if args.quick:
        result = quick_system_check()
        print(json.dumps(result, indent=2))
        return
    
    system = get_pipeline_system(
        enable_cache=not args.no_cache,
        enable_metrics=not args.no_metrics,
        enable_monitoring=args.start_monitoring
    )
    
    if args.start_monitoring:
        system.start_monitoring()
        print("System monitoring started. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            system.stop_monitoring()
            print("\nSystem monitoring stopped.")
    elif args.report:
        report = system.get_comprehensive_report()
        print(json.dumps(report, indent=2, default=str))
    elif args.status:
        status = system.get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        health = system.check_health()
        print(json.dumps(health, indent=2, default=str))


def main():
    """Función principal del CLI"""
    parser = argparse.ArgumentParser(
        description="Pipeline compatibility and monitoring tools"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Comando check
    check_parser = subparsers.add_parser("check", help="Check compatibility")
    check_parser.add_argument(
        "-f", "--format",
        choices=["json", "text"],
        default="text",
        help="Output format"
    )
    check_parser.set_defaults(func=cmd_check)
    
    # Comando monitor
    monitor_parser = subparsers.add_parser("monitor", help="Monitor health")
    monitor_parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick health check"
    )
    monitor_parser.add_argument(
        "-s", "--start",
        action="store_true",
        help="Start continuous monitoring"
    )
    monitor_parser.add_argument(
        "-i", "--interval",
        type=float,
        default=60.0,
        help="Check interval in seconds"
    )
    monitor_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.8,
        help="Alert threshold"
    )
    monitor_parser.set_defaults(func=cmd_monitor)
    
    # Comando export
    export_parser = subparsers.add_parser("export", help="Export report")
    export_parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    export_parser.add_argument(
        "-f", "--format",
        choices=["json", "txt", "md"],
        default="json",
        help="Report format"
    )
    export_parser.set_defaults(func=cmd_export)
    
    # Comando migration
    migration_parser = subparsers.add_parser("migration", help="Migration analysis")
    migration_parser.add_argument(
        "-p", "--project-root",
        help="Project root directory"
    )
    migration_parser.add_argument(
        "-o", "--output",
        help="Output report file"
    )
    migration_parser.set_defaults(func=cmd_migration)
    
    # Comando stats
    stats_parser = subparsers.add_parser("stats", help="Import statistics")
    stats_parser.add_argument(
        "-f", "--format",
        choices=["json", "text"],
        default="text",
        help="Output format"
    )
    stats_parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="Show detailed statistics"
    )
    stats_parser.set_defaults(func=cmd_stats)
    
    # Comando system
    system_parser = subparsers.add_parser("system", help="Full system operations")
    system_parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick system check"
    )
    system_parser.add_argument(
        "-s", "--start-monitoring",
        action="store_true",
        help="Start system monitoring"
    )
    system_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache"
    )
    system_parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metrics"
    )
    system_parser.add_argument(
        "-r", "--report",
        action="store_true",
        help="Get comprehensive report"
    )
    system_parser.add_argument(
        "--status",
        action="store_true",
        help="Get system status"
    )
    system_parser.set_defaults(func=cmd_system)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

