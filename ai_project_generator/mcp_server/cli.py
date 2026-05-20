#!/usr/bin/env python3
"""
MCP Server CLI - Command Line Interface
========================================

Herramienta CLI para administración y diagnóstico del módulo MCP Server.
"""

import argparse
import json
import sys
from typing import Optional, List
from pathlib import Path

try:
    from . import (
        get_version, check_imports, get_missing_imports,
        get_available_features, get_module_info,
        get_diagnostics, check_health, validate_setup
    )
    from .utils.diagnostics import (
        print_diagnostic_report, generate_diagnostic_report,
        get_performance_metrics, get_dependency_tree
    )
except ImportError as e:
    print(f"Error importing MCP Server modules: {e}", file=sys.stderr)
    sys.exit(1)


class MCPCLI:
    """CLI principal para MCP Server."""
    
    def __init__(self):
        """Inicializar CLI."""
        self.parser = argparse.ArgumentParser(
            description="MCP Server - Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s version                    # Show version
  %(prog)s health                     # Check module health
  %(prog)s diagnostics                # Full diagnostics
  %(prog)s validate                   # Validate setup
  %(prog)s imports                    # Check imports status
  %(prog)s report --output report.txt # Generate report
            """
        )
        self.subparsers = self.parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )
        self._setup_commands()
    
    def _setup_commands(self):
        """Configurar comandos."""
        # Version command
        self.subparsers.add_parser('version', help='Show version information')
        
        # Health command
        health_parser = self.subparsers.add_parser('health', help='Check module health')
        health_parser.add_argument(
            '--format', choices=['json', 'text'], default='text',
            help='Output format (default: text)'
        )
        
        # Diagnostics command
        diag_parser = self.subparsers.add_parser('diagnostics', help='Run full diagnostics')
        diag_parser.add_argument(
            '--format', choices=['json', 'text'], default='text',
            help='Output format (default: text)'
        )
        diag_parser.add_argument(
            '--output', type=str,
            help='Output file (optional)'
        )
        
        # Validate command
        validate_parser = self.subparsers.add_parser('validate', help='Validate module setup')
        validate_parser.add_argument(
            '--format', choices=['json', 'text'], default='text',
            help='Output format (default: text)'
        )
        
        # Imports command
        imports_parser = self.subparsers.add_parser('imports', help='Check imports status')
        imports_parser.add_argument(
            '--format', choices=['json', 'text'], default='text',
            help='Output format (default: text)'
        )
        imports_parser.add_argument(
            '--missing', action='store_true',
            help='Show only missing imports'
        )
        
        # Report command
        report_parser = self.subparsers.add_parser('report', help='Generate diagnostic report')
        report_parser.add_argument(
            '--output', type=str,
            help='Output file (default: stdout)'
        )
        
        # Performance command
        perf_parser = self.subparsers.add_parser('performance', help='Show performance metrics')
        perf_parser.add_argument(
            '--format', choices=['json', 'text'], default='text',
            help='Output format (default: text)'
        )
        perf_parser.add_argument(
            '--watch', action='store_true',
            help='Watch mode (continuous updates)'
        )
        perf_parser.add_argument(
            '--interval', type=int, default=5,
            help='Update interval in seconds (default: 5)'
        )
        
        # Dependencies command
        deps_parser = self.subparsers.add_parser('dependencies', help='Show dependency tree')
        deps_parser.add_argument(
            '--format', choices=['json', 'text'], default='text',
            help='Output format (default: text)'
        )
        
        # Info command
        info_parser = self.subparsers.add_parser('info', help='Show module information')
        info_parser.add_argument(
            '--format', choices=['json', 'text'], default='text',
            help='Output format (default: text)'
        )
        
        # Config command
        config_parser = self.subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config commands')
        
        # Config show
        config_show = config_subparsers.add_parser('show', help='Show configuration')
        config_show.add_argument('--path', type=str, help='Config file path')
        config_show.add_argument('--format', choices=['json', 'yaml', 'text'], default='text')
        
        # Config validate
        config_validate = config_subparsers.add_parser('validate', help='Validate configuration')
        config_validate.add_argument('--path', type=str, required=True, help='Config file path')
        
        # Config template
        config_template = config_subparsers.add_parser('template', help='Generate config template')
        config_template.add_argument('--output', type=str, required=True, help='Output file path')
        config_template.add_argument('--format', choices=['json', 'yaml'], default='json')
        
        # Config get
        config_get = config_subparsers.add_parser('get', help='Get config value')
        config_get.add_argument('--path', type=str, help='Config file path')
        config_get.add_argument('key', type=str, help='Config key (dot notation)')
        
        # Config set
        config_set = config_subparsers.add_parser('set', help='Set config value')
        config_set.add_argument('--path', type=str, help='Config file path')
        config_set.add_argument('key', type=str, help='Config key (dot notation)')
        config_set.add_argument('value', type=str, help='Config value (JSON)')
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Ejecutar CLI.
        
        Args:
            args: Argumentos (None = sys.argv)
            
        Returns:
            Código de salida
        """
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return 1
        
        try:
            if parsed_args.command == 'version':
                return self._version()
            elif parsed_args.command == 'health':
                return self._health(parsed_args)
            elif parsed_args.command == 'diagnostics':
                return self._diagnostics(parsed_args)
            elif parsed_args.command == 'validate':
                return self._validate(parsed_args)
            elif parsed_args.command == 'imports':
                return self._imports(parsed_args)
            elif parsed_args.command == 'report':
                return self._report(parsed_args)
            elif parsed_args.command == 'performance':
                return self._performance(parsed_args)
            elif parsed_args.command == 'dependencies':
                return self._dependencies(parsed_args)
            elif parsed_args.command == 'info':
                return self._info(parsed_args)
            elif parsed_args.command == 'config':
                return self._config(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}", file=sys.stderr)
                return 1
        except KeyboardInterrupt:
            print("\nInterrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _version(self) -> int:
        """Mostrar versión."""
        version = get_version()
        print(f"MCP Server v{version}")
        return 0
    
    def _health(self, args) -> int:
        """Verificar salud."""
        health = check_health()
        
        if args.format == 'json':
            print(json.dumps(health, indent=2, default=str))
        else:
            status = health.get('status', 'unknown')
            status_symbol = {
                'healthy': '✓',
                'degraded': '⚠',
                'unhealthy': '✗',
                'unknown': '?'
            }.get(status, '?')
            
            print(f"Status: {status_symbol} {status.upper()}")
            print()
            
            for check_name, check_info in health.get('checks', {}).items():
                if isinstance(check_info, dict):
                    check_status = check_info.get('status', 'unknown')
                    print(f"  {check_name}: {check_status}")
                    if 'availability_rate' in check_info:
                        print(f"    Availability: {check_info['availability_rate']:.1f}%")
                    if 'components' in check_info:
                        for comp_name, comp_available in check_info['components'].items():
                            symbol = '✓' if comp_available else '✗'
                            print(f"    {symbol} {comp_name}")
        
        return 0 if health.get('status') == 'healthy' else 1
    
    def _diagnostics(self, args) -> int:
        """Ejecutar diagnósticos."""
        diagnostics = get_diagnostics()
        
        output = None
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'json':
            content = json.dumps(diagnostics, indent=2, default=str)
        else:
            content = generate_diagnostic_report()
        
        if output:
            output.write_text(content)
            print(f"Diagnostics saved to {output}")
        else:
            print(content)
        
        return 0
    
    def _validate(self, args) -> int:
        """Validar configuración."""
        is_valid, errors = validate_setup()
        
        if args.format == 'json':
            result = {
                'valid': is_valid,
                'errors': errors
            }
            print(json.dumps(result, indent=2))
        else:
            if is_valid:
                print("✓ Module setup is valid")
            else:
                print("✗ Module setup has errors:")
                for error in errors:
                    print(f"  - {error}")
        
        return 0 if is_valid else 1
    
    def _imports(self, args) -> int:
        """Verificar imports."""
        imports_status = check_imports()
        missing = get_missing_imports()
        
        if args.missing:
            if args.format == 'json':
                print(json.dumps(missing, indent=2))
            else:
                if missing:
                    print("Missing imports:")
                    for imp in missing:
                        print(f"  - {imp}")
                else:
                    print("✓ All imports available")
        else:
            if args.format == 'json':
                print(json.dumps(imports_status, indent=2))
            else:
                total = len(imports_status)
                available = sum(1 for v in imports_status.values() if v)
                rate = (available / total * 100) if total > 0 else 0.0
                
                print(f"Imports: {available}/{total} ({rate:.1f}%)")
                if missing:
                    print(f"\nMissing ({len(missing)}):")
                    for imp in missing[:10]:
                        print(f"  - {imp}")
                    if len(missing) > 10:
                        print(f"  ... and {len(missing) - 10} more")
        
        return 0
    
    def _report(self, args) -> int:
        """Generar reporte."""
        report = generate_diagnostic_report()
        
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report)
            print(f"Report saved to {output}")
        else:
            print(report)
        
        return 0
    
    def _performance(self, args) -> int:
        """Mostrar métricas de performance."""
        import time
        
        if args.watch:
            try:
                while True:
                    metrics = get_performance_metrics()
                    self._print_performance(metrics, args.format)
                    time.sleep(args.interval)
                    print()  # Separador
            except KeyboardInterrupt:
                print("\nStopped")
                return 0
        else:
            metrics = get_performance_metrics()
            self._print_performance(metrics, args.format)
            return 0
    
    def _print_performance(self, metrics: dict, format_type: str):
        """Imprimir métricas de performance."""
        if format_type == 'json':
            print(json.dumps(metrics, indent=2, default=str))
        else:
            print("Performance Metrics:")
            print(f"  Memory RSS: {metrics['memory']['rss_mb']:.2f} MB")
            print(f"  Memory VMS: {metrics['memory']['vms_mb']:.2f} MB")
            print(f"  Memory %: {metrics['memory']['percent']:.1f}%")
            print(f"  CPU %: {metrics['cpu']['percent']:.2f}%")
            print(f"  Threads: {metrics['cpu']['num_threads']}")
            print(f"  Timestamp: {metrics['timestamp']}")
    
    def _dependencies(self, args) -> int:
        """Mostrar árbol de dependencias."""
        tree = get_dependency_tree()
        
        if args.format == 'json':
            print(json.dumps(tree, indent=2, default=str))
        else:
            print(f"Total Groups: {tree.get('total_groups', 0)}")
            print(f"Total Modules: {tree.get('total_modules', 0)}")
            print()
            
            for group_name, group_info in tree.get('groups', {}).items():
                print(f"{group_name}:")
                print(f"  Modules: {group_info.get('module_count', 0)}")
                for module in group_info.get('modules', [])[:5]:
                    symbols = module.get('symbols', [])
                    if isinstance(symbols, list):
                        symbols_str = ', '.join(symbols[:3])
                        if len(symbols) > 3:
                            symbols_str += f" ... (+{len(symbols) - 3} more)"
                    else:
                        symbols_str = str(symbols)
                    print(f"    - {module.get('path', 'unknown')}: {symbols_str}")
                if group_info.get('module_count', 0) > 5:
                    print(f"    ... and {group_info.get('module_count', 0) - 5} more modules")
                print()
        
        return 0
    
    def _info(self, args) -> int:
        """Mostrar información del módulo."""
        info = get_module_info()
        
        if args.format == 'json':
            print(json.dumps(info, indent=2, default=str))
        else:
            print(f"Version: {info.get('version', 'unknown')}")
            print(f"Author: {info.get('author', 'unknown')}")
            print(f"License: {info.get('license', 'unknown')}")
            
            if 'statistics' in info:
                stats = info['statistics']
                print(f"\nStatistics:")
                print(f"  Total Components: {stats.get('total_components', 0)}")
                print(f"  Available: {stats.get('available_components', 0)}")
                print(f"  Missing: {stats.get('missing_components', 0)}")
        
        return 0
    
    def _config(self, args) -> int:
        """Gestionar configuración."""
        try:
            from .utils.config_manager import ConfigManager, create_config_template
        except ImportError:
            print("Config manager not available", file=sys.stderr)
            return 1
        
        if args.config_command == 'show':
            return self._config_show(args)
        elif args.config_command == 'validate':
            return self._config_validate(args)
        elif args.config_command == 'template':
            return self._config_template(args)
        elif args.config_command == 'get':
            return self._config_get(args)
        elif args.config_command == 'set':
            return self._config_set(args)
        else:
            print("Unknown config command", file=sys.stderr)
            return 1
    
    def _config_show(self, args) -> int:
        """Mostrar configuración."""
        from .utils.config_manager import ConfigManager
        
        manager = ConfigManager(args.path)
        try:
            config = manager.load()
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1
        
        if args.format == 'json':
            print(json.dumps(config, indent=2, default=str))
        elif args.format == 'yaml':
            try:
                import yaml
                print(yaml.safe_dump(config, default_flow_style=False))
            except ImportError:
                print("PyYAML not installed", file=sys.stderr)
                return 1
        else:
            for section, values in config.items():
                print(f"\n[{section}]")
                if isinstance(values, dict):
                    for key, value in values.items():
                        print(f"  {key} = {value}")
                else:
                    print(f"  {values}")
        
        return 0
    
    def _config_validate(self, args) -> int:
        """Validar configuración."""
        from .utils.config_manager import ConfigManager
        
        manager = ConfigManager(args.path)
        try:
            manager.load()
            errors = manager.validate()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("✓ Configuration is valid")
            return 0
    
    def _config_template(self, args) -> int:
        """Generar plantilla de configuración."""
        from .utils.config_manager import create_config_template
        
        try:
            create_config_template(args.output, args.format)
            print(f"Config template created at {args.output}")
            return 0
        except Exception as e:
            print(f"Error creating template: {e}", file=sys.stderr)
            return 1
    
    def _config_get(self, args) -> int:
        """Obtener valor de configuración."""
        from .utils.config_manager import ConfigManager
        
        manager = ConfigManager(args.path)
        try:
            manager.load()
            value = manager.get(args.key)
            if value is None:
                print(f"Key '{args.key}' not found", file=sys.stderr)
                return 1
            print(json.dumps(value, indent=2, default=str))
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _config_set(self, args) -> int:
        """Establecer valor de configuración."""
        from .utils.config_manager import ConfigManager
        
        manager = ConfigManager(args.path)
        try:
            if args.path and Path(args.path).exists():
                manager.load()
            else:
                manager._config = {}
            
            try:
                value = json.loads(args.value)
            except json.JSONDecodeError:
                value = args.value
            
            manager.set(args.key, value)
            manager.save()
            print(f"Set {args.key} = {value}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1


def main():
    """Función principal."""
    cli = MCPCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

