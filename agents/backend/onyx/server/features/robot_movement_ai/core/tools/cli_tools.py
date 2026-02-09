"""
CLI Tools
=========

Herramientas de línea de comandos.
"""

import argparse
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class CLITool:
    """
    Herramienta CLI base.
    
    Base para crear herramientas de línea de comandos.
    """
    
    def __init__(self, name: str, description: str):
        """
        Inicializar herramienta CLI.
        
        Args:
            name: Nombre de la herramienta
            description: Descripción
        """
        self.name = name
        self.description = description
        self.parser = argparse.ArgumentParser(description=description)
        self._setup_arguments()
    
    def _setup_arguments(self) -> None:
        """Configurar argumentos (sobrescribir en subclases)."""
        self.parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Verbose output'
        )
        self.parser.add_argument(
            '--config',
            type=str,
            help='Configuration file path'
        )
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Ejecutar herramienta.
        
        Args:
            args: Argumentos (None = sys.argv)
            
        Returns:
            Código de salida
        """
        parsed_args = self.parser.parse_args(args)
        
        if parsed_args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        try:
            return self._execute(parsed_args)
        except Exception as e:
            logger.error(f"Error executing {self.name}: {e}", exc_info=True)
            return 1
    
    def _execute(self, args: argparse.Namespace) -> int:
        """
        Ejecutar lógica de la herramienta (sobrescribir en subclases).
        
        Args:
            args: Argumentos parseados
            
        Returns:
            Código de salida
        """
        raise NotImplementedError


class ProjectCLI:
    """
    CLI principal del proyecto.
    
    Herramienta CLI unificada para el proyecto.
    """
    
    def __init__(self):
        """Inicializar CLI principal."""
        self.parser = argparse.ArgumentParser(
            description="Robot Movement AI - Command Line Interface"
        )
        self.subparsers = self.parser.add_subparsers(dest='command', help='Commands')
        self._setup_commands()
    
    def _setup_commands(self) -> None:
        """Configurar comandos."""
        # Comando: run
        run_parser = self.subparsers.add_parser('run', help='Run the server')
        run_parser.add_argument('--host', default='0.0.0.0', help='Host')
        run_parser.add_argument('--port', type=int, default=8010, help='Port')
        run_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
        run_parser.add_argument('--reload', action='store_true', help='Enable reload')
        
        # Comando: test
        test_parser = self.subparsers.add_parser('test', help='Run tests')
        test_parser.add_argument('--path', help='Test path')
        test_parser.add_argument('--coverage', action='store_true', help='Include coverage')
        test_parser.add_argument('--parallel', action='store_true', help='Run in parallel')
        
        # Comando: export
        export_parser = self.subparsers.add_parser('export', help='Export data')
        export_parser.add_argument('--format', choices=['json', 'yaml', 'csv', 'markdown'], required=True)
        export_parser.add_argument('--output', required=True, help='Output file')
        export_parser.add_argument('--data', help='Data file to export')
        
        # Comando: deploy
        deploy_parser = self.subparsers.add_parser('deploy', help='Deployment utilities')
        deploy_parser.add_argument('--generate-script', action='store_true', help='Generate startup script')
        deploy_parser.add_argument('--generate-dockerfile', action='store_true', help='Generate Dockerfile')
        deploy_parser.add_argument('--generate-compose', action='store_true', help='Generate docker-compose')
        deploy_parser.add_argument('--environment', default='production', help='Environment')
        
        # Comando: summary
        summary_parser = self.subparsers.add_parser('summary', help='Generate project summary')
        summary_parser.add_argument('--format', choices=['json', 'markdown'], default='markdown')
        summary_parser.add_argument('--output', help='Output file')
    
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
            if parsed_args.command == 'run':
                return self._run_server(parsed_args)
            elif parsed_args.command == 'test':
                return self._run_tests(parsed_args)
            elif parsed_args.command == 'export':
                return self._export_data(parsed_args)
            elif parsed_args.command == 'deploy':
                return self._deploy(parsed_args)
            elif parsed_args.command == 'summary':
                return self._generate_summary(parsed_args)
            else:
                logger.error(f"Unknown command: {parsed_args.command}")
                return 1
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            return 1
    
    def _run_server(self, args: argparse.Namespace) -> int:
        """Ejecutar servidor."""
        import uvicorn
        from robot_movement_ai import main
        
        uvicorn.run(
            main.app,
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload
        )
        return 0
    
    def _run_tests(self, args: argparse.Namespace) -> int:
        """Ejecutar tests."""
        from robot_movement_ai.core.test_runner import get_test_runner
        
        runner = get_test_runner()
        result = runner.run_tests(
            test_path=args.path,
            coverage=args.coverage,
            parallel=args.parallel
        )
        
        if result.failed > 0 or result.errors > 0:
            return 1
        return 0
    
    def _export_data(self, args: argparse.Namespace) -> int:
        """Exportar datos."""
        from robot_movement_ai.core.export_utils import get_export_manager
        from robot_movement_ai.core.import_utils import get_import_manager
        
        if args.data:
            # Importar datos primero
            import_manager = get_import_manager()
            data = import_manager.import_file(args.data)
        else:
            # Datos de ejemplo
            data = {"message": "No data file provided"}
        
        export_manager = get_export_manager()
        
        if args.format == 'json':
            export_manager.export_json(data, args.output)
        elif args.format == 'yaml':
            export_manager.export_yaml(data, args.output)
        elif args.format == 'csv':
            if isinstance(data, list):
                export_manager.export_csv(data, args.output)
            else:
                logger.error("CSV export requires list data")
                return 1
        elif args.format == 'markdown':
            export_manager.export_markdown(data, args.output)
        
        return 0
    
    def _deploy(self, args: argparse.Namespace) -> int:
        """Utilidades de deployment."""
        from robot_movement_ai.core.deployment_utils import get_deployment_manager
        
        manager = get_deployment_manager()
        config = manager.create_deployment_config(environment=args.environment)
        
        if args.generate_script:
            manager.generate_startup_script(config)
        
        if args.generate_dockerfile:
            manager.generate_dockerfile()
        
        if args.generate_compose:
            manager.generate_docker_compose()
        
        return 0
    
    def _generate_summary(self, args: argparse.Namespace) -> int:
        """Generar resumen."""
        from robot_movement_ai.core.summary_generator import get_summary_generator
        
        generator = get_summary_generator()
        output_file = args.output or f"project_summary.{args.format}"
        generator.export_summary(output_file, format=args.format)
        
        return 0


def main():
    """Función principal de CLI."""
    import sys
    cli = ProjectCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()






