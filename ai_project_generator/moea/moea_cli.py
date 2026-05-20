"""
MOEA CLI - Interfaz de línea de comandos completa
==================================================
CLI unificado para todas las operaciones MOEA
"""
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional

# Add features directory to path
features_dir = Path(__file__).parent.parent
sys.path.insert(0, str(features_dir))


class MOEACLI:
    """CLI principal para operaciones MOEA"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="MOEA Project - Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Ejemplos:
  moea_cli.py generate              # Generar proyecto
  moea_cli.py setup                 # Configurar proyecto
  moea_cli.py test                  # Probar API
  moea_cli.py benchmark             # Hacer benchmark
  moea_cli.py visualize PROJECT_ID  # Visualizar resultados
            """
        )
        
        subparsers = self.parser.add_subparsers(
            dest='command',
            help='Comandos disponibles'
        )
        
        # Comando generate
        gen_parser = subparsers.add_parser(
            'generate',
            help='Generar proyecto MOEA'
        )
        gen_parser.add_argument(
            '--name',
            default='moea_optimization_system',
            help='Nombre del proyecto'
        )
        gen_parser.add_argument(
            '--author',
            default='Blatam Academy',
            help='Autor del proyecto'
        )
        gen_parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Modo verbose'
        )
        
        # Comando setup
        setup_parser = subparsers.add_parser(
            'setup',
            help='Configurar proyecto generado'
        )
        setup_parser.add_argument(
            '--project-dir',
            default='generated_projects/moea_optimization_system',
            help='Directorio del proyecto'
        )
        setup_parser.add_argument(
            '--no-backend',
            action='store_true',
            help='Omitir instalación de backend'
        )
        setup_parser.add_argument(
            '--no-frontend',
            action='store_true',
            help='Omitir instalación de frontend'
        )
        
        # Comando verify
        verify_parser = subparsers.add_parser(
            'verify',
            help='Verificar estructura del proyecto'
        )
        verify_parser.add_argument(
            '--project-dir',
            default='generated_projects/moea_optimization_system',
            help='Directorio del proyecto'
        )
        
        # Comando test
        test_parser = subparsers.add_parser(
            'test',
            help='Probar API del proyecto'
        )
        test_parser.add_argument(
            '--url',
            default='http://localhost:8000',
            help='URL base de la API'
        )
        
        # Comando benchmark
        bench_parser = subparsers.add_parser(
            'benchmark',
            help='Hacer benchmark de algoritmos'
        )
        bench_parser.add_argument(
            '--algorithms',
            nargs='+',
            default=['nsga2', 'nsga3', 'moead', 'spea2'],
            help='Algoritmos a probar'
        )
        bench_parser.add_argument(
            '--problems',
            nargs='+',
            default=['ZDT1', 'ZDT2'],
            help='Problemas a probar'
        )
        bench_parser.add_argument(
            '--population',
            type=int,
            default=100,
            help='Tamaño de población'
        )
        bench_parser.add_argument(
            '--generations',
            type=int,
            default=50,
            help='Número de generaciones'
        )
        bench_parser.add_argument(
            '--runs',
            type=int,
            default=3,
            help='Número de ejecuciones'
        )
        bench_parser.add_argument(
            '--url',
            default='http://localhost:8000',
            help='URL base de la API'
        )
        
        # Comando visualize
        vis_parser = subparsers.add_parser(
            'visualize',
            help='Visualizar resultados'
        )
        vis_parser.add_argument(
            'project_id',
            help='ID del proyecto a visualizar'
        )
        vis_parser.add_argument(
            '--output',
            default='moea_visualizations',
            help='Directorio de salida'
        )
        vis_parser.add_argument(
            '--url',
            default='http://localhost:8000',
            help='URL base de la API'
        )
        
        # Comando status
        status_parser = subparsers.add_parser(
            'status',
            help='Ver estado del sistema'
        )
        status_parser.add_argument(
            '--url',
            default='http://localhost:8000',
            help='URL base de la API'
        )
    
    def run(self):
        """Ejecutar comando"""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return 1
        
        try:
            if args.command == 'generate':
                return self._generate(args)
            elif args.command == 'setup':
                return self._setup(args)
            elif args.command == 'verify':
                return self._verify(args)
            elif args.command == 'test':
                return self._test(args)
            elif args.command == 'benchmark':
                return self._benchmark(args)
            elif args.command == 'visualize':
                return self._visualize(args)
            elif args.command == 'status':
                return self._status(args)
            else:
                print(f"❌ Comando desconocido: {args.command}")
                return 1
        except KeyboardInterrupt:
            print("\n\n⚠️  Operación cancelada por el usuario")
            return 130
        except Exception as e:
            print(f"❌ Error: {e}")
            if args.verbose if hasattr(args, 'verbose') else False:
                import traceback
                traceback.print_exc()
            return 1
    
    def _generate(self, args):
        """Generar proyecto"""
        from quick_moea import quick_generate
        if args.verbose:
            sys.argv.append('--verbose')
        success = asyncio.run(quick_generate())
        return 0 if success else 1
    
    def _setup(self, args):
        """Configurar proyecto"""
        from moea_setup import MOEASetup
        setup = MOEASetup(args.project_dir)
        success = setup.run_setup(
            install_backend=not args.no_backend,
            install_frontend=not args.no_frontend
        )
        return 0 if success else 1
    
    def _verify(self, args):
        """Verificar proyecto"""
        from verify_moea_project import verify_project
        success = verify_project()
        return 0 if success else 1
    
    def _test(self, args):
        """Probar API"""
        from moea_test_api import MOEATester
        tester = MOEATester(args.url)
        success = tester.run_all_tests()
        return 0 if success else 1
    
    def _benchmark(self, args):
        """Hacer benchmark"""
        from moea_benchmark import MOEABenchmark
        benchmark = MOEABenchmark(args.url)
        benchmark.compare_algorithms(
            algorithms=args.algorithms,
            problems=args.problems,
            population_size=args.population,
            generations=args.generations,
            runs=args.runs
        )
        return 0
    
    def _visualize(self, args):
        """Visualizar resultados"""
        from moea_visualize import MOEAVisualizer
        visualizer = MOEAVisualizer(args.url)
        visualizer.visualize_project(args.project_id, args.output)
        return 0
    
    def _status(self, args):
        """Ver estado"""
        import requests
        try:
            response = requests.get(f"{args.url}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Servidor API disponible")
                print(f"   URL: {args.url}")
                
                # Intentar obtener más información
                try:
                    stats = requests.get(f"{args.url}/api/v1/stats", timeout=5)
                    if stats.status_code == 200:
                        data = stats.json()
                        print(f"\n📊 Estadísticas:")
                        print(f"   Proyectos procesados: {data.get('processed_count', 'N/A')}")
                        print(f"   En cola: {data.get('queue_size', 'N/A')}")
                except:
                    pass
                
                return 0
            else:
                print(f"⚠️  Servidor responde con código: {response.status_code}")
                return 1
        except:
            print(f"❌ Servidor no disponible en {args.url}")
            print("   Asegúrate de que el backend esté corriendo")
            return 1


def main():
    """Función principal"""
    cli = MOEACLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()

