"""
Quick MOEA Project Generator
============================
Script rápido y simple para generar el proyecto MOEA
Versión mejorada con progreso, validaciones y mejor UX
"""
import sys
import asyncio
import time
from pathlib import Path
from typing import Optional

# Add features directory to path
features_dir = Path(__file__).parent.parent
sys.path.insert(0, str(features_dir))

try:
    from ai_project_generator.core.project_generator import ProjectGenerator
except ImportError:
    print("❌ Error: No se puede importar ProjectGenerator")
    print("   Asegúrate de estar en el directorio correcto")
    print("   Instala dependencias: pip install -r requirements.txt")
    sys.exit(1)


class Colors:
    """Colores ANSI para terminal"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Imprimir encabezado"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(text: str):
    """Imprimir mensaje de éxito"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text: str):
    """Imprimir mensaje de error"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Imprimir advertencia"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text: str):
    """Imprimir información"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def print_step(step: int, total: int, text: str):
    """Imprimir paso del proceso"""
    print(f"{Colors.OKBLUE}[{step}/{total}]{Colors.ENDC} {text}")


def check_prerequisites() -> bool:
    """Verificar prerrequisitos"""
    print_step(1, 5, "Verificando prerrequisitos...")
    
    # Verificar Python
    try:
        import sys
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print_error(f"Python 3.8+ requerido. Encontrado: {version.major}.{version.minor}")
            return False
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    except Exception as e:
        print_error(f"Error verificando Python: {e}")
        return False
    
    # Verificar directorio de salida
    output_dir = Path("generated_projects")
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print_success(f"Directorio creado: {output_dir}")
        except Exception as e:
            print_error(f"No se puede crear directorio: {e}")
            return False
    
    return True


async def quick_generate(show_progress: bool = True) -> bool:
    """Generación rápida del proyecto MOEA con mejor feedback"""
    print_header("MOEA Project Generator")
    
    # Verificar prerrequisitos
    if not check_prerequisites():
        return False
    
    print_step(2, 5, "Inicializando generador...")
    try:
        generator = ProjectGenerator(
            base_output_dir="generated_projects",
            backend_framework="fastapi",
            frontend_framework="react"
        )
        print_success("Generador inicializado")
    except Exception as e:
        print_error(f"Error inicializando generador: {e}")
        return False
    
    print_step(3, 5, "Configurando proyecto MOEA...")
    description = (
        "A Multi-Objective Evolutionary Algorithm (MOEA) system for solving optimization "
        "problems with multiple conflicting objectives. The system should support various "
        "MOEA algorithms like NSGA-II, NSGA-III, MOEA/D, and SPEA2. It should include "
        "visualization of Pareto fronts, performance metrics calculation (hypervolume, IGD, GD), "
        "comparison tools, and interactive parameter tuning. The system should handle real-time "
        "optimization, batch processing, and export results in various formats."
    )
    
    print_info("Generando proyecto (esto puede tomar varios minutos)...")
    if show_progress:
        print(f"{Colors.OKCYAN}   Por favor espera mientras se genera el código...{Colors.ENDC}\n")
    
    start_time = time.time()
    
    try:
        print_step(4, 5, "Generando código del proyecto...")
        result = await generator.generate_project(
            description=description,
            project_name="moea_optimization_system",
            author="Blatam Academy",
            version="1.0.0"
        )
        
        elapsed = time.time() - start_time
        
        print_step(5, 5, "Verificando proyecto generado...")
        
        # Verificar que el proyecto se generó correctamente
        project_dir = Path(result.get('project_dir', ''))
        if not project_dir.exists():
            print_error("El directorio del proyecto no existe")
            return False
        
        print_header("✅ PROYECTO GENERADO EXITOSAMENTE")
        
        print(f"\n{Colors.BOLD}📁 Información del Proyecto:{Colors.ENDC}")
        print(f"   Directorio: {Colors.OKCYAN}{project_dir.absolute()}{Colors.ENDC}")
        print(f"   Backend:    {Colors.OKCYAN}{result.get('backend_path', 'N/A')}{Colors.ENDC}")
        print(f"   Frontend:   {Colors.OKCYAN}{result.get('frontend_path', 'N/A')}{Colors.ENDC}")
        print(f"   Tiempo:     {Colors.OKGREEN}{elapsed:.2f} segundos{Colors.ENDC}")
        
        project_info = result.get('project_info', {})
        if project_info:
            print(f"\n{Colors.BOLD}📊 Detalles:{Colors.ENDC}")
            print(f"   Nombre:    {project_info.get('project_name', 'N/A')}")
            print(f"   Autor:     {project_info.get('author', 'N/A')}")
            print(f"   Versión:   {project_info.get('version', 'N/A')}")
            print(f"   Tipo IA:   {project_info.get('ai_type', 'N/A')}")
        
        # Mostrar próximos pasos
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}📋 PRÓXIMOS PASOS:{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")
        
        print(f"{Colors.OKBLUE}1. Configurar proyecto:{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}python moea_setup.py{Colors.ENDC}\n")
        
        print(f"{Colors.OKBLUE}2. Verificar estructura:{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}python verify_moea_project.py{Colors.ENDC}\n")
        
        print(f"{Colors.OKBLUE}3. Instalar dependencias manualmente:{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}cd {project_dir / 'backend'}{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}pip install -r requirements.txt{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}cd ../frontend{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}npm install{Colors.ENDC}\n")
        
        print(f"{Colors.OKBLUE}4. Ejecutar servidores:{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}Backend:  cd backend && uvicorn app.main:app --reload{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}Frontend: cd frontend && npm run dev{Colors.ENDC}\n")
        
        print(f"{Colors.OKBLUE}5. Probar API:{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}python moea_test_api.py{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}\n")
        
        return True
        
    except KeyboardInterrupt:
        print_error("\nGeneración cancelada por el usuario")
        return False
    except Exception as e:
        print_error(f"Error generando proyecto: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            print(f"\n{Colors.FAIL}Traceback completo:{Colors.ENDC}")
            traceback.print_exc()
        else:
            print_info("Usa --verbose para ver detalles completos del error")
        return False


def main():
    """Función principal con manejo de argumentos"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generador rápido de proyecto MOEA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python quick_moea.py              # Generación normal
  python quick_moea.py --verbose    # Con detalles de errores
  python quick_moea.py --quiet      # Sin mensajes de progreso
        """
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar detalles completos de errores"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Modo silencioso (menos output)"
    )
    
    args = parser.parse_args()
    
    # Agregar verbose a sys.argv para que quick_generate lo vea
    if args.verbose:
        sys.argv.append("--verbose")
    
    success = asyncio.run(quick_generate(show_progress=not args.quiet))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

