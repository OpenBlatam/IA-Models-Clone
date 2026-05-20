"""
MOEA Wrapper - Wrapper unificado mejorado
=========================================
Wrapper que unifica todas las herramientas MOEA con mejoras adicionales
"""
#!/usr/bin/env python3
"""
MOEA Wrapper - Punto de entrada único mejorado
Ejecuta: python moea_wrapper.py [comando] [opciones]
O: moea [comando] [opciones] (si está instalado)
"""
import sys
import os
from pathlib import Path


def find_moea_scripts():
    """Encontrar todos los scripts MOEA"""
    current_dir = Path(__file__).parent
    scripts = {}
    
    moea_scripts = [
        'quick_moea.py',
        'moea_cli.py',
        'moea_setup.py',
        'moea_test_api.py',
        'moea_benchmark.py',
        'moea_visualize.py',
        'moea_monitor.py',
        'moea_export.py',
        'moea_health.py',
        'moea_utils.py',
        'moea_config.py',
        'verify_moea_project.py'
    ]
    
    for script in moea_scripts:
        script_path = current_dir / script
        if script_path.exists():
            # Extraer nombre base sin extensión
            name = script_path.stem
            scripts[name] = str(script_path)
    
    return scripts


def show_help():
    """Mostrar ayuda"""
    scripts = find_moea_scripts()
    
    print("=" * 70)
    print("MOEA Project - Wrapper Unificado".center(70))
    print("=" * 70)
    print("\n📋 Comandos disponibles:\n")
    
    commands = {
        'quick_moea': 'Generar proyecto MOEA rápidamente',
        'moea_cli': 'CLI unificado (generate, setup, test, etc.)',
        'moea_setup': 'Configurar proyecto generado',
        'moea_test_api': 'Probar API del proyecto',
        'moea_benchmark': 'Hacer benchmark de algoritmos',
        'moea_visualize': 'Visualizar resultados',
        'moea_monitor': 'Monitor en tiempo real',
        'moea_export': 'Exportar proyectos',
        'moea_health': 'Health check del sistema',
        'moea_utils': 'Utilidades varias',
        'moea_config': 'Gestor de configuración',
        'verify_moea_project': 'Verificar estructura'
    }
    
    for cmd, desc in commands.items():
        if cmd in scripts:
            print(f"  {cmd:<25} {desc}")
    
    print("\n" + "=" * 70)
    print("\n💡 Uso:")
    print("   python moea_wrapper.py [comando] [opciones]")
    print("\n📚 Ejemplos:")
    print("   python moea_wrapper.py quick_moea")
    print("   python moea_wrapper.py moea_cli generate")
    print("   python moea_wrapper.py moea_health")
    print("   python moea_wrapper.py moea_monitor --interval 5")
    print("\n" + "=" * 70)


def main():
    """Función principal del wrapper"""
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)
    
    command = sys.argv[1]
    
    # Comandos especiales
    if command in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)
    
    # Buscar script
    scripts = find_moea_scripts()
    
    if command not in scripts:
        print(f"❌ Comando no encontrado: {command}")
        print("\nComandos disponibles:")
        for cmd in scripts.keys():
            print(f"  - {cmd}")
        print("\nUsa 'python moea_wrapper.py help' para ver ayuda completa")
        sys.exit(1)
    
    # Ejecutar script
    script_path = scripts[command]
    script_args = sys.argv[2:]
    
    # Construir comando completo
    import subprocess
    cmd = [sys.executable, script_path] + script_args
    
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operación cancelada por el usuario")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error ejecutando {command}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

