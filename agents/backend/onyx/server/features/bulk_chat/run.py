#!/usr/bin/env python3
"""
Bulk Chat - Command Runner
===========================

Script para ejecutar comandos útiles del sistema.
"""

import sys
import argparse
import subprocess
from pathlib import Path

def run_server(args):
    """Iniciar el servidor."""
    cmd = [
        sys.executable, "-m", "bulk_chat.main",
        "--llm-provider", args.provider,
        "--port", str(args.port),
        "--host", args.host
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    subprocess.run(cmd)

def run_verify():
    """Ejecutar verificación de setup."""
    subprocess.run([sys.executable, "verify_setup.py"])

def run_install():
    """Ejecutar instalación."""
    subprocess.run([sys.executable, "install.py"])

def run_tests():
    """Ejecutar tests."""
    subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short"
    ])

def show_status():
    """Mostrar estado del sistema."""
    import socket
    
    print("=" * 60)
    print("📊 Estado del Sistema - Bulk Chat")
    print("=" * 60)
    print()
    
    # Verificar puerto
    port = 8006
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result == 0:
        print(f"✅ Servidor corriendo en puerto {port}")
        print(f"   URL: http://localhost:{port}")
        print(f"   Docs: http://localhost:{port}/docs")
        print(f"   Dashboard: http://localhost:{port}/dashboard")
    else:
        print(f"⚠️  Servidor no está corriendo en puerto {port}")
    
    print()
    
    # Verificar directorios
    dirs = ['sessions', 'backups', 'logs']
    print("📁 Directorios:")
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            files = list(dir_path.glob('*'))
            print(f"  ✅ {dir_name:15} ({len(files)} archivos)")
        else:
            print(f"  ❌ {dir_name:15} (no existe)")
    
    print()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk Chat - Command Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run.py server              # Iniciar servidor (modo mock)
  python run.py server --provider openai  # Iniciar con OpenAI
  python run.py verify              # Verificar setup
  python run.py install             # Instalar dependencias
  python run.py status              # Ver estado
  python run.py test                # Ejecutar tests
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Iniciar el servidor')
    server_parser.add_argument('--provider', default='mock', choices=['openai', 'anthropic', 'mock'],
                               help='Proveedor de LLM')
    server_parser.add_argument('--port', type=int, default=8006, help='Puerto del servidor')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host del servidor')
    server_parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    # Other commands
    subparsers.add_parser('verify', help='Verificar setup')
    subparsers.add_parser('install', help='Instalar dependencias')
    subparsers.add_parser('test', help='Ejecutar tests')
    subparsers.add_parser('status', help='Mostrar estado del sistema')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Cambiar al directorio del script
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    # Ejecutar comando
    if args.command == 'server':
        run_server(args)
    elif args.command == 'verify':
        run_verify()
    elif args.command == 'install':
        run_install()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'status':
        show_status()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
















