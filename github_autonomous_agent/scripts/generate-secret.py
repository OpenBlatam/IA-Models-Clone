#!/usr/bin/env python3
"""
Script para generar secretos seguros (SECRET_KEY, tokens, etc.).
"""

import secrets
import sys
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_header(msg: str):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def generate_secret_key(length: int = 32) -> str:
    """Genera SECRET_KEY seguro."""
    return secrets.token_urlsafe(length)

def generate_token(length: int = 40) -> str:
    """Genera token seguro."""
    return secrets.token_hex(length)

def generate_password(length: int = 16) -> str:
    """Genera contraseña segura."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar secretos seguros')
    parser.add_argument('--type', choices=['key', 'token', 'password', 'all'],
                       default='all', help='Tipo de secreto a generar')
    parser.add_argument('--length', type=int, default=32,
                       help='Longitud del secreto (default: 32)')
    parser.add_argument('--save', action='store_true',
                       help='Guardar en .env (solo SECRET_KEY)')
    
    args = parser.parse_args()
    
    print_header("Generador de Secretos Seguros")
    
    results = {}
    
    if args.type in ['key', 'all']:
        secret_key = generate_secret_key(args.length)
        results['SECRET_KEY'] = secret_key
        print_info("SECRET_KEY:")
        print(f"  {secret_key}\n")
    
    if args.type in ['token', 'all']:
        token = generate_token(args.length)
        results['TOKEN'] = token
        print_info("Token:")
        print(f"  {token}\n")
    
    if args.type in ['password', 'all']:
        password = generate_password(args.length)
        results['PASSWORD'] = password
        print_info("Contraseña:")
        print(f"  {password}\n")
    
    # Guardar en .env si se solicita
    if args.save and 'SECRET_KEY' in results:
        script_dir = Path(__file__).parent.parent
        env_file = script_dir / '.env'
        
        if env_file.exists():
            # Leer .env actual
            content = env_file.read_text()
            
            # Reemplazar o agregar SECRET_KEY
            if 'SECRET_KEY=' in content:
                import re
                content = re.sub(
                    r'SECRET_KEY=.*',
                    f'SECRET_KEY={results["SECRET_KEY"]}',
                    content
                )
            else:
                content += f'\nSECRET_KEY={results["SECRET_KEY"]}\n'
            
            env_file.write_text(content)
            print_success(f"SECRET_KEY guardado en {env_file}")
        else:
            print(f"⚠️  Archivo .env no encontrado. Crea uno primero.")
    
    print_info("💡 Copia estos valores y guárdalos de forma segura!")
    print_info("⚠️  Nunca commitees secretos al repositorio")

if __name__ == "__main__":
    main()




