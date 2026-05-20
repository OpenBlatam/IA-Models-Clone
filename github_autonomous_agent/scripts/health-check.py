#!/usr/bin/env python3
"""
Script para verificar el estado de salud de todos los servicios.
"""

import sys
import asyncio
import httpx
from pathlib import Path
from typing import Dict, List, Tuple

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_header(msg: str):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def check_redis() -> Tuple[bool, str]:
    """Verifica conexión a Redis."""
    try:
        import redis
        from config.settings import settings
        
        client = redis.from_url(settings.REDIS_URL)
        result = client.ping()
        if result:
            return True, "Redis está respondiendo"
        return False, "Redis no responde"
    except ImportError:
        return None, "redis no está instalado"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_database() -> Tuple[bool, str]:
    """Verifica conexión a base de datos."""
    try:
        from sqlalchemy import create_engine, text
        from config.settings import settings
        
        engine = create_engine(settings.DATABASE_URL.replace('+aiosqlite', '').replace('+asyncpg', ''))
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Base de datos conectada"
    except ImportError:
        return None, "sqlalchemy no está instalado"
    except Exception as e:
        return False, f"Error: {str(e)}"

async def check_api() -> Tuple[bool, str]:
    """Verifica que la API esté respondiendo."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8030/health")
            if response.status_code == 200:
                return True, f"API respondiendo (status: {response.status_code})"
            return False, f"API respondió con status: {response.status_code}"
    except httpx.ConnectError:
        return False, "API no está corriendo"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_github_token() -> Tuple[bool, str]:
    """Verifica que el token de GitHub esté configurado."""
    try:
        from config.settings import settings
        
        if not settings.GITHUB_TOKEN or settings.GITHUB_TOKEN == "your_github_token_here":
            return False, "Token de GitHub no configurado"
        
        # Intentar validar token básico
        if len(settings.GITHUB_TOKEN) < 20:
            return False, "Token de GitHub parece inválido (muy corto)"
        
        return True, "Token de GitHub configurado"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_directories() -> Tuple[bool, str]:
    """Verifica que los directorios necesarios existan."""
    try:
        from config.settings import settings
        from pathlib import Path
        
        dirs_to_check = [
            settings.STORAGE_PATH,
            settings.TASKS_STORAGE_PATH,
            settings.LOGS_STORAGE_PATH,
        ]
        
        missing = []
        for dir_path in dirs_to_check:
            if not Path(dir_path).exists():
                missing.append(dir_path)
        
        if missing:
            return False, f"Directorios faltantes: {', '.join(missing)}"
        
        return True, "Todos los directorios existen"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_celery() -> Tuple[bool, str]:
    """Verifica estado de Celery."""
    try:
        import subprocess
        result = subprocess.run(
            ["celery", "-A", "core.worker", "inspect", "ping"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "Celery workers respondiendo"
        return False, "Celery workers no responden"
    except FileNotFoundError:
        return None, "celery no está en PATH"
    except Exception as e:
        return False, f"Error: {str(e)}"

async def main():
    """Función principal."""
    import os
    
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print_header("Health Check - GitHub Autonomous Agent")
    
    checks = []
    
    # Redis
    print_info("Verificando Redis...")
    redis_ok, redis_msg = check_redis()
    checks.append(("Redis", redis_ok, redis_msg))
    if redis_ok:
        print_success(f"Redis: {redis_msg}")
    elif redis_ok is None:
        print_warning(f"Redis: {redis_msg}")
    else:
        print_error(f"Redis: {redis_msg}")
    
    # Database
    print_info("Verificando Base de Datos...")
    db_ok, db_msg = check_database()
    checks.append(("Database", db_ok, db_msg))
    if db_ok:
        print_success(f"Database: {db_msg}")
    elif db_ok is None:
        print_warning(f"Database: {db_msg}")
    else:
        print_error(f"Database: {db_msg}")
    
    # GitHub Token
    print_info("Verificando Token de GitHub...")
    github_ok, github_msg = check_github_token()
    checks.append(("GitHub Token", github_ok, github_msg))
    if github_ok:
        print_success(f"GitHub Token: {github_msg}")
    else:
        print_error(f"GitHub Token: {github_msg}")
    
    # Directories
    print_info("Verificando Directorios...")
    dirs_ok, dirs_msg = check_directories()
    checks.append(("Directories", dirs_ok, dirs_msg))
    if dirs_ok:
        print_success(f"Directories: {dirs_msg}")
    else:
        print_warning(f"Directories: {dirs_msg}")
    
    # API
    print_info("Verificando API...")
    api_ok, api_msg = await check_api()
    checks.append(("API", api_ok, api_msg))
    if api_ok:
        print_success(f"API: {api_msg}")
    else:
        print_warning(f"API: {api_msg}")
    
    # Celery
    print_info("Verificando Celery...")
    celery_ok, celery_msg = check_celery()
    checks.append(("Celery", celery_ok, celery_msg))
    if celery_ok:
        print_success(f"Celery: {celery_msg}")
    elif celery_ok is None:
        print_warning(f"Celery: {celery_msg}")
    else:
        print_warning(f"Celery: {celery_msg}")
    
    # Resumen
    print_header("Resumen de Health Check")
    
    passed = sum(1 for _, ok, _ in checks if ok is True)
    failed = sum(1 for _, ok, _ in checks if ok is False)
    skipped = sum(1 for _, ok, _ in checks if ok is None)
    
    print(f"✅ Pasados: {passed}")
    print(f"❌ Fallidos: {failed}")
    print(f"⚠️  Omitidos: {skipped}")
    
    if failed == 0:
        print_success("\n✅ Todos los servicios están saludables!")
        return 0
    else:
        print_error(f"\n❌ {failed} servicio(s) con problemas")
        print_info("\nRecomendaciones:")
        for name, ok, msg in checks:
            if ok is False:
                print(f"  - {name}: {msg}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)




