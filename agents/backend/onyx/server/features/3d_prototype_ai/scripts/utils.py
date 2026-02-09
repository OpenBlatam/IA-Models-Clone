"""
Scripts de utilidades para 3D Prototype AI
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def check_health() -> Dict[str, Any]:
    """Verifica salud del sistema"""
    health = {
        "status": "healthy",
        "checks": {}
    }
    
    # Verificar directorios
    required_dirs = ["storage/prototypes", "storage/backups", "logs"]
    for dir_path in required_dirs:
        path = Path(dir_path)
        health["checks"][f"dir_{dir_path.replace('/', '_')}"] = {
            "status": "ok" if path.exists() else "missing",
            "path": str(path.absolute())
        }
        if not path.exists():
            health["status"] = "degraded"
    
    # Verificar archivos de configuración
    config_files = [".env", "requirements.txt", "main.py"]
    for config_file in config_files:
        path = Path(config_file)
        health["checks"][f"file_{config_file.replace('.', '_')}"] = {
            "status": "ok" if path.exists() else "missing"
        }
        if not path.exists() and config_file == ".env":
            health["status"] = "degraded"
    
    return health


def generate_stats() -> Dict[str, Any]:
    """Genera estadísticas del sistema"""
    stats = {
        "files": 0,
        "directories": 0,
        "python_files": 0,
        "total_lines": 0,
        "modules": []
    }
    
    base_path = Path(".")
    
    for path in base_path.rglob("*"):
        if path.is_file():
            stats["files"] += 1
            
            if path.suffix == ".py":
                stats["python_files"] += 1
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = len(f.readlines())
                        stats["total_lines"] += lines
                        stats["modules"].append({
                            "file": str(path),
                            "lines": lines
                        })
                except Exception:
                    pass
        elif path.is_dir() and not path.name.startswith("."):
            stats["directories"] += 1
    
    return stats


def validate_config() -> Dict[str, Any]:
    """Valida configuración"""
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Verificar variables de entorno requeridas
    required_env = ["HOST", "PORT"]
    for env_var in required_env:
        if not os.getenv(env_var):
            validation["warnings"].append(f"Variable de entorno {env_var} no configurada")
    
    # Verificar archivos requeridos
    required_files = ["requirements.txt", "main.py"]
    for file in required_files:
        if not Path(file).exists():
            validation["valid"] = False
            validation["errors"].append(f"Archivo requerido no encontrado: {file}")
    
    return validation


def main():
    """Función principal"""
    if len(sys.argv) < 2:
        print("Uso: python scripts/utils.py [health|stats|validate]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "health":
        health = check_health()
        print(json.dumps(health, indent=2))
    elif command == "stats":
        stats = generate_stats()
        print(json.dumps(stats, indent=2))
    elif command == "validate":
        validation = validate_config()
        print(json.dumps(validation, indent=2))
        if not validation["valid"]:
            sys.exit(1)
    else:
        print(f"Comando desconocido: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()




