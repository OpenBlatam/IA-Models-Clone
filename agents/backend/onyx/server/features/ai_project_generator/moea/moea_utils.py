"""
MOEA Utilities - Utilidades varias
===================================
Funciones de utilidad para el proyecto MOEA
"""
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def format_time(seconds: float) -> str:
    """Formatear tiempo en formato legible"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(bytes_size: int) -> str:
    """Formatear tamaño en formato legible"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def load_json_file(filepath: str) -> Optional[Dict]:
    """Cargar archivo JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error cargando {filepath}: {e}")
        return None


def save_json_file(data: Dict, filepath: str) -> bool:
    """Guardar archivo JSON"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Error guardando {filepath}: {e}")
        return False


def check_server_available(url: str, timeout: int = 2) -> bool:
    """Verificar si el servidor está disponible"""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def get_project_info(project_dir: Path) -> Optional[Dict]:
    """Obtener información del proyecto"""
    info_file = project_dir / "project_info.json"
    if info_file.exists():
        return load_json_file(str(info_file))
    return None


def list_projects(base_dir: str = "generated_projects") -> List[Dict]:
    """Listar todos los proyectos generados"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    projects = []
    for project_dir in base_path.iterdir():
        if project_dir.is_dir():
            info = get_project_info(project_dir)
            projects.append({
                "name": project_dir.name,
                "path": str(project_dir),
                "info": info or {}
            })
    
    return projects


def find_project(name: str, base_dir: str = "generated_projects") -> Optional[Path]:
    """Encontrar proyecto por nombre"""
    base_path = Path(base_dir)
    project_path = base_path / name
    if project_path.exists() and project_path.is_dir():
        return project_path
    return None


def create_summary_report(projects: List[Dict], output_file: str = "moea_summary.json"):
    """Crear reporte resumen de proyectos"""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_projects": len(projects),
        "projects": []
    }
    
    for project in projects:
        summary["projects"].append({
            "name": project["name"],
            "path": project["path"],
            "author": project.get("info", {}).get("author", "Unknown"),
            "version": project.get("info", {}).get("version", "Unknown"),
            "ai_type": project.get("info", {}).get("ai_type", "Unknown")
        })
    
    save_json_file(summary, output_file)
    return summary


def validate_project_structure(project_dir: Path) -> Dict[str, bool]:
    """Validar estructura básica del proyecto"""
    checks = {
        "backend_exists": (project_dir / "backend").exists(),
        "frontend_exists": (project_dir / "frontend").exists(),
        "backend_main": (project_dir / "backend" / "main.py").exists(),
        "frontend_package": (project_dir / "frontend" / "package.json").exists(),
        "readme": (project_dir / "README.md").exists(),
        "project_info": (project_dir / "project_info.json").exists()
    }
    return checks


class MOEAProjectManager:
    """Gestor de proyectos MOEA"""
    
    def __init__(self, base_dir: str = "generated_projects"):
        self.base_dir = Path(base_dir)
    
    def list_all(self) -> List[Dict]:
        """Listar todos los proyectos"""
        return list_projects(str(self.base_dir))
    
    def get_project(self, name: str) -> Optional[Dict]:
        """Obtener información de un proyecto"""
        project_path = find_project(name, str(self.base_dir))
        if project_path:
            return {
                "name": project_path.name,
                "path": str(project_path),
                "info": get_project_info(project_path),
                "valid": all(validate_project_structure(project_path).values())
            }
        return None
    
    def create_summary(self, output_file: str = "moea_projects_summary.json"):
        """Crear resumen de todos los proyectos"""
        projects = self.list_all()
        return create_summary_report(projects, output_file)


def main():
    """CLI para utilidades"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Comandos')
    
    # Comando list
    list_parser = subparsers.add_parser('list', help='Listar proyectos')
    
    # Comando info
    info_parser = subparsers.add_parser('info', help='Información de proyecto')
    info_parser.add_argument('project_name', help='Nombre del proyecto')
    
    # Comando summary
    summary_parser = subparsers.add_parser('summary', help='Crear resumen')
    summary_parser.add_argument('--output', default='moea_projects_summary.json')
    
    args = parser.parse_args()
    
    manager = MOEAProjectManager()
    
    if args.command == 'list':
        projects = manager.list_all()
        print(f"📁 Proyectos encontrados: {len(projects)}")
        for project in projects:
            print(f"   - {project['name']}")
    
    elif args.command == 'info':
        project = manager.get_project(args.project_name)
        if project:
            print(f"📊 Proyecto: {project['name']}")
            print(f"   Ruta: {project['path']}")
            print(f"   Válido: {'✅' if project['valid'] else '❌'}")
            if project['info']:
                print(f"   Autor: {project['info'].get('author', 'N/A')}")
                print(f"   Versión: {project['info'].get('version', 'N/A')}")
        else:
            print(f"❌ Proyecto no encontrado: {args.project_name}")
    
    elif args.command == 'summary':
        summary = manager.create_summary(args.output)
        print(f"✅ Resumen creado: {args.output}")
        print(f"   Proyectos: {summary['total_projects']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

