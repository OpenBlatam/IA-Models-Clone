"""
Project File Generator - Generador de archivos de proyecto
==========================================================

Genera archivos de configuración y documentación para proyectos generados.
"""

from pathlib import Path
from typing import Dict, Any

from .json_utils import json_dumps_pretty
from .shared_utils import get_logger

logger = get_logger(__name__)


def generate_readme(project_dir: Path, project_info: Dict[str, Any]) -> None:
    """Genera README.md principal del proyecto"""
    readme_content = f"""# {project_info['name'].replace('_', ' ').title()}

{project_info['description']}

## 🚀 Inicio Rápido

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port {project_info.get('backend', {}).get('port', 8000)}
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## 📁 Estructura del Proyecto

```
{project_info['name']}/
├── backend/          # Backend API
├── frontend/         # Frontend Application
└── README.md         # Este archivo
```

## 🛠️ Tecnologías

- **Backend**: {project_info.get('backend', {}).get('framework', 'FastAPI')}
- **Frontend**: {project_info.get('frontend', {}).get('framework', 'React')}

## 📝 Características

{chr(10).join(f"- {feature}" for feature in project_info['keywords'].get('features', []))}

## 👤 Autor

{project_info['author']}

## 📄 Licencia

MIT
"""
    (project_dir / "README.md").write_text(readme_content, encoding="utf-8")


def generate_gitignore(project_dir: Path) -> None:
    """Genera .gitignore para el proyecto"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Build
dist/
build/
*.egg-info/

# Logs
*.log
logs/
"""
    (project_dir / ".gitignore").write_text(gitignore_content, encoding="utf-8")


def generate_docker_compose(project_dir: Path, project_info: Dict[str, Any]) -> None:
    """Genera docker-compose.yml para el proyecto"""
    docker_compose = f"""version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "{project_info.get('backend', {}).get('port', 8000)}:8000"
    environment:
      - ENV=development
    volumes:
      - ./backend:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build: ./frontend
    ports:
      - "{project_info.get('frontend', {}).get('port', 3000)}:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:{project_info.get('backend', {}).get('port', 8000)}
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm start
"""
    (project_dir / "docker-compose.yml").write_text(docker_compose, encoding="utf-8")


def generate_project_info_json(project_dir: Path, project_info: Dict[str, Any]) -> None:
    """Genera project_info.json con metadata del proyecto"""
    (project_dir / "project_info.json").write_text(
        json_dumps_pretty(project_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def generate_project_files(project_dir: Path, project_info: Dict[str, Any]) -> None:
    """
    Genera todos los archivos de configuración del proyecto.
    
    Args:
        project_dir: Directorio del proyecto
        project_info: Información del proyecto
    """
    logger.info("Generando archivos de configuración del proyecto...")
    
    generate_readme(project_dir, project_info)
    generate_gitignore(project_dir)
    generate_docker_compose(project_dir, project_info)
    generate_project_info_json(project_dir, project_info)
    
    logger.info("Archivos de configuración generados exitosamente")






