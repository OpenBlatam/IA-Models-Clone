"""
MOEA Documentation Generator - Generador de documentación
=========================================================
Genera documentación automática del proyecto MOEA
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class MOEADocGenerator:
    """Generador de documentación MOEA"""
    
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.project_info = self._load_project_info()
    
    def _load_project_info(self) -> Optional[Dict]:
        """Cargar información del proyecto"""
        info_file = self.project_dir / "project_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def generate_readme(self, output_file: Optional[str] = None) -> str:
        """Generar README.md"""
        if not output_file:
            output_file = self.project_dir / "README.md"
        else:
            output_file = Path(output_file)
        
        project_name = self.project_info.get('project_name', 'MOEA Project') if self.project_info else 'MOEA Project'
        author = self.project_info.get('author', 'Blatam Academy') if self.project_info else 'Blatam Academy'
        version = self.project_info.get('version', '1.0.0') if self.project_info else '1.0.0'
        
        readme_content = f"""# {project_name}

Sistema de algoritmos evolutivos multi-objetivo (MOEA) generado automáticamente.

## 📋 Información del Proyecto

- **Nombre**: {project_name}
- **Autor**: {author}
- **Versión**: {version}
- **Tipo IA**: {self.project_info.get('ai_type', 'N/A') if self.project_info else 'N/A'}

## 🚀 Inicio Rápido

### Instalación

#### Backend
```bash
cd backend
pip install -r requirements.txt
```

#### Frontend
```bash
cd frontend
npm install
```

### Ejecución

#### Backend
```bash
cd backend
uvicorn app.main:app --reload
```

El backend estará disponible en: `http://localhost:8000`

#### Frontend
```bash
cd frontend
npm run dev
```

El frontend estará disponible en: `http://localhost:5173`

## 📚 Documentación de la API

Una vez que el backend esté corriendo:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎯 Características

### Algoritmos MOEA
- NSGA-II
- NSGA-III
- MOEA/D
- SPEA2

### Métricas
- Hypervolume (HV)
- Inverted Generational Distance (IGD)
- Generational Distance (GD)

## 🧪 Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

## 🐳 Docker

```bash
docker-compose up -d
```

## 📝 Licencia

Generado por Blatam Academy

---

**Generado automáticamente el**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        return str(output_file)
    
    def generate_api_docs(self, output_file: Optional[str] = None) -> str:
        """Generar documentación de API"""
        if not output_file:
            output_file = self.project_dir / "API_DOCS.md"
        else:
            output_file = Path(output_file)
        
        api_docs = f"""# API Documentation

Documentación de la API del proyecto MOEA.

## Endpoints Principales

### Optimización

#### POST /api/v1/moea/optimize
Ejecutar optimización MOEA.

**Request Body:**
```json
{{
    "problem": {{
        "name": "ZDT1",
        "objectives": [...],
        "variables": [...]
    }},
    "algorithm": {{
        "algorithm": "nsga2",
        "population_size": 100,
        "generations": 50
    }}
}}
```

**Response:**
```json
{{
    "project_id": "...",
    "pareto_front": [...],
    "metrics": {{
        "hypervolume": 0.123,
        "igd": 0.001,
        "gd": 0.0005
    }}
}}
```

### Métricas

#### GET /api/v1/moea/metrics/{{project_id}}
Obtener métricas de un proyecto.

### Exportación

#### GET /api/v1/moea/export/{{project_id}}
Exportar resultados.

---

**Generado automáticamente el**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(api_docs)
        
        return str(output_file)
    
    def generate_all_docs(self):
        """Generar toda la documentación"""
        print("📝 Generando documentación...")
        
        readme = self.generate_readme()
        print(f"✅ README.md: {readme}")
        
        api_docs = self.generate_api_docs()
        print(f"✅ API_DOCS.md: {api_docs}")
        
        print("\n✅ Documentación generada exitosamente")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Documentation Generator")
    parser.add_argument(
        'project_dir',
        help='Directorio del proyecto'
    )
    parser.add_argument(
        '--readme',
        help='Generar solo README'
    )
    parser.add_argument(
        '--api',
        help='Generar solo documentación de API'
    )
    
    args = parser.parse_args()
    
    generator = MOEADocGenerator(args.project_dir)
    
    if args.readme:
        output = generator.generate_readme(args.readme)
        print(f"✅ README generado: {output}")
    elif args.api:
        output = generator.generate_api_docs(args.api)
        print(f"✅ API docs generado: {output}")
    else:
        generator.generate_all_docs()


if __name__ == "__main__":
    main()

