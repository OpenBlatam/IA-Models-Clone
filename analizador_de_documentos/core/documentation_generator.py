"""
Generador Automático de Documentación
======================================

Sistema para generar documentación automática de APIs y análisis.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """Endpoint de API"""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Any]
    tags: List[str]


class DocumentationGenerator:
    """
    Generador de documentación
    
    Genera:
    - Documentación de API
    - Documentación de análisis
    - Guías de uso
    - Ejemplos de código
    """
    
    def __init__(self, output_dir: str = "docs"):
        """Inicializar generador"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info("DocumentationGenerator inicializado")
    
    def generate_api_docs(
        self,
        endpoints: List[APIEndpoint],
        title: str = "API Documentation"
    ) -> str:
        """
        Generar documentación de API en Markdown
        
        Args:
            endpoints: Lista de endpoints
            title: Título de la documentación
        
        Returns:
            Documentación en Markdown
        """
        md = f"# {title}\n\n"
        md += f"*Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        md += "## Índice\n\n"
        
        # Agrupar por tags
        by_tag = {}
        for endpoint in endpoints:
            for tag in endpoint.tags:
                if tag not in by_tag:
                    by_tag[tag] = []
                by_tag[tag].append(endpoint)
        
        # Generar índice
        for tag in sorted(by_tag.keys()):
            md += f"- [{tag}](#{tag.lower().replace(' ', '-')})\n"
        
        md += "\n---\n\n"
        
        # Generar documentación por tag
        for tag in sorted(by_tag.keys()):
            md += f"## {tag}\n\n"
            
            for endpoint in by_tag[tag]:
                md += f"### {endpoint.method.upper()} {endpoint.path}\n\n"
                md += f"**{endpoint.summary}**\n\n"
                md += f"{endpoint.description}\n\n"
                
                if endpoint.parameters:
                    md += "#### Parámetros\n\n"
                    md += "| Nombre | Tipo | Requerido | Descripción |\n"
                    md += "|--------|------|-----------|-------------|\n"
                    for param in endpoint.parameters:
                        md += f"| {param.get('name', '')} | {param.get('type', '')} | {param.get('required', False)} | {param.get('description', '')} |\n"
                    md += "\n"
                
                if endpoint.responses:
                    md += "#### Respuestas\n\n"
                    for status_code, response in endpoint.responses.items():
                        md += f"**{status_code}**: {response.get('description', '')}\n\n"
                        if 'example' in response:
                            md += "```json\n"
                            md += json.dumps(response['example'], indent=2, ensure_ascii=False)
                            md += "\n```\n\n"
            
            md += "\n---\n\n"
        
        return md
    
    def generate_analysis_docs(
        self,
        analysis_results: Dict[str, Any],
        document_id: str
    ) -> str:
        """
        Generar documentación de análisis
        
        Args:
            analysis_results: Resultados de análisis
            document_id: ID del documento
        
        Returns:
            Documentación en Markdown
        """
        md = f"# Análisis de Documento: {document_id}\n\n"
        md += f"*Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        if "summary" in analysis_results:
            md += "## Resumen\n\n"
            md += f"{analysis_results['summary']}\n\n"
        
        if "classification" in analysis_results:
            md += "## Clasificación\n\n"
            for label, score in analysis_results['classification'].items():
                md += f"- **{label}**: {score:.2%}\n"
            md += "\n"
        
        if "keywords" in analysis_results:
            md += "## Palabras Clave\n\n"
            md += ", ".join(analysis_results['keywords'][:20])
            md += "\n\n"
        
        if "sentiment" in analysis_results:
            md += "## Sentimiento\n\n"
            sentiment = analysis_results['sentiment']
            md += f"- **Positivo**: {sentiment.get('positive', 0):.2%}\n"
            md += f"- **Negativo**: {sentiment.get('negative', 0):.2%}\n"
            md += f"- **Neutro**: {sentiment.get('neutral', 0):.2%}\n\n"
        
        return md
    
    def generate_code_examples(
        self,
        endpoint: APIEndpoint,
        language: str = "python"
    ) -> str:
        """
        Generar ejemplos de código
        
        Args:
            endpoint: Endpoint
            language: Lenguaje (python, curl, javascript)
        
        Returns:
            Código de ejemplo
        """
        if language == "python":
            code = "```python\n"
            code += "import requests\n\n"
            code += f"response = requests.{endpoint.method.lower()}(\n"
            code += f'    "http://localhost:8000{endpoint.path}",\n'
            
            if endpoint.parameters:
                code += "    json={\n"
                for param in endpoint.parameters:
                    if param.get('required', False):
                        code += f'        "{param["name"]}": "...",\n'
                code += "    }\n"
            
            code += ")\n"
            code += "print(response.json())\n"
            code += "```\n"
        
        elif language == "curl":
            code = "```bash\n"
            code += f"curl -X {endpoint.method.upper()} \\\n"
            code += f'  "http://localhost:8000{endpoint.path}"'
            
            if endpoint.method.upper() in ["POST", "PUT", "PATCH"]:
                code += " \\\n"
                code += '  -H "Content-Type: application/json" \\\n'
                code += '  -d \'{"...": "..."}\''
            
            code += "\n```\n"
        
        return code
    
    def save_documentation(
        self,
        content: str,
        filename: str
    ):
        """Guardar documentación en archivo"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Documentación guardada en {filepath}")
















