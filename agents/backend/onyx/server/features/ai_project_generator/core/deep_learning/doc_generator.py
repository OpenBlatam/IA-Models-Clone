"""
Documentation Generator
=======================

Generador automático de documentación para código generado.
"""

import logging
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Documentation:
    """Documentación generada."""
    file_path: Path
    content: str
    sections: Dict[str, str] = None


class DocumentationGenerator:
    """
    Generador de documentación automática.
    """
    
    def __init__(self):
        """Inicializar generador."""
        pass
    
    def generate_docs_for_file(
        self,
        file_path: Path,
        output_dir: Optional[Path] = None
    ) -> Documentation:
        """
        Generar documentación para un archivo.
        
        Args:
            file_path: Archivo a documentar
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Documentación generada
        """
        if not file_path.exists() or file_path.suffix != '.py':
            return Documentation(file_path, "")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            doc_sections = {
                'overview': self._extract_overview(tree),
                'classes': self._extract_classes(tree),
                'functions': self._extract_functions(tree),
                'usage': self._generate_usage_examples(tree, file_path)
            }
            
            doc_content = self._format_documentation(file_path, doc_sections)
            
            # Escribir archivo de documentación
            if output_dir:
                doc_file = output_dir / f"{file_path.stem}.md"
                doc_file.parent.mkdir(parents=True, exist_ok=True)
                doc_file.write_text(doc_content, encoding='utf-8')
                logger.info(f"Documentación generada: {doc_file}")
            
            return Documentation(
                file_path=file_path,
                content=doc_content,
                sections=doc_sections
            )
        except Exception as e:
            logger.warning(f"Error generando documentación para {file_path}: {e}")
            return Documentation(file_path, "")
    
    def _extract_overview(self, tree: ast.AST) -> str:
        """Extraer overview del módulo."""
        docstring = ast.get_docstring(tree)
        if docstring:
            return docstring
        return "Módulo generado automáticamente."
    
    def _extract_classes(self, tree: ast.AST) -> str:
        """Extraer información de clases."""
        classes_info = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node) or "Sin descripción"
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                
                classes_info.append(f"""
### {node.name}

{docstring}

**Métodos:**
{chr(10).join(f"- `{m}`" for m in methods if not m.startswith('_'))}
""")
        
        return '\n'.join(classes_info)
    
    def _extract_functions(self, tree: ast.AST) -> str:
        """Extraer información de funciones."""
        functions_info = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_private(node.name):
                docstring = ast.get_docstring(node) or "Sin descripción"
                
                # Extraer parámetros
                params = [arg.arg for arg in node.args.args]
                
                functions_info.append(f"""
### {node.name}

{docstring}

**Parámetros:**
{chr(10).join(f"- `{p}`" for p in params)}

**Retorna:**
- Tipo: {self._extract_return_type(node)}
""")
        
        return '\n'.join(functions_info)
    
    def _extract_return_type(self, func_node: ast.FunctionDef) -> str:
        """Extraer tipo de retorno."""
        if func_node.returns:
            if isinstance(func_node.returns, ast.Name):
                return func_node.returns.id
            elif isinstance(func_node.returns, ast.Constant):
                return str(func_node.returns.value)
        return "Any"
    
    def _is_private(self, name: str) -> bool:
        """Verificar si es privado."""
        return name.startswith('_')
    
    def _generate_usage_examples(
        self,
        tree: ast.AST,
        file_path: Path
    ) -> str:
        """Generar ejemplos de uso."""
        examples = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                example = f"""
```python
from {self._get_module_path(file_path)} import {node.name}

# Crear instancia
instance = {node.name}()

# Usar métodos
# result = instance.method()
```
"""
                examples.append(example)
        
        return '\n'.join(examples)
    
    def _get_module_path(self, file_path: Path) -> str:
        """Obtener ruta del módulo."""
        parts = file_path.parts
        if 'app' in parts:
            idx = parts.index('app')
            module_parts = parts[idx+1:-1] + (file_path.stem,)
            return '.'.join(module_parts)
        return file_path.stem
    
    def _format_documentation(
        self,
        file_path: Path,
        sections: Dict[str, str]
    ) -> str:
        """Formatear documentación."""
        content = f"""# {file_path.name}

{sections.get('overview', '')}

## Clases

{sections.get('classes', 'No hay clases')}

## Funciones

{sections.get('functions', 'No hay funciones')}

## Ejemplos de Uso

{sections.get('usage', 'No hay ejemplos disponibles')}
"""
        return content
    
    def generate_docs_for_project(
        self,
        project_dir: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Documentation]:
        """
        Generar documentación para todo el proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Diccionario con documentación por archivo
        """
        if output_dir is None:
            output_dir = project_dir / "docs" / "api"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Generar documentación para archivos en app/
        app_dir = project_dir / "app"
        if app_dir.exists():
            for py_file in app_dir.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                doc = self.generate_docs_for_file(py_file, output_dir)
                if doc.content:
                    results[str(py_file.relative_to(project_dir))] = doc
        
        # Generar índice
        self._generate_index(output_dir, results)
        
        return results
    
    def _generate_index(
        self,
        output_dir: Path,
        docs: Dict[str, Documentation]
    ) -> None:
        """Generar índice de documentación."""
        index_content = """# Documentación de la API

Índice de todos los módulos y funciones.

"""
        for file_path, doc in docs.items():
            index_content += f"- [{file_path}]({Path(file_path).stem}.md)\n"
        
        index_file = output_dir / "index.md"
        index_file.write_text(index_content, encoding='utf-8')


# Instancia global
_global_doc_generator: Optional[DocumentationGenerator] = None


def get_doc_generator() -> DocumentationGenerator:
    """
    Obtener instancia global del generador de documentación.
    
    Returns:
        Instancia del generador
    """
    global _global_doc_generator
    
    if _global_doc_generator is None:
        _global_doc_generator = DocumentationGenerator()
    
    return _global_doc_generator

