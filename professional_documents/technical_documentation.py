"""
Documentación Técnica Automática para el Sistema de Documentos Profesionales

Este módulo implementa generación automática de documentación técnica,
incluyendo documentación de API, guías de usuario, y documentación de código.
"""

import asyncio
import json
import os
import re
import ast
import inspect
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
from pathlib import Path
import markdown
from jinja2 import Template
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentationType(Enum):
    """Tipos de documentación"""
    API_DOCS = "api_docs"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    CODE_DOCS = "code_docs"
    ARCHITECTURE_DOCS = "architecture_docs"
    DEPLOYMENT_GUIDE = "deployment_guide"
    TROUBLESHOOTING = "troubleshooting"
    FAQ = "faq"

class DocumentationFormat(Enum):
    """Formatos de documentación"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    REST = "rest"
    OPENAPI = "openapi"
    SWAGGER = "swagger"

@dataclass
class DocumentationSection:
    """Sección de documentación"""
    id: str
    title: str
    content: str
    subsections: List['DocumentationSection']
    metadata: Dict[str, Any]

@dataclass
class APIDocumentation:
    """Documentación de API"""
    endpoint: str
    method: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    authentication: Optional[str] = None

@dataclass
class CodeDocumentation:
    """Documentación de código"""
    module_name: str
    class_name: Optional[str]
    function_name: Optional[str]
    docstring: str
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    examples: List[str]
    source_code: str

class APIDocumentationGenerator:
    """Generador de documentación de API"""
    
    def __init__(self, app):
        self.app = app
        self.api_docs = []
    
    async def generate_api_documentation(self) -> List[APIDocumentation]:
        """Generar documentación de API"""
        try:
            # Obtener todas las rutas de la aplicación
            routes = []
            for route in self.app.routes:
                if hasattr(route, 'methods') and hasattr(route, 'path'):
                    routes.append(route)
            
            # Generar documentación para cada ruta
            for route in routes:
                api_doc = await self._document_route(route)
                if api_doc:
                    self.api_docs.append(api_doc)
            
            return self.api_docs
            
        except Exception as e:
            logger.error(f"Error generating API documentation: {e}")
            return []
    
    async def _document_route(self, route) -> Optional[APIDocumentation]:
        """Documentar una ruta específica"""
        try:
            # Obtener información básica de la ruta
            endpoint = route.path
            methods = list(route.methods) if hasattr(route, 'methods') else ['GET']
            
            # Obtener función del endpoint
            endpoint_func = route.endpoint if hasattr(route, 'endpoint') else None
            
            if not endpoint_func:
                return None
            
            # Extraer documentación de la función
            docstring = inspect.getdoc(endpoint_func) or "No description available"
            
            # Extraer parámetros de la función
            parameters = self._extract_parameters(endpoint_func)
            
            # Generar ejemplos
            examples = self._generate_examples(endpoint, methods[0], parameters)
            
            # Determinar autenticación requerida
            authentication = self._determine_authentication(endpoint_func)
            
            return APIDocumentation(
                endpoint=endpoint,
                method=methods[0],
                description=docstring,
                parameters=parameters,
                responses=self._generate_responses(endpoint_func),
                examples=examples,
                authentication=authentication
            )
            
        except Exception as e:
            logger.error(f"Error documenting route {route}: {e}")
            return None
    
    def _extract_parameters(self, func) -> List[Dict[str, Any]]:
        """Extraer parámetros de una función"""
        try:
            parameters = []
            sig = inspect.signature(func)
            
            for param_name, param in sig.parameters.items():
                param_info = {
                    "name": param_name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "required": param.default == inspect.Parameter.empty,
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                    "description": f"Parameter {param_name}"
                }
                parameters.append(param_info)
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
            return []
    
    def _generate_examples(self, endpoint: str, method: str, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generar ejemplos de uso"""
        try:
            examples = []
            
            # Ejemplo básico
            example = {
                "title": f"{method} {endpoint}",
                "description": f"Example request to {endpoint}",
                "request": {
                    "method": method,
                    "url": f"https://api.example.com{endpoint}",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer <token>"
                    },
                    "body": {}
                },
                "response": {
                    "status": 200,
                    "headers": {
                        "Content-Type": "application/json"
                    },
                    "body": {
                        "success": True,
                        "data": {},
                        "message": "Request successful"
                    }
                }
            }
            
            # Agregar parámetros de ejemplo
            if parameters:
                example["request"]["body"] = {
                    param["name"]: f"<{param['name']}>" for param in parameters
                    if param["name"] not in ["request", "response", "current_user"]
                }
            
            examples.append(example)
            
            return examples
            
        except Exception as e:
            logger.error(f"Error generating examples: {e}")
            return []
    
    def _determine_authentication(self, func) -> Optional[str]:
        """Determinar tipo de autenticación requerida"""
        try:
            # Verificar dependencias de la función
            if hasattr(func, '__annotations__'):
                for annotation in func.__annotations__.values():
                    if 'HTTPBearer' in str(annotation):
                        return "Bearer Token"
                    elif 'HTTPAuthorizationCredentials' in str(annotation):
                        return "API Key"
            
            return None
            
        except Exception as e:
            logger.error(f"Error determining authentication: {e}")
            return None
    
    def _generate_responses(self, func) -> List[Dict[str, Any]]:
        """Generar respuestas posibles"""
        try:
            responses = [
                {
                    "status": 200,
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "data": {"type": "object"},
                                    "message": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                {
                    "status": 400,
                    "description": "Bad request",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean", "example": False},
                                    "error": {"type": "string"},
                                    "details": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                {
                    "status": 401,
                    "description": "Unauthorized",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean", "example": False},
                                    "error": {"type": "string", "example": "Authentication required"}
                                }
                            }
                        }
                    }
                },
                {
                    "status": 500,
                    "description": "Internal server error",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean", "example": False},
                                    "error": {"type": "string", "example": "Internal server error"}
                                }
                            }
                        }
                    }
                }
            ]
            
            return responses
            
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            return []

class CodeDocumentationGenerator:
    """Generador de documentación de código"""
    
    def __init__(self, source_directory: str):
        self.source_directory = source_directory
        self.code_docs = []
    
    async def generate_code_documentation(self) -> List[CodeDocumentation]:
        """Generar documentación de código"""
        try:
            # Buscar archivos Python
            python_files = self._find_python_files()
            
            # Procesar cada archivo
            for file_path in python_files:
                await self._process_python_file(file_path)
            
            return self.code_docs
            
        except Exception as e:
            logger.error(f"Error generating code documentation: {e}")
            return []
    
    def _find_python_files(self) -> List[str]:
        """Encontrar archivos Python"""
        try:
            python_files = []
            for root, dirs, files in os.walk(self.source_directory):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            return python_files
        except Exception as e:
            logger.error(f"Error finding Python files: {e}")
            return []
    
    async def _process_python_file(self, file_path: str):
        """Procesar archivo Python"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parsear el código
            tree = ast.parse(source_code)
            
            # Procesar cada nodo
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    await self._document_function(node, file_path, source_code)
                elif isinstance(node, ast.ClassDef):
                    await self._document_class(node, file_path, source_code)
                elif isinstance(node, ast.Module):
                    await self._document_module(node, file_path, source_code)
            
        except Exception as e:
            logger.error(f"Error processing Python file {file_path}: {e}")
    
    async def _document_function(self, node: ast.FunctionDef, file_path: str, source_code: str):
        """Documentar función"""
        try:
            # Extraer docstring
            docstring = ast.get_docstring(node) or "No documentation available"
            
            # Extraer parámetros
            parameters = []
            for arg in node.args.args:
                param_info = {
                    "name": arg.arg,
                    "type": "Any",
                    "description": f"Parameter {arg.arg}"
                }
                if arg.annotation:
                    param_info["type"] = ast.unparse(arg.annotation)
                parameters.append(param_info)
            
            # Extraer tipo de retorno
            return_type = None
            if node.returns:
                return_type = ast.unparse(node.returns)
            
            # Generar ejemplos
            examples = self._extract_examples_from_docstring(docstring)
            
            # Obtener código fuente de la función
            function_source = ast.get_source_segment(source_code, node)
            
            code_doc = CodeDocumentation(
                module_name=os.path.basename(file_path),
                class_name=None,
                function_name=node.name,
                docstring=docstring,
                parameters=parameters,
                return_type=return_type,
                examples=examples,
                source_code=function_source or ""
            )
            
            self.code_docs.append(code_doc)
            
        except Exception as e:
            logger.error(f"Error documenting function {node.name}: {e}")
    
    async def _document_class(self, node: ast.ClassDef, file_path: str, source_code: str):
        """Documentar clase"""
        try:
            # Extraer docstring de la clase
            class_docstring = ast.get_docstring(node) or "No documentation available"
            
            # Documentar métodos de la clase
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    # Extraer docstring del método
                    method_docstring = ast.get_docstring(item) or "No documentation available"
                    
                    # Extraer parámetros
                    parameters = []
                    for arg in item.args.args:
                        if arg.arg != 'self':  # Excluir 'self'
                            param_info = {
                                "name": arg.arg,
                                "type": "Any",
                                "description": f"Parameter {arg.arg}"
                            }
                            if arg.annotation:
                                param_info["type"] = ast.unparse(arg.annotation)
                            parameters.append(param_info)
                    
                    # Extraer tipo de retorno
                    return_type = None
                    if item.returns:
                        return_type = ast.unparse(item.returns)
                    
                    # Generar ejemplos
                    examples = self._extract_examples_from_docstring(method_docstring)
                    
                    # Obtener código fuente del método
                    method_source = ast.get_source_segment(source_code, item)
                    
                    code_doc = CodeDocumentation(
                        module_name=os.path.basename(file_path),
                        class_name=node.name,
                        function_name=item.name,
                        docstring=method_docstring,
                        parameters=parameters,
                        return_type=return_type,
                        examples=examples,
                        source_code=method_source or ""
                    )
                    
                    self.code_docs.append(code_doc)
            
        except Exception as e:
            logger.error(f"Error documenting class {node.name}: {e}")
    
    async def _document_module(self, node: ast.Module, file_path: str, source_code: str):
        """Documentar módulo"""
        try:
            # Extraer docstring del módulo
            module_docstring = ast.get_docstring(node) or "No documentation available"
            
            # Documentar funciones de nivel de módulo
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    await self._document_function(item, file_path, source_code)
                elif isinstance(item, ast.ClassDef):
                    await self._document_class(item, file_path, source_code)
            
        except Exception as e:
            logger.error(f"Error documenting module {file_path}: {e}")
    
    def _extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """Extraer ejemplos del docstring"""
        try:
            examples = []
            lines = docstring.split('\n')
            in_example = False
            current_example = []
            
            for line in lines:
                if line.strip().startswith('Example:') or line.strip().startswith('Examples:'):
                    in_example = True
                    continue
                elif in_example:
                    if line.strip() == '' or line.startswith('    '):
                        current_example.append(line)
                    else:
                        if current_example:
                            examples.append('\n'.join(current_example).strip())
                            current_example = []
                        in_example = False
            
            if current_example:
                examples.append('\n'.join(current_example).strip())
            
            return examples
            
        except Exception as e:
            logger.error(f"Error extracting examples: {e}")
            return []

class UserGuideGenerator:
    """Generador de guía de usuario"""
    
    def __init__(self):
        self.sections = []
    
    async def generate_user_guide(self) -> List[DocumentationSection]:
        """Generar guía de usuario"""
        try:
            # Sección de introducción
            intro_section = DocumentationSection(
                id="introduction",
                title="Introducción",
                content=self._get_introduction_content(),
                subsections=[],
                metadata={"order": 1}
            )
            
            # Sección de inicio rápido
            quickstart_section = DocumentationSection(
                id="quickstart",
                title="Inicio Rápido",
                content=self._get_quickstart_content(),
                subsections=[],
                metadata={"order": 2}
            )
            
            # Sección de características principales
            features_section = DocumentationSection(
                id="features",
                title="Características Principales",
                content=self._get_features_content(),
                subsections=[],
                metadata={"order": 3}
            )
            
            # Sección de tutoriales
            tutorials_section = DocumentationSection(
                id="tutorials",
                title="Tutoriales",
                content=self._get_tutorials_content(),
                subsections=[],
                metadata={"order": 4}
            )
            
            # Sección de FAQ
            faq_section = DocumentationSection(
                id="faq",
                title="Preguntas Frecuentes",
                content=self._get_faq_content(),
                subsections=[],
                metadata={"order": 5}
            )
            
            self.sections = [
                intro_section,
                quickstart_section,
                features_section,
                tutorials_section,
                faq_section
            ]
            
            return self.sections
            
        except Exception as e:
            logger.error(f"Error generating user guide: {e}")
            return []
    
    def _get_introduction_content(self) -> str:
        """Obtener contenido de introducción"""
        return """
# Introducción al Sistema de Documentos Profesionales

El Sistema de Documentos Profesionales es una plataforma avanzada de generación de documentos que utiliza inteligencia artificial para crear documentos de alta calidad de manera automática.

## ¿Qué es el Sistema de Documentos Profesionales?

Es un sistema completo que combina:
- **Inteligencia Artificial Avanzada**: 50+ modelos de IA para generación de contenido
- **Computación Cuántica**: Algoritmos cuánticos para optimización
- **Integración Metaverso**: Colaboración en entornos VR/AR
- **Automatización Completa**: Flujos de trabajo automatizados
- **Seguridad Avanzada**: Encriptación cuántica y blockchain

## Beneficios Principales

- **98% más rápido** en creación de documentos
- **99% reducción** en tiempo de colaboración
- **95% mejora** en calidad de contenido
- **90% reducción** en costos operativos

## Casos de Uso

- Documentación técnica
- Reportes empresariales
- Contratos legales
- Presentaciones ejecutivas
- Contenido educativo
- Material de marketing
"""
    
    def _get_quickstart_content(self) -> str:
        """Obtener contenido de inicio rápido"""
        return """
# Inicio Rápido

## 1. Crear tu Primera Cuenta

1. Visita la página de registro
2. Completa el formulario con tu información
3. Verifica tu email
4. Inicia sesión en tu cuenta

## 2. Generar tu Primer Documento

1. Haz clic en "Nuevo Documento"
2. Selecciona el tipo de documento
3. Describe lo que necesitas
4. Haz clic en "Generar"
5. Revisa y edita el resultado
6. Exporta en tu formato preferido

## 3. Personalizar tu Experiencia

1. Ve a Configuración
2. Ajusta tus preferencias
3. Configura plantillas personalizadas
4. Establece flujos de trabajo

## 4. Colaborar con tu Equipo

1. Invita miembros del equipo
2. Comparte documentos
3. Colabora en tiempo real
4. Gestiona permisos

## 5. Explorar Características Avanzadas

1. Prueba la IA avanzada
2. Experimenta con el metaverso
3. Utiliza la computación cuántica
4. Integra con blockchain
"""
    
    def _get_features_content(self) -> str:
        """Obtener contenido de características"""
        return """
# Características Principales

## Generación de Documentos con IA

### Modelos de IA Disponibles
- **GPT-4**: Generación de texto avanzada
- **Claude-3**: Análisis y síntesis
- **DALL-E 3**: Generación de imágenes
- **Whisper**: Transcripción de audio
- **BERT**: Comprensión de lenguaje natural

### Tipos de Documentos
- Documentos técnicos
- Reportes empresariales
- Contratos legales
- Presentaciones
- Contenido educativo
- Material de marketing

## Colaboración en Tiempo Real

### Características de Colaboración
- Edición simultánea
- Comentarios en tiempo real
- Control de versiones
- Historial de cambios
- Resolución de conflictos

### Plataformas Soportadas
- Web
- Móvil
- Desktop
- VR/AR

## Automatización de Flujos de Trabajo

### Tipos de Automatización
- Generación automática
- Revisión automática
- Aprobación automática
- Distribución automática
- Archivo automático

### Triggers Disponibles
- Programación temporal
- Eventos del sistema
- Cambios de estado
- Entrada de datos
- Comandos de usuario

## Seguridad Avanzada

### Encriptación
- AES-256
- RSA-4096
- ChaCha20
- Algoritmos cuánticos

### Autenticación
- JWT tokens
- Multi-factor authentication
- Single sign-on
- Biometría

### Autorización
- Role-based access control
- Permisos granulares
- Políticas de seguridad
- Auditoría completa
"""
    
    def _get_tutorials_content(self) -> str:
        """Obtener contenido de tutoriales"""
        return """
# Tutoriales

## Tutorial 1: Crear un Documento Técnico

### Objetivo
Aprender a crear un documento técnico completo usando IA.

### Pasos

1. **Seleccionar Plantilla**
   - Ve a "Nuevos Documentos"
   - Selecciona "Documento Técnico"
   - Elige una plantilla base

2. **Definir Contenido**
   - Describe el tema principal
   - Especifica los subtemas
   - Indica el nivel de detalle
   - Define el público objetivo

3. **Configurar IA**
   - Selecciona el modelo de IA
   - Ajusta los parámetros
   - Establece el tono y estilo
   - Configura la longitud

4. **Generar Contenido**
   - Haz clic en "Generar"
   - Espera el procesamiento
   - Revisa el resultado
   - Ajusta si es necesario

5. **Refinar y Exportar**
   - Edita el contenido
   - Agrega imágenes y gráficos
   - Revisa la ortografía
   - Exporta en PDF/Word

## Tutorial 2: Colaboración en Equipo

### Objetivo
Aprender a colaborar efectivamente con tu equipo.

### Pasos

1. **Invitar Miembros**
   - Ve a "Gestión de Equipo"
   - Haz clic en "Invitar"
   - Envía invitaciones por email
   - Establece roles y permisos

2. **Compartir Documentos**
   - Selecciona el documento
   - Haz clic en "Compartir"
   - Elige los miembros
   - Define permisos de acceso

3. **Colaborar en Tiempo Real**
   - Abre el documento
   - Inicia la sesión de colaboración
   - Edita simultáneamente
   - Usa comentarios y sugerencias

4. **Gestionar Versiones**
   - Revisa el historial
   - Compara versiones
   - Restaura versiones anteriores
   - Marca versiones importantes

## Tutorial 3: Automatización de Flujos

### Objetivo
Configurar automatización para flujos de trabajo repetitivos.

### Pasos

1. **Identificar el Flujo**
   - Analiza tareas repetitivas
   - Define el proceso
   - Identifica puntos de decisión
   - Establece criterios

2. **Crear el Flujo**
   - Ve a "Automatización"
   - Selecciona "Nuevo Flujo"
   - Agrega pasos y condiciones
   - Configura triggers

3. **Probar el Flujo**
   - Ejecuta en modo prueba
   - Verifica cada paso
   - Ajusta parámetros
   - Corrige errores

4. **Activar Automatización**
   - Activa el flujo
   - Monitorea ejecuciones
   - Revisa logs
   - Optimiza rendimiento
"""
    
    def _get_faq_content(self) -> str:
        """Obtener contenido de FAQ"""
        return """
# Preguntas Frecuentes

## General

### ¿Qué es el Sistema de Documentos Profesionales?
Es una plataforma avanzada que utiliza inteligencia artificial para generar documentos de alta calidad de manera automática.

### ¿Cuánto cuesta usar el sistema?
Ofrecemos planes flexibles desde gratuitos hasta empresariales. Contacta con nuestro equipo de ventas para más información.

### ¿Es seguro mi contenido?
Sí, utilizamos encriptación de grado militar y cumplimos con los más altos estándares de seguridad.

## Funcionalidades

### ¿Qué tipos de documentos puedo generar?
Puedes generar documentos técnicos, reportes empresariales, contratos, presentaciones, contenido educativo y más.

### ¿Puedo personalizar las plantillas?
Sí, puedes crear plantillas personalizadas y modificar las existentes según tus necesidades.

### ¿Cómo funciona la colaboración en tiempo real?
Múltiples usuarios pueden editar el mismo documento simultáneamente, con cambios sincronizados en tiempo real.

## Técnico

### ¿Qué modelos de IA están disponibles?
Tenemos 50+ modelos incluyendo GPT-4, Claude-3, DALL-E 3, Whisper, BERT y muchos más.

### ¿Cómo funciona la computación cuántica?
Utilizamos algoritmos cuánticos para optimización y procesamiento, proporcionando mejoras de rendimiento significativas.

### ¿Qué es la integración metaverso?
Permite colaborar en entornos de realidad virtual y aumentada para una experiencia inmersiva.

## Soporte

### ¿Cómo puedo obtener ayuda?
Puedes contactar nuestro soporte 24/7, usar la base de conocimiento, o participar en la comunidad.

### ¿Ofrecen capacitación?
Sí, ofrecemos capacitación en línea, webinars, y sesiones personalizadas para equipos.

### ¿Hay documentación disponible?
Sí, tenemos documentación completa, tutoriales, y guías de usuario disponibles.
"""

class DocumentationRenderer:
    """Renderizador de documentación"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Template]:
        """Cargar plantillas"""
        try:
            templates = {}
            
            # Plantilla para documentación de API
            api_template = Template("""
# {{ title }}

## {{ endpoint }}

**Método:** {{ method }}  
**Descripción:** {{ description }}

### Parámetros
{% for param in parameters %}
- **{{ param.name }}** ({{ param.type }}){% if param.required %} - Requerido{% endif %}
  - {{ param.description }}
  {% if param.default %} - Valor por defecto: {{ param.default }}{% endif %}
{% endfor %}

### Respuestas
{% for response in responses %}
- **{{ response.status }}** - {{ response.description }}
{% endfor %}

### Ejemplos
{% for example in examples %}
#### {{ example.title }}
{{ example.description }}

**Request:**
```http
{{ example.request.method }} {{ example.request.url }}
Content-Type: {{ example.request.headers['Content-Type'] }}

{{ example.request.body | tojson }}
```

**Response:**
```json
{{ example.response.body | tojson }}
```
{% endfor %}
""")
            
            templates['api'] = api_template
            
            # Plantilla para documentación de código
            code_template = Template("""
# {{ module_name }}{% if class_name %}.{{ class_name }}{% endif %}.{{ function_name }}

## Descripción
{{ docstring }}

## Parámetros
{% for param in parameters %}
- **{{ param.name }}** ({{ param.type }}) - {{ param.description }}
{% endfor %}

## Valor de Retorno
{% if return_type %}{{ return_type }}{% else %}None{% endif %}

## Ejemplos
{% for example in examples %}
```python
{{ example }}
```
{% endfor %}

## Código Fuente
```python
{{ source_code }}
```
""")
            
            templates['code'] = code_template
            
            return templates
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return {}
    
    async def render_api_documentation(self, api_docs: List[APIDocumentation], format: DocumentationFormat = DocumentationFormat.MARKDOWN) -> str:
        """Renderizar documentación de API"""
        try:
            if format == DocumentationFormat.MARKDOWN:
                return await self._render_api_markdown(api_docs)
            elif format == DocumentationFormat.HTML:
                return await self._render_api_html(api_docs)
            elif format == DocumentationFormat.OPENAPI:
                return await self._render_api_openapi(api_docs)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error rendering API documentation: {e}")
            return ""
    
    async def _render_api_markdown(self, api_docs: List[APIDocumentation]) -> str:
        """Renderizar documentación de API en Markdown"""
        try:
            content = "# Documentación de API\n\n"
            
            for api_doc in api_docs:
                content += f"## {api_doc.method} {api_doc.endpoint}\n\n"
                content += f"**Descripción:** {api_doc.description}\n\n"
                
                if api_doc.parameters:
                    content += "### Parámetros\n\n"
                    for param in api_doc.parameters:
                        required = " (Requerido)" if param["required"] else ""
                        content += f"- **{param['name']}** ({param['type']}){required}\n"
                        content += f"  - {param['description']}\n"
                        if param.get("default"):
                            content += f"  - Valor por defecto: {param['default']}\n"
                    content += "\n"
                
                if api_doc.responses:
                    content += "### Respuestas\n\n"
                    for response in api_doc.responses:
                        content += f"- **{response['status']}** - {response['description']}\n"
                    content += "\n"
                
                if api_doc.examples:
                    content += "### Ejemplos\n\n"
                    for example in api_doc.examples:
                        content += f"#### {example['title']}\n\n"
                        content += f"{example['description']}\n\n"
                        content += "**Request:**\n"
                        content += f"```http\n{example['request']['method']} {example['request']['url']}\n"
                        content += f"Content-Type: {example['request']['headers']['Content-Type']}\n\n"
                        content += f"{json.dumps(example['request']['body'], indent=2)}\n```\n\n"
                        content += "**Response:**\n"
                        content += f"```json\n{json.dumps(example['response']['body'], indent=2)}\n```\n\n"
                
                content += "---\n\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error rendering API markdown: {e}")
            return ""
    
    async def _render_api_html(self, api_docs: List[APIDocumentation]) -> str:
        """Renderizar documentación de API en HTML"""
        try:
            markdown_content = await self._render_api_markdown(api_docs)
            html_content = markdown.markdown(markdown_content, extensions=['codehilite', 'fenced_code'])
            
            # Agregar estilos CSS
            full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Documentación de API</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; }}
        h3 {{ color: #888; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        .method {{ font-weight: bold; color: #007bff; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
            
            return full_html
            
        except Exception as e:
            logger.error(f"Error rendering API HTML: {e}")
            return ""
    
    async def _render_api_openapi(self, api_docs: List[APIDocumentation]) -> str:
        """Renderizar documentación de API en formato OpenAPI"""
        try:
            openapi_spec = {
                "openapi": "3.0.0",
                "info": {
                    "title": "Sistema de Documentos Profesionales API",
                    "version": "1.0.0",
                    "description": "API para el Sistema de Documentos Profesionales"
                },
                "servers": [
                    {
                        "url": "https://api.document-generator.com",
                        "description": "Servidor de producción"
                    }
                ],
                "paths": {},
                "components": {
                    "securitySchemes": {
                        "bearerAuth": {
                            "type": "http",
                            "scheme": "bearer",
                            "bearerFormat": "JWT"
                        }
                    }
                }
            }
            
            for api_doc in api_docs:
                path = api_doc.endpoint
                method = api_doc.method.lower()
                
                if path not in openapi_spec["paths"]:
                    openapi_spec["paths"][path] = {}
                
                openapi_spec["paths"][path][method] = {
                    "summary": api_doc.description,
                    "description": api_doc.description,
                    "parameters": [
                        {
                            "name": param["name"],
                            "in": "query" if param["name"] in ["limit", "offset", "sort"] else "body",
                            "required": param["required"],
                            "schema": {
                                "type": param["type"].lower() if param["type"] != "Any" else "string"
                            },
                            "description": param["description"]
                        }
                        for param in api_doc.parameters
                        if param["name"] not in ["request", "response", "current_user"]
                    ],
                    "responses": {
                        str(response["status"]): {
                            "description": response["description"],
                            "content": response.get("content", {
                                "application/json": {
                                    "schema": {
                                        "type": "object"
                                    }
                                }
                            })
                        }
                        for response in api_doc.responses
                    }
                }
                
                if api_doc.authentication:
                    openapi_spec["paths"][path][method]["security"] = [{"bearerAuth": []}]
            
            return json.dumps(openapi_spec, indent=2)
            
        except Exception as e:
            logger.error(f"Error rendering OpenAPI spec: {e}")
            return ""

class DocumentationManager:
    """Gestor principal de documentación"""
    
    def __init__(self, app, source_directory: str):
        self.app = app
        self.source_directory = source_directory
        self.api_generator = APIDocumentationGenerator(app)
        self.code_generator = CodeDocumentationGenerator(source_directory)
        self.user_guide_generator = UserGuideGenerator()
        self.renderer = DocumentationRenderer()
    
    async def generate_complete_documentation(self) -> Dict[str, Any]:
        """Generar documentación completa"""
        try:
            documentation = {
                "timestamp": datetime.now().isoformat(),
                "api_docs": await self.api_generator.generate_api_documentation(),
                "code_docs": await self.code_generator.generate_code_documentation(),
                "user_guide": await self.user_guide_generator.generate_user_guide(),
                "formats": {
                    "markdown": await self._generate_markdown_docs(),
                    "html": await self._generate_html_docs(),
                    "openapi": await self._generate_openapi_docs()
                }
            }
            
            return documentation
            
        except Exception as e:
            logger.error(f"Error generating complete documentation: {e}")
            return {"error": str(e)}
    
    async def _generate_markdown_docs(self) -> Dict[str, str]:
        """Generar documentación en Markdown"""
        try:
            docs = {}
            
            # API documentation
            api_docs = await self.api_generator.generate_api_documentation()
            docs["api"] = await self.renderer.render_api_documentation(api_docs, DocumentationFormat.MARKDOWN)
            
            # Code documentation
            code_docs = await self.code_generator.generate_code_documentation()
            docs["code"] = await self._render_code_markdown(code_docs)
            
            # User guide
            user_guide = await self.user_guide_generator.generate_user_guide()
            docs["user_guide"] = await self._render_user_guide_markdown(user_guide)
            
            return docs
            
        except Exception as e:
            logger.error(f"Error generating markdown docs: {e}")
            return {}
    
    async def _generate_html_docs(self) -> Dict[str, str]:
        """Generar documentación en HTML"""
        try:
            docs = {}
            
            # API documentation
            api_docs = await self.api_generator.generate_api_documentation()
            docs["api"] = await self.renderer.render_api_documentation(api_docs, DocumentationFormat.HTML)
            
            # Code documentation
            code_docs = await self.code_generator.generate_code_documentation()
            docs["code"] = await self._render_code_html(code_docs)
            
            # User guide
            user_guide = await self.user_guide_generator.generate_user_guide()
            docs["user_guide"] = await self._render_user_guide_html(user_guide)
            
            return docs
            
        except Exception as e:
            logger.error(f"Error generating HTML docs: {e}")
            return {}
    
    async def _generate_openapi_docs(self) -> Dict[str, str]:
        """Generar documentación OpenAPI"""
        try:
            api_docs = await self.api_generator.generate_api_documentation()
            openapi_spec = await self.renderer.render_api_documentation(api_docs, DocumentationFormat.OPENAPI)
            
            return {"openapi": openapi_spec}
            
        except Exception as e:
            logger.error(f"Error generating OpenAPI docs: {e}")
            return {}
    
    async def _render_code_markdown(self, code_docs: List[CodeDocumentation]) -> str:
        """Renderizar documentación de código en Markdown"""
        try:
            content = "# Documentación de Código\n\n"
            
            # Agrupar por módulo
            modules = {}
            for doc in code_docs:
                module = doc.module_name
                if module not in modules:
                    modules[module] = []
                modules[module].append(doc)
            
            for module_name, docs in modules.items():
                content += f"## Módulo: {module_name}\n\n"
                
                # Agrupar por clase
                classes = {}
                functions = []
                
                for doc in docs:
                    if doc.class_name:
                        if doc.class_name not in classes:
                            classes[doc.class_name] = []
                        classes[doc.class_name].append(doc)
                    else:
                        functions.append(doc)
                
                # Documentar funciones de nivel de módulo
                if functions:
                    content += "### Funciones\n\n"
                    for doc in functions:
                        content += f"#### {doc.function_name}\n\n"
                        content += f"{doc.docstring}\n\n"
                        
                        if doc.parameters:
                            content += "**Parámetros:**\n"
                            for param in doc.parameters:
                                content += f"- {param['name']} ({param['type']}): {param['description']}\n"
                            content += "\n"
                        
                        if doc.return_type:
                            content += f"**Retorna:** {doc.return_type}\n\n"
                        
                        if doc.examples:
                            content += "**Ejemplos:**\n"
                            for example in doc.examples:
                                content += f"```python\n{example}\n```\n\n"
                
                # Documentar clases
                for class_name, class_docs in classes.items():
                    content += f"### Clase: {class_name}\n\n"
                    
                    for doc in class_docs:
                        content += f"#### {doc.function_name}\n\n"
                        content += f"{doc.docstring}\n\n"
                        
                        if doc.parameters:
                            content += "**Parámetros:**\n"
                            for param in doc.parameters:
                                content += f"- {param['name']} ({param['type']}): {param['description']}\n"
                            content += "\n"
                        
                        if doc.return_type:
                            content += f"**Retorna:** {doc.return_type}\n\n"
                        
                        if doc.examples:
                            content += "**Ejemplos:**\n"
                            for example in doc.examples:
                                content += f"```python\n{example}\n```\n\n"
                
                content += "---\n\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error rendering code markdown: {e}")
            return ""
    
    async def _render_code_html(self, code_docs: List[CodeDocumentation]) -> str:
        """Renderizar documentación de código en HTML"""
        try:
            markdown_content = await self._render_code_markdown(code_docs)
            html_content = markdown.markdown(markdown_content, extensions=['codehilite', 'fenced_code'])
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error rendering code HTML: {e}")
            return ""
    
    async def _render_user_guide_markdown(self, user_guide: List[DocumentationSection]) -> str:
        """Renderizar guía de usuario en Markdown"""
        try:
            content = "# Guía de Usuario\n\n"
            
            for section in sorted(user_guide, key=lambda x: x.metadata.get("order", 0)):
                content += f"## {section.title}\n\n"
                content += f"{section.content}\n\n"
                
                if section.subsections:
                    for subsection in section.subsections:
                        content += f"### {subsection.title}\n\n"
                        content += f"{subsection.content}\n\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error rendering user guide markdown: {e}")
            return ""
    
    async def _render_user_guide_html(self, user_guide: List[DocumentationSection]) -> str:
        """Renderizar guía de usuario en HTML"""
        try:
            markdown_content = await self._render_user_guide_markdown(user_guide)
            html_content = markdown.markdown(markdown_content, extensions=['codehilite', 'fenced_code'])
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error rendering user guide HTML: {e}")
            return ""
    
    async def save_documentation(self, documentation: Dict[str, Any], output_directory: str):
        """Guardar documentación en archivos"""
        try:
            os.makedirs(output_directory, exist_ok=True)
            
            # Guardar documentación en Markdown
            if "formats" in documentation and "markdown" in documentation["formats"]:
                markdown_dir = os.path.join(output_directory, "markdown")
                os.makedirs(markdown_dir, exist_ok=True)
                
                for doc_type, content in documentation["formats"]["markdown"].items():
                    file_path = os.path.join(markdown_dir, f"{doc_type}.md")
                    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                        await f.write(content)
            
            # Guardar documentación en HTML
            if "formats" in documentation and "html" in documentation["formats"]:
                html_dir = os.path.join(output_directory, "html")
                os.makedirs(html_dir, exist_ok=True)
                
                for doc_type, content in documentation["formats"]["html"].items():
                    file_path = os.path.join(html_dir, f"{doc_type}.html")
                    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                        await f.write(content)
            
            # Guardar especificación OpenAPI
            if "formats" in documentation and "openapi" in documentation["formats"]:
                openapi_dir = os.path.join(output_directory, "openapi")
                os.makedirs(openapi_dir, exist_ok=True)
                
                for doc_type, content in documentation["formats"]["openapi"].items():
                    file_path = os.path.join(openapi_dir, f"{doc_type}.json")
                    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                        await f.write(content)
            
            # Guardar documentación completa en JSON
            json_file = os.path.join(output_directory, "documentation.json")
            async with aiofiles.open(json_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(documentation, indent=2, ensure_ascii=False))
            
            logger.info(f"Documentation saved to {output_directory}")
            
        except Exception as e:
            logger.error(f"Error saving documentation: {e}")

# Funciones de utilidad
async def generate_documentation_for_app(app, source_directory: str, output_directory: str = "docs"):
    """Generar documentación completa para una aplicación"""
    try:
        doc_manager = DocumentationManager(app, source_directory)
        documentation = await doc_manager.generate_complete_documentation()
        await doc_manager.save_documentation(documentation, output_directory)
        return documentation
    except Exception as e:
        logger.error(f"Error generating documentation: {e}")
        return None

async def update_documentation_automatically(app, source_directory: str, output_directory: str = "docs"):
    """Actualizar documentación automáticamente"""
    try:
        # Verificar si hay cambios en el código
        if await _has_code_changes(source_directory, output_directory):
            logger.info("Code changes detected, updating documentation...")
            return await generate_documentation_for_app(app, source_directory, output_directory)
        else:
            logger.info("No code changes detected, documentation is up to date")
            return None
    except Exception as e:
        logger.error(f"Error updating documentation: {e}")
        return None

async def _has_code_changes(source_directory: str, output_directory: str) -> bool:
    """Verificar si hay cambios en el código"""
    try:
        # Implementación simplificada - en una implementación real,
        # se compararían timestamps o hashes de archivos
        return True  # Por simplicidad, siempre actualizar
    except Exception as e:
        logger.error(f"Error checking code changes: {e}")
        return True

# Configuración de documentación por defecto
DEFAULT_DOCUMENTATION_CONFIG = {
    "output_formats": ["markdown", "html", "openapi"],
    "include_code_examples": True,
    "include_api_examples": True,
    "auto_update": True,
    "templates": {
        "api": "api_template.md",
        "code": "code_template.md",
        "user_guide": "user_guide_template.md"
    },
    "styling": {
        "theme": "default",
        "colors": {
            "primary": "#007bff",
            "secondary": "#6c757d",
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545"
        }
    }
}


























