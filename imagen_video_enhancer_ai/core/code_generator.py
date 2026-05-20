"""
Code Generator System
====================

Advanced system for automatic code generation.
"""

import ast
import inspect
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CodeTemplateType(Enum):
    """Code template types."""
    CLASS = "class"
    FUNCTION = "function"
    MODULE = "module"
    API_ROUTE = "api_route"
    TEST = "test"
    CONFIG = "config"
    SERVICE = "service"
    MODEL = "model"


@dataclass
class CodeTemplate:
    """Code template definition."""
    name: str
    template_type: CodeTemplateType
    template: str
    variables: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class GeneratedCode:
    """Generated code result."""
    code: str
    file_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeGenerator:
    """Advanced code generator system."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize code generator.
        
        Args:
            output_dir: Output directory for generated code
        """
        self.output_dir = output_dir or Path("generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, CodeTemplate] = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register default code templates."""
        # Class template
        self.register_template(CodeTemplate(
            name="base_class",
            template_type=CodeTemplateType.CLASS,
            template="""class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self{init_params}):
        \"\"\"Initialize {class_name}.\"\"\"
{init_body}
""",
            description="Base class template"
        ))
        
        # Function template
        self.register_template(CodeTemplate(
            name="async_function",
            template_type=CodeTemplateType.FUNCTION,
            template="""async def {function_name}({params}):
    \"\"\"
    {description}
    
    Args:
{args_doc}
    
    Returns:
{returns_doc}
    \"\"\"
{body}
""",
            description="Async function template"
        ))
        
        # API route template
        self.register_template(CodeTemplate(
            name="api_route",
            template_type=CodeTemplateType.API_ROUTE,
            template="""@router.{method}("{path}")
async def {function_name}({params}):
    \"\"\"
    {description}
    \"\"\"
{body}
""",
            description="API route template"
        ))
        
        # Test template
        self.register_template(CodeTemplate(
            name="test_function",
            template_type=CodeTemplateType.TEST,
            template="""def test_{test_name}():
    \"\"\"
    Test {description}
    \"\"\"
{body}
""",
            description="Test function template"
        ))
    
    def register_template(self, template: CodeTemplate):
        """
        Register a code template.
        
        Args:
            template: Template to register
        """
        self.templates[template.name] = template
        logger.info(f"Registered template: {template.name}")
    
    def generate_from_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        output_file: Optional[Path] = None
    ) -> GeneratedCode:
        """
        Generate code from template.
        
        Args:
            template_name: Template name
            variables: Template variables
            output_file: Optional output file path
            
        Returns:
            Generated code
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        code = template.template.format(**variables)
        
        if output_file:
            output_path = self.output_dir / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding='utf-8')
            logger.info(f"Generated code: {output_path}")
        else:
            output_path = None
        
        return GeneratedCode(
            code=code,
            file_path=output_path,
            metadata={
                "template": template_name,
                "variables": variables,
                "generated_at": datetime.now().isoformat()
            }
        )
    
    def generate_class(
        self,
        class_name: str,
        description: str,
        methods: List[Dict[str, Any]],
        base_classes: Optional[List[str]] = None,
        output_file: Optional[Path] = None
    ) -> GeneratedCode:
        """
        Generate a class.
        
        Args:
            class_name: Class name
            description: Class description
            methods: List of method definitions
            base_classes: Optional base classes
            output_file: Optional output file path
            
        Returns:
            Generated code
        """
        base_str = ""
        if base_classes:
            base_str = f"({', '.join(base_classes)})"
        
        methods_code = []
        for method in methods:
            method_name = method.get("name", "method")
            method_params = method.get("params", "")
            method_body = method.get("body", "pass")
            method_doc = method.get("doc", "")
            
            method_code = f"    def {method_name}({method_params}):\n"
            if method_doc:
                method_code += f'        """{method_doc}"""\n'
            method_code += f"        {method_body}\n"
            methods_code.append(method_code)
        
        code = f'''class {class_name}{base_str}:
    """
    {description}
    """
    
{chr(10).join(methods_code)}
'''
        
        if output_file:
            output_path = self.output_dir / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding='utf-8')
            logger.info(f"Generated class: {output_path}")
        else:
            output_path = None
        
        return GeneratedCode(
            code=code,
            file_path=output_path,
            metadata={
                "type": "class",
                "class_name": class_name,
                "generated_at": datetime.now().isoformat()
            }
        )
    
    def generate_api_routes(
        self,
        routes: List[Dict[str, Any]],
        router_name: str = "router",
        output_file: Optional[Path] = None
    ) -> GeneratedCode:
        """
        Generate API routes.
        
        Args:
            routes: List of route definitions
            router_name: Router variable name
            output_file: Optional output file path
            
        Returns:
            Generated code
        """
        imports = """from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
"""
        
        routes_code = []
        for route in routes:
            method = route.get("method", "get").lower()
            path = route.get("path", "/")
            function_name = route.get("function_name", f"handle_{method}_{path.replace('/', '_')}")
            params = route.get("params", "")
            body = route.get("body", "pass")
            description = route.get("description", "")
            
            route_code = f'''@router.{method}("{path}")
async def {function_name}({params}):
    """
    {description}
    """
    {body}
'''
            routes_code.append(route_code)
        
        code = f"""{imports}

{router_name} = APIRouter()

{chr(10).join(routes_code)}
"""
        
        if output_file:
            output_path = self.output_dir / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding='utf-8')
            logger.info(f"Generated API routes: {output_path}")
        else:
            output_path = None
        
        return GeneratedCode(
            code=code,
            file_path=output_path,
            metadata={
                "type": "api_routes",
                "route_count": len(routes),
                "generated_at": datetime.now().isoformat()
            }
        )
    
    def generate_tests(
        self,
        test_cases: List[Dict[str, Any]],
        test_file_name: str,
        output_file: Optional[Path] = None
    ) -> GeneratedCode:
        """
        Generate test code.
        
        Args:
            test_cases: List of test case definitions
            test_file_name: Test file name
            output_file: Optional output file path
            
        Returns:
            Generated code
        """
        imports = """import pytest
import asyncio
from typing import Dict, Any
"""
        
        tests_code = []
        for test_case in test_cases:
            test_name = test_case.get("name", "test_case")
            description = test_case.get("description", "")
            body = test_case.get("body", "pass")
            is_async = test_case.get("async", False)
            
            async_prefix = "async " if is_async else ""
            test_code = f'''def {async_prefix}test_{test_name}():
    """
    {description}
    """
    {body}
'''
            tests_code.append(test_code)
        
        code = f"""{imports}

{chr(10).join(tests_code)}
"""
        
        if output_file:
            output_path = self.output_dir / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code, encoding='utf-8')
            logger.info(f"Generated tests: {output_path}")
        else:
            output_path = None
        
        return GeneratedCode(
            code=code,
            file_path=output_path,
            metadata={
                "type": "tests",
                "test_count": len(test_cases),
                "generated_at": datetime.now().isoformat()
            }
        )
    
    def extract_from_existing(
        self,
        source_file: Path,
        extract_type: str = "class"
    ) -> Dict[str, Any]:
        """
        Extract code structure from existing file.
        
        Args:
            source_file: Source file path
            extract_type: Type to extract (class, function, etc.)
            
        Returns:
            Extracted structure
        """
        if not source_file.exists():
            raise ValueError(f"File not found: {source_file}")
        
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        tree = ast.parse(source_code)
        
        extracted = {
            "classes": [],
            "functions": [],
            "imports": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                extracted["classes"].append({
                    "name": node.name,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                })
            elif isinstance(node, ast.FunctionDef):
                extracted["functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "is_async": isinstance(node, ast.AsyncFunctionDef)
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                extracted["imports"].append(ast.unparse(node))
        
        return extracted



