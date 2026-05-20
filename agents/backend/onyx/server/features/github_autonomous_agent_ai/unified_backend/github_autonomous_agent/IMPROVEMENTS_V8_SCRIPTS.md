# Scripts de Automatización - Mejoras V8

## Scripts Útiles para Implementar las Mejoras

---

## 🔍 Script 1: Buscar Strings Hardcodeados

**Archivo**: `scripts/find-hardcoded-strings.py`

```python
#!/usr/bin/env python3
"""
Script para encontrar strings hardcodeados que deberían ser constantes.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

# Strings comunes que deberían ser constantes
HARDCODED_PATTERNS = [
    (r'"main"', 'GitConfig.DEFAULT_BASE_BRANCH'),
    (r"'main'", 'GitConfig.DEFAULT_BASE_BRANCH'),
    (r'"GitHub token no configurado"', 'ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED'),
    (r'"Repository not found"', 'ErrorMessages.REPOSITORY_NOT_FOUND'),
    # Agregar más patrones según necesidad
]

def find_hardcoded_strings(directory: str = '.') -> List[Tuple[str, int, str, str]]:
    """
    Buscar strings hardcodeados en el código.
    
    Returns:
        Lista de tuplas: (archivo, línea, patrón, sugerencia)
    """
    results = []
    
    for root, dirs, files in os.walk(directory):
        # Ignorar directorios
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.venv']]
        
        for file in files:
            if not file.endswith('.py'):
                continue
            
            filepath = os.path.join(root, file)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        for pattern, suggestion in HARDCODED_PATTERNS:
                            if re.search(pattern, line):
                                results.append((filepath, line_num, pattern, suggestion))
            except Exception as e:
                print(f"Error leyendo {filepath}: {e}")
    
    return results

def main():
    """Ejecutar búsqueda y mostrar resultados"""
    results = find_hardcoded_strings('core')
    results.extend(find_hardcoded_strings('api'))
    
    if not results:
        print("✅ No se encontraron strings hardcodeados")
        return
    
    print(f"⚠️  Se encontraron {len(results)} strings hardcodeados:\n")
    
    for filepath, line_num, pattern, suggestion in results:
        print(f"📄 {filepath}:{line_num}")
        print(f"   Patrón: {pattern}")
        print(f"   Sugerencia: Usar {suggestion}\n")

if __name__ == '__main__':
    main()
```

**Uso:**
```bash
python scripts/find-hardcoded-strings.py
```

---

## 🔄 Script 2: Migrar Strings a Constantes

**Archivo**: `scripts/migrate-to-constants.py`

```python
#!/usr/bin/env python3
"""
Script para migrar strings hardcodeados a constantes.
"""

import re
import sys
from pathlib import Path

# Mapeo de strings a constantes
MIGRATION_MAP = {
    '"main"': 'GitConfig.DEFAULT_BASE_BRANCH',
    "'main'": 'GitConfig.DEFAULT_BASE_BRANCH',
    '"GitHub token no configurado"': 'ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED',
    # Agregar más mapeos
}

def migrate_file(filepath: Path, dry_run: bool = True) -> int:
    """
    Migrar strings en un archivo.
    
    Returns:
        Número de reemplazos realizados
    """
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        replacements = 0
        
        # Verificar si necesita imports
        needs_gitconfig = False
        needs_errors = False
        
        for old, new in MIGRATION_MAP.items():
            if old in content:
                if 'GitConfig' in new:
                    needs_gitconfig = True
                if 'ErrorMessages' in new:
                    needs_errors = True
                
                content = content.replace(old, new)
                replacements += content.count(new) - original_content.count(new)
        
        # Agregar imports si es necesario
        if replacements > 0:
            imports_to_add = []
            if needs_gitconfig and 'from core.constants import GitConfig' not in content:
                imports_to_add.append('from core.constants import GitConfig')
            if needs_errors and 'from core.constants import ErrorMessages' not in content:
                imports_to_add.append('from core.constants import ErrorMessages')
            
            if imports_to_add:
                # Encontrar dónde agregar imports
                import_section = re.search(r'^(from|import)', content, re.MULTILINE)
                if import_section:
                    insert_pos = import_section.start()
                    content = content[:insert_pos] + '\n'.join(imports_to_add) + '\n' + content[insert_pos:]
        
        if not dry_run and replacements > 0:
            filepath.write_text(content, encoding='utf-8')
            print(f"✅ Migrado {filepath}: {replacements} reemplazos")
        
        return replacements
    
    except Exception as e:
        print(f"❌ Error procesando {filepath}: {e}")
        return 0

def main():
    """Ejecutar migración"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrar strings a constantes')
    parser.add_argument('--dry-run', action='store_true', help='Solo mostrar cambios sin aplicar')
    parser.add_argument('files', nargs='*', help='Archivos a migrar (si no se especifica, migra todos)')
    
    args = parser.parse_args()
    
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = list(Path('core').rglob('*.py')) + list(Path('api').rglob('*.py'))
    
    total_replacements = 0
    for filepath in files:
        replacements = migrate_file(filepath, dry_run=args.dry_run)
        total_replacements += replacements
    
    if args.dry_run:
        print(f"\n🔍 Dry run: {total_replacements} reemplazos encontrados")
        print("Ejecuta sin --dry-run para aplicar cambios")
    else:
        print(f"\n✅ Migración completa: {total_replacements} reemplazos realizados")

if __name__ == '__main__':
    main()
```

**Uso:**
```bash
# Dry run (ver cambios sin aplicar)
python scripts/migrate-to-constants.py --dry-run

# Aplicar cambios
python scripts/migrate-to-constants.py

# Migrar archivos específicos
python scripts/migrate-to-constants.py core/utils.py api/utils.py
```

---

## ✅ Script 3: Verificar Uso de Constantes

**Archivo**: `scripts/verify-constants-usage.py`

```python
#!/usr/bin/env python3
"""
Script para verificar que se usan constantes correctamente.
"""

import ast
import re
from pathlib import Path
from typing import Set, List, Tuple

class ConstantsChecker(ast.NodeVisitor):
    """Visitor para verificar uso de constantes"""
    
    def __init__(self):
        self.hardcoded_strings: List[Tuple[int, str]] = []
        self.constant_usage: Set[str] = []
        self.imports: Set[str] = set()
    
    def visit_Str(self, node):
        """Visitar strings literales"""
        if node.s in ['main', 'master']:
            self.hardcoded_strings.append((node.lineno, node.s))
        super().generic_visit(node)
    
    def visit_Constant(self, node):
        """Visitar constantes (Python 3.8+)"""
        if isinstance(node.value, str) and node.value in ['main', 'master']:
            self.hardcoded_strings.append((node.lineno, node.value))
        super().generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visitar imports"""
        if node.module == 'core.constants':
            for alias in node.names:
                self.imports.add(alias.name)
        super().generic_visit(node)
    
    def visit_Attribute(self, node):
        """Visitar atributos (como GitConfig.DEFAULT_BASE_BRANCH)"""
        if isinstance(node.value, ast.Name):
            if node.value.id in ['GitConfig', 'ErrorMessages']:
                self.constant_usage.add(f"{node.value.id}.{node.attr}")
        super().generic_visit(node)

def check_file(filepath: Path) -> dict:
    """Verificar un archivo"""
    try:
        content = filepath.read_text(encoding='utf-8')
        tree = ast.parse(content, filename=str(filepath))
        
        checker = ConstantsChecker()
        checker.visit(tree)
        
        return {
            'file': str(filepath),
            'hardcoded': checker.hardcoded_strings,
            'constants_used': list(checker.constant_usage),
            'imports': list(checker.imports),
            'needs_import': bool(checker.hardcoded_strings) and 'GitConfig' not in checker.imports
        }
    except Exception as e:
        return {'file': str(filepath), 'error': str(e)}

def main():
    """Ejecutar verificación"""
    files = list(Path('core').rglob('*.py')) + list(Path('api').rglob('*.py'))
    
    issues = []
    good_files = []
    
    for filepath in files:
        result = check_file(filepath)
        
        if 'error' in result:
            print(f"❌ Error en {result['file']}: {result['error']}")
            continue
        
        if result['hardcoded']:
            issues.append(result)
        else:
            good_files.append(result['file'])
    
    # Reporte
    print(f"📊 Verificación de Constantes\n")
    print(f"✅ Archivos correctos: {len(good_files)}")
    print(f"⚠️  Archivos con problemas: {len(issues)}\n")
    
    if issues:
        print("Archivos que necesitan corrección:\n")
        for issue in issues:
            print(f"📄 {issue['file']}")
            print(f"   Strings hardcodeados: {issue['hardcoded']}")
            if issue['needs_import']:
                print(f"   ⚠️  Necesita importar constantes")
            print()

if __name__ == '__main__':
    main()
```

**Uso:**
```bash
python scripts/verify-constants-usage.py
```

---

## 🧪 Script 4: Generar Tests para Decoradores

**Archivo**: `scripts/generate-decorator-tests.py`

```python
#!/usr/bin/env python3
"""
Script para generar tests para decoradores.
"""

TEMPLATE = '''"""
Tests generados para {decorator_name}
"""

import pytest
from unittest.mock import patch, MagicMock
from {module} import {decorator_name}


def test_{decorator_name}_sync_success():
    """Test decorador con función sync exitosa"""
    @{decorator_name}
    def sync_func():
        return "success"
    
    result = sync_func()
    assert result == "success"


def test_{decorator_name}_sync_error():
    """Test decorador con función sync que falla"""
    @{decorator_name}
    def sync_func():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        sync_func()


@pytest.mark.asyncio
async def test_{decorator_name}_async_success():
    """Test decorador con función async exitosa"""
    @{decorator_name}
    async def async_func():
        return "success"
    
    result = await async_func()
    assert result == "success"


@pytest.mark.asyncio
async def test_{decorator_name}_async_error():
    """Test decorador con función async que falla"""
    @{decorator_name}
    async def async_func():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        await async_func()


@patch('{module}.logger')
def test_{decorator_name}_logs_error(mock_logger):
    """Test que el decorador loguea errores"""
    @{decorator_name}
    def failing_func():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        failing_func()
    
    # Verificar que se llamó logger.error
    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args
    
    # Verificar que incluye exc_info=True
    assert call_args[1].get('exc_info') is True
'''

def generate_test(decorator_name: str, module: str, output_file: str):
    """Generar archivo de test"""
    content = TEMPLATE.format(
        decorator_name=decorator_name,
        module=module
    )
    
    Path(output_file).write_text(content, encoding='utf-8')
    print(f"✅ Test generado: {output_file}")

def main():
    """Generar tests para decoradores principales"""
    decorators = [
        ('handle_github_exception', 'core.utils', 'tests/unit/test_handle_github_exception.py'),
        ('handle_api_errors', 'api.utils', 'tests/unit/test_handle_api_errors.py'),
    ]
    
    for decorator_name, module, output_file in decorators:
        generate_test(decorator_name, module, output_file)
    
    print("\n✅ Tests generados. Revisa y completa según necesidad.")

if __name__ == '__main__':
    from pathlib import Path
    main()
```

**Uso:**
```bash
python scripts/generate-decorator-tests.py
```

---

## 📊 Script 5: Analizar Uso de Decoradores

**Archivo**: `scripts/analyze-decorator-usage.py`

```python
#!/usr/bin/env python3
"""
Script para analizar uso de decoradores en el código.
"""

import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

class DecoratorAnalyzer(ast.NodeVisitor):
    """Analizar uso de decoradores"""
    
    def __init__(self):
        self.decorators: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.functions_without_decorators: List[Tuple[str, int, bool]] = []
    
    def visit_FunctionDef(self, node):
        """Visitar definiciones de función"""
        decorator_names = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorator_names.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorator_names.append(decorator.attr)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorator_names.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    decorator_names.append(decorator.func.attr)
        
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        if decorator_names:
            for name in decorator_names:
                self.decorators[name].append((node.name, node.lineno))
        else:
            # Funciones que podrían necesitar decoradores
            if 'github' in node.name.lower() or 'api' in node.name.lower():
                self.functions_without_decorators.append((node.name, node.lineno, is_async))
        
        super().generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Visitar funciones async"""
        self.visit_FunctionDef(node)

def analyze_file(filepath: Path) -> Dict:
    """Analizar un archivo"""
    try:
        content = filepath.read_text(encoding='utf-8')
        tree = ast.parse(content, filename=str(filepath))
        
        analyzer = DecoratorAnalyzer()
        analyzer.visit(tree)
        
        return {
            'file': str(filepath),
            'decorators': dict(analyzer.decorators),
            'without_decorators': analyzer.functions_without_decorators
        }
    except Exception as e:
        return {'file': str(filepath), 'error': str(e)}

def main():
    """Ejecutar análisis"""
    files = list(Path('core').rglob('*.py')) + list(Path('api').rglob('*.py'))
    
    all_decorators = defaultdict(list)
    all_without = []
    
    for filepath in files:
        result = analyze_file(filepath)
        
        if 'error' in result:
            continue
        
        for decorator, functions in result['decorators'].items():
            all_decorators[decorator].extend([
                (result['file'], func, line) for func, line in functions
            ])
        
        all_without.extend([
            (result['file'], func, line, is_async)
            for func, line, is_async in result['without_decorators']
        ])
    
    # Reporte
    print("📊 Análisis de Decoradores\n")
    
    print("Decoradores más usados:")
    for decorator, usages in sorted(all_decorators.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {decorator}: {len(usages)} usos")
    
    if all_without:
        print(f"\n⚠️  Funciones que podrían necesitar decoradores: {len(all_without)}")
        for file, func, line, is_async in all_without[:10]:  # Mostrar primeros 10
            async_str = "async " if is_async else ""
            print(f"  {file}:{line} - {async_str}{func}()")

if __name__ == '__main__':
    main()
```

**Uso:**
```bash
python scripts/analyze-decorator-usage.py
```

---

## 🔧 Script 6: Refactorizar Automáticamente

**Archivo**: `scripts/auto-refactor-v8.py`

```python
#!/usr/bin/env python3
"""
Script para refactorizar automáticamente código a estándares V8.
"""

import re
from pathlib import Path
from typing import List

REFACTORING_RULES = [
    # Reemplazar "main" hardcodeado
    (r'\b"main"\b', 'GitConfig.DEFAULT_BASE_BRANCH'),
    (r"\b'main'\b", 'GitConfig.DEFAULT_BASE_BRANCH'),
    
    # Agregar imports si es necesario (esto requiere análisis AST más complejo)
]

def refactor_file(filepath: Path, dry_run: bool = True) -> dict:
    """Refactorizar un archivo"""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content
        changes = []
        
        # Aplicar reglas
        for pattern, replacement in REFACTORING_RULES:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                changes.append(f"Reemplazado {len(matches)} instancias de {pattern}")
        
        # Agregar imports si es necesario
        if 'GitConfig.DEFAULT_BASE_BRANCH' in content and 'from core.constants import GitConfig' not in content:
            # Encontrar primera línea de import
            import_match = re.search(r'^(from|import)', content, re.MULTILINE)
            if import_match:
                insert_pos = import_match.start()
                content = content[:insert_pos] + 'from core.constants import GitConfig\n' + content[insert_pos:]
                changes.append("Agregado import de GitConfig")
        
        if not dry_run and content != original:
            filepath.write_text(content, encoding='utf-8')
        
        return {
            'file': str(filepath),
            'changed': content != original,
            'changes': changes
        }
    
    except Exception as e:
        return {'file': str(filepath), 'error': str(e)}

def main():
    """Ejecutar refactorización"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Solo mostrar cambios')
    parser.add_argument('--files', nargs='+', help='Archivos específicos a refactorizar')
    
    args = parser.parse_args()
    
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = list(Path('core').rglob('*.py')) + list(Path('api').rglob('*.py'))
    
    results = [refactor_file(f, dry_run=args.dry_run) for f in files]
    
    changed = [r for r in results if r.get('changed')]
    
    print(f"📊 Refactorización V8\n")
    print(f"Archivos procesados: {len(results)}")
    print(f"Archivos modificados: {len(changed)}\n")
    
    for result in changed:
        print(f"✅ {result['file']}")
        for change in result['changes']:
            print(f"   - {change}")

if __name__ == '__main__':
    main()
```

**Uso:**
```bash
# Ver cambios
python scripts/auto-refactor-v8.py --dry-run

# Aplicar cambios
python scripts/auto-refactor-v8.py
```

---

## 📝 Script 7: Generar Documentación de Constantes

**Archivo**: `scripts/generate-constants-docs.py`

```python
#!/usr/bin/env python3
"""
Script para generar documentación de constantes.
"""

import ast
from pathlib import Path
from typing import Dict, List

def extract_constants(filepath: Path) -> Dict[str, List[str]]:
    """Extraer constantes de un archivo"""
    try:
        content = filepath.read_text(encoding='utf-8')
        tree = ast.parse(content)
        
        constants = {}
        current_class = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                current_class = node.name
                constants[current_class] = []
            
            if isinstance(node, ast.Assign) and current_class:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():  # Convención: constantes en mayúsculas
                            constants[current_class].append(target.id)
        
        return constants
    except Exception as e:
        return {}

def generate_docs():
    """Generar documentación"""
    constants_file = Path('core/constants.py')
    
    if not constants_file.exists():
        print("❌ core/constants.py no encontrado")
        return
    
    constants = extract_constants(constants_file)
    
    doc = "# Constantes Disponibles\n\n"
    doc += "Este documento lista todas las constantes disponibles en el proyecto.\n\n"
    
    for class_name, const_list in constants.items():
        doc += f"## {class_name}\n\n"
        doc += f"Clase que contiene constantes relacionadas con {class_name.lower()}.\n\n"
        
        for const in const_list:
            doc += f"### `{class_name}.{const}`\n\n"
            doc += f"**Descripción**: [Agregar descripción]\n\n"
            doc += f"**Uso**:\n```python\nfrom core.constants import {class_name}\n\nvalue = {class_name}.{const}\n```\n\n"
        
        doc += "\n---\n\n"
    
    output_file = Path('docs/CONSTANTS_REFERENCE.md')
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(doc, encoding='utf-8')
    
    print(f"✅ Documentación generada: {output_file}")

if __name__ == '__main__':
    generate_docs()
```

**Uso:**
```bash
python scripts/generate-constants-docs.py
```

---

## 🎯 Uso de los Scripts

### Workflow Recomendado

1. **Encontrar problemas**:
   ```bash
   python scripts/find-hardcoded-strings.py
   python scripts/verify-constants-usage.py
   ```

2. **Analizar código**:
   ```bash
   python scripts/analyze-decorator-usage.py
   ```

3. **Migrar código** (dry run primero):
   ```bash
   python scripts/migrate-to-constants.py --dry-run
   python scripts/migrate-to-constants.py
   ```

4. **Generar tests**:
   ```bash
   python scripts/generate-decorator-tests.py
   ```

5. **Refactorizar automáticamente**:
   ```bash
   python scripts/auto-refactor-v8.py --dry-run
   python scripts/auto-refactor-v8.py
   ```

6. **Generar documentación**:
   ```bash
   python scripts/generate-constants-docs.py
   ```

---

## 🔗 Integración con Makefile

Agregar al `Makefile`:

```makefile
# Verificar constantes
check-constants:
	python scripts/verify-constants-usage.py

# Migrar a constantes
migrate-constants:
	python scripts/migrate-to-constants.py --dry-run

# Analizar decoradores
analyze-decorators:
	python scripts/analyze-decorator-usage.py

# Refactorizar V8
refactor-v8:
	python scripts/auto-refactor-v8.py --dry-run

# Generar documentación
docs-constants:
	python scripts/generate-constants-docs.py
```

**Uso:**
```bash
make check-constants
make migrate-constants
make analyze-decorators
```

---

**Nota**: Estos scripts son herramientas de ayuda. Siempre revisa los cambios antes de commitear.



