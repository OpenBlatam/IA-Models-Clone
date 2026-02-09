"""
Helper para migración del código legacy de pipelines al nuevo sistema modular

Este módulo proporciona herramientas para facilitar la migración gradual
del código que usa el wrapper de compatibilidad al nuevo sistema modular.
"""

import ast
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class PipelineMigrationAnalyzer:
    """Analiza código para encontrar usos del wrapper legacy"""
    
    def __init__(self):
        self.legacy_import_pattern = re.compile(
            r'from\s+core\.architecture\.pipelines\s+import'
        )
        self.legacy_usage_patterns = [
            r'ParallelPipeline',
            r'ConditionalPipeline',
        ]
    
    def find_legacy_imports(self, file_path: Path) -> List[Dict[str, any]]:
        """
        Encuentra imports legacy en un archivo.
        
        Args:
            file_path: Ruta al archivo a analizar
            
        Returns:
            Lista de imports legacy encontrados
        """
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    if self.legacy_import_pattern.search(line):
                        imports.append({
                            'line': i,
                            'content': line.strip(),
                            'file': str(file_path)
                        })
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return imports
    
    def find_legacy_usage(self, file_path: Path) -> List[Dict[str, any]]:
        """
        Encuentra usos de alias legacy en un archivo.
        
        Args:
            file_path: Ruta al archivo a analizar
            
        Returns:
            Lista de usos legacy encontrados
        """
        usages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    for pattern in self.legacy_usage_patterns:
                        if re.search(pattern, line):
                            usages.append({
                                'line': i,
                                'content': line.strip(),
                                'pattern': pattern,
                                'file': str(file_path)
                            })
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return usages
    
    def analyze_directory(self, directory: Path) -> Dict[str, any]:
        """
        Analiza un directorio completo buscando código legacy.
        
        Args:
            directory: Directorio a analizar
            
        Returns:
            Reporte con todos los usos legacy encontrados
        """
        report = {
            'legacy_imports': [],
            'legacy_usages': [],
            'files_analyzed': 0,
            'files_with_legacy': 0
        }
        
        python_files = list(directory.rglob('*.py'))
        report['files_analyzed'] = len(python_files)
        
        for file_path in python_files:
            # Saltar archivos de tests y __pycache__
            if 'test' in str(file_path) or '__pycache__' in str(file_path):
                continue
            
            imports = self.find_legacy_imports(file_path)
            usages = self.find_legacy_usage(file_path)
            
            if imports or usages:
                report['files_with_legacy'] += 1
                report['legacy_imports'].extend(imports)
                report['legacy_usages'].extend(usages)
        
        return report


class PipelineMigrationGenerator:
    """Genera código migrado del sistema legacy al nuevo"""
    
    MIGRATION_MAP = {
        'Pipeline': 'pipelines.pipeline.Pipeline',
        'PipelineStage': 'pipelines.stages.PipelineStage',
        'FunctionStage': 'pipelines.stages.FunctionStage',
        'PipelineBuilder': 'pipelines.builders.PipelineBuilder',
        'SequentialExecutor': 'pipelines.executors.SequentialExecutor',
        'ParallelExecutor': 'pipelines.executors.ParallelExecutor',
        'ConditionalExecutor': 'pipelines.executors.ConditionalExecutor',
        'ParallelPipeline': 'pipelines.executors.ParallelExecutor',
        'ConditionalPipeline': 'pipelines.executors.ConditionalExecutor',
        'LoggingMiddleware': 'pipelines.middleware.LoggingMiddleware',
        'MetricsMiddleware': 'pipelines.middleware.MetricsMiddleware',
        'CachingMiddleware': 'pipelines.middleware.CachingMiddleware',
        'RetryMiddleware': 'pipelines.middleware.RetryMiddleware',
        'ValidationMiddleware': 'pipelines.middleware.ValidationMiddleware',
    }
    
    def generate_migration_suggestion(self, legacy_import: str) -> str:
        """
        Genera una sugerencia de migración para un import legacy.
        
        Args:
            legacy_import: Línea de import legacy
            
        Returns:
            Sugerencia de código migrado
        """
        # Extraer los items importados
        match = re.search(r'import\s+(.+)', legacy_import)
        if not match:
            return legacy_import
        
        imported_items = [item.strip() for item in match.group(1).split(',')]
        
        # Agrupar por módulo
        module_groups = {}
        for item in imported_items:
            if item in self.MIGRATION_MAP:
                module = self.MIGRATION_MAP[item].split('.')[0]
                if module not in module_groups:
                    module_groups[module] = []
                module_groups[module].append(item)
        
        # Generar imports nuevos
        new_imports = []
        for module, items in module_groups.items():
            module_path = self.MIGRATION_MAP[items[0]].rsplit('.', 1)[0]
            new_imports.append(f"from core.architecture.{module_path} import {', '.join(items)}")
        
        return '\n'.join(new_imports) if new_imports else legacy_import
    
    def generate_migration_report(self, analysis_report: Dict) -> str:
        """
        Genera un reporte de migración en formato texto.
        
        Args:
            analysis_report: Reporte de análisis
            
        Returns:
            Reporte formateado
        """
        report_lines = [
            "=" * 80,
            "Pipeline Migration Report",
            "=" * 80,
            f"\nFiles analyzed: {analysis_report['files_analyzed']}",
            f"Files with legacy code: {analysis_report['files_with_legacy']}",
            f"Legacy imports found: {len(analysis_report['legacy_imports'])}",
            f"Legacy usages found: {len(analysis_report['legacy_usages'])}",
            "\n" + "=" * 80,
            "Legacy Imports:",
            "=" * 80,
        ]
        
        for imp in analysis_report['legacy_imports']:
            report_lines.append(f"\nFile: {imp['file']}:{imp['line']}")
            report_lines.append(f"  {imp['content']}")
            suggestion = self.generate_migration_suggestion(imp['content'])
            report_lines.append(f"\n  Suggested migration:")
            report_lines.append(f"  {suggestion}")
        
        if analysis_report['legacy_usages']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("Legacy Usages (Aliases):")
            report_lines.append("=" * 80)
            
            for usage in analysis_report['legacy_usages']:
                report_lines.append(f"\nFile: {usage['file']}:{usage['line']}")
                report_lines.append(f"  {usage['content']}")
                if usage['pattern'] == 'ParallelPipeline':
                    report_lines.append(f"  -> Replace with: ParallelExecutor")
                elif usage['pattern'] == 'ConditionalPipeline':
                    report_lines.append(f"  -> Replace with: ConditionalExecutor")
        
        report_lines.append("\n" + "=" * 80)
        
        return '\n'.join(report_lines)


def analyze_project_for_migration(project_root: Path) -> Dict:
    """
    Analiza un proyecto completo para migración de pipelines.
    
    Args:
        project_root: Raíz del proyecto
        
    Returns:
        Reporte completo de análisis
    """
    analyzer = PipelineMigrationAnalyzer()
    return analyzer.analyze_directory(project_root)


def generate_migration_report(project_root: Path, output_file: Optional[Path] = None) -> str:
    """
    Genera un reporte completo de migración.
    
    Args:
        project_root: Raíz del proyecto
        output_file: Archivo donde guardar el reporte (opcional)
        
    Returns:
        Reporte en formato texto
    """
    analysis = analyze_project_for_migration(project_root)
    generator = PipelineMigrationGenerator()
    report = generator.generate_migration_report(analysis)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        project_root = Path(__file__).parent.parent.parent.parent
    
    print("Analyzing project for pipeline migration...")
    report = generate_migration_report(project_root)
    print(report)

