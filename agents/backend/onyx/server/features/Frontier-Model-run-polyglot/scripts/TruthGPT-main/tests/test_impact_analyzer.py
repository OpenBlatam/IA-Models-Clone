"""
Test Impact Analyzer
Determines which tests need to run based on code changes
"""

import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import json


class TestImpactAnalyzer:
    """Analyze code changes to determine which tests to run"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.source_root = project_root.parent  # Assuming tests/ is in project root
        self.test_mapping = {}  # Maps source files to test files
        self._build_test_mapping()
    
    def _build_test_mapping(self):
        """Build mapping between source files and test files"""
        # Find all Python source files
        source_files = list(self.source_root.rglob("*.py"))
        source_files = [f for f in source_files if "test" not in str(f) and "__pycache__" not in str(f)]
        
        # Find all test files
        test_files = list(self.project_root.rglob("test_*.py"))
        test_files.extend(list(self.project_root.rglob("*_test.py")))
        
        # Build mapping by analyzing imports and test content
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    # Extract imports
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                    
                    # Map imports to source files
                    for imp in imports:
                        # Try to find corresponding source file
                        for source_file in source_files:
                            module_path = str(source_file.relative_to(self.source_root)).replace('/', '.').replace('\\', '.').replace('.py', '')
                            if imp in module_path or module_path.endswith(imp):
                                if source_file not in self.test_mapping:
                                    self.test_mapping[source_file] = []
                                if test_file not in self.test_mapping[source_file]:
                                    self.test_mapping[source_file].append(test_file)
            except Exception as e:
                print(f"Warning: Could not parse {test_file}: {e}")
    
    def get_changed_files(self, base_ref: str = "HEAD", target_ref: str = None) -> List[Path]:
        """Get list of changed files using git"""
        try:
            if target_ref:
                cmd = ["git", "diff", "--name-only", base_ref, target_ref]
            else:
                cmd = ["git", "diff", "--name-only", base_ref]
            
            result = subprocess.run(
                cmd,
                cwd=self.source_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                changed_files = [
                    self.source_root / f.strip()
                    for f in result.stdout.strip().split('\n')
                    if f.strip() and f.endswith('.py')
                ]
                return changed_files
        except Exception as e:
            print(f"Warning: Could not get changed files: {e}")
        
        return []
    
    def get_affected_tests(self, changed_files: List[Path]) -> Set[Path]:
        """Get tests that should run based on changed files"""
        affected_tests = set()
        
        for changed_file in changed_files:
            # Direct mapping
            if changed_file in self.test_mapping:
                affected_tests.update(self.test_mapping[changed_file])
            
            # Check for related files (same module, parent modules)
            module_parts = changed_file.parts
            for source_file, test_files in self.test_mapping.items():
                if any(part in source_file.parts for part in module_parts[:-1]):
                    affected_tests.update(test_files)
        
        return affected_tests
    
    def analyze_impact(self, base_ref: str = "HEAD", target_ref: str = None) -> Dict:
        """Analyze impact of changes and return recommended tests"""
        changed_files = self.get_changed_files(base_ref, target_ref)
        affected_tests = self.get_affected_tests(changed_files)
        
        # Categorize tests
        categorized = {
            'unit': [],
            'integration': [],
            'analyzers': [],
            'systems': [],
            'other': []
        }
        
        for test_file in affected_tests:
            test_str = str(test_file)
            if 'unit' in test_str or 'test_' in test_str:
                if 'integration' in test_str:
                    categorized['integration'].append(str(test_file.relative_to(self.project_root)))
                elif 'analyzer' in test_str or 'analyzers' in test_str:
                    categorized['analyzers'].append(str(test_file.relative_to(self.project_root)))
                elif 'system' in test_str or 'systems' in test_str:
                    categorized['systems'].append(str(test_file.relative_to(self.project_root)))
                else:
                    categorized['unit'].append(str(test_file.relative_to(self.project_root)))
            else:
                categorized['other'].append(str(test_file.relative_to(self.project_root)))
        
        return {
            'changed_files': [str(f.relative_to(self.source_root)) for f in changed_files],
            'affected_tests': [str(t.relative_to(self.project_root)) for t in affected_tests],
            'categorized_tests': categorized,
            'summary': {
                'total_changed_files': len(changed_files),
                'total_affected_tests': len(affected_tests),
                'unit_tests': len(categorized['unit']),
                'integration_tests': len(categorized['integration']),
                'analyzer_tests': len(categorized['analyzers']),
                'system_tests': len(categorized['systems'])
            }
        }
    
    def generate_test_command(self, impact_result: Dict, pytest_args: str = "") -> str:
        """Generate pytest command for affected tests"""
        affected = impact_result['affected_tests']
        if not affected:
            return "pytest"  # Run all if no specific tests
        
        # Build pytest command
        test_paths = " ".join(affected)
        return f"pytest {test_paths} {pytest_args}".strip()
    
    def save_impact_report(self, impact_result: Dict, output_file: Path = None):
        """Save impact analysis report"""
        if output_file is None:
            output_file = self.project_root / "test_impact_report.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(impact_result, f, indent=2)
        
        print(f"✅ Impact report saved to {output_file}")


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Impact Analyzer')
    parser.add_argument('--base', default='HEAD', help='Base git reference')
    parser.add_argument('--target', help='Target git reference (optional)')
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--generate-command', action='store_true', help='Generate pytest command')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    print("🔍 Analyzing test impact...")
    analyzer = TestImpactAnalyzer(project_root)
    
    impact = analyzer.analyze_impact(args.base, args.target)
    
    print(f"\n📊 Impact Analysis Results:")
    print(f"  Changed Files: {impact['summary']['total_changed_files']}")
    print(f"  Affected Tests: {impact['summary']['total_affected_tests']}")
    print(f"    - Unit: {impact['summary']['unit_tests']}")
    print(f"    - Integration: {impact['summary']['integration_tests']}")
    print(f"    - Analyzers: {impact['summary']['analyzer_tests']}")
    print(f"    - Systems: {impact['summary']['system_tests']}")
    
    if args.generate_command:
        cmd = analyzer.generate_test_command(impact)
        print(f"\n💻 Recommended Command:")
        print(f"  {cmd}")
    
    if args.output:
        output_path = Path(args.output)
        analyzer.save_impact_report(impact, output_path)
    else:
        analyzer.save_impact_report(impact)


if __name__ == '__main__':
    main()

