"""
System Analysis Script for Bulk TruthGPT
==========================================

Analyzes and reports on the current state of the ultra-advanced system.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import ast

def analyze_system() -> Dict[str, Any]:
    """Analyze the entire bulk_truthgpt system."""
    
    base_path = Path(__file__).parent
    analysis = {
        'total_files': 0,
        'total_lines': 0,
        'modules': [],
        'systems': [],
        'utilities': [],
        'errors': [],
        'metrics': {}
    }
    
    # Analyze utils directory
    utils_path = base_path / 'utils'
    if utils_path.exists():
        for file in utils_path.glob('*.py'):
            analysis['total_files'] += 1
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    analysis['total_lines'] += len(content.splitlines())
                    
                    # Check for ultra-advanced systems
                    if 'ultra_' in file.stem:
                        analysis['systems'].append({
                            'name': file.stem,
                            'path': str(file.relative_to(base_path)),
                            'lines': len(content.splitlines())
                        })
                    
                    # Check for utilities
                    if any(keyword in file.stem for keyword in ['optimizer', 'monitor', 'cache', 'system']):
                        analysis['utilities'].append({
                            'name': file.stem,
                            'path': str(file.relative_to(base_path)),
                            'lines': len(content.splitlines())
                        })
                    
            except Exception as e:
                analysis['errors'].append({
                    'file': str(file),
                    'error': str(e)
                })
    
    # Generate summary
    analysis['metrics'] = {
        'total_systems': len(analysis['systems']),
        'total_utilities': len(analysis['utilities']),
        'average_lines_per_file': analysis['total_lines'] / max(analysis['total_files'], 1),
        'system_coverage': 'ultra-advanced' if len(analysis['systems']) > 10 else 'basic'
    }
    
    return analysis

def print_report(analysis: Dict[str, Any]):
    """Print analysis report."""
    print("=" * 80)
    print("BULK TRUTHGPT SYSTEM ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nTotal Files: {analysis['total_files']}")
    print(f"Total Lines: {analysis['total_lines']:,}")
    print(f"Total Systems: {analysis['metrics']['total_systems']}")
    print(f"Total Utilities: {analysis['metrics']['total_utilities']}")
    print(f"System Coverage: {analysis['metrics']['system_coverage'].upper()}")
    print(f"Average Lines per File: {analysis['metrics']['average_lines_per_file']:.1f}")
    
    if analysis['systems']:
        print("\n" + "=" * 80)
        print("ULTRA-ADVANCED SYSTEMS:")
        print("=" * 80)
        for i, system in enumerate(analysis['systems'], 1):
            print(f"{i}. {system['name']} ({system['lines']} lines)")
    
    if analysis['utilities']:
        print("\n" + "=" * 80)
        print("UTILITIES:")
        print("=" * 80)
        for i, util in enumerate(analysis['utilities'], 1):
            print(f"{i}. {util['name']} ({util['lines']} lines)")
    
    if analysis['errors']:
        print("\n" + "=" * 80)
        print("ERRORS:")
        print("=" * 80)
        for error in analysis['errors']:
            print(f"- {error['file']}: {error['error']}")

if __name__ == "__main__":
    analysis = analyze_system()
    print_report(analysis)








