#!/usr/bin/env python3
"""
Generate Refactoring Report
Generates a comprehensive report of refactoring status
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def count_files(directory: Path, extension: str = None) -> int:
    """Count files in directory"""
    if not directory.exists():
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if extension is None or file.endswith(extension):
                count += 1
    return count


def analyze_structure(root_dir: Path) -> Dict:
    """Analyze project structure"""
    stats = {
        'services': {},
        'utils': {},
        'docs': {},
        'scripts': {},
        'config': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Services
    services_dir = root_dir / 'services'
    if services_dir.exists():
        categories = ['analysis', 'recommendations', 'tracking', 'products', 'ml', 
                     'notifications', 'integrations', 'reporting', 'social', 'shared']
        for category in categories:
            cat_dir = services_dir / category
            if cat_dir.exists():
                stats['services'][category] = count_files(cat_dir, '.py')
    
    # Utils
    utils_dir = root_dir / 'utils'
    if utils_dir.exists():
        categories = ['logging', 'caching', 'validation', 'security', 'performance',
                     'database', 'async', 'monitoring', 'helpers']
        for category in categories:
            cat_dir = utils_dir / category
            if cat_dir.exists():
                stats['utils'][category] = count_files(cat_dir, '.py')
    
    # Docs
    docs_dir = root_dir / 'docs'
    if docs_dir.exists():
        categories = ['architecture', 'dependencies', 'features', 'guides', 'api']
        for category in categories:
            cat_dir = docs_dir / category
            if cat_dir.exists():
                stats['docs'][category] = count_files(cat_dir, '.md')
    
    # Scripts
    scripts_dir = root_dir / 'scripts'
    if scripts_dir.exists():
        req_dir = scripts_dir / 'requirements'
        if req_dir.exists():
            categories = ['analysis', 'validation', 'management', 'utils']
            for category in categories:
                cat_dir = req_dir / category
                if cat_dir.exists():
                    stats['scripts'][category] = count_files(cat_dir)
    
    # Config
    config_dir = root_dir / 'config'
    if config_dir.exists():
        env_dir = config_dir / 'environments'
        if env_dir.exists():
            stats['config']['environments'] = count_files(env_dir, '.yaml')
    
    return stats


def generate_report(stats: Dict) -> str:
    """Generate markdown report"""
    report = "# Refactoring Report\n\n"
    report += f"Generated: {stats['timestamp']}\n\n"
    
    report += "## Services Organization\n\n"
    if stats['services']:
        report += "| Category | Files |\n"
        report += "|----------|-------|\n"
        total_services = 0
        for category, count in sorted(stats['services'].items()):
            report += f"| {category} | {count} |\n"
            total_services += count
        report += f"| **Total** | **{total_services}** |\n\n"
    else:
        report += "No services organized yet.\n\n"
    
    report += "## Utils Organization\n\n"
    if stats['utils']:
        report += "| Category | Files |\n"
        report += "|----------|-------|\n"
        total_utils = 0
        for category, count in sorted(stats['utils'].items()):
            report += f"| {category} | {count} |\n"
            total_utils += count
        report += f"| **Total** | **{total_utils}** |\n\n"
    else:
        report += "No utils organized yet.\n\n"
    
    report += "## Documentation Organization\n\n"
    if stats['docs']:
        report += "| Category | Files |\n"
        report += "|----------|-------|\n"
        total_docs = 0
        for category, count in sorted(stats['docs'].items()):
            report += f"| {category} | {count} |\n"
            total_docs += count
        report += f"| **Total** | **{total_docs}** |\n\n"
    else:
        report += "No documentation organized yet.\n\n"
    
    report += "## Scripts Organization\n\n"
    if stats['scripts']:
        report += "| Category | Files |\n"
        report += "|----------|-------|\n"
        total_scripts = 0
        for category, count in sorted(stats['scripts'].items()):
            report += f"| {category} | {count} |\n"
            total_scripts += count
        report += f"| **Total** | **{total_scripts}** |\n\n"
    else:
        report += "No scripts organized yet.\n\n"
    
    report += "## Summary\n\n"
    report += f"- **Services**: {sum(stats['services'].values())} files organized\n"
    report += f"- **Utils**: {sum(stats['utils'].values())} files organized\n"
    report += f"- **Documentation**: {sum(stats['docs'].values())} files organized\n"
    report += f"- **Scripts**: {sum(stats['scripts'].values())} files organized\n"
    
    return report


def main():
    """Main function"""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent.parent
    
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist")
        sys.exit(1)
    
    print("Analyzing project structure...")
    stats = analyze_structure(root_dir)
    
    report = generate_report(stats)
    
    # Write report
    report_file = root_dir / 'docs' / 'REFACTORING_REPORT.md'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Refactoring report generated: {report_file}")
    print(f"\nSummary:")
    print(f"  Services: {sum(stats['services'].values())} files")
    print(f"  Utils: {sum(stats['utils'].values())} files")
    print(f"  Documentation: {sum(stats['docs'].values())} files")
    print(f"  Scripts: {sum(stats['scripts'].values())} files")


if __name__ == '__main__':
    main()



