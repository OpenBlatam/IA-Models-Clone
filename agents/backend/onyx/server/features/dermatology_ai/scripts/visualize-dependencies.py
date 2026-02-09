#!/usr/bin/env python3
"""
Dependency Visualizer
Creates visual representation of dependencies
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import subprocess

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, skipping visualizations")


def get_package_size(package_name):
    """Get installed package size"""
    try:
        result = subprocess.run(
            ['pip', 'show', package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Location:'):
                    location = line.split(':', 1)[1].strip()
                    # Estimate size (simplified)
                    return "installed"
    except:
        pass
    return "unknown"


def analyze_dependencies():
    """Analyze and visualize dependencies"""
    files = {
        'requirements.txt': 'Production',
        'requirements-optimized.txt': 'Optimized',
        'requirements-dev.txt': 'Development',
        'requirements-minimal.txt': 'Minimal',
        'requirements-gpu.txt': 'GPU',
        'requirements-docker.txt': 'Docker'
    }
    
    categories = defaultdict(lambda: defaultdict(int))
    total_packages = defaultdict(int)
    
    for filename, label in files.items():
        filepath = Path(filename)
        if not filepath.exists():
            continue
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-r'):
                    continue
                
                package = line.split('>=')[0].split('==')[0].split('[')[0].strip()
                if package:
                    total_packages[label] += 1
                    
                    # Categorize
                    if any(x in package.lower() for x in ['torch', 'tensor', 'transformers', 'onnx']):
                        categories['ML/AI'][label] += 1
                    elif any(x in package.lower() for x in ['fastapi', 'uvicorn', 'starlette', 'pydantic']):
                        categories['Web Framework'][label] += 1
                    elif any(x in package.lower() for x in ['sql', 'alchemy', 'asyncpg', 'motor', 'redis']):
                        categories['Database'][label] += 1
                    elif any(x in package.lower() for x in ['pytest', 'test', 'coverage']):
                        categories['Testing'][label] += 1
                    elif any(x in package.lower() for x in ['black', 'ruff', 'mypy', 'lint']):
                        categories['Code Quality'][label] += 1
                    elif any(x in package.lower() for x in ['opencv', 'pillow', 'numpy', 'scipy']):
                        categories['Image Processing'][label] += 1
                    else:
                        categories['Other'][label] += 1
    
    return categories, total_packages, files


def create_text_report(categories, total_packages, files):
    """Create text-based report"""
    print("=" * 80)
    print("DEPENDENCY VISUALIZATION REPORT")
    print("=" * 80)
    print()
    
    print("📊 Total Packages per File:")
    print("-" * 80)
    for label, count in sorted(total_packages.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * (count // 5)
        print(f"  {label:20} {count:4} packages {bar}")
    print()
    
    print("📦 Packages by Category:")
    print("-" * 80)
    for category in sorted(categories.keys()):
        print(f"\n  {category}:")
        for label in files.values():
            count = categories[category].get(label, 0)
            if count > 0:
                bar = "█" * count
                print(f"    {label:20} {count:3} {bar}")
    print()
    
    print("=" * 80)


def create_chart(categories, total_packages):
    """Create matplotlib chart"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Chart 1: Total packages
    labels = list(total_packages.keys())
    sizes = list(total_packages.values())
    colors = plt.cm.Set3(range(len(labels)))
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Total Packages by File', fontsize=14, fontweight='bold')
    
    # Chart 2: Categories
    category_data = {}
    for category, file_counts in categories.items():
        category_data[category] = sum(file_counts.values())
    
    cat_labels = list(category_data.keys())
    cat_sizes = list(category_data.values())
    cat_colors = plt.cm.Pastel1(range(len(cat_labels)))
    
    ax2.pie(cat_sizes, labels=cat_labels, autopct='%1.1f%%', colors=cat_colors, startangle=90)
    ax2.set_title('Packages by Category', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dependency-visualization.png', dpi=150, bbox_inches='tight')
    print("📊 Chart saved to: dependency-visualization.png")


def main():
    """Main function"""
    print("Analyzing dependencies...")
    categories, total_packages, files = analyze_dependencies()
    
    create_text_report(categories, total_packages, files)
    
    if HAS_MATPLOTLIB:
        try:
            create_chart(categories, total_packages)
        except Exception as e:
            print(f"Warning: Could not create chart: {e}")
    
    # Save JSON data
    data = {
        'categories': {k: dict(v) for k, v in categories.items()},
        'totals': dict(total_packages)
    }
    
    with open('dependency-visualization.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("\n📄 Data saved to: dependency-visualization.json")


if __name__ == '__main__':
    main()



