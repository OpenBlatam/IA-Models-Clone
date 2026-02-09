#!/usr/bin/env python3
"""
Dependency Dashboard
Interactive dashboard for dependency management
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("rich not available, using basic output")
    print("Install with: pip install rich")


def analyze_all_files():
    """Analyze all requirements files"""
    files = {}
    
    for req_file in Path('.').glob('requirements*.txt'):
        if req_file.name == 'requirements-lock.txt':
            continue
        
        packages = []
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-r'):
                    package = line.split('>=')[0].split('==')[0].split('[')[0].strip()
                    if package:
                        packages.append(package)
        
        files[req_file.name] = {
            'count': len(packages),
            'packages': packages
        }
    
    return files


def create_dashboard():
    """Create interactive dashboard"""
    if not HAS_RICH:
        print("Rich library required for dashboard")
        print("Install with: pip install rich")
        return
    
    console = Console()
    files = analyze_all_files()
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="files"),
        Layout(name="stats")
    )
    
    # Header
    layout["header"].update(
        Panel.fit(
            "[bold blue]Dependency Management Dashboard[/bold blue]",
            border_style="blue"
        )
    )
    
    # Files table
    files_table = Table(title="Requirements Files", show_header=True)
    files_table.add_column("File", style="cyan")
    files_table.add_column("Packages", justify="right", style="green")
    files_table.add_column("Status", style="yellow")
    
    for filename, data in files.items():
        status = "✓" if Path(filename).exists() else "✗"
        files_table.add_row(filename, str(data['count']), status)
    
    layout["files"].update(Panel(files_table, title="Files"))
    
    # Statistics
    total_packages = sum(data['count'] for data in files.values())
    unique_packages = len(set(
        pkg for data in files.values() for pkg in data['packages']
    ))
    
    stats_text = f"""
[bold]Total Packages:[/bold] {total_packages}
[bold]Unique Packages:[/bold] {unique_packages}
[bold]Files:[/bold] {len(files)}

[bold]Quick Commands:[/bold]
  make check          - Security check
  make outdated       - Check outdated
  make update         - Update packages
  make test           - Run tests
    """
    
    layout["stats"].update(Panel(stats_text, title="Statistics"))
    
    # Footer
    layout["footer"].update(
        Panel.fit(
            f"[dim]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="dim"
        )
    )
    
    console.print(layout)


def create_simple_dashboard():
    """Create simple text dashboard"""
    files = analyze_all_files()
    
    print("=" * 60)
    print("Dependency Management Dashboard")
    print("=" * 60)
    print()
    
    print("Requirements Files:")
    print("-" * 60)
    for filename, data in files.items():
        status = "✓" if Path(filename).exists() else "✗"
        print(f"  {status} {filename:30} {data['count']:4} packages")
    print()
    
    total = sum(data['count'] for data in files.values())
    unique = len(set(pkg for data in files.values() for pkg in data['packages']))
    
    print("Statistics:")
    print("-" * 60)
    print(f"  Total packages: {total}")
    print(f"  Unique packages: {unique}")
    print(f"  Files: {len(files)}")
    print()
    
    print("Quick Commands:")
    print("-" * 60)
    print("  make check          - Security check")
    print("  make outdated       - Check outdated")
    print("  make update         - Update packages")
    print("  make test           - Run tests")
    print()


def main():
    """Main function"""
    if HAS_RICH:
        create_dashboard()
    else:
        create_simple_dashboard()


if __name__ == '__main__':
    main()



