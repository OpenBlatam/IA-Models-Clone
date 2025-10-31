"""
BUL Toolkit - Master Control Script
===================================

Master script that provides access to all BUL system tools and utilities.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

class BULToolkit:
    """Master toolkit for BUL system management."""
    
    def __init__(self):
        self.tools = {
            'start': {
                'script': 'start_optimized.py',
                'description': 'Start the BUL system',
                'category': 'system'
            },
            'demo': {
                'script': 'demo_optimized.py',
                'description': 'Run system demonstration',
                'category': 'demo'
            },
            'test': {
                'script': 'test_optimized.py',
                'description': 'Run comprehensive tests',
                'category': 'testing'
            },
            'validate': {
                'script': 'validate_system.py',
                'description': 'Validate system integrity',
                'category': 'testing'
            },
            'install': {
                'script': 'install_optimized.py',
                'description': 'Install and setup system',
                'category': 'setup'
            },
            'monitor': {
                'script': 'monitor_system.py',
                'description': 'Monitor system performance',
                'category': 'monitoring'
            },
            'performance': {
                'script': 'performance_analyzer.py',
                'description': 'Analyze system performance',
                'category': 'monitoring'
            },
            'load-test': {
                'script': 'load_tester.py',
                'description': 'Run load tests',
                'category': 'testing'
            },
            'security': {
                'script': 'security_audit.py',
                'description': 'Run security audit',
                'category': 'security'
            },
            'cleanup': {
                'script': 'cleanup_final.py',
                'description': 'Clean up system files',
                'category': 'maintenance'
            },
            'deploy': {
                'script': 'deployment_manager.py',
                'description': 'Manage deployments (Docker, K8s, CI/CD)',
                'category': 'deployment'
            },
            'backup': {
                'script': 'backup_manager.py',
                'description': 'Backup and restore system',
                'category': 'maintenance'
            },
            'docs': {
                'script': 'api_documentation_generator.py',
                'description': 'Generate API documentation',
                'category': 'documentation'
            },
            'analytics': {
                'script': 'analytics_dashboard.py',
                'description': 'Real-time analytics dashboard',
                'category': 'monitoring'
            },
            'ai': {
                'script': 'ai_integration_manager.py',
                'description': 'AI integration management',
                'category': 'ai'
            },
            'workflow': {
                'script': 'workflow_automation.py',
                'description': 'Workflow automation system',
                'category': 'automation'
            },
            'cloud': {
                'script': 'cloud_integration_manager.py',
                'description': 'Cloud integration management',
                'category': 'cloud'
            },
            'notifications': {
                'script': 'notification_system.py',
                'description': 'Advanced notification system',
                'category': 'communication'
            },
            'ml': {
                'script': 'machine_learning_pipeline.py',
                'description': 'Machine learning pipeline',
                'category': 'ml'
            },
            'data': {
                'script': 'data_processing_engine.py',
                'description': 'Data processing engine',
                'category': 'data'
            },
            'bi': {
                'script': 'business_intelligence_dashboard.py',
                'description': 'Business intelligence dashboard',
                'category': 'bi'
            },
            'integration': {
                'script': 'enterprise_integration_hub.py',
                'description': 'Enterprise integration hub',
                'category': 'integration'
            },
            'security': {
                'script': 'advanced_security_manager.py',
                'description': 'Advanced security management',
                'category': 'security'
            },
            'compliance': {
                'script': 'compliance_manager.py',
                'description': 'Compliance management system',
                'category': 'compliance'
            },
            'governance': {
                'script': 'advanced_governance_manager.py',
                'description': 'Advanced governance management',
                'category': 'governance'
            }
        }
        
        self.categories = {
            'system': 'System Management',
            'demo': 'Demonstrations',
            'testing': 'Testing & Validation',
            'setup': 'Installation & Setup',
            'monitoring': 'Performance Monitoring',
            'security': 'Security & Auditing',
            'maintenance': 'Maintenance',
            'deployment': 'Deployment & DevOps',
            'documentation': 'Documentation',
            'ai': 'AI Integration',
            'automation': 'Workflow Automation',
            'cloud': 'Cloud Integration',
            'communication': 'Communication & Notifications',
            'ml': 'Machine Learning',
            'data': 'Data Processing',
            'bi': 'Business Intelligence',
            'integration': 'Enterprise Integration',
            'compliance': 'Compliance Management',
            'governance': 'Advanced Governance'
        }
    
    def list_tools(self, category: str = None):
        """List available tools."""
        print("🛠️  BUL Toolkit - Available Tools")
        print("=" * 50)
        
        if category and category in self.categories:
            print(f"\n📁 {self.categories[category]}:")
            print("-" * 30)
            for tool_name, tool_info in self.tools.items():
                if tool_info['category'] == category:
                    print(f"  {tool_name:<15} - {tool_info['description']}")
        else:
            # Group by category
            for cat_name, cat_desc in self.categories.items():
                print(f"\n📁 {cat_desc}:")
                print("-" * 30)
                for tool_name, tool_info in self.tools.items():
                    if tool_info['category'] == cat_name:
                        print(f"  {tool_name:<15} - {tool_info['description']}")
        
        print(f"\n💡 Usage: python bul_toolkit.py <tool_name> [options]")
        print(f"💡 Example: python bul_toolkit.py start --debug")
    
    def run_tool(self, tool_name: str, args: List[str] = None):
        """Run a specific tool."""
        if tool_name not in self.tools:
            print(f"❌ Unknown tool: {tool_name}")
            print(f"💡 Available tools: {', '.join(self.tools.keys())}")
            return 1
        
        tool_info = self.tools[tool_name]
        script_path = Path(tool_info['script'])
        
        if not script_path.exists():
            print(f"❌ Tool script not found: {script_path}")
            return 1
        
        print(f"🚀 Running {tool_name}: {tool_info['description']}")
        print("=" * 50)
        
        # Prepare command
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        try:
            # Run the tool
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except KeyboardInterrupt:
            print("\n⏹️  Tool execution interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error running tool: {e}")
            return 1
    
    def system_status(self):
        """Check system status."""
        print("📊 BUL System Status")
        print("=" * 30)
        
        # Check if main files exist
        critical_files = [
            'bul_optimized.py',
            'config_optimized.py',
            'modules/__init__.py',
            'requirements_optimized.txt'
        ]
        
        print("📁 Critical Files:")
        for file_path in critical_files:
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path}")
        
        # Check modules
        print("\n🧩 Modules:")
        modules = [
            'modules/document_processor.py',
            'modules/query_analyzer.py',
            'modules/business_agents.py',
            'modules/api_handler.py'
        ]
        
        for module in modules:
            if Path(module).exists():
                print(f"  ✅ {module}")
            else:
                print(f"  ❌ {module}")
        
        # Check tools
        print("\n🛠️  Tools:")
        for tool_name, tool_info in self.tools.items():
            script_path = Path(tool_info['script'])
            if script_path.exists():
                print(f"  ✅ {tool_name} - {tool_info['description']}")
            else:
                print(f"  ❌ {tool_name} - {tool_info['description']}")
    
    def quick_setup(self):
        """Quick setup wizard."""
        print("🚀 BUL Quick Setup Wizard")
        print("=" * 40)
        
        steps = [
            ("Install Dependencies", "install"),
            ("Validate System", "validate"),
            ("Run Demo", "demo"),
            ("Start System", "start")
        ]
        
        for step_name, tool_name in steps:
            print(f"\n📋 {step_name}...")
            response = input(f"   Run {tool_name}? (y/N): ").lower().strip()
            
            if response == 'y':
                result = self.run_tool(tool_name)
                if result != 0:
                    print(f"   ⚠️  {step_name} completed with issues")
                else:
                    print(f"   ✅ {step_name} completed successfully")
            else:
                print(f"   ⏭️  Skipped {step_name}")
        
        print("\n🎉 Quick setup completed!")
        print("💡 Use 'python bul_toolkit.py start' to start the system")
    
    def help(self):
        """Show help information."""
        print("""
🛠️  BUL Toolkit - Master Control Script
=======================================

The BUL Toolkit provides a unified interface to all BUL system tools and utilities.

USAGE:
  python bul_toolkit.py <command> [options]

COMMANDS:
  list [category]     - List available tools (optionally by category)
  run <tool> [args]   - Run a specific tool with arguments
  status              - Check system status
  setup               - Run quick setup wizard
  help                - Show this help message

CATEGORIES:
  system              - System management tools
  demo                - Demonstration tools
  testing             - Testing and validation tools
  setup               - Installation and setup tools
  monitoring          - Performance monitoring tools
  security            - Security and auditing tools
  maintenance         - Maintenance tools

EXAMPLES:
  python bul_toolkit.py list                    # List all tools
  python bul_toolkit.py list testing            # List testing tools
  python bul_toolkit.py run start --debug       # Start system in debug mode
  python bul_toolkit.py run test                # Run tests
  python bul_toolkit.py run monitor --interval 10  # Monitor with 10s interval
  python bul_toolkit.py status                  # Check system status
  python bul_toolkit.py setup                   # Run setup wizard

TOOLS:
""")
        
        # Show tools by category
        for cat_name, cat_desc in self.categories.items():
            print(f"  {cat_desc}:")
            for tool_name, tool_info in self.tools.items():
                if tool_info['category'] == cat_name:
                    print(f"    {tool_name:<15} - {tool_info['description']}")
            print()

def main():
    """Main toolkit function."""
    parser = argparse.ArgumentParser(
        description="BUL Toolkit - Master Control Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available tools')
    list_parser.add_argument('category', nargs='?', help='Filter by category')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a specific tool')
    run_parser.add_argument('tool', help='Tool name to run')
    run_parser.add_argument('args', nargs='*', help='Tool arguments')
    
    # Status command
    subparsers.add_parser('status', help='Check system status')
    
    # Setup command
    subparsers.add_parser('setup', help='Run quick setup wizard')
    
    # Help command
    subparsers.add_parser('help', help='Show help information')
    
    args = parser.parse_args()
    
    toolkit = BULToolkit()
    
    if not args.command:
        # No command specified, show help
        toolkit.help()
        return 0
    
    if args.command == 'list':
        toolkit.list_tools(args.category)
    elif args.command == 'run':
        return toolkit.run_tool(args.tool, args.args)
    elif args.command == 'status':
        toolkit.system_status()
    elif args.command == 'setup':
        toolkit.quick_setup()
    elif args.command == 'help':
        toolkit.help()
    else:
        print(f"❌ Unknown command: {args.command}")
        toolkit.help()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
