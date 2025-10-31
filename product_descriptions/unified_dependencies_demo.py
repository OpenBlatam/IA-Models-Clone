from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import structlog
from unified_dependencies_manager import (
        import platform
                    import random
        import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional
"""
Unified Dependencies Management Demo

This demo showcases comprehensive dependency management capabilities:

- Dependency analysis and reporting
- Environment validation
- Automatic dependency installation
- Conflict resolution
- Security vulnerability scanning
- Performance impact analysis
- Requirements file generation
- Real-world dependency management scenarios
"""



    UnifiedDependenciesManager, DependencyCategory, DependencyPriority,
    create_requirements_file, install_group_dependencies, validate_system_dependencies
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class UnifiedDependenciesDemo:
    """Comprehensive demo for unified dependencies management."""
    
    def __init__(self) -> Any:
        self.results = {}
        self.manager = UnifiedDependenciesManager()
        
    async def run_dependency_analysis_demo(self) -> Dict:
        """Demonstrate comprehensive dependency analysis."""
        logger.info("Starting Dependency Analysis Demo")
        
        # Get comprehensive dependency report
        report = self.manager.get_dependency_report()
        
        # Analyze by category
        category_analysis = {}
        for category, deps in report['by_category'].items():
            total = len(deps)
            installed = sum(1 for d in deps if d['installed'])
            critical = sum(1 for d in deps if d['priority'] == 'critical')
            
            category_analysis[category] = {
                'total': total,
                'installed': installed,
                'missing': total - installed,
                'critical': critical,
                'installation_rate': installed / total if total > 0 else 0
            }
        
        # Analyze by priority
        priority_analysis = {}
        for dep in self.manager.dependencies.values():
            priority = dep.priority.value
            if priority not in priority_analysis:
                priority_analysis[priority] = {'total': 0, 'installed': 0}
            
            priority_analysis[priority]['total'] += 1
            if self.manager.check_dependency_installed(dep.name):
                priority_analysis[priority]['installed'] += 1
        
        # Calculate installation rates
        for priority in priority_analysis:
            total = priority_analysis[priority]['total']
            installed = priority_analysis[priority]['installed']
            priority_analysis[priority]['rate'] = installed / total if total > 0 else 0
        
        return {
            'summary': report['summary'],
            'category_analysis': category_analysis,
            'priority_analysis': priority_analysis,
            'missing_dependencies': report['missing_dependencies'],
            'outdated_dependencies': report['outdated_dependencies'],
            'conflicts': report['conflicts'],
            'vulnerabilities': report['vulnerabilities']
        }
    
    async def run_environment_validation_demo(self) -> Dict:
        """Demonstrate environment validation capabilities."""
        logger.info("Starting Environment Validation Demo")
        
        # Validate current environment
        validation = self.manager.validate_environment()
        
        # Check each group separately
        group_validations = {}
        for group_name in self.manager.groups.keys():
            missing = self.manager.get_missing_dependencies(group_name)
            group_validations[group_name] = {
                'missing_count': len(missing),
                'is_valid': len(missing) == 0,
                'missing_dependencies': [
                    {
                        'name': dep.name,
                        'version': dep.version,
                        'priority': dep.priority.value,
                        'description': dep.description
                    }
                    for dep in missing
                ]
            }
        
        # Check platform-specific dependencies
        platform_analysis = self._analyze_platform_dependencies()
        
        return {
            'overall_validation': validation,
            'group_validations': group_validations,
            'platform_analysis': platform_analysis
        }
    
    def _analyze_platform_dependencies(self) -> Dict:
        """Analyze platform-specific dependencies."""
        
        current_platform = platform.system().lower()
        platform_analysis = {
            'current_platform': current_platform,
            'platform_specific_deps': [],
            'compatible_deps': [],
            'incompatible_deps': []
        }
        
        for dep in self.manager.dependencies.values():
            if PlatformType.ALL in dep.platforms:
                platform_analysis['compatible_deps'].append(dep.name)
            elif any(p.value == current_platform for p in dep.platforms):
                platform_analysis['platform_specific_deps'].append(dep.name)
            else:
                platform_analysis['incompatible_deps'].append(dep.name)
        
        return platform_analysis
    
    async def run_requirements_generation_demo(self) -> Dict:
        """Demonstrate requirements file generation."""
        logger.info("Starting Requirements Generation Demo")
        
        generated_files = {}
        
        # Generate requirements for each group
        for group_name in self.manager.groups.keys():
            requirements = self.manager.generate_requirements_file(group_name)
            filename = f"requirements-{group_name}.txt"
            
            with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(requirements)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            generated_files[group_name] = {
                'filename': filename,
                'dependency_count': len(requirements.strip().split('\n')) if requirements.strip() else 0,
                'content': requirements
            }
        
        # Generate comprehensive requirements file
        all_requirements = self.manager.generate_requirements_file(include_optional=False)
        with open("requirements-all.txt", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(all_requirements)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        generated_files['all'] = {
            'filename': 'requirements-all.txt',
            'dependency_count': len(all_requirements.strip().split('\n')) if all_requirements.strip() else 0,
            'content': all_requirements
        }
        
        # Generate requirements with optional dependencies
        optional_requirements = self.manager.generate_requirements_file(include_optional=True)
        with open("requirements-optional.txt", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(optional_requirements)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        generated_files['optional'] = {
            'filename': 'requirements-optional.txt',
            'dependency_count': len(optional_requirements.strip().split('\n')) if optional_requirements.strip() else 0,
            'content': optional_requirements
        }
        
        return {
            'generated_files': generated_files,
            'total_files': len(generated_files)
        }
    
    async def run_performance_impact_analysis_demo(self) -> Dict:
        """Demonstrate performance impact analysis of dependencies."""
        logger.info("Starting Performance Impact Analysis Demo")
        
        # Analyze performance impact by category
        performance_analysis = {}
        
        for category in DependencyCategory:
            category_deps = [d for d in self.manager.dependencies.values() 
                           if d.category == category]
            
            if category_deps:
                impact_counts = {'low': 0, 'medium': 0, 'high': 0}
                memory_counts = {'low': 0, 'medium': 0, 'high': 0}
                cpu_counts = {'low': 0, 'medium': 0, 'high': 0}
                
                for dep in category_deps:
                    impact_counts[dep.performance_impact] += 1
                    memory_counts[dep.memory_usage] += 1
                    cpu_counts[dep.cpu_usage] += 1
                
                performance_analysis[category.value] = {
                    'total_dependencies': len(category_deps),
                    'performance_impact': impact_counts,
                    'memory_usage': memory_counts,
                    'cpu_usage': cpu_counts,
                    'high_impact_deps': [
                        dep.name for dep in category_deps 
                        if dep.performance_impact == 'high'
                    ],
                    'high_memory_deps': [
                        dep.name for dep in category_deps 
                        if dep.memory_usage == 'high'
                    ],
                    'high_cpu_deps': [
                        dep.name for dep in category_deps 
                        if dep.cpu_usage == 'high'
                    ]
                }
        
        # Analyze installed dependencies performance
        installed_performance = {
            'total_installed': 0,
            'high_performance_impact': 0,
            'high_memory_usage': 0,
            'high_cpu_usage': 0,
            'performance_heavy_deps': []
        }
        
        for dep in self.manager.dependencies.values():
            if self.manager.check_dependency_installed(dep.name):
                installed_performance['total_installed'] += 1
                
                if dep.performance_impact == 'high':
                    installed_performance['high_performance_impact'] += 1
                    installed_performance['performance_heavy_deps'].append(dep.name)
                
                if dep.memory_usage == 'high':
                    installed_performance['high_memory_usage'] += 1
                
                if dep.cpu_usage == 'high':
                    installed_performance['high_cpu_usage'] += 1
        
        return {
            'category_performance': performance_analysis,
            'installed_performance': installed_performance
        }
    
    async def run_security_analysis_demo(self) -> Dict:
        """Demonstrate security vulnerability analysis."""
        logger.info("Starting Security Analysis Demo")
        
        # Check for security vulnerabilities
        vulnerabilities = self.manager.check_security_vulnerabilities()
        
        # Analyze by severity
        severity_analysis = {'high': 0, 'medium': 0, 'low': 0}
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'medium')
            severity_analysis[severity] += 1
        
        # Check for outdated dependencies (security risk)
        outdated = self.manager.get_outdated_dependencies()
        security_outdated = [
            (dep, current, latest) for dep, current, latest in outdated
            if dep.priority in [DependencyPriority.CRITICAL, DependencyPriority.HIGH]
        ]
        
        # Check for conflicts (potential security issues)
        conflicts = self.manager.check_dependency_conflicts()
        
        return {
            'vulnerabilities': vulnerabilities,
            'severity_analysis': severity_analysis,
            'security_outdated': [
                {
                    'name': dep.name,
                    'current_version': current,
                    'latest_version': latest,
                    'priority': dep.priority.value,
                    'description': dep.description
                }
                for dep, current, latest in security_outdated
            ],
            'conflicts': conflicts,
            'security_score': self._calculate_security_score(vulnerabilities, security_outdated, conflicts)
        }
    
    def _calculate_security_score(self, vulnerabilities: List, outdated: List, conflicts: List) -> float:
        """Calculate security score (0-100, higher is better)."""
        score = 100.0
        
        # Deduct points for vulnerabilities
        score -= len(vulnerabilities) * 10
        
        # Deduct points for outdated critical dependencies
        score -= len(outdated) * 5
        
        # Deduct points for conflicts
        score -= len(conflicts) * 3
        
        return max(0.0, score)
    
    async def run_installation_simulation_demo(self) -> Dict:
        """Demonstrate dependency installation simulation."""
        logger.info("Starting Installation Simulation Demo")
        
        # Simulate installation for each group
        installation_simulation = {}
        
        for group_name in self.manager.groups.keys():
            missing = self.manager.get_missing_dependencies(group_name)
            
            if missing:
                # Simulate installation
                simulation_results = {}
                for dep in missing:
                    # Simulate success/failure based on priority
                    success_rate = {
                        DependencyPriority.CRITICAL: 0.95,
                        DependencyPriority.HIGH: 0.90,
                        DependencyPriority.MEDIUM: 0.85,
                        DependencyPriority.LOW: 0.80,
                        DependencyPriority.OPTIONAL: 0.75
                    }
                    
                    success = random.random() < success_rate[dep.priority]
                    simulation_results[dep.name] = success
                
                installation_simulation[group_name] = {
                    'missing_count': len(missing),
                    'simulation_results': simulation_results,
                    'success_count': sum(1 for success in simulation_results.values() if success),
                    'failure_count': sum(1 for success in simulation_results.values() if not success),
                    'success_rate': sum(1 for success in simulation_results.values() if success) / len(simulation_results) if simulation_results else 0
                }
            else:
                installation_simulation[group_name] = {
                    'missing_count': 0,
                    'simulation_results': {},
                    'success_count': 0,
                    'failure_count': 0,
                    'success_rate': 1.0
                }
        
        return installation_simulation
    
    async def run_comprehensive_demo(self) -> Dict:
        """Run comprehensive dependencies management demo."""
        logger.info("Starting Comprehensive Unified Dependencies Management Demo")
        
        results = {}
        
        try:
            # Run individual demos
            results['dependency_analysis'] = await self.run_dependency_analysis_demo()
            results['environment_validation'] = await self.run_environment_validation_demo()
            results['requirements_generation'] = await self.run_requirements_generation_demo()
            results['performance_analysis'] = await self.run_performance_impact_analysis_demo()
            results['security_analysis'] = await self.run_security_analysis_demo()
            results['installation_simulation'] = await self.run_installation_simulation_demo()
            
            # Generate comprehensive report
            comprehensive_report = self._generate_comprehensive_report(results)
            results['comprehensive_report'] = comprehensive_report
            
            # Save results
            self._save_results(results)
            
            # Generate visualizations
            self.plot_results(results)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_comprehensive_report(self, results: Dict) -> Dict:
        """Generate comprehensive report from all demo results."""
        report = {
            'summary': {},
            'recommendations': [],
            'action_items': [],
            'risk_assessment': {}
        }
        
        # Summary statistics
        dep_analysis = results['dependency_analysis']
        summary = dep_analysis['summary']
        
        report['summary'] = {
            'total_dependencies': summary['total_dependencies'],
            'installed_dependencies': summary['installed_dependencies'],
            'missing_dependencies': summary['missing_dependencies'],
            'outdated_dependencies': summary['outdated_dependencies'],
            'conflicts': summary['conflicts'],
            'vulnerabilities': summary['vulnerabilities'],
            'installation_rate': summary['installation_rate']
        }
        
        # Generate recommendations
        if summary['missing_dependencies'] > 0:
            report['recommendations'].append(
                f"Install {summary['missing_dependencies']} missing dependencies"
            )
        
        if summary['outdated_dependencies'] > 0:
            report['recommendations'].append(
                f"Update {summary['outdated_dependencies']} outdated dependencies"
            )
        
        if summary['conflicts'] > 0:
            report['recommendations'].append(
                f"Resolve {summary['conflicts']} dependency conflicts"
            )
        
        if summary['vulnerabilities'] > 0:
            report['recommendations'].append(
                f"Address {summary['vulnerabilities']} security vulnerabilities"
            )
        
        # Action items
        for group_name, validation in results['environment_validation']['group_validations'].items():
            if validation['missing_count'] > 0:
                report['action_items'].append({
                    'action': f"Install missing dependencies for group '{group_name}'",
                    'priority': 'high' if group_name in ['core', 'profiling'] else 'medium',
                    'count': validation['missing_count']
                })
        
        # Risk assessment
        security_analysis = results['security_analysis']
        report['risk_assessment'] = {
            'security_score': security_analysis['security_score'],
            'vulnerability_count': len(security_analysis['vulnerabilities']),
            'outdated_critical': len(security_analysis['security_outdated']),
            'conflict_count': len(security_analysis['conflicts']),
            'overall_risk': 'high' if security_analysis['security_score'] < 70 else 'medium' if security_analysis['security_score'] < 90 else 'low'
        }
        
        return report
    
    def plot_results(self, results: Dict, save_path: str = "dependencies_analysis_results.png"):
        """Plot comprehensive results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Dependency Installation Status
        dep_analysis = results['dependency_analysis']
        summary = dep_analysis['summary']
        
        labels = ['Installed', 'Missing', 'Outdated']
        sizes = [summary['installed_dependencies'], summary['missing_dependencies'], summary['outdated_dependencies']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Dependency Installation Status')
        
        # Plot 2: Dependencies by Category
        category_analysis = dep_analysis['category_analysis']
        categories = list(category_analysis.keys())
        installed_counts = [cat['installed'] for cat in category_analysis.values()]
        total_counts = [cat['total'] for cat in category_analysis.values()]
        
        x = range(len(categories))
        axes[0, 1].bar(x, total_counts, label='Total', alpha=0.7)
        axes[0, 1].bar(x, installed_counts, label='Installed', alpha=0.9)
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Dependencies by Category')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(categories, rotation=45)
        axes[0, 1].legend()
        
        # Plot 3: Performance Impact Analysis
        perf_analysis = results['performance_analysis']
        installed_perf = perf_analysis['installed_performance']
        
        impact_labels = ['Low', 'Medium', 'High']
        impact_counts = [
            installed_perf['total_installed'] - installed_perf['high_performance_impact'],
            installed_perf['high_performance_impact'] - (installed_perf['high_memory_usage'] + installed_perf['high_cpu_usage']),
            installed_perf['high_memory_usage'] + installed_perf['high_cpu_usage']
        ]
        
        axes[0, 2].pie(impact_counts, labels=impact_labels, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Performance Impact Distribution')
        
        # Plot 4: Security Analysis
        security_analysis = results['security_analysis']
        severity_analysis = security_analysis['severity_analysis']
        
        severity_labels = list(severity_analysis.keys())
        severity_counts = list(severity_analysis.values())
        severity_colors = ['#e74c3c', '#f39c12', '#2ecc71']
        
        axes[1, 0].bar(severity_labels, severity_counts, color=severity_colors)
        axes[1, 0].set_title('Security Vulnerabilities by Severity')
        axes[1, 0].set_ylabel('Count')
        
        # Plot 5: Installation Simulation Results
        install_sim = results['installation_simulation']
        groups = list(install_sim.keys())
        success_rates = [group['success_rate'] for group in install_sim.values()]
        
        axes[1, 1].bar(groups, success_rates, color='#3498db')
        axes[1, 1].set_title('Installation Success Rates by Group')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_xticklabels(groups, rotation=45)
        axes[1, 1].set_ylim(0, 1)
        
        # Plot 6: Security Score
        security_score = security_analysis['security_score']
        
        axes[1, 2].pie([security_score, 100-security_score], 
                      labels=['Secure', 'Risk'], 
                      colors=['#2ecc71', '#e74c3c'],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title(f'Security Score: {security_score:.1f}/100')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results plot saved to {save_path}")
    
    def _save_results(self, results: Dict):
        """Save demo results to file."""
        output_path = Path("unified_dependencies_results.json")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj) -> Any:
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return str(obj)
        
        serializable_results = convert_numpy(results)
        
        with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Demo results saved to {output_path}")


async def main():
    """Main demo function."""
    logger.info("Unified Dependencies Management Demo")
    
    # Create demo instance
    demo = UnifiedDependenciesDemo()
    
    # Run comprehensive demo
    results = await demo.run_comprehensive_demo()
    
    # Print summary
    logger.info("Demo completed successfully!")
    
    if 'comprehensive_report' in results:
        report = results['comprehensive_report']
        logger.info("Dependencies Summary:")
        logger.info(f"  Total Dependencies: {report['summary']['total_dependencies']}")
        logger.info(f"  Installed: {report['summary']['installed_dependencies']}")
        logger.info(f"  Missing: {report['summary']['missing_dependencies']}")
        logger.info(f"  Security Score: {report['risk_assessment']['security_score']:.1f}/100")
        
        if report['recommendations']:
            logger.info("Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 