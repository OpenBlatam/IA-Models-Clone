"""
BUL Security Audit Tool
======================

Comprehensive security audit tool for the BUL system.
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityAuditor:
    """Security audit tool for BUL system."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.recommendations = []
        self.security_score = 100
    
    def audit_file_permissions(self) -> Dict[str, Any]:
        """Audit file permissions and accessibility."""
        print("🔒 Auditing file permissions...")
        
        issues = []
        critical_files = [
            'bul_optimized.py',
            'config_optimized.py',
            'env_optimized.txt',
            '.env'
        ]
        
        for file_path in critical_files:
            if Path(file_path).exists():
                stat = Path(file_path).stat()
                mode = oct(stat.st_mode)[-3:]
                
                # Check if file is world-readable
                if mode[-1] in ['4', '5', '6', '7']:
                    issues.append({
                        'file': file_path,
                        'issue': 'World-readable file',
                        'severity': 'medium',
                        'mode': mode,
                        'recommendation': 'Set file permissions to 600 or 640'
                    })
                    self.security_score -= 10
        
        return {
            'component': 'file_permissions',
            'issues_found': len(issues),
            'issues': issues
        }
    
    def audit_environment_variables(self) -> Dict[str, Any]:
        """Audit environment variable security."""
        print("🌍 Auditing environment variables...")
        
        issues = []
        
        # Check for hardcoded secrets
        sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        # Check Python files for hardcoded secrets
        python_files = list(Path('.').glob('*.py'))
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in sensitive_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        issues.append({
                            'file': str(file_path),
                            'issue': 'Potential hardcoded secret',
                            'severity': 'high',
                            'match': match,
                            'recommendation': 'Use environment variables instead of hardcoded secrets'
                        })
                        self.security_score -= 15
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        # Check .env file
        env_file = Path('.env')
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                
                # Check for default/weak values
                weak_patterns = [
                    r'password\s*=\s*["\']?password["\']?',
                    r'secret\s*=\s*["\']?secret["\']?',
                    r'key\s*=\s*["\']?your.*key.*here["\']?'
                ]
                
                for pattern in weak_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append({
                            'file': '.env',
                            'issue': 'Weak default value',
                            'severity': 'high',
                            'recommendation': 'Change default values to strong, unique values'
                        })
                        self.security_score -= 20
            except Exception as e:
                logger.warning(f"Could not read .env file: {e}")
        
        return {
            'component': 'environment_variables',
            'issues_found': len(issues),
            'issues': issues
        }
    
    def audit_dependencies(self) -> Dict[str, Any]:
        """Audit dependencies for known vulnerabilities."""
        print("📦 Auditing dependencies...")
        
        issues = []
        
        # Check requirements file
        req_file = Path('requirements_optimized.txt')
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    content = f.read()
                
                # Check for pinned versions (good practice)
                lines = content.strip().split('\n')
                unpinned_deps = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '>=' in line or '==' in line or '~=' in line:
                            continue  # Pinned version
                        else:
                            unpinned_deps.append(line)
                
                if unpinned_deps:
                    issues.append({
                        'file': 'requirements_optimized.txt',
                        'issue': 'Unpinned dependencies',
                        'severity': 'medium',
                        'dependencies': unpinned_deps,
                        'recommendation': 'Pin dependency versions for reproducible builds'
                    })
                    self.security_score -= 5
                
                # Check for known vulnerable packages
                vulnerable_packages = [
                    'flask<1.0',
                    'django<2.0',
                    'requests<2.20'
                ]
                
                for vuln_pkg in vulnerable_packages:
                    if vuln_pkg.split('<')[0] in content.lower():
                        issues.append({
                            'file': 'requirements_optimized.txt',
                            'issue': 'Potentially vulnerable package',
                            'severity': 'high',
                            'package': vuln_pkg,
                            'recommendation': 'Update to latest secure version'
                        })
                        self.security_score -= 15
                        
            except Exception as e:
                logger.warning(f"Could not read requirements file: {e}")
        
        return {
            'component': 'dependencies',
            'issues_found': len(issues),
            'issues': issues
        }
    
    def audit_api_security(self) -> Dict[str, Any]:
        """Audit API security configurations."""
        print("🔐 Auditing API security...")
        
        issues = []
        
        # Check main application file
        main_file = Path('bul_optimized.py')
        if main_file.exists():
            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for CORS configuration
                if 'allow_origins=["*"]' in content:
                    issues.append({
                        'file': 'bul_optimized.py',
                        'issue': 'Overly permissive CORS',
                        'severity': 'medium',
                        'recommendation': 'Restrict CORS origins to specific domains'
                    })
                    self.security_score -= 10
                
                # Check for rate limiting
                if 'rate_limit' not in content.lower():
                    issues.append({
                        'file': 'bul_optimized.py',
                        'issue': 'No rate limiting detected',
                        'severity': 'medium',
                        'recommendation': 'Implement rate limiting to prevent abuse'
                    })
                    self.security_score -= 10
                
                # Check for input validation
                if 'validation' not in content.lower() and 'pydantic' not in content.lower():
                    issues.append({
                        'file': 'bul_optimized.py',
                        'issue': 'Limited input validation',
                        'severity': 'high',
                        'recommendation': 'Implement comprehensive input validation'
                    })
                    self.security_score -= 15
                
            except Exception as e:
                logger.warning(f"Could not read main application file: {e}")
        
        return {
            'component': 'api_security',
            'issues_found': len(issues),
            'issues': issues
        }
    
    def audit_data_handling(self) -> Dict[str, Any]:
        """Audit data handling and storage security."""
        print("💾 Auditing data handling...")
        
        issues = []
        
        # Check for sensitive data in logs
        log_files = list(Path('.').glob('*.log'))
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Check for potential secrets in logs
                secret_patterns = [
                    r'password[=:]\s*\S+',
                    r'secret[=:]\s*\S+',
                    r'key[=:]\s*\S+',
                    r'token[=:]\s*\S+'
                ]
                
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append({
                            'file': str(log_file),
                            'issue': 'Potential secret in log file',
                            'severity': 'high',
                            'recommendation': 'Remove sensitive data from logs'
                        })
                        self.security_score -= 20
            except Exception as e:
                logger.warning(f"Could not read log file {log_file}: {e}")
        
        # Check output directory permissions
        output_dir = Path('generated_documents')
        if output_dir.exists():
            stat = output_dir.stat()
            mode = oct(stat.st_mode)[-3:]
            
            if mode[-1] in ['6', '7']:  # World-writable
                issues.append({
                    'file': 'generated_documents/',
                    'issue': 'World-writable output directory',
                    'severity': 'high',
                    'mode': mode,
                    'recommendation': 'Restrict directory permissions'
                })
                self.security_score -= 15
        
        return {
            'component': 'data_handling',
            'issues_found': len(issues),
            'issues': issues
        }
    
    def audit_configuration(self) -> Dict[str, Any]:
        """Audit configuration security."""
        print("⚙️ Auditing configuration...")
        
        issues = []
        
        # Check configuration file
        config_file = Path('config_optimized.py')
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for default/weak configuration
                if 'your-secret-key-here' in content:
                    issues.append({
                        'file': 'config_optimized.py',
                        'issue': 'Default secret key',
                        'severity': 'high',
                        'recommendation': 'Generate a strong, unique secret key'
                    })
                    self.security_score -= 20
                
                # Check for debug mode in production
                if 'debug_mode: bool = True' in content:
                    issues.append({
                        'file': 'config_optimized.py',
                        'issue': 'Debug mode enabled by default',
                        'severity': 'medium',
                        'recommendation': 'Disable debug mode in production'
                    })
                    self.security_score -= 10
                
            except Exception as e:
                logger.warning(f"Could not read configuration file: {e}")
        
        return {
            'component': 'configuration',
            'issues_found': len(issues),
            'issues': issues
        }
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete security audit."""
        print("🔍 Starting BUL Security Audit")
        print("=" * 50)
        
        audit_results = {}
        
        # Run all audit components
        audit_results['file_permissions'] = self.audit_file_permissions()
        audit_results['environment_variables'] = self.audit_environment_variables()
        audit_results['dependencies'] = self.audit_dependencies()
        audit_results['api_security'] = self.audit_api_security()
        audit_results['data_handling'] = self.audit_data_handling()
        audit_results['configuration'] = self.audit_configuration()
        
        # Calculate overall security score
        self.security_score = max(0, self.security_score)
        
        # Generate recommendations
        self._generate_recommendations(audit_results)
        
        return {
            'audit_results': audit_results,
            'security_score': self.security_score,
            'total_issues': sum(r['issues_found'] for r in audit_results.values()),
            'recommendations': self.recommendations,
            'audit_timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, audit_results: Dict[str, Any]):
        """Generate security recommendations based on audit results."""
        self.recommendations = [
            "Implement comprehensive input validation for all API endpoints",
            "Use environment variables for all sensitive configuration",
            "Enable rate limiting to prevent abuse",
            "Implement proper logging without exposing sensitive data",
            "Regularly update dependencies to latest secure versions",
            "Use strong, unique secret keys and passwords",
            "Restrict file and directory permissions appropriately",
            "Implement proper error handling without information disclosure",
            "Use HTTPS in production environments",
            "Implement authentication and authorization as needed"
        ]
    
    def generate_report(self, audit_results: Dict[str, Any]) -> str:
        """Generate security audit report."""
        report = f"""
BUL Security Audit Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SECURITY SCORE
--------------
Overall Security Score: {audit_results['security_score']}/100
Total Issues Found: {audit_results['total_issues']}

AUDIT RESULTS
-------------
"""
        
        for component, results in audit_results['audit_results'].items():
            report += f"""
{component.replace('_', ' ').title()}:
  Issues Found: {results['issues_found']}
"""
            
            if results['issues']:
                for issue in results['issues']:
                    severity_icon = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢"
                    report += f"  {severity_icon} {issue['issue']} ({issue['severity']})\n"
                    report += f"     File: {issue['file']}\n"
                    report += f"     Recommendation: {issue['recommendation']}\n"
        
        report += f"""
SECURITY RECOMMENDATIONS
-----------------------
{chr(10).join(f"- {rec}" for rec in audit_results['recommendations'])}

NEXT STEPS
----------
1. Address high-severity issues immediately
2. Review and implement security recommendations
3. Run regular security audits
4. Keep dependencies updated
5. Monitor for new security vulnerabilities
"""
        
        return report

def main():
    """Main security audit function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Security Audit Tool")
    parser.add_argument("--component", choices=['all', 'files', 'env', 'deps', 'api', 'data', 'config'],
                       default='all', help="Component to audit")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor()
    
    if args.component == 'all':
        results = auditor.run_full_audit()
    else:
        # Run specific component audit
        if args.component == 'files':
            results = {'audit_results': {'file_permissions': auditor.audit_file_permissions()}}
        elif args.component == 'env':
            results = {'audit_results': {'environment_variables': auditor.audit_environment_variables()}}
        elif args.component == 'deps':
            results = {'audit_results': {'dependencies': auditor.audit_dependencies()}}
        elif args.component == 'api':
            results = {'audit_results': {'api_security': auditor.audit_api_security()}}
        elif args.component == 'data':
            results = {'audit_results': {'data_handling': auditor.audit_data_handling()}}
        elif args.component == 'config':
            results = {'audit_results': {'configuration': auditor.audit_configuration()}}
    
    # Display results
    print("\n" + "=" * 50)
    print("🔒 SECURITY AUDIT RESULTS")
    print("=" * 50)
    
    if 'security_score' in results:
        print(f"Security Score: {results['security_score']}/100")
        print(f"Total Issues: {results['total_issues']}")
    
    for component, data in results['audit_results'].items():
        print(f"\n{component.replace('_', ' ').title()}: {data['issues_found']} issues")
        
        for issue in data['issues']:
            severity_icon = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢"
            print(f"  {severity_icon} {issue['issue']} - {issue['file']}")
    
    # Generate report if requested
    if args.report:
        report = auditor.generate_report(results)
        report_file = f"security_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
