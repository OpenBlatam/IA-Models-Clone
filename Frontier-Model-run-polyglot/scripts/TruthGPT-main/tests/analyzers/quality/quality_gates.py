"""
Quality Gates
Define and check quality gates for test suite
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class QualityGate:
    """Quality gate definition"""
    
    def __init__(
        self,
        name: str,
        metric: str,
        operator: str,
        threshold: float,
        required: bool = True
    ):
        self.name = name
        self.metric = metric
        self.operator = operator  # <, >, <=, >=, ==
        self.threshold = threshold
        self.required = required
    
    def check(self, value: float) -> bool:
        """Check if gate passes"""
        if self.operator == '<':
            return value < self.threshold
        elif self.operator == '>':
            return value > self.threshold
        elif self.operator == '<=':
            return value <= self.threshold
        elif self.operator == '>=':
            return value >= self.threshold
        elif self.operator == '==':
            return abs(value - self.threshold) < 0.01
        return False

class QualityGatesSystem:
    """Quality gates system"""
    
    def __init__(self, project_root: Path, config_file: str = "quality_gates.json"):
        self.project_root = project_root
        self.config_file = project_root / config_file
        self.gates: List[QualityGate] = []
        self._load_gates()
    
    def _load_gates(self):
        """Load quality gates"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    for gate_config in config.get('gates', []):
                        gate = QualityGate(**gate_config)
                        self.gates.append(gate)
            except Exception:
                pass
        
        # Create default gates if none exist
        if not self.gates:
            self._create_default_gates()
    
    def _create_default_gates(self):
        """Create default quality gates"""
        self.gates = [
            QualityGate("Success Rate", "success_rate", ">=", 90.0, True),
            QualityGate("Execution Time", "execution_time", "<=", 300.0, False),
            QualityGate("Failure Count", "failure_count", "<=", 10, True),
            QualityGate("Error Count", "error_count", "<=", 5, True)
        ]
        self._save_gates()
    
    def _save_gates(self):
        """Save quality gates"""
        config = {
            'gates': [
                {
                    'name': g.name,
                    'metric': g.metric,
                    'operator': g.operator,
                    'threshold': g.threshold,
                    'required': g.required
                }
                for g in self.gates
            ]
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def add_gate(
        self,
        name: str,
        metric: str,
        operator: str,
        threshold: float,
        required: bool = True
    ):
        """Add quality gate"""
        gate = QualityGate(name, metric, operator, threshold, required)
        self.gates.append(gate)
        self._save_gates()
    
    def check_gates(self, test_results: Dict) -> Dict:
        """Check all quality gates"""
        results = []
        passed = 0
        failed = 0
        required_failed = 0
        
        metrics = {
            'success_rate': test_results.get('success_rate', 0),
            'execution_time': test_results.get('execution_time', 0),
            'failure_count': test_results.get('failures', 0),
            'error_count': test_results.get('errors', 0),
            'total_tests': test_results.get('total_tests', 0)
        }
        
        for gate in self.gates:
            value = metrics.get(gate.metric, 0)
            passed_check = gate.check(value)
            
            result = {
                'gate': gate.name,
                'metric': gate.metric,
                'value': value,
                'threshold': gate.threshold,
                'operator': gate.operator,
                'required': gate.required,
                'passed': passed_check
            }
            
            results.append(result)
            
            if passed_check:
                passed += 1
            else:
                failed += 1
                if gate.required:
                    required_failed += 1
        
        all_passed = required_failed == 0
        
        return {
            'all_passed': all_passed,
            'passed': passed,
            'failed': failed,
            'required_failed': required_failed,
            'total_gates': len(self.gates),
            'results': results
        }
    
    def generate_gates_report(self, check_results: Dict) -> str:
        """Generate quality gates report"""
        lines = []
        lines.append("=" * 80)
        lines.append("QUALITY GATES REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        status_emoji = "✅" if check_results['all_passed'] else "❌"
        lines.append(f"{status_emoji} Overall Status: {'PASSED' if check_results['all_passed'] else 'FAILED'}")
        lines.append(f"   Passed: {check_results['passed']}/{check_results['total_gates']}")
        lines.append(f"   Failed: {check_results['failed']}/{check_results['total_gates']}")
        lines.append(f"   Required Failed: {check_results['required_failed']}")
        lines.append("")
        
        lines.append("📋 GATE RESULTS")
        lines.append("-" * 80)
        
        for result in check_results['results']:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            required = " (REQUIRED)" if result['required'] else ""
            
            lines.append(f"{status} {result['gate']}{required}")
            lines.append(f"   Metric: {result['metric']}")
            lines.append(f"   Value: {result['value']} {result['operator']} {result['threshold']}")
            lines.append("")
        
        if not check_results['all_passed']:
            lines.append("⚠️  ACTION REQUIRED")
            lines.append("-" * 80)
            lines.append("Some quality gates failed. Please address the issues before proceeding.")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    gates_system = QualityGatesSystem(project_root)
    
    # Example test results
    test_results = {
        'total_tests': 204,
        'passed': 200,
        'failures': 2,
        'errors': 0,
        'skipped': 2,
        'success_rate': 98.0,
        'execution_time': 45.3
    }
    
    check_results = gates_system.check_gates(test_results)
    report = gates_system.generate_gates_report(check_results)
    
    print(report)
    
    # Save report
    report_file = project_root / "quality_gates_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Quality gates report saved to: {report_file}")

if __name__ == "__main__":
    main()







