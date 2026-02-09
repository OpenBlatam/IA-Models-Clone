"""
Advanced Quality Gates
Advanced quality gate system with multiple criteria
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class AdvancedQualityGates:
    """Advanced quality gates"""
    
    DEFAULT_GATES = {
        'success_rate': {
            'threshold': 95,
            'operator': '>=',
            'weight': 0.3
        },
        'execution_time': {
            'threshold': 300,
            'operator': '<=',
            'weight': 0.2
        },
        'failure_rate': {
            'threshold': 5,
            'operator': '<=',
            'weight': 0.3
        },
        'stability': {
            'threshold': 80,
            'operator': '>=',
            'weight': 0.2
        }
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.gates_config_file = project_root / "quality_gates_config.json"
    
    def evaluate_gates(self, lookback_days: int = 30) -> Dict:
        """Evaluate quality gates"""
        history = self._load_history()
        gates = self._load_gates_config()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Calculate metrics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        failures = [r.get('failures', 0) + r.get('errors', 0) for r in recent]
        
        avg_success = mean(success_rates) if success_rates else 0
        avg_time = mean(execution_times) if execution_times else 0
        total_tests_sum = sum(total_tests)
        total_failures = sum(failures)
        failure_rate = (total_failures / total_tests_sum * 100) if total_tests_sum > 0 else 0
        
        # Calculate stability (simplified)
        sr_std = stdev(success_rates) if len(success_rates) > 1 else 0
        stability = max(0, 100 - (sr_std * 10))
        
        # Evaluate each gate
        gate_results = {}
        all_passed = True
        
        for gate_name, gate_config in gates.items():
            threshold = gate_config['threshold']
            operator = gate_config['operator']
            weight = gate_config.get('weight', 0.25)
            
            # Get metric value
            if gate_name == 'success_rate':
                value = avg_success
            elif gate_name == 'execution_time':
                value = avg_time
            elif gate_name == 'failure_rate':
                value = failure_rate
            elif gate_name == 'stability':
                value = stability
            else:
                continue
            
            # Evaluate gate
            passed = self._evaluate_gate(value, threshold, operator)
            if not passed:
                all_passed = False
            
            gate_results[gate_name] = {
                'passed': passed,
                'value': round(value, 2),
                'threshold': threshold,
                'operator': operator,
                'weight': weight
            }
        
        # Calculate overall score
        overall_score = sum(
            (100 if result['passed'] else 0) * result['weight']
            for result in gate_results.values()
        )
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'all_gates_passed': all_passed,
            'overall_score': round(overall_score, 1),
            'gate_results': gate_results,
            'recommendations': self._generate_recommendations(gate_results)
        }
    
    def _evaluate_gate(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate a single gate"""
        if operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '==':
            return abs(value - threshold) < 0.01
        return False
    
    def _generate_recommendations(self, gate_results: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        for gate_name, result in gate_results.items():
            if not result['passed']:
                recommendations.append(
                    f"{gate_name.replace('_', ' ').title()} gate failed: "
                    f"{result['value']} {result['operator']} {result['threshold']} (required)"
                )
        
        if not recommendations:
            recommendations.append("All quality gates passed - maintain current standards")
        
        return recommendations
    
    def _load_gates_config(self) -> Dict:
        """Load quality gates configuration"""
        if self.gates_config_file.exists():
            try:
                with open(self.gates_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Save default config
        with open(self.gates_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.DEFAULT_GATES, f, indent=2)
        
        return self.DEFAULT_GATES
    
    def generate_gates_report(self, evaluation: Dict) -> str:
        """Generate gates report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED QUALITY GATES REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in evaluation:
            lines.append(f"❌ {evaluation['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {evaluation['period']}")
        lines.append(f"Total Runs: {evaluation['total_runs']}")
        lines.append("")
        
        status_emoji = "✅" if evaluation['all_gates_passed'] else "❌"
        lines.append(f"{status_emoji} All Gates Passed: {'YES' if evaluation['all_gates_passed'] else 'NO'}")
        lines.append(f"Overall Score: {evaluation['overall_score']}/100")
        lines.append("")
        
        lines.append("🚪 QUALITY GATES")
        lines.append("-" * 80)
        for gate_name, result in evaluation['gate_results'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            lines.append(f"{status} {gate_name.replace('_', ' ').title()}")
            lines.append(f"   Value: {result['value']}")
            lines.append(f"   Threshold: {result['threshold']} ({result['operator']})")
            lines.append(f"   Weight: {result['weight'] * 100}%")
            lines.append("")
        
        if evaluation['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in evaluation['recommendations']:
                lines.append(f"• {rec}")
        
        return "\n".join(lines)
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

def main():
    """Main function"""
    from pathlib import Path
    from statistics import stdev
    
    project_root = Path(__file__).parent.parent
    
    gates = AdvancedQualityGates(project_root)
    evaluation = gates.evaluate_gates(lookback_days=30)
    
    report = gates.generate_gates_report(evaluation)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_quality_gates_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced quality gates report saved to: {report_file}")

if __name__ == "__main__":
    main()







