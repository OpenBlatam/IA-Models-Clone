"""
Anomaly Detector
Detects anomalies in test results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from statistics import mean, stdev
from datetime import datetime

class AnomalyDetector:
    """Detect anomalies in test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def detect_anomalies(self, lookback: int = 20) -> Dict:
        """Detect anomalies in test results"""
        history = self._load_history()
        
        if len(history) < lookback:
            return {'error': f'Need at least {lookback} runs, found {len(history)}'}
        
        recent = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:lookback]
        recent.reverse()  # Oldest first
        
        # Calculate statistics
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        total_tests = [r.get('total_tests', 0) for r in recent]
        
        avg_success = mean(success_rates)
        std_success = stdev(success_rates) if len(success_rates) > 1 else 0
        
        avg_time = mean(execution_times)
        std_time = stdev(execution_times) if len(execution_times) > 1 else 0
        
        avg_tests = mean(total_tests)
        std_tests = stdev(total_tests) if len(total_tests) > 1 else 0
        
        # Detect anomalies (values > 2 standard deviations from mean)
        anomalies = []
        
        for i, run in enumerate(recent):
            anomalies_in_run = []
            
            # Success rate anomaly
            if std_success > 0:
                z_score = abs((run.get('success_rate', 0) - avg_success) / std_success)
                if z_score > 2:
                    anomalies_in_run.append({
                        'type': 'success_rate',
                        'value': run.get('success_rate', 0),
                        'expected': avg_success,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
            
            # Execution time anomaly
            if std_time > 0:
                z_score = abs((run.get('execution_time', 0) - avg_time) / std_time)
                if z_score > 2:
                    anomalies_in_run.append({
                        'type': 'execution_time',
                        'value': run.get('execution_time', 0),
                        'expected': avg_time,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
            
            # Total tests anomaly
            if std_tests > 0:
                z_score = abs((run.get('total_tests', 0) - avg_tests) / std_tests)
                if z_score > 2:
                    anomalies_in_run.append({
                        'type': 'total_tests',
                        'value': run.get('total_tests', 0),
                        'expected': avg_tests,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
            
            if anomalies_in_run:
                anomalies.append({
                    'run': run.get('run_name', f'run_{i}'),
                    'timestamp': run.get('timestamp'),
                    'anomalies': anomalies_in_run
                })
        
        return {
            'total_anomalies': len(anomalies),
            'anomalies': anomalies,
            'statistics': {
                'avg_success_rate': avg_success,
                'std_success_rate': std_success,
                'avg_execution_time': avg_time,
                'std_execution_time': std_time,
                'avg_total_tests': avg_tests,
                'std_total_tests': std_tests
            }
        }
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def generate_anomaly_report(self, analysis: Dict) -> str:
        """Generate anomaly detection report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ANOMALY DETECTION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Total Anomalies Detected: {analysis['total_anomalies']}")
        lines.append("")
        
        if not analysis['anomalies']:
            lines.append("✅ No anomalies detected!")
            lines.append("All test runs are within expected ranges.")
            return "\n".join(lines)
        
        lines.append("🔴 DETECTED ANOMALIES")
        lines.append("-" * 80)
        
        for anomaly_group in analysis['anomalies']:
            lines.append(f"\nRun: {anomaly_group['run']}")
            lines.append(f"Timestamp: {anomaly_group['timestamp']}")
            lines.append("Anomalies:")
            
            for anomaly in anomaly_group['anomalies']:
                severity_emoji = "🔴" if anomaly['severity'] == 'high' else "🟡"
                lines.append(f"  {severity_emoji} {anomaly['type']}:")
                lines.append(f"     Value: {anomaly['value']:.2f}")
                lines.append(f"     Expected: {anomaly['expected']:.2f}")
                lines.append(f"     Z-Score: {anomaly['z_score']:.2f}")
                lines.append(f"     Severity: {anomaly['severity'].upper()}")
        
        lines.append("")
        lines.append("📊 STATISTICS")
        lines.append("-" * 80)
        stats = analysis['statistics']
        lines.append(f"Average Success Rate:    {stats['avg_success_rate']:.1f}% (±{stats['std_success_rate']:.1f}%)")
        lines.append(f"Average Execution Time:   {stats['avg_execution_time']:.2f}s (±{stats['std_execution_time']:.2f}s)")
        lines.append(f"Average Total Tests:      {stats['avg_total_tests']:.1f} (±{stats['std_total_tests']:.1f})")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    detector = AnomalyDetector(project_root)
    analysis = detector.detect_anomalies(lookback=20)
    
    report = detector.generate_anomaly_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "anomaly_detection_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Anomaly report saved to: {report_file}")

if __name__ == "__main__":
    main()







