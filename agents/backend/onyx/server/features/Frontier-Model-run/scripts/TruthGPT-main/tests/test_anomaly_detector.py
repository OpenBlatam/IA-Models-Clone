"""
Test Result Anomaly Detector
Detects anomalies in test results using statistical methods
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class TestAnomalyDetector:
    """Detect anomalies in test execution results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_result_history.json"
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load test result history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """Save test result history"""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
    
    def record_result(self, result: Dict):
        """Record a test result"""
        result['timestamp'] = datetime.now().isoformat()
        self.history.append(result)
        
        # Keep last 500 results
        if len(self.history) > 500:
            self.history = self.history[-500:]
        
        self._save_history()
    
    def _calculate_statistics(self, values: List[float]) -> Dict:
        """Calculate statistical measures"""
        if not values:
            return {}
        
        mean = statistics.mean(values)
        median = statistics.median(values)
        
        if len(values) > 1:
            stdev = statistics.stdev(values)
        else:
            stdev = 0
        
        return {
            'mean': mean,
            'median': median,
            'stdev': stdev,
            'min': min(values),
            'max': max(values)
        }
    
    def detect_duration_anomalies(self, test_name: str = None, threshold: float = 2.0) -> List[Dict]:
        """Detect tests with anomalous execution duration"""
        # Group by test name
        by_test = defaultdict(list)
        for record in self.history:
            test = record.get('test_name', 'unknown')
            if test_name and test != test_name:
                continue
            
            duration = record.get('duration', 0)
            if duration > 0:
                by_test[test].append(duration)
        
        anomalies = []
        
        for test, durations in by_test.items():
            if len(durations) < 3:  # Need at least 3 data points
                continue
            
            stats = self._calculate_statistics(durations)
            mean = stats['mean']
            stdev = stats['stdev']
            
            # Check latest duration
            latest = durations[-1]
            
            # Z-score
            if stdev > 0:
                z_score = abs((latest - mean) / stdev)
            else:
                z_score = 0
            
            if z_score > threshold:
                anomalies.append({
                    'test_name': test,
                    'type': 'duration_anomaly',
                    'latest_duration': latest,
                    'average_duration': mean,
                    'z_score': round(z_score, 2),
                    'severity': 'high' if z_score > 3 else 'medium',
                    'message': f"Duration {latest:.2f}s is {z_score:.1f}σ from mean ({mean:.2f}s)"
                })
        
        return sorted(anomalies, key=lambda x: x['z_score'], reverse=True)
    
    def detect_failure_spikes(self, window_days: int = 7) -> List[Dict]:
        """Detect sudden spikes in test failures"""
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = [
            r for r in self.history
            if datetime.fromisoformat(r.get('timestamp', '2000-01-01')) > cutoff
        ]
        
        # Group by test and day
        by_test_day = defaultdict(lambda: defaultdict(int))
        
        for record in recent:
            test = record.get('test_name', 'unknown')
            timestamp = datetime.fromisoformat(record.get('timestamp', '2000-01-01'))
            day = timestamp.date()
            
            if record.get('status') in ('failed', 'error'):
                by_test_day[test][day] += 1
        
        spikes = []
        
        for test, day_counts in by_test_day.items():
            if len(day_counts) < 2:
                continue
            
            counts = list(day_counts.values())
            avg_failures = statistics.mean(counts)
            
            if avg_failures == 0:
                continue
            
            # Check if latest day has spike
            latest_day = max(day_counts.keys())
            latest_count = day_counts[latest_day]
            
            if latest_count > avg_failures * 2 and latest_count >= 3:
                spikes.append({
                    'test_name': test,
                    'type': 'failure_spike',
                    'latest_failures': latest_count,
                    'average_failures': round(avg_failures, 1),
                    'spike_ratio': round(latest_count / avg_failures, 1),
                    'severity': 'high' if latest_count > avg_failures * 3 else 'medium',
                    'message': f"{latest_count} failures on {latest_day} (avg: {avg_failures:.1f})"
                })
        
        return sorted(spikes, key=lambda x: x['spike_ratio'], reverse=True)
    
    def detect_success_rate_drops(self, min_runs: int = 5) -> List[Dict]:
        """Detect tests with dropping success rates"""
        # Group by test
        by_test = defaultdict(list)
        for record in self.history:
            test = record.get('test_name', 'unknown')
            status = record.get('status', 'unknown')
            by_test[test].append(status)
        
        drops = []
        
        for test, statuses in by_test.items():
            if len(statuses) < min_runs * 2:  # Need enough data
                continue
            
            # Split into two halves
            first_half = statuses[:len(statuses)//2]
            second_half = statuses[len(statuses)//2:]
            
            first_success = sum(1 for s in first_half if s == 'passed') / len(first_half)
            second_success = sum(1 for s in second_half if s == 'passed') / len(second_half)
            
            drop = first_success - second_success
            
            if drop > 0.2:  # 20% drop
                drops.append({
                    'test_name': test,
                    'type': 'success_rate_drop',
                    'previous_rate': round(first_success * 100, 1),
                    'current_rate': round(second_success * 100, 1),
                    'drop': round(drop * 100, 1),
                    'severity': 'high' if drop > 0.4 else 'medium',
                    'message': f"Success rate dropped from {first_success*100:.1f}% to {second_success*100:.1f}%"
                })
        
        return sorted(drops, key=lambda x: x['drop'], reverse=True)
    
    def detect_all_anomalies(self) -> Dict:
        """Detect all types of anomalies"""
        return {
            'duration_anomalies': self.detect_duration_anomalies(),
            'failure_spikes': self.detect_failure_spikes(),
            'success_rate_drops': self.detect_success_rate_drops(),
            'summary': {
                'total_anomalies': 0,
                'high_severity': 0,
                'medium_severity': 0
            }
        }
    
    def generate_anomaly_report(self, output_file: Path = None) -> str:
        """Generate human-readable anomaly report"""
        anomalies = self.detect_all_anomalies()
        
        lines = []
        lines.append("🔍 TEST RESULT ANOMALY REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Duration anomalies
        if anomalies['duration_anomalies']:
            lines.append("⏱️ DURATION ANOMALIES")
            lines.append("-" * 80)
            for anomaly in anomalies['duration_anomalies'][:10]:
                lines.append(f"  [{anomaly['severity'].upper()}] {anomaly['test_name']}")
                lines.append(f"    {anomaly['message']}")
            lines.append("")
        
        # Failure spikes
        if anomalies['failure_spikes']:
            lines.append("📈 FAILURE SPIKES")
            lines.append("-" * 80)
            for spike in anomalies['failure_spikes'][:10]:
                lines.append(f"  [{spike['severity'].upper()}] {spike['test_name']}")
                lines.append(f"    {spike['message']}")
            lines.append("")
        
        # Success rate drops
        if anomalies['success_rate_drops']:
            lines.append("📉 SUCCESS RATE DROPS")
            lines.append("-" * 80)
            for drop in anomalies['success_rate_drops'][:10]:
                lines.append(f"  [{drop['severity'].upper()}] {drop['test_name']}")
                lines.append(f"    {drop['message']}")
            lines.append("")
        
        # Summary
        total = (
            len(anomalies['duration_anomalies']) +
            len(anomalies['failure_spikes']) +
            len(anomalies['success_rate_drops'])
        )
        
        high_severity = sum(
            1 for a in anomalies['duration_anomalies'] + 
            anomalies['failure_spikes'] + 
            anomalies['success_rate_drops']
            if a.get('severity') == 'high'
        )
        
        lines.append("📊 SUMMARY")
        lines.append("-" * 80)
        lines.append(f"  Total Anomalies: {total}")
        lines.append(f"  High Severity: {high_severity}")
        lines.append(f"  Medium Severity: {total - high_severity}")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to {output_file}")
        
        return report


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Anomaly Detector')
    parser.add_argument('--detect', action='store_true', help='Detect all anomalies')
    parser.add_argument('--duration', action='store_true', help='Detect duration anomalies')
    parser.add_argument('--spikes', action='store_true', help='Detect failure spikes')
    parser.add_argument('--drops', action='store_true', help='Detect success rate drops')
    parser.add_argument('--report', type=str, help='Generate report file')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    detector = TestAnomalyDetector(project_root)
    
    if args.report:
        print("🔍 Generating anomaly report...")
        detector.generate_anomaly_report(Path(args.report))
    elif args.duration:
        print("🔍 Detecting duration anomalies...")
        anomalies = detector.detect_duration_anomalies()
        for a in anomalies[:10]:
            print(f"  {a['test_name']}: {a['message']}")
    elif args.spikes:
        print("🔍 Detecting failure spikes...")
        spikes = detector.detect_failure_spikes()
        for s in spikes[:10]:
            print(f"  {s['test_name']}: {s['message']}")
    elif args.drops:
        print("🔍 Detecting success rate drops...")
        drops = detector.detect_success_rate_drops()
        for d in drops[:10]:
            print(f"  {d['test_name']}: {d['message']}")
    elif args.detect:
        print("🔍 Detecting all anomalies...")
        anomalies = detector.detect_all_anomalies()
        print(f"\n📊 Found {len(anomalies['duration_anomalies'])} duration anomalies")
        print(f"📊 Found {len(anomalies['failure_spikes'])} failure spikes")
        print(f"📊 Found {len(anomalies['success_rate_drops'])} success rate drops")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

