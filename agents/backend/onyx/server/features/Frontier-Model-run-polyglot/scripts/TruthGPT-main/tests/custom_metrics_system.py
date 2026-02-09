"""
Custom Metrics System
Define and track custom metrics for test results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class CustomMetricsSystem:
    """System for defining and tracking custom metrics"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.metrics_file = project_root / "config" / "custom_metrics.json"
        self.metrics_definitions = self._load_metrics()
        self.history_file = project_root / "custom_metrics_history.json"
        self.history = self._load_history()
    
    def _load_metrics(self) -> Dict:
        """Load metric definitions"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metrics(self):
        """Save metric definitions"""
        self.metrics_file.parent.mkdir(exist_ok=True)
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_definitions, f, indent=2)
    
    def _load_history(self) -> List[Dict]:
        """Load metrics history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """Save metrics history"""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
    
    def define_metric(
        self,
        name: str,
        description: str,
        calculation: str,
        unit: str = None
    ):
        """Define a custom metric"""
        self.metrics_definitions[name] = {
            'description': description,
            'calculation': calculation,
            'unit': unit,
            'created_at': datetime.now().isoformat()
        }
        self._save_metrics()
        print(f"✅ Metric '{name}' defined")
    
    def calculate_metric(
        self,
        metric_name: str,
        test_results: Dict
    ) -> Optional[float]:
        """Calculate a custom metric"""
        if metric_name not in self.metrics_definitions:
            print(f"⚠️ Metric '{metric_name}' not defined")
            return None
        
        definition = self.metrics_definitions[metric_name]
        calculation = definition['calculation']
        
        # Build context for calculation
        context = {
            'total_tests': test_results.get('total_tests', 0),
            'passed': test_results.get('passed', 0),
            'failed': test_results.get('failed', 0),
            'errors': test_results.get('errors', 0),
            'skipped': test_results.get('skipped', 0),
            'execution_time': test_results.get('execution_time', 0),
            'success_rate': test_results.get('success_rate', 0)
        }
        
        # Simple calculation evaluation
        try:
            # Replace variables in calculation
            for key, value in context.items():
                calculation = calculation.replace(key, str(value))
            
            result = eval(calculation)
            return float(result)
        except Exception as e:
            print(f"Error calculating metric: {e}")
            return None
    
    def record_metrics(
        self,
        test_results: Dict,
        run_name: str = None
    ):
        """Record metrics for a test run"""
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metrics_values = {}
        
        for metric_name in self.metrics_definitions.keys():
            value = self.calculate_metric(metric_name, test_results)
            if value is not None:
                metrics_values[metric_name] = value
        
        record = {
            'run_name': run_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_values
        }
        
        self.history.append(record)
        
        # Keep last 1000 records
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        self._save_history()
        return metrics_values
    
    def get_metric_trend(
        self,
        metric_name: str,
        days: int = 30
    ) -> Dict:
        """Get trend for a metric"""
        cutoff = datetime.now() - timedelta(days=days)
        
        values = []
        timestamps = []
        
        for record in self.history:
            timestamp = datetime.fromisoformat(record['timestamp'])
            if timestamp > cutoff and metric_name in record['metrics']:
                values.append(record['metrics'][metric_name])
                timestamps.append(record['timestamp'])
        
        if not values:
            return {}
        
        return {
            'metric_name': metric_name,
            'period_days': days,
            'data_points': len(values),
            'current': values[-1],
            'average': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'trend': self._calculate_trend(values),
            'timeline': [
                {'timestamp': ts, 'value': val}
                for ts, val in zip(timestamps, values)
            ]
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change = (second_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0
        
        if abs(change) < 5:
            return 'stable'
        elif change > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def list_metrics(self) -> Dict:
        """List all defined metrics"""
        return self.metrics_definitions


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom Metrics System')
    parser.add_argument('--define', nargs=3, metavar=('NAME', 'DESC', 'CALC'), help='Define a metric')
    parser.add_argument('--list', action='store_true', help='List all metrics')
    parser.add_argument('--calculate', type=str, help='Calculate metric for results')
    parser.add_argument('--trend', type=str, help='Get trend for metric')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    metrics_system = CustomMetricsSystem(project_root)
    
    if args.define:
        name, desc, calc = args.define
        metrics_system.define_metric(name, desc, calc)
    elif args.list:
        print("📊 Defined Metrics:")
        for name, definition in metrics_system.list_metrics().items():
            print(f"  {name}: {definition['description']}")
            print(f"    Calculation: {definition['calculation']}")
    elif args.trend:
        print(f"📈 Trend for '{args.trend}':")
        trend = metrics_system.get_metric_trend(args.trend)
        if trend:
            print(f"  Current: {trend['current']}")
            print(f"  Average: {trend['average']:.2f}")
            print(f"  Trend: {trend['trend']}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

