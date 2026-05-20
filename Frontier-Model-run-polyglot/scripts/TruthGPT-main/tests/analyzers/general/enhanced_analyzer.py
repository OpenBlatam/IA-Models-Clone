"""
Enhanced Analyzer
Enhanced analysis with deep insights
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from collections import defaultdict

class EnhancedAnalyzer:
    """Enhanced analysis with deep insights"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def enhanced_analysis(self, lookback_days: int = 30) -> Dict:
        """Perform enhanced analysis"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Deep analysis
        analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'timestamp': datetime.now().isoformat(),
            'metrics': self._analyze_metrics(recent),
            'patterns': self._analyze_patterns(recent),
            'trends': self._analyze_trends(recent),
            'anomalies': self._detect_anomalies(recent),
            'insights': self._generate_insights(recent),
            'recommendations': self._generate_recommendations(recent)
        }
        
        return analysis
    
    def _analyze_metrics(self, recent: List[Dict]) -> Dict:
        """Analyze metrics in detail"""
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        return {
            'success_rate': {
                'mean': round(mean(success_rates), 2) if success_rates else 0,
                'median': round(median(success_rates), 2) if len(success_rates) > 1 else (round(success_rates[0], 2) if success_rates else 0),
                'stdev': round(stdev(success_rates), 2) if len(success_rates) > 1 else 0,
                'min': round(min(success_rates), 2) if success_rates else 0,
                'max': round(max(success_rates), 2) if success_rates else 0,
                'consistency': round(100 - (stdev(success_rates) if len(success_rates) > 1 else 0), 2)
            },
            'execution_time': {
                'mean': round(mean(execution_times), 2) if execution_times else 0,
                'median': round(median(execution_times), 2) if len(execution_times) > 1 else (round(execution_times[0], 2) if execution_times else 0),
                'stdev': round(stdev(execution_times), 2) if len(execution_times) > 1 else 0,
                'min': round(min(execution_times), 2) if execution_times else 0,
                'max': round(max(execution_times), 2) if execution_times else 0
            }
        }
    
    def _analyze_patterns(self, recent: List[Dict]) -> Dict:
        """Analyze patterns"""
        day_patterns = defaultdict(int)
        hour_patterns = defaultdict(int)
        
        for run in recent:
            try:
                timestamp = run.get('timestamp', '')
                if timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    day_patterns[dt.strftime('%A')] += 1
                    hour_patterns[dt.hour] += 1
            except Exception:
                pass
        
        return {
            'day_distribution': dict(day_patterns),
            'hour_distribution': dict(hour_patterns),
            'most_active_day': max(day_patterns.items(), key=lambda x: x[1])[0] if day_patterns else None,
            'most_active_hour': max(hour_patterns.items(), key=lambda x: x[1])[0] if hour_patterns else None
        }
    
    def _analyze_trends(self, recent: List[Dict]) -> Dict:
        """Analyze trends"""
        if len(recent) < 2:
            return {}
        
        first_third = recent[:len(recent)//3]
        middle_third = recent[len(recent)//3:2*len(recent)//3]
        last_third = recent[2*len(recent)//3:]
        
        def avg_metric(runs, metric):
            values = [r.get(metric, 0) for r in runs]
            return mean(values) if values else 0
        
        return {
            'success_rate_trend': {
                'first': round(avg_metric(first_third, 'success_rate'), 2),
                'middle': round(avg_metric(middle_third, 'success_rate'), 2),
                'last': round(avg_metric(last_third, 'success_rate'), 2),
                'direction': self._determine_direction(
                    avg_metric(first_third, 'success_rate'),
                    avg_metric(last_third, 'success_rate')
                )
            },
            'execution_time_trend': {
                'first': round(avg_metric(first_third, 'execution_time'), 2),
                'middle': round(avg_metric(middle_third, 'execution_time'), 2),
                'last': round(avg_metric(last_third, 'execution_time'), 2),
                'direction': self._determine_direction(
                    avg_metric(first_third, 'execution_time'),
                    avg_metric(last_third, 'execution_time'),
                    reverse=True  # Lower is better for execution time
                )
            }
        }
    
    def _determine_direction(self, first: float, last: float, reverse: bool = False) -> str:
        """Determine trend direction"""
        if reverse:
            if last < first:
                return 'improving'
            elif last > first:
                return 'declining'
        else:
            if last > first:
                return 'improving'
            elif last < first:
                return 'declining'
        return 'stable'
    
    def _detect_anomalies(self, recent: List[Dict]) -> List[Dict]:
        """Detect anomalies"""
        anomalies = []
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        if len(success_rates) < 2:
            return anomalies
        
        avg_success = mean(success_rates)
        std_success = stdev(success_rates)
        
        avg_time = mean(execution_times)
        std_time = stdev(execution_times) if len(execution_times) > 1 else 0
        
        for i, run in enumerate(recent):
            timestamp = run.get('timestamp', '')
            success_rate = run.get('success_rate', 0)
            execution_time = run.get('execution_time', 0)
            
            # Check for anomalies
            if std_success > 0:
                z_score = abs((success_rate - avg_success) / std_success)
                if z_score > 2:
                    anomalies.append({
                        'timestamp': timestamp,
                        'type': 'success_rate',
                        'value': success_rate,
                        'z_score': round(z_score, 2),
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
            
            if std_time > 0:
                z_score = abs((execution_time - avg_time) / std_time)
                if z_score > 2:
                    anomalies.append({
                        'timestamp': timestamp,
                        'type': 'execution_time',
                        'value': execution_time,
                        'z_score': round(z_score, 2),
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
        
        return anomalies
    
    def _generate_insights(self, recent: List[Dict]) -> List[str]:
        """Generate insights"""
        insights = []
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        if avg_success >= 98:
            insights.append("Exceptional test quality - maintain excellence")
        elif avg_success >= 95:
            insights.append("High test quality - minor improvements possible")
        elif avg_success >= 90:
            insights.append("Good test quality - focus on stability")
        else:
            insights.append("Test quality needs attention - prioritize fixes")
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        if avg_time < 30:
            insights.append("Excellent execution speed")
        elif avg_time < 120:
            insights.append("Good execution speed")
        else:
            insights.append("Execution speed could be improved")
        
        if len(success_rates) > 1:
            variance = max(success_rates) - min(success_rates)
            if variance < 2:
                insights.append("Very consistent test results")
            elif variance < 5:
                insights.append("Consistent test results")
            else:
                insights.append("Inconsistent test results - investigate variance")
        
        return insights
    
    def _generate_recommendations(self, recent: List[Dict]) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        success_rates = [r.get('success_rate', 0) for r in recent]
        avg_success = mean(success_rates) if success_rates else 0
        
        if avg_success < 95:
            recommendations.append("Focus on improving test success rate to 95%+")
        
        execution_times = [r.get('execution_time', 0) for r in recent]
        avg_time = mean(execution_times) if execution_times else 0
        
        if avg_time > 300:
            recommendations.append("Optimize test execution - consider parallel execution")
        
        if len(success_rates) > 1:
            variance = max(success_rates) - min(success_rates)
            if variance > 10:
                recommendations.append("Reduce test result variance - fix flaky tests")
        
        return recommendations
    
    def generate_analysis_report(self, analysis: Dict) -> str:
        """Generate enhanced analysis report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        # Metrics
        lines.append("📊 METRICS ANALYSIS")
        lines.append("-" * 80)
        metrics = analysis['metrics']
        lines.append("Success Rate:")
        lines.append(f"  Mean: {metrics['success_rate']['mean']:.2f}%")
        lines.append(f"  Consistency: {metrics['success_rate']['consistency']:.2f}%")
        lines.append("")
        
        # Patterns
        if 'patterns' in analysis:
            lines.append("📈 PATTERNS")
            lines.append("-" * 80)
            patterns = analysis['patterns']
            if patterns.get('most_active_day'):
                lines.append(f"Most Active Day: {patterns['most_active_day']}")
            if patterns.get('most_active_hour'):
                lines.append(f"Most Active Hour: {patterns['most_active_hour']}:00")
            lines.append("")
        
        # Trends
        if 'trends' in analysis:
            lines.append("📊 TRENDS")
            lines.append("-" * 80)
            trends = analysis['trends']
            if 'success_rate_trend' in trends:
                trend = trends['success_rate_trend']
                lines.append(f"Success Rate: {trend['direction'].upper()}")
                lines.append(f"  First: {trend['first']:.2f}% → Last: {trend['last']:.2f}%")
            lines.append("")
        
        # Anomalies
        if 'anomalies' in analysis and analysis['anomalies']:
            lines.append("⚠️  ANOMALIES")
            lines.append("-" * 80)
            for anomaly in analysis['anomalies'][:10]:
                lines.append(f"{anomaly['type']}: {anomaly['value']} (Z-score: {anomaly['z_score']})")
            lines.append("")
        
        # Insights
        if 'insights' in analysis:
            lines.append("💡 INSIGHTS")
            lines.append("-" * 80)
            for insight in analysis['insights']:
                lines.append(f"• {insight}")
            lines.append("")
        
        # Recommendations
        if 'recommendations' in analysis:
            lines.append("🎯 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in analysis['recommendations']:
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
    project_root = Path(__file__).parent.parent
    
    analyzer = EnhancedAnalyzer(project_root)
    analysis = analyzer.enhanced_analysis(lookback_days=30)
    
    report = analyzer.generate_analysis_report(analysis)
    print(report)
    
    # Save analysis
    analysis_file = project_root / "enhanced_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n📄 Enhanced analysis saved to: {analysis_file}")

if __name__ == "__main__":
    main()







