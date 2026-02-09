"""
Advanced Test Result Visualizer
Creates interactive visualizations and charts from test results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AdvancedTestResultVisualizer:
    """Advanced visualization of test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        self.output_dir = project_root / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_timeline_chart(
        self,
        results: List[Dict],
        output_file: Path = None
    ) -> str:
        """Generate timeline chart data"""
        # Group by date
        by_date = defaultdict(lambda: {'passed': 0, 'failed': 0, 'total': 0})
        
        for result in results:
            timestamp = result.get('timestamp', '')
            if timestamp:
                date = timestamp[:10]  # YYYY-MM-DD
                status = result.get('status', 'unknown')
                by_date[date]['total'] += 1
                if status == 'passed':
                    by_date[date]['passed'] += 1
                elif status in ('failed', 'error'):
                    by_date[date]['failed'] += 1
        
        # Generate chart data
        chart_data = {
            'type': 'timeline',
            'labels': sorted(by_date.keys()),
            'datasets': [
                {
                    'label': 'Passed',
                    'data': [by_date[date]['passed'] for date in sorted(by_date.keys())],
                    'backgroundColor': 'rgba(75, 192, 192, 0.6)'
                },
                {
                    'label': 'Failed',
                    'data': [by_date[date]['failed'] for date in sorted(by_date.keys())],
                    'backgroundColor': 'rgba(255, 99, 132, 0.6)'
                }
            ]
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chart_data, f, indent=2)
            print(f"✅ Timeline chart data saved to {output_file}")
        
        return json.dumps(chart_data, indent=2)
    
    def generate_duration_distribution(
        self,
        results: List[Dict],
        output_file: Path = None
    ) -> str:
        """Generate duration distribution chart"""
        durations = [r.get('duration', 0) for r in results if r.get('duration', 0) > 0]
        
        if not durations:
            return json.dumps({'error': 'No duration data'})
        
        # Create buckets
        max_dur = max(durations)
        bucket_size = max(1, max_dur / 10)
        buckets = defaultdict(int)
        
        for dur in durations:
            bucket = int(dur / bucket_size) * bucket_size
            buckets[bucket] += 1
        
        chart_data = {
            'type': 'histogram',
            'labels': [f"{k:.1f}s" for k in sorted(buckets.keys())],
            'data': [buckets[k] for k in sorted(buckets.keys())],
            'statistics': {
                'mean': statistics.mean(durations),
                'median': statistics.median(durations),
                'min': min(durations),
                'max': max(durations),
                'stdev': statistics.stdev(durations) if len(durations) > 1 else 0
            }
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chart_data, f, indent=2)
            print(f"✅ Duration distribution saved to {output_file}")
        
        return json.dumps(chart_data, indent=2)
    
    def generate_status_pie_chart(
        self,
        results: List[Dict],
        output_file: Path = None
    ) -> str:
        """Generate status distribution pie chart"""
        status_counts = defaultdict(int)
        
        for result in results:
            status = result.get('status', 'unknown')
            status_counts[status] += 1
        
        colors = {
            'passed': '#4CAF50',
            'failed': '#F44336',
            'error': '#FF9800',
            'skipped': '#9E9E9E',
            'unknown': '#607D8B'
        }
        
        chart_data = {
            'type': 'pie',
            'labels': list(status_counts.keys()),
            'data': list(status_counts.values()),
            'backgroundColor': [colors.get(s, '#607D8B') for s in status_counts.keys()]
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chart_data, f, indent=2)
            print(f"✅ Status pie chart saved to {output_file}")
        
        return json.dumps(chart_data, indent=2)
    
    def generate_heatmap(
        self,
        results: List[Dict],
        output_file: Path = None
    ) -> str:
        """Generate test execution heatmap"""
        # Group by test and date
        by_test_date = defaultdict(lambda: defaultdict(int))
        
        for result in results:
            test_name = result.get('test_name', 'unknown')
            timestamp = result.get('timestamp', '')
            if timestamp:
                date = timestamp[:10]
                status = result.get('status', 'unknown')
                # 1 for passed, -1 for failed, 0 for skipped
                value = 1 if status == 'passed' else (-1 if status in ('failed', 'error') else 0)
                by_test_date[test_name][date] += value
        
        # Get all dates
        all_dates = set()
        for dates in by_test_date.values():
            all_dates.update(dates.keys())
        all_dates = sorted(all_dates)
        
        # Generate heatmap data
        heatmap_data = {
            'type': 'heatmap',
            'tests': list(by_test_date.keys())[:50],  # Limit to 50 tests
            'dates': all_dates,
            'data': [
                [by_test_date[test].get(date, 0) for date in all_dates]
                for test in list(by_test_date.keys())[:50]
            ]
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(heatmap_data, f, indent=2)
            print(f"✅ Heatmap saved to {output_file}")
        
        return json.dumps(heatmap_data, indent=2)
    
    def generate_dashboard_html(
        self,
        results: List[Dict],
        output_file: Path = None
    ) -> str:
        """Generate interactive HTML dashboard"""
        if output_file is None:
            output_file = self.output_dir / "dashboard.html"
        
        # Generate chart data
        timeline_data = self.generate_timeline_chart(results)
        duration_data = self.generate_duration_distribution(results)
        status_data = self.generate_status_pie_chart(results)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Results Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .chart-container {{ margin: 20px 0; }}
        canvas {{ max-width: 800px; }}
    </style>
</head>
<body>
    <h1>Test Results Dashboard</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="chart-container">
        <h2>Timeline</h2>
        <canvas id="timelineChart"></canvas>
    </div>
    
    <div class="chart-container">
        <h2>Status Distribution</h2>
        <canvas id="statusChart"></canvas>
    </div>
    
    <div class="chart-container">
        <h2>Duration Distribution</h2>
        <canvas id="durationChart"></canvas>
    </div>
    
    <script>
        // Timeline Chart
        const timelineData = {timeline_data};
        new Chart(document.getElementById('timelineChart'), {{
            type: 'line',
            data: timelineData,
            options: {{ responsive: true }}
        }});
        
        // Status Chart
        const statusData = {status_data};
        new Chart(document.getElementById('statusChart'), {{
            type: 'pie',
            data: statusData,
            options: {{ responsive: true }}
        }});
        
        // Duration Chart
        const durationData = {duration_data};
        new Chart(document.getElementById('durationChart'), {{
            type: 'bar',
            data: durationData,
            options: {{ responsive: true }}
        }});
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Dashboard saved to {output_file}")
        return html


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Test Result Visualizer')
    parser.add_argument('--results', type=str, help='Results file to visualize')
    parser.add_argument('--timeline', action='store_true', help='Generate timeline chart')
    parser.add_argument('--duration', action='store_true', help='Generate duration distribution')
    parser.add_argument('--status', action='store_true', help='Generate status pie chart')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap')
    parser.add_argument('--dashboard', action='store_true', help='Generate HTML dashboard')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    visualizer = AdvancedTestResultVisualizer(project_root)
    
    if args.results:
        with open(args.results, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = list(data.get('test_details', {}).values()) if isinstance(data.get('test_details'), dict) else data.get('test_details', [])
        
        if args.dashboard:
            print("📊 Generating dashboard...")
            visualizer.generate_dashboard_html(results)
        elif args.timeline:
            print("📊 Generating timeline chart...")
            visualizer.generate_timeline_chart(results, project_root / "timeline_chart.json")
        elif args.duration:
            print("📊 Generating duration distribution...")
            visualizer.generate_duration_distribution(results, project_root / "duration_chart.json")
        elif args.status:
            print("📊 Generating status chart...")
            visualizer.generate_status_pie_chart(results, project_root / "status_chart.json")
        elif args.heatmap:
            print("📊 Generating heatmap...")
            visualizer.generate_heatmap(results, project_root / "heatmap.json")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

