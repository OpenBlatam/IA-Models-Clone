"""
Test Timeline
Visualize test execution timeline
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

class TestTimeline:
    """Generate test execution timeline"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.results_dir = project_root / "test_results"
    
    def generate_timeline(self, lookback_days: int = 30) -> Dict:
        """Generate test execution timeline"""
        history = self._load_history()
        
        from datetime import timedelta
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = sorted(
            [r for r in history if r.get('timestamp', '') >= cutoff_date],
            key=lambda x: x.get('timestamp', '')
        )
        
        timeline_events = []
        
        for run in recent:
            timestamp = run.get('timestamp', '')
            success_rate = run.get('success_rate', 0)
            total_tests = run.get('total_tests', 0)
            failures = run.get('failures', 0) + run.get('errors', 0)
            
            # Determine event type
            if success_rate >= 95:
                event_type = 'success'
            elif success_rate >= 85:
                event_type = 'warning'
            else:
                event_type = 'failure'
            
            timeline_events.append({
                'timestamp': timestamp,
                'type': event_type,
                'success_rate': success_rate,
                'total_tests': total_tests,
                'failures': failures,
                'execution_time': run.get('execution_time', 0)
            })
        
        return {
            'events': timeline_events,
            'total_events': len(timeline_events),
            'period_days': lookback_days
        }
    
    def generate_timeline_html(self, timeline_data: Dict) -> Path:
        """Generate HTML timeline visualization"""
        output_path = self.project_root / "test_timeline.html"
        
        events = timeline_data['events']
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Execution Timeline</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        .timeline {{ position: relative; padding: 20px 0; }}
        .event {{
            margin: 10px 0;
            padding: 10px;
            border-left: 4px solid;
            background: #f5f5f5;
        }}
        .event.success {{ border-color: #4CAF50; }}
        .event.warning {{ border-color: #ff9800; }}
        .event.failure {{ border-color: #f44336; }}
        .timestamp {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Test Execution Timeline</h1>
    <p>Period: Last {timeline_data['period_days']} days | Total Events: {timeline_data['total_events']}</p>
    <div class="timeline">
"""
        
        for event in events:
            html += f"""
        <div class="event {event['type']}">
            <div class="timestamp">{event['timestamp'][:19]}</div>
            <div>Success Rate: {event['success_rate']:.1f}%</div>
            <div>Tests: {event['total_tests']} | Failures: {event['failures']}</div>
            <div>Time: {event['execution_time']:.2f}s</div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
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
    
    timeline = TestTimeline(project_root)
    timeline_data = timeline.generate_timeline(lookback_days=30)
    
    html_path = timeline.generate_timeline_html(timeline_data)
    print(f"✅ Timeline generated: {html_path}")

if __name__ == "__main__":
    main()

