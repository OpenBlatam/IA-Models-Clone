"""
Test Maturity Model
Assess test suite maturity level
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class TestMaturityModel:
    """Assess test suite maturity"""
    
    # Maturity levels
    LEVELS = {
        1: {
            'name': 'Initial',
            'description': 'Ad-hoc testing, no formal process',
            'characteristics': [
                'No test automation',
                'Manual testing only',
                'No metrics tracking'
            ]
        },
        2: {
            'name': 'Managed',
            'description': 'Basic test automation and tracking',
            'characteristics': [
                'Some test automation',
                'Basic metrics tracking',
                'Regular test execution'
            ]
        },
        3: {
            'name': 'Defined',
            'description': 'Structured testing process',
            'characteristics': [
                'Comprehensive test automation',
                'Test metrics and reporting',
                'CI/CD integration'
            ]
        },
        4: {
            'name': 'Quantitatively Managed',
            'description': 'Data-driven testing decisions',
            'characteristics': [
                'Advanced analytics',
                'Predictive capabilities',
                'Optimization based on data'
            ]
        },
        5: {
            'name': 'Optimizing',
            'description': 'Continuous improvement',
            'characteristics': [
                'ML/AI integration',
                'Automated optimization',
                'Continuous innovation'
            ]
        }
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def assess_maturity(self, lookback_days: int = 30) -> Dict:
        """Assess test suite maturity"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Assess different dimensions
        dimensions = {
            'automation': self._assess_automation(recent),
            'metrics': self._assess_metrics(recent),
            'reporting': self._assess_reporting(),
            'integration': self._assess_integration(),
            'analytics': self._assess_analytics()
        }
        
        # Calculate overall maturity
        avg_maturity = mean(dimensions.values())
        
        # Determine level
        if avg_maturity >= 4.5:
            level = 5
        elif avg_maturity >= 3.5:
            level = 4
        elif avg_maturity >= 2.5:
            level = 3
        elif avg_maturity >= 1.5:
            level = 2
        else:
            level = 1
        
        return {
            'overall_level': level,
            'overall_score': round(avg_maturity, 1),
            'dimensions': dimensions,
            'level_info': self.LEVELS[level],
            'period_days': lookback_days
        }
    
    def _assess_automation(self, recent: List[Dict]) -> float:
        """Assess automation level"""
        if not recent:
            return 1.0
        
        # Check if tests are automated (assuming they are if we have history)
        total_runs = len(recent)
        if total_runs > 20:
            return 3.0  # High automation
        elif total_runs > 10:
            return 2.5  # Medium automation
        else:
            return 2.0  # Basic automation
    
    def _assess_metrics(self, recent: List[Dict]) -> float:
        """Assess metrics tracking level"""
        if not recent:
            return 1.0
        
        # Check if metrics are tracked
        has_metrics = any(
            r.get('success_rate') is not None or
            r.get('execution_time') is not None
            for r in recent
        )
        
        if has_metrics:
            return 3.0  # Good metrics tracking
        else:
            return 1.5  # Basic tracking
    
    def _assess_reporting(self) -> float:
        """Assess reporting capabilities"""
        # Check if reporting tools exist
        report_files = [
            'html_report_generator.py',
            'executive_report.py',
            'advanced_visualizer.py'
        ]
        
        found = sum(1 for f in report_files if (self.project_root / 'tests' / f).exists())
        
        if found >= 3:
            return 4.0  # Excellent reporting
        elif found >= 2:
            return 3.0  # Good reporting
        elif found >= 1:
            return 2.0  # Basic reporting
        else:
            return 1.0  # No reporting
    
    def _assess_integration(self) -> float:
        """Assess CI/CD integration"""
        # Check for CI/CD files
        ci_files = [
            '.github/workflows/tests.yml',
            '.gitlab-ci.yml'
        ]
        
        found = sum(1 for f in ci_files if (self.project_root / f).exists())
        
        if found >= 1:
            return 3.0  # CI/CD integrated
        else:
            return 2.0  # No CI/CD
    
    def _assess_analytics(self) -> float:
        """Assess analytics capabilities"""
        # Check for analytics tools
        analytics_files = [
            'health_scorer.py',
            'trend_forecaster.py',
            'correlation_analyzer.py',
            'test_optimizer.py'
        ]
        
        found = sum(1 for f in analytics_files if (self.project_root / 'tests' / f).exists())
        
        if found >= 3:
            return 5.0  # Advanced analytics
        elif found >= 2:
            return 4.0  # Good analytics
        elif found >= 1:
            return 3.0  # Basic analytics
        else:
            return 2.0  # Limited analytics
    
    def generate_maturity_report(self, assessment: Dict) -> str:
        """Generate maturity assessment report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST SUITE MATURITY ASSESSMENT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in assessment:
            lines.append(f"❌ {assessment['error']}")
            return "\n".join(lines)
        
        level = assessment['overall_level']
        level_info = assessment['level_info']
        
        lines.append(f"📊 Overall Maturity Level: {level} - {level_info['name']}")
        lines.append(f"   Score: {assessment['overall_score']}/5.0")
        lines.append(f"   Period: Last {assessment['period_days']} days")
        lines.append("")
        
        lines.append(f"📝 Description: {level_info['description']}")
        lines.append("")
        
        lines.append("Characteristics:")
        for char in level_info['characteristics']:
            lines.append(f"  • {char}")
        lines.append("")
        
        lines.append("📈 DIMENSION SCORES")
        lines.append("-" * 80)
        dimensions = assessment['dimensions']
        lines.append(f"Automation:     {dimensions['automation']:.1f}/5.0")
        lines.append(f"Metrics:        {dimensions['metrics']:.1f}/5.0")
        lines.append(f"Reporting:      {dimensions['reporting']:.1f}/5.0")
        lines.append(f"Integration:    {dimensions['integration']:.1f}/5.0")
        lines.append(f"Analytics:      {dimensions['analytics']:.1f}/5.0")
        lines.append("")
        
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 80)
        
        if level < 5:
            next_level = level + 1
            next_level_info = self.LEVELS[next_level]
            lines.append(f"To reach Level {next_level} ({next_level_info['name']}):")
            lines.append(f"  {next_level_info['description']}")
            lines.append("")
            lines.append("Focus areas:")
            for char in next_level_info['characteristics']:
                lines.append(f"  • {char}")
        
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
    
    model = TestMaturityModel(project_root)
    assessment = model.assess_maturity(lookback_days=30)
    
    report = model.generate_maturity_report(assessment)
    print(report)
    
    # Save report
    report_file = project_root / "maturity_assessment_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Maturity assessment report saved to: {report_file}")

if __name__ == "__main__":
    main()







