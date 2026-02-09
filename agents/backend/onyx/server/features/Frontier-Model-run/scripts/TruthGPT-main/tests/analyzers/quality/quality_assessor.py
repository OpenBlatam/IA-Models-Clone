"""
Quality Assessor
Assess overall test quality
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
from statistics import mean

class QualityAssessor:
    """Assess test quality"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def assess_quality(self, lookback_days: int = 30) -> Dict:
        """Assess test quality"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Quality dimensions
        success_rates = [r.get('success_rate', 0) for r in recent]
        execution_times = [r.get('execution_time', 0) for r in recent]
        
        # Calculate quality scores
        quality_scores = {
            'reliability': self._calculate_reliability_score(success_rates),
            'performance': self._calculate_performance_score(execution_times),
            'consistency': self._calculate_consistency_score(success_rates),
            'stability': self._calculate_stability_score(recent)
        }
        
        # Overall quality
        overall_quality = mean(quality_scores.values())
        
        # Quality level
        if overall_quality >= 90:
            quality_level = 'Excellent'
        elif overall_quality >= 75:
            quality_level = 'Good'
        elif overall_quality >= 60:
            quality_level = 'Fair'
        else:
            quality_level = 'Needs Improvement'
        
        return {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'quality_scores': quality_scores,
            'overall_quality': round(overall_quality, 1),
            'quality_level': quality_level,
            'strengths': self._identify_strengths(quality_scores),
            'weaknesses': self._identify_weaknesses(quality_scores)
        }
    
    def _calculate_reliability_score(self, success_rates: List[float]) -> float:
        """Calculate reliability score"""
        if not success_rates:
            return 0.0
        avg_success = mean(success_rates)
        return min(100, avg_success)
    
    def _calculate_performance_score(self, execution_times: List[float]) -> float:
        """Calculate performance score"""
        if not execution_times:
            return 0.0
        avg_time = mean(execution_times)
        # Lower time = higher score
        if avg_time <= 60:
            return 100
        elif avg_time <= 120:
            return 80
        elif avg_time <= 300:
            return 60
        else:
            return max(0, 100 - (avg_time / 10))
    
    def _calculate_consistency_score(self, success_rates: List[float]) -> float:
        """Calculate consistency score"""
        if len(success_rates) < 2:
            return 50.0
        from statistics import stdev
        variance = stdev(success_rates)
        # Lower variance = higher score
        return max(0, 100 - (variance * 10))
    
    def _calculate_stability_score(self, recent: List[Dict]) -> float:
        """Calculate stability score"""
        if len(recent) < 5:
            return 50.0
        
        # Check for trends
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_avg = mean([r.get('success_rate', 0) for r in first_half])
        second_avg = mean([r.get('success_rate', 0) for r in second_half])
        
        change = abs(second_avg - first_avg)
        # Lower change = higher score
        return max(0, 100 - (change * 2))
    
    def _identify_strengths(self, quality_scores: Dict) -> List[str]:
        """Identify strengths"""
        strengths = []
        for dimension, score in quality_scores.items():
            if score >= 80:
                strengths.append(f"Strong {dimension} ({score:.1f}/100)")
        return strengths
    
    def _identify_weaknesses(self, quality_scores: Dict) -> List[str]:
        """Identify weaknesses"""
        weaknesses = []
        for dimension, score in quality_scores.items():
            if score < 70:
                weaknesses.append(f"Weak {dimension} ({score:.1f}/100)")
        return weaknesses
    
    def generate_quality_report(self, assessment: Dict) -> str:
        """Generate quality assessment report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST QUALITY ASSESSMENT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in assessment:
            lines.append(f"❌ {assessment['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {assessment['period']}")
        lines.append(f"Total Runs: {assessment['total_runs']}")
        lines.append(f"Overall Quality: {assessment['overall_quality']}/100")
        lines.append(f"Quality Level: {assessment['quality_level']}")
        lines.append("")
        
        lines.append("📊 QUALITY DIMENSIONS")
        lines.append("-" * 80)
        for dimension, score in assessment['quality_scores'].items():
            lines.append(f"{dimension.capitalize()}: {score:.1f}/100")
        lines.append("")
        
        if assessment['strengths']:
            lines.append("✅ STRENGTHS")
            lines.append("-" * 80)
            for strength in assessment['strengths']:
                lines.append(f"• {strength}")
            lines.append("")
        
        if assessment['weaknesses']:
            lines.append("⚠️  WEAKNESSES")
            lines.append("-" * 80)
            for weakness in assessment['weaknesses']:
                lines.append(f"• {weakness}")
        
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
    
    assessor = QualityAssessor(project_root)
    assessment = assessor.assess_quality(lookback_days=30)
    
    report = assessor.generate_quality_report(assessment)
    print(report)
    
    # Save report
    report_file = project_root / "quality_assessment_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Quality assessment report saved to: {report_file}")

if __name__ == "__main__":
    main()







