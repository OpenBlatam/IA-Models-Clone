"""
Enhanced Dependency Analyzer
Enhanced dependency analysis with advanced insights
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from statistics import mean

class EnhancedDependencyAnalyzer:
    """Enhanced dependency analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
    
    def analyze_dependencies(self, lookback_days: int = 30) -> Dict:
        """Analyze test dependencies"""
        history = self._load_history()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze dependencies
        dependency_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'dependency_graph': self._build_dependency_graph(recent),
            'circular_dependencies': self._detect_circular_dependencies(recent),
            'critical_paths': self._identify_critical_paths(recent),
            'dependency_chains': self._analyze_dependency_chains(recent),
            'dependency_metrics': self._calculate_dependency_metrics(recent),
            'recommendations': []
        }
        
        # Generate recommendations
        dependency_analysis['recommendations'] = self._generate_dependency_recommendations(dependency_analysis)
        
        return dependency_analysis
    
    def _build_dependency_graph(self, recent: List[Dict]) -> Dict:
        """Build dependency graph"""
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        # Analyze failure patterns to infer dependencies
        for i, r in enumerate(recent):
            if i == 0:
                continue
            
            prev = recent[i-1]
            current_failures = r.get('failures', 0) + r.get('errors', 0)
            prev_failures = prev.get('failures', 0) + prev.get('errors', 0)
            
            # If failures increased significantly, might indicate dependency
            if current_failures > prev_failures * 1.5:
                # Infer dependency relationship
                graph[prev.get('timestamp', '')[:10]].add(r.get('timestamp', '')[:10])
                in_degree[r.get('timestamp', '')[:10]] += 1
        
        return {
            'nodes': len(set().union(*graph.values()) | set(graph.keys())),
            'edges': sum(len(deps) for deps in graph.values()),
            'graph': {str(k): list(v) for k, v in graph.items()},
            'in_degrees': dict(in_degree)
        }
    
    def _detect_circular_dependencies(self, recent: List[Dict]) -> List[Dict]:
        """Detect circular dependencies"""
        circular = []
        
        if len(recent) < 3:
            return circular
        
        # Simple circular detection based on failure patterns
        for i in range(len(recent) - 2):
            r1 = recent[i]
            r2 = recent[i+1]
            r3 = recent[i+2]
            
            f1 = r1.get('failures', 0) + r1.get('errors', 0)
            f2 = r2.get('failures', 0) + r2.get('errors', 0)
            f3 = r3.get('failures', 0) + r3.get('errors', 0)
            
            # Pattern: high -> low -> high (potential circular dependency)
            if f1 > f2 * 1.5 and f3 > f2 * 1.5:
                circular.append({
                    'start_index': i,
                    'end_index': i+2,
                    'pattern': 'high_low_high',
                    'failures': [f1, f2, f3],
                    'severity': 'medium'
                })
        
        return circular
    
    def _identify_critical_paths(self, recent: List[Dict]) -> List[Dict]:
        """Identify critical paths"""
        critical_paths = []
        
        if len(recent) < 2:
            return critical_paths
        
        # Find sequences of high-failure runs
        current_path = []
        for i, r in enumerate(recent):
            failures = r.get('failures', 0) + r.get('errors', 0)
            avg_failures = mean([r2.get('failures', 0) + r2.get('errors', 0) for r2 in recent])
            
            if failures > avg_failures * 1.5:
                current_path.append(i)
            else:
                if len(current_path) >= 2:
                    critical_paths.append({
                        'start_index': current_path[0],
                        'end_index': current_path[-1],
                        'length': len(current_path),
                        'avg_failures': round(mean([recent[j].get('failures', 0) + recent[j].get('errors', 0) for j in current_path]), 2),
                        'severity': 'high' if len(current_path) >= 3 else 'medium'
                    })
                current_path = []
        
        # Check if path extends to end
        if len(current_path) >= 2:
            critical_paths.append({
                'start_index': current_path[0],
                'end_index': current_path[-1],
                'length': len(current_path),
                'avg_failures': round(mean([recent[j].get('failures', 0) + recent[j].get('errors', 0) for j in current_path]), 2),
                'severity': 'high' if len(current_path) >= 3 else 'medium'
            })
        
        return critical_paths
    
    def _analyze_dependency_chains(self, recent: List[Dict]) -> Dict:
        """Analyze dependency chains"""
        if len(recent) < 2:
            return {}
        
        chains = []
        current_chain = [0]
        
        for i in range(1, len(recent)):
            prev_failures = recent[i-1].get('failures', 0) + recent[i-1].get('errors', 0)
            current_failures = recent[i].get('failures', 0) + recent[i].get('errors', 0)
            
            # If failures increased, continue chain
            if current_failures > prev_failures * 1.2:
                current_chain.append(i)
            else:
                if len(current_chain) >= 2:
                    chains.append({
                        'length': len(current_chain),
                        'start_index': current_chain[0],
                        'end_index': current_chain[-1]
                    })
                current_chain = [i]
        
        # Check if chain extends to end
        if len(current_chain) >= 2:
            chains.append({
                'length': len(current_chain),
                'start_index': current_chain[0],
                'end_index': current_chain[-1]
            })
        
        return {
            'total_chains': len(chains),
            'max_chain_length': max([c['length'] for c in chains], default=0),
            'avg_chain_length': round(mean([c['length'] for c in chains]), 2) if chains else 0,
            'chains': chains[:5]  # Top 5
        }
    
    def _calculate_dependency_metrics(self, recent: List[Dict]) -> Dict:
        """Calculate dependency metrics"""
        if len(recent) < 2:
            return {}
        
        # Calculate dependency strength
        dependency_strengths = []
        for i in range(1, len(recent)):
            prev_failures = recent[i-1].get('failures', 0) + recent[i-1].get('errors', 0)
            current_failures = recent[i].get('failures', 0) + recent[i].get('errors', 0)
            
            if prev_failures > 0:
                strength = (current_failures / prev_failures) if prev_failures > 0 else 0
                dependency_strengths.append(strength)
        
        avg_strength = mean(dependency_strengths) if dependency_strengths else 0
        
        # Calculate coupling
        high_coupling_runs = sum(1 for s in dependency_strengths if s > 1.5)
        coupling_ratio = (high_coupling_runs / len(dependency_strengths) * 100) if dependency_strengths else 0
        
        return {
            'avg_dependency_strength': round(avg_strength, 2),
            'high_coupling_runs': high_coupling_runs,
            'coupling_ratio': round(coupling_ratio, 1),
            'total_dependencies': len(dependency_strengths),
            'coupling_level': 'high' if coupling_ratio > 30 else 'medium' if coupling_ratio > 15 else 'low'
        }
    
    def _generate_dependency_recommendations(self, analysis: Dict) -> List[str]:
        """Generate dependency recommendations"""
        recommendations = []
        
        if analysis['circular_dependencies']:
            recommendations.append(f"🚨 {len(analysis['circular_dependencies'])} circular dependency pattern(s) detected - break cycles")
        
        critical_paths = analysis['critical_paths']
        if critical_paths:
            high_severity = sum(1 for p in critical_paths if p['severity'] == 'high')
            if high_severity > 0:
                recommendations.append(f"🚨 {high_severity} high-severity critical path(s) - optimize dependencies")
        
        chains = analysis['dependency_chains']
        if chains.get('max_chain_length', 0) > 5:
            recommendations.append(f"Long dependency chain detected (length: {chains['max_chain_length']}) - reduce chain length")
        
        metrics = analysis['dependency_metrics']
        if metrics.get('coupling_level') == 'high':
            recommendations.append(f"High coupling ratio ({metrics['coupling_ratio']:.1f}%) - reduce test dependencies")
        
        if not recommendations:
            recommendations.append("✅ Dependencies are well-structured - maintain current practices")
        
        return recommendations
    
    def generate_dependency_report(self, analysis: Dict) -> str:
        """Generate dependency report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ENHANCED DEPENDENCY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        lines.append("📊 DEPENDENCY GRAPH")
        lines.append("-" * 80)
        graph = analysis['dependency_graph']
        lines.append(f"Nodes: {graph['nodes']}")
        lines.append(f"Edges: {graph['edges']}")
        lines.append("")
        
        if analysis['circular_dependencies']:
            lines.append("🔴 CIRCULAR DEPENDENCIES")
            lines.append("-" * 80)
            for circ in analysis['circular_dependencies']:
                lines.append(f"Circular Pattern: Runs {circ['start_index']}-{circ['end_index']}")
                lines.append(f"   Pattern: {circ['pattern']}")
                lines.append(f"   Failures: {circ['failures']}")
                lines.append(f"   Severity: {circ['severity'].upper()}")
            lines.append("")
        
        if analysis['critical_paths']:
            lines.append("🔴 CRITICAL PATHS")
            lines.append("-" * 80)
            severity_emoji = {'high': '🔴', 'medium': '🟡'}
            for path in analysis['critical_paths']:
                emoji = severity_emoji.get(path['severity'], '⚪')
                lines.append(f"{emoji} Path: Runs {path['start_index']}-{path['end_index']}")
                lines.append(f"   Length: {path['length']} runs")
                lines.append(f"   Avg Failures: {path['avg_failures']}")
            lines.append("")
        
        if analysis.get('dependency_chains'):
            chains = analysis['dependency_chains']
            lines.append("🔗 DEPENDENCY CHAINS")
            lines.append("-" * 80)
            lines.append(f"Total Chains: {chains['total_chains']}")
            lines.append(f"Max Chain Length: {chains['max_chain_length']}")
            lines.append(f"Average Chain Length: {chains['avg_chain_length']}")
            lines.append("")
        
        if analysis.get('dependency_metrics'):
            metrics = analysis['dependency_metrics']
            lines.append("📈 DEPENDENCY METRICS")
            lines.append("-" * 80)
            lines.append(f"Average Dependency Strength: {metrics['avg_dependency_strength']}")
            lines.append(f"High Coupling Runs: {metrics['high_coupling_runs']}")
            lines.append(f"Coupling Ratio: {metrics['coupling_ratio']}%")
            lines.append(f"Coupling Level: {metrics['coupling_level'].upper()}")
            lines.append("")
        
        if analysis['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
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
    
    analyzer = EnhancedDependencyAnalyzer(project_root)
    analysis = analyzer.analyze_dependencies(lookback_days=30)
    
    report = analyzer.generate_dependency_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "enhanced_dependency_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Enhanced dependency analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()

