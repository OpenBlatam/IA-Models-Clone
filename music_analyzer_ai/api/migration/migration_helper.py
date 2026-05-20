"""
Migration helper for transitioning from old to new API structure
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class APIMigrationHelper:
    """Helper for API migration"""
    
    @staticmethod
    def compare_endpoints(old_endpoints: List[str], new_endpoints: List[str]) -> Dict[str, Any]:
        """
        Compare old and new endpoint lists
        
        Args:
            old_endpoints: List of old endpoint paths
            new_endpoints: List of new endpoint paths
        
        Returns:
            Comparison results
        """
        old_set = set(old_endpoints)
        new_set = set(new_endpoints)
        
        return {
            "total_old": len(old_endpoints),
            "total_new": len(new_endpoints),
            "migrated": len(old_set & new_set),
            "missing": list(old_set - new_set),
            "new": list(new_set - old_set),
            "coverage": len(old_set & new_set) / len(old_set) * 100 if old_set else 0
        }
    
    @staticmethod
    def generate_migration_report(comparison: Dict[str, Any]) -> str:
        """
        Generate migration report
        
        Args:
            comparison: Comparison results
        
        Returns:
            Migration report string
        """
        report = "# API Migration Report\n\n"
        report += f"## Summary\n\n"
        report += f"- Old Endpoints: {comparison['total_old']}\n"
        report += f"- New Endpoints: {comparison['total_new']}\n"
        report += f"- Migrated: {comparison['migrated']}\n"
        report += f"- Coverage: {comparison['coverage']:.1f}%\n\n"
        
        if comparison['missing']:
            report += f"## Missing Endpoints ({len(comparison['missing'])})\n\n"
            for endpoint in comparison['missing']:
                report += f"- {endpoint}\n"
            report += "\n"
        
        if comparison['new']:
            report += f"## New Endpoints ({len(comparison['new'])})\n\n"
            for endpoint in comparison['new']:
                report += f"- {endpoint}\n"
        
        return report


migration_helper = APIMigrationHelper()

