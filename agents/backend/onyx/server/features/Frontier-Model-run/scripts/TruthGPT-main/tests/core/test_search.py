"""
Test Result Search and Filter
Advanced search and filtering for test results
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import defaultdict

class TestResultSearcher:
    """Search and filter test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Search test results with filters"""
        results = []
        
        # Load all result files
        for result_file in sorted(self.results_dir.glob("*.json"), reverse=True):
            try:
                import json
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Apply filters
                if filters and not self._matches_filters(data, filters):
                    continue
                
                # Search in test details
                if self._matches_query(data, query):
                    results.append({
                        'file': result_file.name,
                        'data': data
                    })
                
                if len(results) >= limit:
                    break
            except Exception:
                continue
        
        return results
    
    def _matches_query(self, data: Dict, query: str) -> bool:
        """Check if data matches search query"""
        query_lower = query.lower()
        
        # Search in run name
        if 'run_name' in data and query_lower in data['run_name'].lower():
            return True
        
        # Search in test names
        test_details = data.get('test_details', {})
        for test_list in [test_details.get('failures', []), 
                         test_details.get('errors', []),
                         test_details.get('skipped', [])]:
            for test in test_list:
                if query_lower in str(test.get('test', '')).lower():
                    return True
        
        return False
    
    def _matches_filters(self, data: Dict, filters: Dict) -> bool:
        """Check if data matches filters"""
        summary = data.get('summary', {})
        
        # Status filter
        if 'status' in filters:
            status = filters['status']
            if status == 'passed' and summary.get('failed', 0) + summary.get('errors', 0) > 0:
                return False
            if status == 'failed' and summary.get('failed', 0) == 0 and summary.get('errors', 0) == 0:
                return False
        
        # Success rate filter
        if 'min_success_rate' in filters:
            if summary.get('success_rate', 0) < filters['min_success_rate']:
                return False
        
        # Date range filter
        if 'date_from' in filters:
            timestamp = data.get('timestamp', '')
            if timestamp < filters['date_from']:
                return False
        
        if 'date_to' in filters:
            timestamp = data.get('timestamp', '')
            if timestamp > filters['date_to']:
                return False
        
        return True
    
    def filter_by_status(self, status: str, limit: int = 100) -> List[Dict]:
        """Filter results by status"""
        filters = {'status': status}
        return self.search('', filters=filters, limit=limit)
    
    def filter_by_date_range(
        self,
        date_from: str,
        date_to: str,
        limit: int = 100
    ) -> List[Dict]:
        """Filter results by date range"""
        filters = {
            'date_from': date_from,
            'date_to': date_to
        }
        return self.search('', filters=filters, limit=limit)
    
    def filter_by_success_rate(
        self,
        min_rate: float,
        limit: int = 100
    ) -> List[Dict]:
        """Filter results by minimum success rate"""
        filters = {'min_success_rate': min_rate}
        return self.search('', filters=filters, limit=limit)
    
    def get_test_timeline(self, test_name: str) -> List[Dict]:
        """Get timeline of a specific test"""
        timeline = []
        
        for result_file in sorted(self.results_dir.glob("*.json")):
            try:
                import json
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                test_details = data.get('test_details', {})
                
                # Check all test lists
                for status, test_list in [
                    ('failed', test_details.get('failures', [])),
                    ('error', test_details.get('errors', [])),
                    ('skipped', test_details.get('skipped', []))
                ]:
                    for test in test_list:
                        if test_name in str(test.get('test', '')):
                            timeline.append({
                                'timestamp': data.get('timestamp'),
                                'run_name': data.get('run_name'),
                                'status': status,
                                'test': test
                            })
            except Exception:
                continue
        
        return sorted(timeline, key=lambda x: x['timestamp'])

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    searcher = TestResultSearcher(project_root)
    
    # Search for tests
    results = searcher.search("inference", limit=10)
    print(f"Found {len(results)} results matching 'inference'")
    
    # Filter by status
    failed_runs = searcher.filter_by_status("failed", limit=5)
    print(f"Found {len(failed_runs)} failed runs")
    
    # Filter by success rate
    good_runs = searcher.filter_by_success_rate(95.0, limit=10)
    print(f"Found {len(good_runs)} runs with >95% success rate")

if __name__ == "__main__":
    main()







