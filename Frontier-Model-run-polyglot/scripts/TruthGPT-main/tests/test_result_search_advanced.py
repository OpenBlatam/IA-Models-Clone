"""
Advanced Test Result Search
Full-text search, fuzzy matching, and advanced filtering
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import difflib


class AdvancedTestResultSearch:
    """Advanced search for test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        self.index = {}  # In-memory search index
    
    def build_index(self, results_files: List[Path] = None):
        """Build search index from result files"""
        if results_files is None:
            results_files = list(self.results_dir.glob("*.json"))
        
        index = {
            'tests': {},  # test_name -> [result_ids]
            'errors': defaultdict(list),  # error_text -> [result_ids]
            'files': defaultdict(list),  # file_path -> [result_ids]
            'statuses': defaultdict(list),  # status -> [result_ids]
            'dates': defaultdict(list),  # date -> [result_ids]
            'results': {}  # result_id -> full_result
        }
        
        result_id = 0
        
        for result_file in results_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    run_name = data.get('run_name', 'unknown')
                    timestamp = data.get('timestamp', '')
                    
                    for test_name, test_data in data.get('test_details', {}).items():
                        result_id += 1
                        
                        # Index by test name
                        if test_name not in index['tests']:
                            index['tests'][test_name] = []
                        index['tests'][test_name].append(result_id)
                        
                        # Index by error
                        error_msg = test_data.get('error_message', '')
                        if error_msg:
                            # Extract keywords from error
                            keywords = self._extract_keywords(error_msg)
                            for keyword in keywords:
                                index['errors'][keyword].append(result_id)
                        
                        # Index by file
                        test_file = test_data.get('test_file', '')
                        if test_file:
                            index['files'][test_file].append(result_id)
                        
                        # Index by status
                        status = test_data.get('status', 'unknown')
                        index['statuses'][status].append(result_id)
                        
                        # Index by date
                        date = timestamp[:10] if timestamp else ''
                        if date:
                            index['dates'][date].append(result_id)
                        
                        # Store full result
                        index['results'][result_id] = {
                            'run_name': run_name,
                            'timestamp': timestamp,
                            'test_name': test_name,
                            **test_data
                        }
            except Exception as e:
                print(f"Error indexing {result_file}: {e}")
        
        self.index = index
        print(f"✅ Indexed {result_id} test results")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Remove common words, extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # Filter common words
        stop_words = {'test', 'error', 'failed', 'assertion', 'exception', 'traceback'}
        keywords = [w for w in words if w not in stop_words]
        return keywords[:10]  # Top 10 keywords
    
    def search(
        self,
        query: str = None,
        test_name: str = None,
        status: str = None,
        error_contains: str = None,
        file_path: str = None,
        date_from: datetime = None,
        date_to: datetime = None,
        fuzzy: bool = False,
        limit: int = 100
    ) -> List[Dict]:
        """Advanced search with multiple filters"""
        if not self.index:
            print("⚠️ Index not built. Run build_index() first.")
            return []
        
        # Start with all result IDs
        candidate_ids = set(self.index['results'].keys())
        
        # Apply filters
        if test_name:
            if fuzzy:
                # Fuzzy match test names
                matching_tests = [
                    name for name in self.index['tests'].keys()
                    if difflib.SequenceMatcher(None, test_name.lower(), name.lower()).ratio() > 0.7
                ]
                test_ids = set()
                for name in matching_tests:
                    test_ids.update(self.index['tests'][name])
                candidate_ids &= test_ids
            else:
                # Exact match
                if test_name in self.index['tests']:
                    candidate_ids &= set(self.index['tests'][test_name])
                else:
                    return []
        
        if status:
            if status in self.index['statuses']:
                candidate_ids &= set(self.index['statuses'][status])
            else:
                return []
        
        if error_contains:
            matching_ids = set()
            for keyword, ids in self.index['errors'].items():
                if error_contains.lower() in keyword.lower():
                    matching_ids.update(ids)
            candidate_ids &= matching_ids
        
        if file_path:
            if file_path in self.index['files']:
                candidate_ids &= set(self.index['files'][file_path])
            else:
                return []
        
        if date_from or date_to:
            date_ids = set()
            for date, ids in self.index['dates'].items():
                date_obj = datetime.fromisoformat(date) if date else datetime.min
                if date_from and date_obj < date_from:
                    continue
                if date_to and date_obj > date_to:
                    continue
                date_ids.update(ids)
            candidate_ids &= date_ids
        
        # Full-text search on query
        if query:
            query_ids = set()
            query_lower = query.lower()
            
            # Search in test names
            for test_name, ids in self.index['tests'].items():
                if query_lower in test_name.lower():
                    query_ids.update(ids)
            
            # Search in errors
            for keyword, ids in self.index['errors'].items():
                if query_lower in keyword.lower():
                    query_ids.update(ids)
            
            candidate_ids &= query_ids
        
        # Get results
        results = [
            self.index['results'][rid]
            for rid in list(candidate_ids)[:limit]
        ]
        
        return results
    
    def search_similar_errors(
        self,
        error_message: str,
        threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict]:
        """Find tests with similar error messages"""
        if not self.index:
            return []
        
        similar_results = []
        
        for result_id, result in self.index['results'].items():
            result_error = result.get('error_message', '')
            if not result_error:
                continue
            
            similarity = difflib.SequenceMatcher(
                None,
                error_message.lower(),
                result_error.lower()
            ).ratio()
            
            if similarity >= threshold:
                similar_results.append({
                    **result,
                    'similarity': similarity
                })
        
        # Sort by similarity
        similar_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_results[:limit]
    
    def get_search_statistics(self) -> Dict:
        """Get search index statistics"""
        if not self.index:
            return {}
        
        return {
            'total_results': len(self.index['results']),
            'unique_tests': len(self.index['tests']),
            'unique_errors': len(self.index['errors']),
            'unique_files': len(self.index['files']),
            'status_distribution': {
                status: len(ids)
                for status, ids in self.index['statuses'].items()
            }
        }


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Test Result Search')
    parser.add_argument('--build-index', action='store_true', help='Build search index')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--test', type=str, help='Filter by test name')
    parser.add_argument('--status', type=str, help='Filter by status')
    parser.add_argument('--error', type=str, help='Search in errors')
    parser.add_argument('--fuzzy', action='store_true', help='Use fuzzy matching')
    parser.add_argument('--similar', type=str, help='Find similar errors')
    parser.add_argument('--stats', action='store_true', help='Show index statistics')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    search = AdvancedTestResultSearch(project_root)
    
    if args.build_index:
        print("🔍 Building search index...")
        search.build_index()
    elif args.stats:
        search.build_index()
        stats = search.get_search_statistics()
        print(f"\n📊 Search Index Statistics:")
        print(f"  Total Results: {stats['total_results']}")
        print(f"  Unique Tests: {stats['unique_tests']}")
        print(f"  Unique Errors: {stats['unique_errors']}")
    elif args.similar:
        search.build_index()
        print(f"🔍 Finding similar errors to: {args.similar[:50]}...")
        similar = search.search_similar_errors(args.similar)
        print(f"\nFound {len(similar)} similar errors:")
        for result in similar[:5]:
            print(f"  {result['test_name']}: {result['similarity']:.2%} similar")
    elif args.search or args.test or args.status:
        search.build_index()
        print("🔍 Searching...")
        results = search.search(
            query=args.search,
            test_name=args.test,
            status=args.status,
            error_contains=args.error,
            fuzzy=args.fuzzy
        )
        print(f"\nFound {len(results)} results:")
        for r in results[:10]:
            print(f"  {r['test_name']}: {r.get('status', 'unknown')}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

