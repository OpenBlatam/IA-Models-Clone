"""
Test Tagging System
Tag and categorize tests for better organization
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict

class TestTaggingSystem:
    """Tag and categorize tests"""
    
    def __init__(self, project_root: Path, tags_file: str = "test_tags.json"):
        self.project_root = project_root
        self.tags_file = project_root / tags_file
        self.tags: Dict[str, List[str]] = {}  # test_name -> [tags]
        self.tag_definitions: Dict[str, str] = {}  # tag -> description
        self._load_tags()
    
    def _load_tags(self):
        """Load tags from file"""
        if self.tags_file.exists():
            try:
                with open(self.tags_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tags = data.get('tags', {})
                    self.tag_definitions = data.get('definitions', {})
            except Exception:
                self.tags = {}
                self.tag_definitions = {}
    
    def _save_tags(self):
        """Save tags to file"""
        data = {
            'tags': self.tags,
            'definitions': self.tag_definitions
        }
        
        with open(self.tags_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def add_tag(self, test_name: str, tag: str, description: Optional[str] = None):
        """Add tag to test"""
        if test_name not in self.tags:
            self.tags[test_name] = []
        
        if tag not in self.tags[test_name]:
            self.tags[test_name].append(tag)
        
        if description and tag not in self.tag_definitions:
            self.tag_definitions[tag] = description
        
        self._save_tags()
    
    def remove_tag(self, test_name: str, tag: str):
        """Remove tag from test"""
        if test_name in self.tags:
            if tag in self.tags[test_name]:
                self.tags[test_name].remove(tag)
                if not self.tags[test_name]:
                    del self.tags[test_name]
                self._save_tags()
    
    def get_tests_by_tag(self, tag: str) -> List[str]:
        """Get all tests with a specific tag"""
        return [test for test, tags in self.tags.items() if tag in tags]
    
    def get_tags_for_test(self, test_name: str) -> List[str]:
        """Get all tags for a test"""
        return self.tags.get(test_name, [])
    
    def get_all_tags(self) -> Set[str]:
        """Get all unique tags"""
        all_tags = set()
        for tags in self.tags.values():
            all_tags.update(tags)
        return all_tags
    
    def auto_tag_tests(self):
        """Automatically tag tests based on name patterns"""
        # Common patterns
        patterns = {
            'integration': ['test_integration', 'integration'],
            'unit': ['test_unit', 'unit_test'],
            'performance': ['test_performance', 'performance', 'benchmark'],
            'security': ['test_security', 'security'],
            'edge_case': ['test_edge', 'edge_case'],
            'regression': ['test_regression', 'regression'],
            'core': ['test_core', 'core'],
            'optimization': ['test_optimization', 'optimization'],
            'training': ['test_training', 'training'],
            'inference': ['test_inference', 'inference'],
            'monitoring': ['test_monitoring', 'monitoring']
        }
        
        # Load test names from results
        results_dir = self.project_root / "test_results"
        all_tests = set()
        
        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    test_details = data.get('test_details', {})
                    
                    for test_list in [
                        test_details.get('failures', []),
                        test_details.get('errors', []),
                        test_details.get('skipped', [])
                    ]:
                        for test in test_list:
                            test_name = str(test.get('test', ''))
                            if test_name:
                                all_tests.add(test_name)
            except Exception:
                continue
        
        # Auto-tag based on patterns
        for test_name in all_tests:
            test_lower = test_name.lower()
            for tag, patterns_list in patterns.items():
                if any(pattern in test_lower for pattern in patterns_list):
                    self.add_tag(test_name, tag)
        
        self._save_tags()
    
    def generate_tag_report(self) -> str:
        """Generate tag report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST TAGGING REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        all_tags = self.get_all_tags()
        
        lines.append(f"Total Tags: {len(all_tags)}")
        lines.append(f"Tagged Tests: {len(self.tags)}")
        lines.append("")
        
        lines.append("📋 TAGS AND TESTS")
        lines.append("-" * 80)
        
        for tag in sorted(all_tags):
            tests = self.get_tests_by_tag(tag)
            description = self.tag_definitions.get(tag, '')
            
            lines.append(f"\n🏷️  {tag.upper()}")
            if description:
                lines.append(f"   Description: {description}")
            lines.append(f"   Tests: {len(tests)}")
            for test in tests[:10]:  # Show first 10
                lines.append(f"     • {test}")
            if len(tests) > 10:
                lines.append(f"     ... and {len(tests) - 10} more")
        
        return "\n".join(lines)

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    tagging = TestTaggingSystem(project_root)
    
    # Auto-tag tests
    tagging.auto_tag_tests()
    print("✅ Auto-tagged tests based on patterns")
    
    # Generate report
    report = tagging.generate_tag_report()
    print("\n" + report)
    
    # Save report
    report_file = project_root / "tagging_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Tagging report saved to: {report_file}")

if __name__ == "__main__":
    main()







