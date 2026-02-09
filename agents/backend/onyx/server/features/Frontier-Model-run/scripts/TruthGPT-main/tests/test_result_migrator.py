"""
Test Result Migrator
Migrate test results between formats and versions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class TestResultMigrator:
    """Migrate test results between formats"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.migrations_dir = project_root / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)
    
    def migrate_to_v2(
        self,
        old_result: Dict
    ) -> Dict:
        """Migrate from v1 to v2 format"""
        # v1 format assumed to have different structure
        # v2 format has: timestamp, run_name, summary, test_details
        
        migrated = {
            'version': '2.0',
            'timestamp': old_result.get('timestamp', datetime.now().isoformat()),
            'run_name': old_result.get('run_name', 'migrated_run'),
            'summary': {
                'total_tests': old_result.get('total_tests', 0),
                'passed': old_result.get('passed', 0),
                'failed': old_result.get('failed', 0),
                'errors': old_result.get('errors', 0),
                'skipped': old_result.get('skipped', 0),
                'success_rate': old_result.get('success_rate', 0),
                'execution_time': old_result.get('execution_time', 0)
            },
            'test_details': {}
        }
        
        # Migrate test details
        if 'tests' in old_result:
            # Old format: list of tests
            for test in old_result['tests']:
                test_name = test.get('name', 'unknown')
                migrated['test_details'][test_name] = {
                    'status': test.get('status', 'unknown'),
                    'duration': test.get('duration', 0),
                    'error_message': test.get('error', ''),
                    'test_file': test.get('file', ''),
                    'test_class': test.get('class', '')
                }
        elif 'test_results' in old_result:
            # Another old format
            for test_name, test_data in old_result['test_results'].items():
                migrated['test_details'][test_name] = {
                    'status': test_data.get('status', 'unknown'),
                    'duration': test_data.get('duration', 0),
                    'error_message': test_data.get('error', ''),
                    'test_file': test_data.get('file', ''),
                    'test_class': test_data.get('class', '')
                }
        
        return migrated
    
    def migrate_file(
        self,
        source_file: Path,
        target_file: Path = None,
        target_version: str = '2.0'
    ) -> Path:
        """Migrate a result file"""
        # Load old format
        with open(source_file, 'r', encoding='utf-8') as f:
            old_result = json.load(f)
        
        # Detect version
        version = old_result.get('version', '1.0')
        
        if version == target_version:
            print(f"✅ File already at version {target_version}")
            return source_file
        
        # Migrate
        if version == '1.0' and target_version == '2.0':
            migrated = self.migrate_to_v2(old_result)
        else:
            print(f"⚠️ Unknown migration path: {version} -> {target_version}")
            return source_file
        
        # Save migrated file
        if target_file is None:
            target_file = source_file.parent / f"{source_file.stem}_v{target_version}.json"
        
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(migrated, f, indent=2)
        
        print(f"✅ Migrated {source_file.name} to version {target_version}")
        return target_file
    
    def migrate_directory(
        self,
        source_dir: Path,
        target_dir: Path = None,
        target_version: str = '2.0'
    ) -> int:
        """Migrate all files in a directory"""
        if target_dir is None:
            target_dir = source_dir.parent / f"{source_dir.name}_v{target_version}"
        
        target_dir.mkdir(exist_ok=True)
        
        migrated_count = 0
        
        for result_file in source_dir.glob("*.json"):
            try:
                target_file = target_dir / result_file.name
                self.migrate_file(result_file, target_file, target_version)
                migrated_count += 1
            except Exception as e:
                print(f"Error migrating {result_file}: {e}")
        
        print(f"✅ Migrated {migrated_count} files")
        return migrated_count
    
    def create_migration_script(
        self,
        from_version: str,
        to_version: str,
        output_file: Path = None
    ) -> Path:
        """Create a migration script"""
        if output_file is None:
            output_file = self.migrations_dir / f"migrate_{from_version}_to_{to_version}.py"
        
        script_content = f'''"""
Migration script: {from_version} -> {to_version}
Generated: {datetime.now().isoformat()}
"""

import json
from pathlib import Path

def migrate(data):
    """Migrate data from {from_version} to {to_version}"""
    # TODO: Implement migration logic
    return data

if __name__ == '__main__':
    import sys
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file.parent / f"{input_file.stem}_migrated.json"
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    migrated = migrate(data)
    
    with open(output_file, 'w') as f:
        json.dump(migrated, f, indent=2)
    
    print(f"✅ Migrated: {{output_file}}")
'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ Migration script created: {output_file}")
        return output_file


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result Migrator')
    parser.add_argument('--migrate-file', type=str, help='Migrate a single file')
    parser.add_argument('--migrate-dir', type=str, help='Migrate directory')
    parser.add_argument('--to-version', type=str, default='2.0', help='Target version')
    parser.add_argument('--create-script', nargs=2, metavar=('FROM', 'TO'), help='Create migration script')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    migrator = TestResultMigrator(project_root)
    
    if args.migrate_file:
        print(f"🔄 Migrating file: {args.migrate_file}")
        migrator.migrate_file(Path(args.migrate_file), target_version=args.to_version)
    elif args.migrate_dir:
        print(f"🔄 Migrating directory: {args.migrate_dir}")
        migrator.migrate_directory(Path(args.migrate_dir), target_version=args.to_version)
    elif args.create_script:
        from_version, to_version = args.create_script
        migrator.create_migration_script(from_version, to_version)
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

