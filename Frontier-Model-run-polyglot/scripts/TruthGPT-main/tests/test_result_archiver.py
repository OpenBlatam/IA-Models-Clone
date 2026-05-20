"""
Test Result Archiver
Automatically archives old test results with compression and organization
"""

import json
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import tarfile


class TestResultArchiver:
    """Archive old test results automatically"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.archive_dir = project_root / "test_archives"
        self.archive_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def archive_old_results(
        self,
        days_old: int = 30,
        compress: bool = True,
        organize_by_date: bool = True
    ) -> Dict:
        """Archive results older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        archived = []
        total_size = 0
        
        # Find old result files
        for result_file in self.results_dir.glob("*.json"):
            try:
                # Check file modification time
                file_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                
                if file_time < cutoff_date:
                    # Load to check timestamp
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        result_timestamp = data.get('timestamp', '')
                        
                        if result_timestamp:
                            result_date = datetime.fromisoformat(result_timestamp)
                            if result_date < cutoff_date:
                                # Archive this file
                                archive_path = self._get_archive_path(
                                    result_file,
                                    result_date,
                                    organize_by_date
                                )
                                
                                if compress:
                                    archive_path = self._compress_file(result_file, archive_path)
                                else:
                                    shutil.move(str(result_file), str(archive_path))
                                
                                archived.append({
                                    'original': str(result_file.name),
                                    'archived': str(archive_path),
                                    'size': result_file.stat().st_size,
                                    'date': result_timestamp
                                })
                                
                                total_size += result_file.stat().st_size
            except Exception as e:
                print(f"Error archiving {result_file}: {e}")
        
        return {
            'archived_count': len(archived),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'archived_files': archived
        }
    
    def _get_archive_path(
        self,
        result_file: Path,
        result_date: datetime,
        organize_by_date: bool
    ) -> Path:
        """Get archive path for a file"""
        if organize_by_date:
            # Organize by year/month
            year_month = result_date.strftime("%Y-%m")
            archive_subdir = self.archive_dir / year_month
            archive_subdir.mkdir(exist_ok=True)
            return archive_subdir / result_file.name
        else:
            return self.archive_dir / result_file.name
    
    def _compress_file(self, source_file: Path, target_file: Path) -> Path:
        """Compress a file using gzip"""
        compressed_path = target_file.with_suffix(target_file.suffix + '.gz')
        
        with open(source_file, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original
        source_file.unlink()
        
        return compressed_path
    
    def create_archive_package(
        self,
        date_from: datetime = None,
        date_to: datetime = None,
        output_file: Path = None
    ) -> Path:
        """Create a tar.gz archive package"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.archive_dir / f"test_results_{timestamp}.tar.gz"
        
        with tarfile.open(output_file, 'w:gz') as tar:
            # Add result files
            for result_file in self.results_dir.glob("*.json"):
                file_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                
                if date_from and file_time < date_from:
                    continue
                if date_to and file_time > date_to:
                    continue
                
                tar.add(result_file, arcname=result_file.name)
        
        print(f"✅ Archive package created: {output_file}")
        return output_file
    
    def list_archives(self) -> List[Dict]:
        """List all archived files"""
        archives = []
        
        for archive_file in self.archive_dir.rglob("*"):
            if archive_file.is_file():
                archives.append({
                    'path': str(archive_file.relative_to(self.archive_dir)),
                    'size': archive_file.stat().st_size,
                    'modified': datetime.fromtimestamp(archive_file.stat().st_mtime).isoformat(),
                    'compressed': archive_file.suffix == '.gz'
                })
        
        return sorted(archives, key=lambda x: x['modified'], reverse=True)
    
    def restore_archive(
        self,
        archive_path: Path,
        extract_to: Path = None
    ) -> Path:
        """Restore files from archive"""
        if extract_to is None:
            extract_to = self.results_dir
        
        if archive_path.suffix == '.gz':
            # Decompress gzip file
            target_file = extract_to / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(target_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return target_file
        elif archive_path.suffixes == ['.tar', '.gz']:
            # Extract tar.gz
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
            return extract_to
        else:
            # Just copy
            target_file = extract_to / archive_path.name
            shutil.copy2(archive_path, target_file)
            return target_file
    
    def get_archive_statistics(self) -> Dict:
        """Get statistics about archives"""
        archives = self.list_archives()
        
        total_size = sum(a['size'] for a in archives)
        compressed_count = sum(1 for a in archives if a['compressed'])
        
        return {
            'total_archives': len(archives),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'compressed_count': compressed_count,
            'uncompressed_count': len(archives) - compressed_count
        }


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result Archiver')
    parser.add_argument('--archive', type=int, help='Archive results older than N days')
    parser.add_argument('--package', action='store_true', help='Create archive package')
    parser.add_argument('--list', action='store_true', help='List archives')
    parser.add_argument('--restore', type=str, help='Restore archive file')
    parser.add_argument('--stats', action='store_true', help='Show archive statistics')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    archiver = TestResultArchiver(project_root)
    
    if args.archive:
        print(f"📦 Archiving results older than {args.archive} days...")
        result = archiver.archive_old_results(args.archive)
        print(f"✅ Archived {result['archived_count']} files ({result['total_size_mb']} MB)")
    elif args.package:
        print("📦 Creating archive package...")
        package = archiver.create_archive_package()
        print(f"✅ Package created: {package}")
    elif args.list:
        print("📦 Listing archives...")
        archives = archiver.list_archives()
        for arch in archives[:20]:
            print(f"  {arch['path']}: {arch['size'] / 1024:.1f} KB")
    elif args.restore:
        print(f"📦 Restoring archive: {args.restore}...")
        restored = archiver.restore_archive(Path(args.restore))
        print(f"✅ Restored to: {restored}")
    elif args.stats:
        print("📊 Archive Statistics:")
        stats = archiver.get_archive_statistics()
        print(f"  Total Archives: {stats['total_archives']}")
        print(f"  Total Size: {stats['total_size_mb']} MB")
        print(f"  Compressed: {stats['compressed_count']}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

