"""
Continuous Test Runner (Watch Mode)
Automatically runs tests when files change
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import List, Set

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestWatcher:
    """Watch for file changes and run tests automatically"""
    
    def __init__(self, watch_dirs: List[str] = None, watch_extensions: Set[str] = None):
        self.project_root = project_root
        self.watch_dirs = watch_dirs or ['core', 'tests']
        self.watch_extensions = watch_extensions or {'.py'}
        self.last_run_time = {}
        self.running = False
        
    def get_files_to_watch(self) -> Set[Path]:
        """Get all files to watch"""
        files = set()
        for watch_dir in self.watch_dirs:
            dir_path = self.project_root / watch_dir
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix in self.watch_extensions:
                        files.add(file_path)
        return files
    
    def check_for_changes(self) -> bool:
        """Check if any watched files have changed"""
        files = self.get_files_to_watch()
        changed = False
        
        for file_path in files:
            mtime = file_path.stat().st_mtime
            if file_path not in self.last_run_time or mtime > self.last_run_time[file_path]:
                self.last_run_time[file_path] = mtime
                changed = True
        
        return changed
    
    def run_tests(self, category: str = None):
        """Run tests"""
        print("\n" + "=" * 60)
        print(f"🔄 Running tests... ({time.strftime('%H:%M:%S')})")
        print("=" * 60)
        
        cmd = [sys.executable, str(self.project_root / 'run_unified_tests.py')]
        if category:
            cmd.append(category)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=False,
                text=True
            )
            return result.returncode == 0
        except KeyboardInterrupt:
            print("\n⏹️  Test run interrupted")
            return False
    
    def watch(self, category: str = None, interval: float = 1.0):
        """Start watching for changes"""
        print("👀 Test Watcher Started")
        print("=" * 60)
        print(f"📁 Watching directories: {', '.join(self.watch_dirs)}")
        print(f"📝 Watching extensions: {', '.join(self.watch_extensions)}")
        if category:
            print(f"🎯 Test category: {category}")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")
        
        # Initialize file times
        files = self.get_files_to_watch()
        for file_path in files:
            self.last_run_time[file_path] = file_path.stat().st_mtime
        
        # Run tests initially
        self.run_tests(category)
        
        try:
            while True:
                time.sleep(interval)
                if self.check_for_changes():
                    self.run_tests(category)
        except KeyboardInterrupt:
            print("\n\n👋 Test watcher stopped")
            sys.exit(0)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch for file changes and run tests')
    parser.add_argument('category', nargs='?', help='Test category to run (optional)')
    parser.add_argument('--interval', type=float, default=1.0, 
                       help='Check interval in seconds (default: 1.0)')
    parser.add_argument('--dirs', nargs='+', default=['core', 'tests'],
                       help='Directories to watch (default: core tests)')
    
    args = parser.parse_args()
    
    watcher = TestWatcher(watch_dirs=args.dirs)
    watcher.watch(category=args.category, interval=args.interval)


if __name__ == "__main__":
    main()







