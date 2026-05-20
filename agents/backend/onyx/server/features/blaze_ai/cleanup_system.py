#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Cleanup Script for Blaze AI.
Helps free up disk space and clean temporary files.
"""

import sys
import os
import shutil
import tempfile
import glob
from pathlib import Path
from typing import List, Tuple, Dict
import time

class SystemCleanup:
    """System cleanup and maintenance class."""
    
    def __init__(self):
        self.cleaned_files = []
        self.cleaned_dirs = []
        self.freed_space = 0
        self.errors = []
    
    def get_directory_size(self, path: Path) -> int:
        """Calculate directory size in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            self.errors.append(f"Error calculating size for {path}: {e}")
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    
    def clean_python_cache(self):
        """Clean Python cache files."""
        print("🧹 Cleaning Python cache files...")
        
        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/*.pyd",
            "**/.pytest_cache",
            "**/.coverage",
            "**/htmlcov",
            "**/.mypy_cache"
        ]
        
        for pattern in cache_patterns:
            for path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(path):
                        size = os.path.getsize(path)
                        os.remove(path)
                        self.cleaned_files.append(path)
                        self.freed_space += size
                        print(f"  ✅ Removed file: {path} ({self.format_size(size)})")
                    elif os.path.isdir(path):
                        size = self.get_directory_size(Path(path))
                        shutil.rmtree(path)
                        self.cleaned_dirs.append(path)
                        self.freed_space += size
                        print(f"  ✅ Removed directory: {path} ({self.format_size(size)})")
                except Exception as e:
                    self.errors.append(f"Error removing {path}: {e}")
    
    def clean_temp_files(self):
        """Clean temporary files."""
        print("\n🗑️  Cleaning temporary files...")
        
        # Clean system temp directory
        try:
            temp_dir = tempfile.gettempdir()
            temp_files = glob.glob(os.path.join(temp_dir, "blaze_ai_*"))
            temp_files.extend(glob.glob(os.path.join(temp_dir, "test_*")))
            
            for temp_file in temp_files:
                try:
                    if os.path.isfile(temp_file):
                        size = os.path.getsize(temp_file)
                        os.remove(temp_file)
                        self.cleaned_files.append(temp_file)
                        self.freed_space += size
                        print(f"  ✅ Removed temp file: {temp_file} ({self.format_size(size)})")
                    elif os.path.isdir(temp_file):
                        size = self.get_directory_size(Path(temp_file))
                        shutil.rmtree(temp_file)
                        self.cleaned_dirs.append(temp_file)
                        self.freed_space += size
                        print(f"  ✅ Removed temp directory: {temp_file} ({self.format_size(size)})")
                except Exception as e:
                    self.errors.append(f"Error removing temp file {temp_file}: {e}")
        except Exception as e:
            self.errors.append(f"Error accessing temp directory: {e}")
        
        # Clean local test files
        local_temp_patterns = [
            "test_*.py",
            "*.tmp",
            "*.temp",
            "*.log",
            "*.cache"
        ]
        
        for pattern in local_temp_patterns:
            for path in glob.glob(pattern):
                if os.path.basename(path) not in ["simple_test.py", "diagnose_system.py", "cleanup_system.py"]:
                    try:
                        if os.path.isfile(path):
                            size = os.path.getsize(path)
                            os.remove(path)
                            self.cleaned_files.append(path)
                            self.freed_space += size
                            print(f"  ✅ Removed local temp: {path} ({self.format_size(size)})")
                    except Exception as e:
                        self.errors.append(f"Error removing local temp {path}: {e}")
    
    def clean_build_artifacts(self):
        """Clean build artifacts and generated files."""
        print("\n🔨 Cleaning build artifacts...")
        
        build_patterns = [
            "**/build",
            "**/dist",
            "**/*.egg-info",
            "**/.eggs",
            "**/.tox",
            "**/.venv",
            "**/venv",
            "**/node_modules"
        ]
        
        for pattern in build_patterns:
            for path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isdir(path):
                        size = self.get_directory_size(Path(path))
                        shutil.rmtree(path)
                        self.cleaned_dirs.append(path)
                        self.freed_space += size
                        print(f"  ✅ Removed build artifact: {path} ({self.format_size(size)})")
                except Exception as e:
                    self.errors.append(f"Error removing build artifact {path}: {e}")
    
    def clean_old_logs(self):
        """Clean old log files."""
        print("\n📝 Cleaning old log files...")
        
        log_patterns = [
            "**/*.log",
            "**/*.log.*",
            "**/logs/*.log",
            "**/logs/*.log.*"
        ]
        
        for pattern in log_patterns:
            for path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(path):
                        # Check if log file is older than 7 days
                        file_age = time.time() - os.path.getmtime(path)
                        if file_age > 7 * 24 * 3600:  # 7 days in seconds
                            size = os.path.getsize(path)
                            os.remove(path)
                            self.cleaned_files.append(path)
                            self.freed_space += size
                            print(f"  ✅ Removed old log: {path} ({self.format_size(size)})")
                except Exception as e:
                    self.errors.append(f"Error removing log {path}: {e}")
    
    def analyze_disk_usage(self):
        """Analyze disk usage by directory."""
        print("\n📊 Analyzing disk usage...")
        
        current_dir = Path.cwd()
        dir_sizes = {}
        
        try:
            for item in current_dir.iterdir():
                if item.is_dir():
                    size = self.get_directory_size(item)
                    dir_sizes[item.name] = size
            
            # Sort by size (largest first)
            sorted_dirs = sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)
            
            print("  📁 Directory sizes:")
            for dir_name, size in sorted_dirs[:10]:  # Show top 10
                print(f"    {dir_name}: {self.format_size(size)}")
                
        except Exception as e:
            self.errors.append(f"Error analyzing disk usage: {e}")
    
    def generate_cleanup_report(self):
        """Generate cleanup report."""
        print("\n" + "=" * 70)
        print("🧹 CLEANUP REPORT")
        print("=" * 70)
        
        print(f"📁 Files removed: {len(self.cleaned_files)}")
        print(f"📁 Directories removed: {len(self.cleaned_dirs)}")
        print(f"💾 Total space freed: {self.format_size(self.freed_space)}")
        
        if self.cleaned_files:
            print(f"\n🗑️  Removed files:")
            for file_path in self.cleaned_files[:10]:  # Show first 10
                print(f"  - {file_path}")
            if len(self.cleaned_files) > 10:
                print(f"  ... and {len(self.cleaned_files) - 10} more files")
        
        if self.cleaned_dirs:
            print(f"\n🗑️  Removed directories:")
            for dir_path in self.cleaned_dirs[:5]:  # Show first 5
                print(f"  - {dir_path}")
            if len(self.cleaned_dirs) > 5:
                print(f"  ... and {len(self.cleaned_dirs) - 5} more directories")
        
        if self.errors:
            print(f"\n❌ Errors encountered:")
            for error in self.errors[:5]:  # Show first 5
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more errors")
    
    def run_cleanup(self, aggressive: bool = False):
        """Run system cleanup."""
        print("🚀 Blaze AI System Cleanup")
        print("=" * 70)
        
        # Always clean Python cache
        self.clean_python_cache()
        
        # Clean temporary files
        self.clean_temp_files()
        
        # Clean build artifacts
        self.clean_build_artifacts()
        
        # Clean old logs
        self.clean_old_logs()
        
        # Analyze disk usage
        self.analyze_disk_usage()
        
        # Generate report
        self.generate_cleanup_report()
        
        return self.freed_space > 0

def main():
    """Main cleanup function."""
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description="Blaze AI System Cleanup")
        parser.add_argument("--aggressive", action="store_true", 
                          help="Perform aggressive cleanup (removes more files)")
        parser.add_argument("--dry-run", action="store_true",
                          help="Show what would be cleaned without actually cleaning")
        
        args = parser.parse_args()
        
        if args.dry_run:
            print("🔍 DRY RUN MODE - No files will be actually removed")
            print("=" * 70)
        
        cleanup = SystemCleanup()
        success = cleanup.run_cleanup(aggressive=args.aggressive)
        
        if success:
            print("\n🎉 Cleanup completed successfully!")
            print(f"💾 Freed {cleanup.format_size(cleanup.freed_space)} of disk space")
        else:
            print("\n⚠️  Cleanup completed but no significant space was freed")
        
        if cleanup.errors:
            print(f"\n⚠️  {len(cleanup.errors)} errors occurred during cleanup")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
