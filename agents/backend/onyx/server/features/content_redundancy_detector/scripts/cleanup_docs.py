"""
Script to clean up duplicate documentation files
Moves deprecated docs to archive/ directory
"""

import os
import shutil
from pathlib import Path

# Main documentation files to keep
KEEP_FILES = {
    "README.md",
    "README_REFACTORING.md",
    "REFACTORING_COMPLETE_FINAL.md",
    "REFACTORING_FINAL_REPORT.md",
    "MIGRATION_GUIDE.md",
    "QUICK_START.md",
    "DEPLOYMENT_GUIDE.md",
    "ORGANIZATION_GUIDE.md",
}

# Files to archive (duplicates)
ARCHIVE_PATTERNS = [
    "*SUMMARY*.md",
    "*ULTIMATE*.md",
    "*REFACTORING*.md",
    "*IMPROVEMENTS*.md",
    "*SYSTEM*.md",
    "*ADVANCED*.md",
    "*OPTIMIZED*.md",
    "*ENHANCED*.md",
    "*ENTERPRISE*.md",
    "*MEJORAS*.md",
    "*STATUS*.md",
    "*PROGRESS*.md",
    "*GUIDE*.md",
    "*ARCHITECTURE*.md",
    "*COMPLETE*.md",
    "*FINAL*.md",
]

def should_archive(filename: str) -> bool:
    """Check if file should be archived"""
    if filename in KEEP_FILES:
        return False
    
    # Check against patterns
    for pattern in ARCHIVE_PATTERNS:
        pattern_clean = pattern.replace("*", "").replace(".md", "")
        if pattern_clean.lower() in filename.lower():
            return True
    
    return False

def main():
    """Main cleanup function"""
    base_dir = Path(__file__).parent.parent
    archive_dir = base_dir / "docs" / "archive"
    
    # Create archive directory
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all markdown files
    md_files = list(base_dir.glob("*.md"))
    
    archived_count = 0
    kept_count = 0
    
    for md_file in md_files:
        filename = md_file.name
        
        if should_archive(filename):
            # Move to archive
            dest = archive_dir / filename
            if not dest.exists():
                shutil.move(str(md_file), str(dest))
                print(f"📦 Archived: {filename}")
                archived_count += 1
            else:
                # File already in archive, just remove
                md_file.unlink()
                print(f"🗑️  Removed duplicate: {filename}")
                archived_count += 1
        else:
            kept_count += 1
            print(f"✅ Kept: {filename}")
    
    print(f"\n📊 Summary:")
    print(f"   Archived: {archived_count} files")
    print(f"   Kept: {kept_count} files")
    print(f"   Archive location: {archive_dir}")

if __name__ == "__main__":
    main()




