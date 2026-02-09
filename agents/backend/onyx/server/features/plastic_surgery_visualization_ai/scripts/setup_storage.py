#!/usr/bin/env python3
"""Setup storage directories."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


def setup_storage():
    """Create storage directories."""
    directories = [
        Path(settings.upload_dir),
        Path(settings.output_dir),
        Path("./storage/cache"),
        Path("./storage/metrics"),
        Path("./storage/logs"),
    ]
    
    print("Setting up storage directories...")
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {directory}")
    
    # Create .gitkeep files
    for directory in directories:
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            print(f"✓ Created .gitkeep in {directory}")
    
    print("\nStorage setup complete!")


if __name__ == "__main__":
    setup_storage()

