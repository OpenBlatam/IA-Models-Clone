from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import shutil
from pathlib import Path
import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Script to rename files and directories to follow lowercase with underscores convention.
This script will recursively process the instagram_captions directory and rename all
files and directories to use lowercase with underscores.
"""


def to_snake_case(name) -> Any:
    """Convert a string to snake_case (lowercase with underscores)."""
    # Remove file extensions first
    name_without_ext = name
    extension = ""
    if "." in name:
        name_without_ext, extension = name.rsplit(".", 1)
    
    # Convert to snake_case
    # Handle camelCase and PascalCase
    snake_case = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name_without_ext)
    # Handle consecutive uppercase letters
    snake_case = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', snake_case)
    # Convert to lowercase and replace spaces/hyphens with underscores
    snake_case = re.sub(r'[-\s]+', '_', snake_case.lower())
    # Remove any remaining non-alphanumeric characters except underscores
    snake_case = re.sub(r'[^a-z0-9_]', '_', snake_case)
    # Remove multiple consecutive underscores
    snake_case = re.sub(r'_+', '_', snake_case)
    # Remove leading/trailing underscores
    snake_case = snake_case.strip('_')
    
    # Add extension back if it existed
    if extension:
        snake_case = f"{snake_case}.{extension.lower()}"
    
    return snake_case

def should_skip_directory(dir_name) -> Any:
    """Check if directory should be skipped (like __pycache__, .git, etc.)."""
    skip_dirs = {
        '__pycache__', '.git', '.vscode', '.idea', 'node_modules',
        'venv', 'env', '.env', 'dist', 'build', 'target'
    }
    return dir_name in skip_dirs

def rename_files_and_directories(root_path) -> Any:
    """Recursively rename files and directories to snake_case."""
    root_path = Path(root_path)
    
    # First, collect all paths that need to be renamed
    paths_to_rename = []
    
    for root, dirs, files in os.walk(root_path, topdown=False):
        root_path_obj = Path(root)
        
        # Skip certain directories
        dirs[:] = [d for d in dirs if not should_skip_directory(d)]
        
        # Collect directories to rename
        for dir_name in dirs:
            dir_path = root_path_obj / dir_name
            new_name = to_snake_case(dir_name)
            if new_name != dir_name:
                paths_to_rename.append((dir_path, new_name))
        
        # Collect files to rename
        for file_name in files:
            file_path = root_path_obj / file_name
            new_name = to_snake_case(file_name)
            if new_name != file_name:
                paths_to_rename.append((file_path, new_name))
    
    # Sort paths by depth (deepest first) to avoid conflicts
    paths_to_rename.sort(key=lambda x: len(str(x[0]).split(os.sep)), reverse=True)
    
    # Perform the renames
    renamed_count = 0
    for old_path, new_name in paths_to_rename:
        new_path = old_path.parent / new_name
        
        try:
            if old_path.exists():
                old_path.rename(new_path)
                print(f"Renamed: {old_path} -> {new_path}")
                renamed_count += 1
        except Exception as e:
            print(f"Error renaming {old_path}: {e}")
    
    print(f"\nTotal files/directories renamed: {renamed_count}")

def main():
    """Main function to execute the renaming process."""
    # Get the current directory (instagram_captions)
    current_dir = Path(__file__).parent
    
    print("Starting file and directory renaming process...")
    print(f"Working directory: {current_dir}")
    print("Converting all names to lowercase with underscores...")
    print()
    
    # Confirm before proceeding
    response = input("Do you want to proceed with renaming? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    try:
        rename_files_and_directories(current_dir)
        print("\nRenaming process completed successfully!")
        print("\nNote: You may need to update import statements in your code files.")
        print("Consider running a search and replace for old import paths.")
        
    except Exception as e:
        print(f"Error during renaming process: {e}")

match __name__:
    case "__main__":
    main() 