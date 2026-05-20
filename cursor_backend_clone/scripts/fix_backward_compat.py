"""Fix backward compatibility import paths"""
from pathlib import Path
import re

CORE_DIR = Path(__file__).parent.parent / "core"

def fix_import_path(content: str, dest_path: Path) -> str:
    """Fix the import path in backward compat file"""
    # Calculate relative import path
    rel_path = dest_path.relative_to(CORE_DIR)
    import_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
    import_path = ".".join(import_parts)
    
    # Fix the import statement
    pattern = r'from \.\.domain/exceptions import \*'
    replacement = f'from ..domain.exceptions import *'
    
    # More general pattern
    pattern2 = r'from \.\.([^"]+)/'
    def replace_slash(match):
        path = match.group(1).replace('/', '.')
        return f'from ..{path}.'
    
    content = re.sub(pattern2, replace_slash, content)
    
    # Fix the warning message
    content = re.sub(
        r'Use domain\.exceptions instead',
        f'Use {import_path} instead',
        content
    )
    
    return content

def fix_all_backward_compat():
    """Fix all backward compatibility files"""
    fixed = 0
    
    # Map of source files to their destinations
    mappings = {
        "exceptions.py": "domain/exceptions.py",
    }
    
    for source_name, dest_path_str in mappings.items():
        source_file = CORE_DIR / source_name
        dest_file = CORE_DIR / dest_path_str
        
        if not source_file.exists():
            continue
        
        content = source_file.read_text()
        
        # Calculate correct import
        rel_path = dest_file.relative_to(CORE_DIR)
        import_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        import_path = ".".join(import_parts)
        
        # Fix content
        new_content = f'''"""
Backward compatibility re-export for {source_name}

This file is deprecated. Use {import_path} instead.
"""
import warnings

warnings.warn(
    "{source_name} is deprecated. Use {import_path} instead.",
    DeprecationWarning,
    stacklevel=2
)

from ..{import_path.replace(".", "/")} import *
'''
        
        source_file.write_text(new_content)
        fixed += 1
        print(f"  ✓ Fixed {source_name}")
    
    # Fix all other backward compat files
    for file in CORE_DIR.glob("*.py"):
        if file.name in ["__init__.py", "agent.py", "task_executor.py", 
                        "command_executor.py", "command_listener.py", 
                        "command_validator.py", "constants.py", "signal_handler.py",
                        "resource_manager.py", "gradio_interface.py"]:
            continue
        
        content = file.read_text()
        if "Backward compatibility" in content and "from .." in content:
            # Try to find the correct destination
            # This is a simplified fix - we'll regenerate properly
            if "/" in content and "from .." in content:
                # Fix slashes in import paths
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith('from ..') and '/' in line:
                        # Convert path slashes to dots
                        line = line.replace('/', '.')
                    new_lines.append(line)
                file.write_text('\n'.join(new_lines))
                fixed += 1
                print(f"  ✓ Fixed {file.name}")
    
    print(f"\n✅ Fixed {fixed} backward compatibility files")
    return fixed

if __name__ == "__main__":
    print("🔧 Fixing backward compatibility imports...\n")
    fix_all_backward_compat()






