"""Fix backward compatibility import paths"""
from pathlib import Path
import re

CORE_DIR = Path(__file__).parent.parent / "core"

def fix_imports_in_file(file_path: Path):
    """Fix imports in a backward compat file"""
    if not file_path.exists():
        return False
    
    content = file_path.read_text()
    original_content = content
    
    # Fix patterns like "from ..domain" to "from .domain"
    # Fix patterns like "from ..infrastructure" to "from .infrastructure"
    # Fix patterns like "from ..utils" to "from .utils"
    # Fix patterns like "from ..services" to "from .services"
    # Fix patterns like "from ..ai" to "from .ai"
    # Fix patterns like "from ..mcp" to "from .mcp"
    
    patterns = [
        (r'from \.\.domain\.', r'from .domain.'),
        (r'from \.\.infrastructure\.', r'from .infrastructure.'),
        (r'from \.\.utils\.', r'from .utils.'),
        (r'from \.\.services\.', r'from .services.'),
        (r'from \.\.ai\.', r'from .ai.'),
        (r'from \.\.mcp\.', r'from .mcp.'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        file_path.write_text(content)
        return True
    return False

def fix_all_backward_compat():
    """Fix all backward compatibility files"""
    fixed = 0
    
    for file in CORE_DIR.glob("*.py"):
        if file.name in ["__init__.py", "agent.py", "task_executor.py", 
                        "command_executor.py", "command_listener.py", 
                        "command_validator.py", "constants.py", "signal_handler.py",
                        "resource_manager.py", "gradio_interface.py"]:
            continue
        
        if fix_imports_in_file(file):
            fixed += 1
            print(f"  ✓ Fixed {file.name}")
    
    print(f"\n✅ Fixed {fixed} backward compatibility files")
    return fixed

if __name__ == "__main__":
    print("🔧 Fixing backward compatibility imports...\n")
    fix_all_backward_compat()






