#!/usr/bin/env python3
"""
Quick Start Script for Bulk Chat
=================================

Script simple para iniciar el servidor de chat continuo.
"""

import sys
import os
from pathlib import Path

# Asegurar que el directorio esté en el path
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent.parent

# Agregar directorios al path
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Cambiar al directorio del script
os.chdir(current_dir)

try:
    from bulk_chat.main import main
except ImportError as e:
    print(f"❌ Error importing bulk_chat: {e}")
    print(f"   Current directory: {current_dir}")
    print(f"   Python path: {sys.path[:3]}")
    print("\n💡 Try: python -m bulk_chat.main")
    sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

