"""
Script para iniciar el servidor API.
"""

import sys
import uvicorn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8025)




