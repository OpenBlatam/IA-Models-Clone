#!/usr/bin/env python3
"""
Modular Ultimate Enhanced Supreme Production Ultra-Optimal Bulk TruthGPT AI System
Flask application with blueprints and proper separation of concerns
"""

import os
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent / 'app'
sys.path.insert(0, str(app_dir))

from app import create_app

def main():
    """Main application entry point."""
    # Get configuration from environment
    config_name = os.getenv('FLASK_ENV', 'development')
    
    # Create Flask application
    app = create_app(config_name)
    
    # Get host and port from environment
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"🚀 Starting Ultimate Enhanced Supreme Production Ultra-Optimal Bulk TruthGPT AI System")
    print(f"📊 Environment: {config_name}")
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug: {debug}")
    print(f"📁 App Directory: {app_dir}")
    
    # Run the application
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main()









