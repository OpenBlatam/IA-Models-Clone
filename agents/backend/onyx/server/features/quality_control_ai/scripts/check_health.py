#!/usr/bin/env python3
"""
Health Check Script

Script to check the health of the Quality Control AI service.
"""

import requests
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from quality_control_ai.config.app_settings import get_settings

def check_health():
    """Check service health."""
    settings = get_settings()
    url = f"http://{settings.api_host}:{settings.api_port}/api/v1/health"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Service is healthy")
        print(f"   Status: {data.get('status')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Uptime: {data.get('uptime_seconds', 0):.0f} seconds")
        print(f"   Total Inspections: {data.get('total_inspections', 0)}")
        print(f"   Success Rate: {data.get('success_rate', 0):.2f}%")
        
        return 0
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Service is unhealthy: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(check_health())



