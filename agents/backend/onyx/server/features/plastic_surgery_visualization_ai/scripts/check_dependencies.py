#!/usr/bin/env python3
"""Check if all dependencies are installed."""

import sys
from importlib import util


REQUIRED = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "aiofiles",
    "aiohttp",
    "PIL",
    "numpy",
]

OPTIONAL = [
    "cv2",  # opencv-python-headless
    "skimage",  # scikit-image
    "imageio",
    "validators",
    "prometheus_client",
    "sentry_sdk",
]


def check_package(name: str) -> bool:
    """Check if package is installed."""
    # Handle special cases
    package_map = {
        "PIL": "Pillow",
        "cv2": "cv2",
        "skimage": "skimage",
    }
    
    import_name = package_map.get(name, name)
    
    spec = util.find_spec(import_name)
    return spec is not None


def main():
    """Check dependencies."""
    print("Checking dependencies...\n")
    
    # Required
    print("Required dependencies:")
    all_required = True
    for package in REQUIRED:
        installed = check_package(package)
        status = "✓" if installed else "✗"
        print(f"  {status} {package}")
        if not installed:
            all_required = False
    
    print("\nOptional dependencies:")
    for package in OPTIONAL:
        installed = check_package(package)
        status = "✓" if installed else "○"
        print(f"  {status} {package}")
    
    if not all_required:
        print("\n✗ Some required dependencies are missing!")
        print("  Install with: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✓ All required dependencies are installed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

