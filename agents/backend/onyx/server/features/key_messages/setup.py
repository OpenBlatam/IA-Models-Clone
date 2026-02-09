from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import sys
from setuptools import setup, find_packages
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Setup script for Key Messages feature.
"""

# Read the README file
def read_readme():
    """Read README.md file."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    return "Advanced Key Messages feature with cybersecurity capabilities"

# Read requirements from file
def read_requirements(filename) -> Any:
    """Read requirements from file."""
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(requirements_path):
        return []
    
    with open(requirements_path, "r", encoding="utf-8") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements

# Get version from file or default
def get_version():
    """Get version from file or default."""
    version_file = os.path.join(os.path.dirname(__file__), "key_messages", "_version.py")
    if os.path.exists(version_file):
        with open(version_file, "r", encoding="utf-8") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"\'')
    return "2.0.0"

# Setup configuration
setup(
    name="key-messages",
    version=get_version(),
    author="Blatam Academy",
    author_email="dev@blatam.academy",
    description="Advanced Key Messages feature with cybersecurity capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/blatam-academy/key-messages",
    project_urls={
        "Bug Tracker": "https://github.com/blatam-academy/key-messages/issues",
        "Documentation": "https://key-messages.readthedocs.io",
        "Source Code": "https://github.com/blatam-academy/key-messages",
    },
    packages=find_packages(include=["key_messages", "key_messages.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "test": read_requirements("requirements-test.txt"),
        "cyber": read_requirements("requirements-cyber.txt"),
        "ml": [
            "diffusers>=0.25.0,<1.0.0",
            "matplotlib>=3.7.0,<4.0.0",
            "seaborn>=0.12.0,<1.0.0",
            "plotly>=5.17.0,<6.0.0",
            "scipy>=1.11.0,<2.0.0",
            "tensorboard>=2.15.0,<3.0.0",
            "wandb>=0.16.0,<1.0.0",
            "mlflow>=2.8.0,<3.0.0",
            "tqdm>=4.66.0,<5.0.0",
            "rich>=13.7.0,<14.0.0",
            "click>=8.1.0,<9.0.0",
            "gradio>=4.0.0,<5.0.0",
        ],
        "minimal": read_requirements("requirements-minimal.txt"),
        "all": [
            "key-messages[dev,test,cyber,ml]"
        ],
    },
    entry_points={
        "console_scripts": [
            "key-messages=key_messages.cli:main",
            "key-messages-api=key_messages.api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "key_messages": [
            "py.typed",
            "*.pyi",
            "*.txt",
            "*.md",
            "*.yaml",
            "*.yml",
        ],
    },
    zip_safe=False,
    keywords=[
        "key-messages",
        "cybersecurity",
        "machine-learning",
        "fastapi",
        "artificial-intelligence",
        "security",
        "penetration-testing",
        "vulnerability-scanning",
        "network-security",
        "web-security",
    ],
    platforms=["any"],
    license="MIT",
    download_url="https://github.com/blatam-academy/key-messages/archive/v2.0.0.tar.gz",
) 