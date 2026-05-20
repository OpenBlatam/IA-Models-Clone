#!/usr/bin/env python3
"""
Setup script for Optimized Email Sequence AI System
"""

import os
from setuptools import setup, find_packages


def read_file(filename: str) -> str:
    """Read file contents safely"""
    file_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Warning: Could not read {filename}: {e}")
    return ""


def read_requirements(filename: str) -> list:
    """Read requirements from file"""
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        if os.path.exists(requirements_path):
            with open(requirements_path, "r", encoding="utf-8") as f:
                return [
                    line.strip() 
                    for line in f 
                    if line.strip() and not line.startswith("#")
                ]
    except Exception as e:
        print(f"Warning: Could not read requirements from {filename}: {e}")
    return []


# Read files
README = read_file("README.md")
LONG_DESCRIPTION = README if README else "Optimized Email Sequence AI System with Advanced Performance and Profiling"

# Read requirements
INSTALL_REQUIRES = read_requirements("requirements-minimal.txt")
DEV_REQUIRES = read_requirements("requirements-dev.txt")

setup(
    name="email-sequence-ai-optimized",
    version="2.0.0",
    author="Blatam Academy",
    author_email="contact@blatamacademy.com",
    description="Optimized Email Sequence AI System with Advanced Performance and Profiling",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/blatamacademy/email-sequence-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Communications :: Email",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ],
        "distributed": [
            "torch>=2.0.0",
            "horovod>=0.28.0",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "google-cloud-storage>=2.10.0",
            "azure-storage-blob>=12.17.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
            "sentry-sdk>=1.28.0",
        ],
        "profiling": [
            "memory-profiler>=0.61.0",
            "line-profiler>=4.1.0",
            "py-spy>=0.3.0",
            "pyinstrument>=4.6.0",
        ],
        "full": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.12.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "gradio>=3.35.0",
            "python-dotenv>=1.0.0",
            "pyyaml>=6.0",
            "loguru>=0.7.0",
            "tqdm>=4.65.0",
            "pydantic>=2.0.0",
            "pytest>=7.4.0",
            "better-exceptions>=0.3.0",
            "prometheus-client>=0.17.0",
            "memory-profiler>=0.61.0",
            "line-profiler>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "email-sequence-ai=email_sequence.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "email",
        "sequence",
        "ai",
        "langchain",
        "automation",
        "personalization",
        "machine-learning",
        "nlp",
        "optimization",
        "performance"
    ],
    project_urls={
        "Bug Reports": "https://github.com/blatamacademy/email-sequence-ai/issues",
        "Source": "https://github.com/blatamacademy/email-sequence-ai",
        "Documentation": "https://github.com/blatamacademy/email-sequence-ai/docs",
    },
) 