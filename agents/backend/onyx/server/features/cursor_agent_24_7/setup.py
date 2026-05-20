"""
Setup script para Cursor Agent 24/7
====================================

Instala el paquete y crea el comando 'cursor-agent'
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="cursor-agent-24-7",
    version="1.0.0",
    description="Agente persistente 24/7 para ejecutar comandos desde Cursor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Blatam Academy",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "pydantic>=2.9.0",
        "typer>=0.12.0",
        "click>=8.1.7",
    ],
    entry_points={
        "console_scripts": [
            "cursor-agent=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)




