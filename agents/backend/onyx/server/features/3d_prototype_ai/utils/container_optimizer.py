"""
Container Optimizer - Optimizaciones para contenedores ligeros
==============================================================

Incluye:
- Multi-stage builds
- Layer caching
- Minimal base images
- Binary optimization
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ContainerOptimizer:
    """Optimizador de contenedores"""
    
    @staticmethod
    def generate_optimized_dockerfile(base_image: str = "python:3.11-slim",
                                     multi_stage: bool = True) -> str:
        """Genera Dockerfile optimizado"""
        if multi_stage:
            return f"""
# Multi-stage build
FROM {base_image} as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc g++ make libffi-dev libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM {base_image}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Crear usuario no-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiar dependencias
COPY --from=builder /root/.local /home/appuser/.local

WORKDIR /app
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8030
CMD ["python", "-m", "uvicorn", "api.prototype_api:app", "--host", "0.0.0.0", "--port", "8030"]
"""
        else:
            return f"""
FROM {base_image}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8030
CMD ["python", "-m", "uvicorn", "api.prototype_api:app", "--host", "0.0.0.0", "--port", "8030"]
"""
    
    @staticmethod
    def optimize_image_size() -> Dict[str, str]:
        """Recomendaciones para optimizar tamaño de imagen"""
        return {
            "use_multi_stage": "Usar multi-stage builds para reducir tamaño final",
            "minimal_base": "Usar imágenes base minimalistas (alpine, slim)",
            "clean_apt": "Limpiar cache de apt después de instalar",
            "no_cache_pip": "Usar --no-cache-dir con pip",
            "combine_runs": "Combinar múltiples RUN en uno solo",
            "copy_order": "Copiar requirements.txt antes del código",
            "remove_dev": "Remover dependencias de desarrollo en runtime"
        }


class StandaloneBinaryPackager:
    """Empaquetador para binarios standalone"""
    
    @staticmethod
    def generate_pyinstaller_spec() -> str:
        """Genera spec file para PyInstaller"""
        return """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('api', 'api'),
        ('core', 'core'),
        ('models', 'models'),
        ('utils', 'utils'),
        ('config', 'config'),
    ],
    hiddenimports=[
        'uvicorn',
        'fastapi',
        'pydantic',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='3d_prototype_ai',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    
    @staticmethod
    def generate_nuitka_config() -> str:
        """Genera configuración para Nuitka"""
        return """
# Nuitka configuration for standalone binary
python -m nuitka \\
    --standalone \\
    --onefile \\
    --enable-plugin=fastapi \\
    --include-module=uvicorn \\
    --include-module=fastapi \\
    --include-module=pydantic \\
    --output-dir=dist \\
    main.py
"""


# Instancia global
container_optimizer = ContainerOptimizer()
binary_packager = StandaloneBinaryPackager()




