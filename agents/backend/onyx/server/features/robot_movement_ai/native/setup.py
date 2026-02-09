"""
Setup script for native C++ extensions
=======================================

Compila las extensiones C++ usando pybind11 y setuptools.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11
import os
import sys

# Obtener ruta de Eigen (si está instalado)
def find_eigen():
    """Buscar instalación de Eigen."""
    possible_paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",
        os.path.expanduser("~/eigen3"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Intentar con pkg-config
    try:
        import subprocess
        result = subprocess.run(
            ["pkg-config", "--cflags", "eigen3"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            flags = result.stdout.strip()
            for flag in flags.split():
                if flag.startswith("-I"):
                    return flag[2:]
    except:
        pass
    
    return None

# Configuración de la extensión
ext_modules = [
    Pybind11Extension(
        "robot_movement_ai.native.cpp_extensions",
        [
            "native/cpp_extensions.cpp",
        ],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../../../include",
            # Agregar Eigen si está disponible
        ] + ([find_eigen()] if find_eigen() else []),
        language='c++',
        cxx_std=17,
        extra_compile_args=[
            '-O3',  # Optimización máxima
            '-march=native',  # Optimización para CPU específica
            '-ffast-math',  # Matemáticas rápidas (puede afectar precisión)
            '-fopenmp',  # Paralelización OpenMP
            '-Wall',  # Warnings
            '-Wextra',  # Más warnings
            '-Wno-unused-parameter',  # Ignorar parámetros no usados
        ] + (['-std=c++17'] if sys.platform != 'win32' else ['/std:c++17']),
        extra_link_args=['-fopenmp'] if sys.platform != 'win32' else [],
    ),
]

# Si Eigen no está disponible, usar implementación sin Eigen
if not find_eigen():
    print("Warning: Eigen3 not found. Using simplified implementation.")
    # Se puede crear una versión simplificada sin Eigen

setup(
    name="robot_movement_ai_native",
    version="1.0.0",
    author="Blatam Academy",
    description="Native C++ extensions for Robot Movement AI",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)

