@echo off
REM Build script para extensiones nativas (Windows)
REM ===============================================

echo ==========================================
echo Building Native Extensions for Robot Movement AI
echo ==========================================

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Por favor instala Python 3.8+
    exit /b 1
)

echo [INFO] Python encontrado
python --version

REM Verificar pybind11
python -c "import pybind11" >nul 2>&1
if errorlevel 1 (
    echo [WARN] pybind11 no encontrado. Instalando...
    pip install pybind11
)

REM Compilar extensiones C++
echo [INFO] Compilando extensiones C++...
if exist "cpp_extensions.cpp" (
    python setup.py build_ext --inplace
    if errorlevel 1 (
        echo [WARN] Error compilando extensiones C++. Continuando sin ellas...
    ) else (
        echo [INFO] Extensiones C++ compiladas exitosamente
    )
) else (
    echo [WARN] cpp_extensions.cpp no encontrado. Saltando compilación C++...
)

REM Compilar extensiones Rust
where cargo >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Compilando extensiones Rust...
    cd rust_extensions
    
    if exist "Cargo.toml" (
        where maturin >nul 2>&1
        if errorlevel 1 (
            echo [WARN] maturin no encontrado. Instalando...
            pip install maturin
        )
        
        maturin develop --release
        if errorlevel 1 (
            echo [WARN] Error compilando extensiones Rust. Continuando sin ellas...
        ) else (
            echo [INFO] Extensiones Rust compiladas exitosamente
        )
    ) else (
        echo [WARN] Cargo.toml no encontrado. Saltando compilación Rust...
    )
    cd ..
) else (
    echo [WARN] Rust no encontrado. Saltando compilación Rust...
    echo [WARN] Para instalar Rust: https://rustup.rs/
)

echo.
echo ==========================================
echo [INFO] Build completado!
echo ==========================================
echo.
echo Para verificar la instalación, ejecuta:
echo   python -c "from robot_movement_ai.native import CPP_AVAILABLE, RUST_AVAILABLE; print(f'C++: {CPP_AVAILABLE}, Rust: {RUST_AVAILABLE}')"

