# MCP Server CLI - Guía de Uso

## Resumen

Herramienta CLI completa para administración y diagnóstico del módulo MCP Server.

## Instalación

El CLI está disponible como script Python:

```bash
python -m mcp_server.cli [COMMAND] [OPTIONS]
```

O directamente:

```bash
python mcp_server/cli.py [COMMAND] [OPTIONS]
```

## Comandos Disponibles

### `version`
Muestra la versión del módulo.

```bash
python -m mcp_server.cli version
```

**Salida:**
```
MCP Server v2.2.0
```

### `health`
Verifica la salud del módulo.

```bash
# Formato texto (default)
python -m mcp_server.cli health

# Formato JSON
python -m mcp_server.cli health --format json
```

**Salida:**
```
Status: ✓ HEALTHY

  imports: ok
    Availability: 95.5%
  core_components: ok
    ✓ MCPServer
    ✓ ConnectorRegistry
    ✓ MCPSecurityManager
```

### `diagnostics`
Ejecuta diagnóstico completo del módulo.

```bash
# Mostrar en consola
python -m mcp_server.cli diagnostics

# Guardar en archivo
python -m mcp_server.cli diagnostics --output report.txt

# Formato JSON
python -m mcp_server.cli diagnostics --format json --output report.json
```

### `validate`
Valida la configuración del módulo.

```bash
# Formato texto
python -m mcp_server.cli validate

# Formato JSON
python -m mcp_server.cli validate --format json
```

**Salida:**
```
✓ Module setup is valid
```

O si hay errores:
```
✗ Module setup has errors:
  - Core imports failed: ...
  - ImportManager not available: ...
```

### `imports`
Verifica el estado de los imports.

```bash
# Mostrar todos los imports
python -m mcp_server.cli imports

# Mostrar solo imports faltantes
python -m mcp_server.cli imports --missing

# Formato JSON
python -m mcp_server.cli imports --format json
```

**Salida:**
```
Imports: 95/100 (95.0%)

Missing (5):
  - Component1
  - Component2
  ...
```

### `report`
Genera reporte de diagnóstico completo.

```bash
# Mostrar en consola
python -m mcp_server.cli report

# Guardar en archivo
python -m mcp_server.cli report --output diagnostic_report.txt
```

### `performance`
Muestra métricas de performance.

```bash
# Una vez
python -m mcp_server.cli performance

# Modo watch (actualización continua)
python -m mcp_server.cli performance --watch --interval 5

# Formato JSON
python -m mcp_server.cli performance --format json
```

**Salida:**
```
Performance Metrics:
  Memory RSS: 125.50 MB
  Memory VMS: 256.30 MB
  Memory %: 12.5%
  CPU %: 5.25%
  Threads: 8
  Timestamp: 2024-01-01T12:00:00
```

### `dependencies`
Muestra el árbol de dependencias.

```bash
# Formato texto
python -m mcp_server.cli dependencies

# Formato JSON
python -m mcp_server.cli dependencies --format json
```

**Salida:**
```
Total Groups: 5
Total Modules: 25

core:
  Modules: 2
    - server: MCPServer, MCPRequest, MCPResponse
    - exceptions: MCPError, MCPAuthenticationError ...
...
```

### `info`
Muestra información del módulo.

```bash
# Formato texto
python -m mcp_server.cli info

# Formato JSON
python -m mcp_server.cli info --format json
```

**Salida:**
```
Version: 2.2.0
Author: Blatam Academy
License: Proprietary

Statistics:
  Total Components: 100
  Available: 95
  Missing: 5
```

### `config`
Gestiona configuración del módulo.

#### `config show`
Muestra configuración completa.

```bash
# Mostrar configuración
python -m mcp_server.cli config show --path config.json

# Formato JSON
python -m mcp_server.cli config show --path config.json --format json

# Formato YAML
python -m mcp_server.cli config show --path config.yaml --format yaml
```

#### `config validate`
Valida configuración.

```bash
python -m mcp_server.cli config validate --path config.json
```

**Salida:**
```
✓ Configuration is valid
```

O si hay errores:
```
Configuration validation failed:
  - server.port must be <= 65535
  - security.secret_key is required
```

#### `config template`
Genera plantilla de configuración.

```bash
# Generar plantilla JSON
python -m mcp_server.cli config template --output config.json --format json

# Generar plantilla YAML
python -m mcp_server.cli config template --output config.yaml --format yaml
```

#### `config get`
Obtiene valor de configuración.

```bash
python -m mcp_server.cli config get --path config.json "server.host"
```

**Salida:**
```
"0.0.0.0"
```

#### `config set`
Establece valor de configuración.

```bash
python -m mcp_server.cli config set --path config.json "server.port" "8080"
```

**Salida:**
```
Set server.port = 8080
```

## Ejemplos de Uso

### Verificación Rápida

```bash
# Verificar que todo está bien
python -m mcp_server.cli health && \
python -m mcp_server.cli validate
```

### Generar Reporte Completo

```bash
# Generar reporte y guardarlo
python -m mcp_server.cli report --output mcp_diagnostic_$(date +%Y%m%d).txt
```

### Monitoreo Continuo

```bash
# Monitorear performance cada 10 segundos
python -m mcp_server.cli performance --watch --interval 10
```

### Integración con CI/CD

```bash
#!/bin/bash
# Script de validación para CI/CD

python -m mcp_server.cli validate
if [ $? -ne 0 ]; then
    echo "Module validation failed"
    exit 1
fi

python -m mcp_server.cli health --format json > health_check.json
```

### Debugging

```bash
# Obtener información completa para debugging
python -m mcp_server.cli diagnostics --format json > debug_info.json
python -m mcp_server.cli imports --missing > missing_imports.txt
```

## Códigos de Salida

- `0`: Éxito
- `1`: Error general
- `130`: Interrumpido por usuario (Ctrl+C)

## Integración

### Con Scripts de Inicialización

```python
#!/usr/bin/env python3
import subprocess
import sys

# Validar antes de iniciar
result = subprocess.run(
    ['python', '-m', 'mcp_server.cli', 'validate'],
    capture_output=True
)

if result.returncode != 0:
    print("Validation failed:", result.stdout.decode())
    sys.exit(1)

# Continuar con inicialización...
```

### Con Health Checks

```python
from fastapi import APIRouter
import subprocess
import json

router = APIRouter()

@router.get("/health/cli")
async def health_cli():
    result = subprocess.run(
        ['python', '-m', 'mcp_server.cli', 'health', '--format', 'json'],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)
```

## Próximos Pasos

1. Agregar más comandos específicos
2. Integrar con sistemas de monitoreo
3. Agregar comandos de administración
4. Crear comandos de configuración
5. Agregar comandos de testing

