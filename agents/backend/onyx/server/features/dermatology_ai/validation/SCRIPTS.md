# 🛠️ Scripts de Automatización - Validación

Scripts útiles para automatizar tareas comunes en el proceso de validación.

## 📊 Script: Analizar Métricas de Validación

### `analyze_metrics.py`

```python
#!/usr/bin/env python3
"""
Script para analizar métricas de validación.
Recopila datos de Google Analytics, formularios, etc.
"""

import json
from datetime import datetime
from pathlib import Path

def load_metrics():
    """Carga métricas desde archivo JSON."""
    metrics_file = Path("results/metrics.json")
    if not metrics_file.exists():
        return {}
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def calculate_conversion_rate(visits, uploads):
    """Calcula tasa de conversión."""
    if visits == 0:
        return 0
    return (uploads / visits) * 100

def analyze_metrics():
    """Analiza métricas y genera reporte."""
    metrics = load_metrics()
    
    if not metrics:
        print("❌ No hay métricas para analizar.")
        print("💡 Ejecuta 'collect_metrics.py' primero.")
        return
    
    print("\n📊 REPORTE DE MÉTRICAS DE VALIDACIÓN\n")
    print("=" * 50)
    
    # Métricas de tráfico
    visits = metrics.get('visits', 0)
    unique_users = metrics.get('unique_users', 0)
    bounce_rate = metrics.get('bounce_rate', 0)
    
    print(f"\n🌐 TRÁFICO:")
    print(f"  Visitas: {visits}")
    print(f"  Usuarios únicos: {unique_users}")
    print(f"  Tasa de rebote: {bounce_rate:.1f}%")
    
    # Métricas de conversión
    uploads = metrics.get('uploads', 0)
    completions = metrics.get('completions', 0)
    conversion_rate = calculate_conversion_rate(visits, uploads)
    completion_rate = calculate_conversion_rate(uploads, completions)
    
    print(f"\n📈 CONVERSIÓN:")
    print(f"  Subidas de foto: {uploads}")
    print(f"  Análisis completados: {completions}")
    print(f"  Tasa de conversión (visitas → uploads): {conversion_rate:.1f}%")
    print(f"  Tasa de completación (uploads → completions): {completion_rate:.1f}%")
    
    # Métricas de satisfacción
    avg_feedback = metrics.get('avg_feedback', 0)
    nps = metrics.get('nps', 0)
    payment_intent = metrics.get('payment_intent', 0)
    
    print(f"\n😊 SATISFACCIÓN:")
    print(f"  Feedback promedio: {avg_feedback:.1f}/5")
    print(f"  NPS: {nps}")
    print(f"  % que pagarían: {payment_intent:.1f}%")
    
    # Métricas técnicas
    avg_analysis_time = metrics.get('avg_analysis_time', 0)
    error_rate = metrics.get('error_rate', 0)
    
    print(f"\n⚙️ TÉCNICAS:")
    print(f"  Tiempo promedio de análisis: {avg_analysis_time:.1f}s")
    print(f"  Tasa de error: {error_rate:.1f}%")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    
    if conversion_rate < 20:
        print("  ⚠️ Tasa de conversión baja. Revisa UX y propuesta de valor.")
    
    if completion_rate < 50:
        print("  ⚠️ Muchos usuarios no completan. Revisa tiempo de análisis.")
    
    if avg_feedback < 3.5:
        print("  ⚠️ Feedback bajo. Revisa calidad de resultados.")
    
    if payment_intent < 30:
        print("  ⚠️ Baja intención de pago. Revisa propuesta de valor.")
    
    if avg_analysis_time > 15:
        print("  ⚠️ Análisis muy lento. Optimiza backend.")
    
    if error_rate > 5:
        print("  ⚠️ Alta tasa de errores. Revisa estabilidad.")
    
    # Decisión
    print(f"\n🎯 DECISIÓN:")
    
    if (conversion_rate >= 20 and 
        completion_rate >= 50 and 
        avg_feedback >= 3.5 and 
        payment_intent >= 30):
        print("  ✅ CONTINUAR - Métricas positivas")
    elif (conversion_rate >= 10 and 
          avg_feedback >= 3.0):
        print("  ⚠️ ITERAR - Hay potencial pero necesita mejoras")
    else:
        print("  ❌ RECONSIDERAR - Métricas bajas, necesita pivotar")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    analyze_metrics()
```

---

## 📝 Script: Recopilar Métricas Manualmente

### `collect_metrics.py`

```python
#!/usr/bin/env python3
"""
Script interactivo para recopilar métricas manualmente.
Útil cuando no tienes analytics configurado.
"""

import json
from pathlib import Path
from datetime import datetime

def collect_metrics():
    """Recopila métricas de forma interactiva."""
    print("\n📊 RECOPILACIÓN DE MÉTRICAS\n")
    print("=" * 50)
    
    metrics = {}
    
    # Métricas de tráfico
    print("\n🌐 TRÁFICO:")
    metrics['visits'] = int(input("  Visitas totales: ") or "0")
    metrics['unique_users'] = int(input("  Usuarios únicos: ") or "0")
    metrics['bounce_rate'] = float(input("  Tasa de rebote (%): ") or "0")
    
    # Métricas de conversión
    print("\n📈 CONVERSIÓN:")
    metrics['uploads'] = int(input("  Subidas de foto: ") or "0")
    metrics['completions'] = int(input("  Análisis completados: ") or "0")
    
    # Métricas de satisfacción
    print("\n😊 SATISFACCIÓN:")
    metrics['avg_feedback'] = float(input("  Feedback promedio (1-5): ") or "0")
    metrics['nps'] = int(input("  NPS (-100 a 100): ") or "0")
    metrics['payment_intent'] = float(input("  % que pagarían: ") or "0")
    
    # Métricas técnicas
    print("\n⚙️ TÉCNICAS:")
    metrics['avg_analysis_time'] = float(input("  Tiempo promedio de análisis (segundos): ") or "0")
    metrics['error_rate'] = float(input("  Tasa de error (%): ") or "0")
    
    # Guardar
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    metrics_file = results_dir / "metrics.json"
    metrics['last_updated'] = datetime.now().isoformat()
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Métricas guardadas en: {metrics_file}")
    print("💡 Ejecuta 'analyze_metrics.py' para ver el análisis.")

if __name__ == "__main__":
    collect_metrics()
```

---

## 📧 Script: Generar Reporte de Validación

### `generate_report.py`

```python
#!/usr/bin/env python3
"""
Genera un reporte completo de validación en Markdown.
"""

import json
from pathlib import Path
from datetime import datetime

def load_metrics():
    """Carga métricas."""
    metrics_file = Path("results/metrics.json")
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def generate_report():
    """Genera reporte de validación."""
    metrics = load_metrics()
    
    if not metrics:
        print("❌ No hay métricas. Ejecuta 'collect_metrics.py' primero.")
        return
    
    report = f"""# 📊 Reporte de Validación - Dermatology AI

**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 Resumen Ejecutivo

Este reporte resume los resultados de la validación de la idea Dermatology AI.

## 🌐 Métricas de Tráfico

- **Visitas**: {metrics.get('visits', 0)}
- **Usuarios únicos**: {metrics.get('unique_users', 0)}
- **Tasa de rebote**: {metrics.get('bounce_rate', 0):.1f}%

## 📊 Métricas de Conversión

- **Subidas de foto**: {metrics.get('uploads', 0)}
- **Análisis completados**: {metrics.get('completions', 0)}
- **Tasa de conversión**: {(metrics.get('uploads', 0) / max(metrics.get('visits', 1), 1) * 100):.1f}%

## 😊 Métricas de Satisfacción

- **Feedback promedio**: {metrics.get('avg_feedback', 0):.1f}/5
- **NPS**: {metrics.get('nps', 0)}
- **% que pagarían**: {metrics.get('payment_intent', 0):.1f}%

## ⚙️ Métricas Técnicas

- **Tiempo promedio de análisis**: {metrics.get('avg_analysis_time', 0):.1f}s
- **Tasa de error**: {metrics.get('error_rate', 0):.1f}%

## 🎯 Conclusión

[Escribe tu conclusión aquí basada en las métricas]

## 📝 Próximos Pasos

1. [ ] [Próximo paso 1]
2. [ ] [Próximo paso 2]
3. [ ] [Próximo paso 3]

## 💡 Aprendizajes Clave

- [Aprendizaje 1]
- [Aprendizaje 2]
- [Aprendizaje 3]

---
*Generado automáticamente el {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    report_file = results_dir / f"report_{datetime.now().strftime('%Y%m%d')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Reporte generado: {report_file}")
    print("💡 Edita el reporte para agregar conclusiones y próximos pasos.")

if __name__ == "__main__":
    generate_report()
```

---

## 🔍 Script: Verificar Backend

### `check_backend.sh` (Linux/Mac) o `check_backend.bat` (Windows)

**check_backend.sh:**
```bash
#!/bin/bash
# Verifica que el backend esté funcionando

BACKEND_URL="http://localhost:8006"

echo "🔍 Verificando backend en $BACKEND_URL..."

# Verificar health endpoint
response=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/health")

if [ "$response" -eq 200 ]; then
    echo "✅ Backend está funcionando correctamente"
    exit 0
else
    echo "❌ Backend no responde (código: $response)"
    echo "💡 Asegúrate de que el backend esté corriendo:"
    echo "   cd agents/backend/onyx/server/features/dermatology_ai"
    echo "   python main.py"
    exit 1
fi
```

**check_backend.bat:**
```batch
@echo off
REM Verifica que el backend esté funcionando

set BACKEND_URL=http://localhost:8006

echo 🔍 Verificando backend en %BACKEND_URL%...

curl -s -o nul -w "%%{http_code}" %BACKEND_URL%/health > temp_response.txt
set /p response=<temp_response.txt
del temp_response.txt

if "%response%"=="200" (
    echo ✅ Backend está funcionando correctamente
    exit /b 0
) else (
    echo ❌ Backend no responde (código: %response%)
    echo 💡 Asegúrate de que el backend esté corriendo:
    echo    cd agents\backend\onyx\server\features\dermatology_ai
    echo    python main.py
    exit /b 1
)
```

---

## 📋 Script: Setup Inicial

### `setup_validation.sh` (Linux/Mac) o `setup_validation.bat` (Windows)

**setup_validation.sh:**
```bash
#!/bin/bash
# Configura el entorno de validación

echo "🚀 Configurando entorno de validación..."

# Crear directorio de resultados
mkdir -p results
echo "✅ Directorio 'results' creado"

# Crear archivo de métricas vacío
cat > results/metrics.json << EOF
{
  "visits": 0,
  "unique_users": 0,
  "bounce_rate": 0,
  "uploads": 0,
  "completions": 0,
  "avg_feedback": 0,
  "nps": 0,
  "payment_intent": 0,
  "avg_analysis_time": 0,
  "error_rate": 0
}
EOF
echo "✅ Archivo de métricas creado"

# Hacer scripts ejecutables
chmod +x check_backend.sh
chmod +x collect_metrics.py
chmod +x analyze_metrics.py
chmod +x generate_report.py

echo "✅ Scripts configurados"
echo ""
echo "🎯 Próximos pasos:"
echo "1. Ejecuta 'check_backend.sh' para verificar el backend"
echo "2. Abre 'frontend/index.html' en el navegador"
echo "3. Lee 'QUICK_GUIDE.md' para empezar"
```

**setup_validation.bat:**
```batch
@echo off
REM Configura el entorno de validación

echo 🚀 Configurando entorno de validación...

REM Crear directorio de resultados
if not exist results mkdir results
echo ✅ Directorio 'results' creado

REM Crear archivo de métricas vacío
(
echo {
echo   "visits": 0,
echo   "unique_users": 0,
echo   "bounce_rate": 0,
echo   "uploads": 0,
echo   "completions": 0,
echo   "avg_feedback": 0,
echo   "nps": 0,
echo   "payment_intent": 0,
echo   "avg_analysis_time": 0,
echo   "error_rate": 0
echo }
) > results\metrics.json
echo ✅ Archivo de métricas creado

echo ✅ Scripts configurados
echo.
echo 🎯 Próximos pasos:
echo 1. Ejecuta 'check_backend.bat' para verificar el backend
echo 2. Abre 'frontend\index.html' en el navegador
echo 3. Lee 'QUICK_GUIDE.md' para empezar
```

---

## 📖 Uso de los Scripts

### 1. Setup Inicial
```bash
# Linux/Mac
chmod +x setup_validation.sh
./setup_validation.sh

# Windows
setup_validation.bat
```

### 2. Verificar Backend
```bash
# Linux/Mac
./check_backend.sh

# Windows
check_backend.bat
```

### 3. Recopilar Métricas
```bash
python collect_metrics.py
```

### 4. Analizar Métricas
```bash
python analyze_metrics.py
```

### 5. Generar Reporte
```bash
python generate_report.py
```

---

**💡 Tip**: Personaliza estos scripts según tus necesidades específicas.






