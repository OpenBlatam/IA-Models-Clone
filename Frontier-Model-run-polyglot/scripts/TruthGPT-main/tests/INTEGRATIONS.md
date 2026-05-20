# 🔗 Integraciones y Notificaciones - TruthGPT Test Suite

## Sistema de Notificaciones

### Configuración

1. Copiar archivo de ejemplo:
```bash
cp config/notifications.json.example config/notifications.json
```

2. Editar `config/notifications.json` con tus credenciales

### Email

```bash
# Enviar notificación por email
python notify_results.py --input results.json --email

# Configurar en notifications.json:
{
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "your-email@gmail.com",
    "password": "your-app-password",
    "from": "tests@yourcompany.com",
    "to": ["team@yourcompany.com"]
  }
}
```

### Slack

```bash
# Enviar notificación a Slack
python notify_results.py --input results.json --slack

# Obtener webhook URL:
# 1. Ir a https://api.slack.com/apps
# 2. Crear nueva app
# 3. Habilitar Incoming Webhooks
# 4. Copiar webhook URL a notifications.json
```

### Microsoft Teams

```bash
# Enviar notificación a Teams
python notify_results.py --input results.json --teams

# Obtener webhook URL:
# 1. En Teams, ir a canal deseado
# 2. Configuración > Conectores
# 3. Agregar "Incoming Webhook"
# 4. Copiar URL a notifications.json
```

## Dashboard en Tiempo Real

### Iniciar Servidor

```bash
# Iniciar dashboard
python dashboard_server.py --port 8080 --data monitor_stats.json
make dashboard

# Abrir en navegador: http://localhost:8080
```

### Características

- 📊 Visualización en tiempo real
- 🔄 Auto-refresh cada 5 segundos
- 📈 Gráficos interactivos
- 📱 Responsive design

### Integración con Monitor

```bash
# Terminal 1: Monitorear tests
python monitor_tests.py --output monitor_stats.json

# Terminal 2: Dashboard
python dashboard_server.py --data monitor_stats.json
```

## Métricas Avanzadas

### Colector de Métricas

```bash
# Recopilar métricas
python metrics_collector.py
make metrics

# Genera:
# - Score de salud de tests
# - Tendencias de rendimiento
# - Detección de anomalías
# - Recomendaciones
```

### Score de Salud

El score de salud considera:
- **Tasa de éxito** (40%): Porcentaje de tests que pasan
- **Velocidad de ejecución** (20%): Tiempo promedio
- **Cobertura** (20%): Cobertura de código
- **Flakiness** (20%): Consistencia de tests

### Estados

- **Excellent** (90-100): Todo funcionando perfectamente
- **Good** (75-89): Buen estado, mejoras menores posibles
- **Fair** (60-74): Requiere atención
- **Poor** (<60): Acción urgente requerida

## Workflows Integrados

### Workflow Completo con Notificaciones

```bash
# 1. Ejecutar tests
make test

# 2. Monitorear
python monitor_tests.py --duration 3600 --output monitor_stats.json

# 3. Generar reportes
make report

# 4. Enviar notificaciones
python notify_results.py --input monitor_stats.json

# 5. Visualizar en dashboard
make dashboard
```

### CI/CD con Notificaciones

```yaml
# En .github/workflows/tests.yml, agregar:
- name: Send notifications
  if: always()
  run: |
    python notify_results.py --input test-results/results.json
  env:
    SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
```

## Ejemplos de Uso

### Ejemplo 1: Notificación Diaria

```bash
# Script diario (cron job)
#!/bin/bash
cd /path/to/tests
python run_tests.py all
python generate_report.py --output daily_report.json
python notify_results.py --input daily_report.json
```

### Ejemplo 2: Dashboard Continuo

```bash
# Iniciar dashboard como servicio
nohup python dashboard_server.py --port 8080 --data monitor_stats.json > dashboard.log 2>&1 &

# Monitorear en background
nohup python monitor_tests.py --output monitor_stats.json > monitor.log 2>&1 &
```

### Ejemplo 3: Alertas en Tiempo Real

```bash
# Monitorear y notificar solo en fallos
python monitor_tests.py --output monitor_stats.json &
while true; do
  if [ $(jq '.stats.failed' monitor_stats.json) -gt 0 ]; then
    python notify_results.py --input monitor_stats.json --slack
  fi
  sleep 60
done
```

## Seguridad

### Credenciales

- **Nunca** commitees `notifications.json` con credenciales reales
- Usa variables de entorno en CI/CD
- Usa app passwords para email (no contraseñas principales)
- Rota webhooks periódicamente

### Variables de Entorno

```bash
# En lugar de archivo de configuración
export SLACK_WEBHOOK="https://hooks.slack.com/..."
export EMAIL_PASSWORD="your-app-password"

# El script puede leer de variables de entorno
```

## Troubleshooting

### Email no se envía
- Verificar credenciales SMTP
- Usar app password (no contraseña principal)
- Verificar firewall/puerto 587

### Slack no recibe mensajes
- Verificar webhook URL
- Verificar formato del payload
- Revisar logs de Slack API

### Dashboard no actualiza
- Verificar que el archivo de datos existe
- Verificar permisos de lectura
- Revisar consola del navegador para errores

## Recursos

- **README.md**: Documentación completa
- **ADVANCED_FEATURES.md**: Características avanzadas
- **TOOLS.md**: Guía de herramientas
- **config/notifications.json.example**: Template de configuración

