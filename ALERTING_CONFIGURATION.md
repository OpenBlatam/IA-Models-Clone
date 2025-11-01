# 🔔 Configuración de Alertas - Blatam Academy Features

## 📊 Alertas Críticas (Severidad Alta)

### Alerta: Sistema Caído

```yaml
# prometheus/alerts.yml
groups:
  - name: critical_alerts
    rules:
      - alert: SystemDown
        expr: up{job="bul"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "BUL System is down"
          description: "BUL service has been down for more than 1 minute"
```

**Acción**: Reiniciar servicio inmediatamente

---

### Alerta: Latencia Crítica

```yaml
- alert: CriticalLatency
  expr: histogram_quantile(0.95, kv_cache_latency_seconds_bucket) > 2
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Critical latency detected"
    description: "P95 latency is above 2 seconds for 5 minutes"
```

**Acción**: Verificar carga, escalar si necesario

---

### Alerta: Error Rate Alto

```yaml
- alert: HighErrorRate
  expr: rate(kv_cache_errors_total[5m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate"
    description: "Error rate is above 10% for 5 minutes"
```

**Acción**: Investigar logs de errores

---

## ⚠️ Alertas Importantes (Severidad Media)

### Alerta: Cache Hit Rate Bajo

```yaml
- alert: LowCacheHitRate
  expr: rate(kv_cache_requests_total{status="hit"}[5m]) / rate(kv_cache_requests_total[5m]) < 0.5
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Low cache hit rate"
    description: "Cache hit rate below 50% for 15 minutes"
```

**Acción**: Considerar aumentar cache size

---

### Alerta: Memoria Alta

```yaml
- alert: HighMemoryUsage
  expr: kv_cache_memory_usage_bytes / kv_cache_memory_limit_bytes > 0.9
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage"
    description: "Memory usage above 90% for 10 minutes"
```

**Acción**: Considerar compresión o reducir cache

---

### Alerta: Throughput Bajo

```yaml
- alert: LowThroughput
  expr: rate(kv_cache_requests_total[5m]) < 20
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Low throughput"
    description: "Throughput below 20 req/s for 15 minutes"
```

**Acción**: Verificar bottlenecks

---

## 📈 Alertas Informativas (Severidad Baja)

### Alerta: Cache Warming Recomendado

```yaml
- alert: CacheWarmingRecommended
  expr: rate(kv_cache_requests_total{status="miss"}[1h]) > 100
  for: 1h
  labels:
    severity: info
  annotations:
    summary: "Cache warming recommended"
    description: "High cache miss rate, consider cache warming"
```

---

### Alerta: Backup Pendiente

```yaml
- alert: BackupPending
  expr: time() - last_backup_timestamp > 86400  # 24 horas
  labels:
    severity: info
  annotations:
    summary: "Backup pending"
    description: "No backup in last 24 hours"
```

---

## 🔧 Configuración de Alertas en Prometheus

### prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'bul'
    static_configs:
      - targets: ['bul:8002']
```

### Alertas Completas

```yaml
# alerts.yml completo
groups:
  - name: kv_cache_alerts
    interval: 30s
    rules:
      # Críticas
      - alert: SystemDown
        expr: up{job="bul"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "System is down"
      
      - alert: CriticalLatency
        expr: histogram_quantile(0.95, kv_cache_latency_seconds_bucket) > 2
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical latency"
      
      # Advertencias
      - alert: LowCacheHitRate
        expr: rate(kv_cache_requests_total{status="hit"}[5m]) / rate(kv_cache_requests_total[5m]) < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
      
      - alert: HighMemoryUsage
        expr: kv_cache_memory_usage_bytes / kv_cache_memory_limit_bytes > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
```

## 📧 Configuración de Notificaciones

### Slack Integration

```yaml
# alertmanager.yml
route:
  receiver: 'slack-notifications'
  routes:
    - match:
        severity: critical
      receiver: 'slack-critical'

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'
  
  - name: 'slack-critical'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
        title: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'
```

### Email Integration

```yaml
receivers:
  - name: 'email-notifications'
    email_configs:
      - to: 'team@example.com'
        from: 'alerts@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alerts@example.com'
        auth_password: 'password'
        headers:
          Subject: 'Alert: {{ .GroupLabels.alertname }}'
```

### PagerDuty Integration

```yaml
receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .CommonAnnotations.description }}'
```

## 📊 Dashboard de Alertas en Grafana

### Panel de Alertas

```json
{
  "panels": [
    {
      "title": "Active Alerts",
      "targets": [{
        "expr": "ALERTS{alertstate='firing'}"
      }],
      "type": "table"
    },
    {
      "title": "Alert History",
      "targets": [{
        "expr": "changes(ALERTS[1h])"
      }],
      "type": "graph"
    }
  ]
}
```

## ✅ Checklist de Configuración de Alertas

### Pre-Producción
- [ ] Alertas críticas configuradas
- [ ] Alertas importantes configuradas
- [ ] Canales de notificación configurados
- [ ] Alertas testeadas
- [ ] Runbook para cada alerta

### Post-Producción
- [ ] Alertas monitoreadas
- [ ] False positives identificados y ajustados
- [ ] Thresholds ajustados según experiencia
- [ ] Alertas documentadas

---

**Más información:**
- [Production Ready](bulk/PRODUCTION_READY.md)
- [Monitoring Guide](../monitoring/)
- [Operational Runbook](OPERATIONAL_RUNBOOK.md)

