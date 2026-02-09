# 🛠️ Scripts de Utilidad - Blatam Academy Features

## 📋 Scripts Disponibles

### setup_complete.sh
Script completo de setup del sistema desde cero.

**Uso:**
```bash
chmod +x scripts/setup_complete.sh
./scripts/setup_complete.sh
```

**Funciones:**
- Verifica prerrequisitos
- Crea archivo .env
- Construye imágenes Docker
- Inicia servicios
- Verifica estado

### health_check.sh
Health check completo de todos los servicios.

**Uso:**
```bash
chmod +x scripts/health_check.sh
./scripts/health_check.sh
```

**Verifica:**
- Docker containers
- HTTP endpoints
- Recursos del sistema

### benchmark.sh
Script de benchmarking del sistema.

**Uso:**
```bash
chmod +x scripts/benchmark.sh
./scripts/benchmark.sh
```

**Tests:**
- Latencia
- Throughput
- KV Cache performance

## 🔧 Scripts Personalizados

Puedes crear tus propios scripts basándote en estos templates.

---

**Más información:**
- [README Principal](../README.md)
- [Guía de Troubleshooting](../TROUBLESHOOTING_GUIDE.md)



