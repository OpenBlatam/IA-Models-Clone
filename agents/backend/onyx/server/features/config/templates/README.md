# 📋 Plantillas de Configuración

## 📁 Archivos Disponibles

### production.env.template
Template completo de variables de entorno para producción.

**Uso:**
```bash
cp config/templates/production.env.template .env
# Editar .env con valores reales
```

**Incluye:**
- Configuración de servicios
- API keys
- Database y Redis
- KV Cache settings
- Monitoring
- Security

### kv_cache_production.yaml
Configuración optimizada del KV Cache para producción.

**Uso:**
```python
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigManager

config_manager = ConfigManager(
    engine,
    config_file='config/templates/kv_cache_production.yaml'
)
await config_manager.reload_from_file()
```

**Presets disponibles:**
- `development`
- `production`
- `high_performance`
- `memory_efficient`
- `bulk_processing`

## ⚙️ Personalización

Edita los templates según tus necesidades específicas.

---

**Más información:**
- [Guía de Configuración](../README.md#-configuración)
- [Mejores Prácticas](../BEST_PRACTICES.md)



