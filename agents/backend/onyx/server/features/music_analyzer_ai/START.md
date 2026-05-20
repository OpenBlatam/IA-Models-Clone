# 🚀 Inicio Rápido - Un Solo Comando

## Iniciar Todo

### Windows

```cmd
deployment\start.bat
```

### Linux/Mac

```bash
./deployment/start.sh
```

### Python (Multiplataforma)

```bash
python deployment/start.py
```

### Make (Linux/Mac)

```bash
make start
```

## Detener Todo

### Windows

```cmd
deployment\stop.bat
```

### Linux/Mac

```bash
./deployment/stop.sh
```

### Make

```bash
make stop
```

## ¿Qué se Inicia?

Con un solo comando se inician:

- ✅ **Music Analyzer AI** - API principal
- ✅ **Redis** - Caché
- ✅ **PostgreSQL** - Base de datos (opcional)
- ✅ **Nginx** - Reverse proxy
- ✅ **Prometheus** - Métricas
- ✅ **Grafana** - Dashboards

## URLs Disponibles

Una vez iniciado:

- 🌐 **API**: http://localhost:8010
- ❤️ **Health**: http://localhost:8010/health
- 📖 **Docs**: http://localhost:8010/docs
- 📈 **Grafana**: http://localhost:3000
- 📊 **Prometheus**: http://localhost:9090

## Configuración

El script crea automáticamente un archivo `.env` si no existe. Para producción, edita `.env` con tus credenciales reales.

## Más Información

Ver `deployment/README_START.md` para documentación completa.




