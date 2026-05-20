# Deployment Guide - Community Manager AI

## 🚀 Guía de Despliegue

### Requisitos Previos

- Python 3.9+
- PostgreSQL o SQLite
- Redis (opcional, para cache)
- Variables de entorno configuradas

### 1. Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd community_manager_ai

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar variables de entorno
nano .env
```

Variables importantes:
```env
DATABASE_URL=postgresql://user:password@localhost/dbname
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key
OPENAI_API_KEY=your-openai-key
DEBUG=false
```

### 3. Inicializar Base de Datos

```bash
# Inicializar tablas
python scripts/init_database.py

# Agregar índices (opcional)
python -c "from database.migrations import add_indexes; add_indexes()"
```

### 4. Ejecutar Servidor

#### Desarrollo
```bash
python main.py
```

#### Producción con Gunicorn
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.app:create_app --factory
```

#### Docker
```bash
docker build -t community-manager-ai .
docker run -p 8000:8000 --env-file .env community-manager-ai
```

### 5. Verificar Instalación

```bash
# Health check
curl http://localhost:8000/health

# Versión de API
curl http://localhost:8000/version
```

## 📊 Monitoreo

### Health Checks
- `/health` - Health check básico
- `/monitoring/health` - Health check detallado

### Métricas
- `/monitoring/metrics` - Métricas del sistema
- `/monitoring/counters` - Contadores
- `/monitoring/timings/{operation}` - Estadísticas de timing

## 🔒 Seguridad

### Recomendaciones
1. Cambiar `JWT_SECRET_KEY` en producción
2. Cambiar `ENCRYPTION_KEY` en producción
3. Usar HTTPS
4. Configurar CORS apropiadamente
5. Rate limiting activado
6. Validación de inputs activada

## 📈 Escalabilidad

### Opciones
1. **Horizontal**: Múltiples instancias con load balancer
2. **Vertical**: Más recursos en servidor
3. **Cache**: Redis para cache distribuido
4. **Database**: Connection pooling, read replicas

## 🔄 Backup

### Backup Automático
```bash
# Crear backup
curl -X POST http://localhost:8000/backup/create

# Listar backups
curl http://localhost:8000/backup/list
```

### Backup Manual
```bash
# Backup de base de datos
pg_dump -U user dbname > backup.sql

# Backup de archivos
tar -czf backup.tar.gz data/
```

## 🐛 Troubleshooting

### Problemas Comunes

1. **Error de conexión a BD**
   - Verificar `DATABASE_URL`
   - Verificar que la BD esté corriendo

2. **Error de Redis**
   - Verificar que Redis esté corriendo
   - O usar cache en memoria

3. **Error de autenticación**
   - Verificar `JWT_SECRET_KEY`
   - Verificar tokens

## 📝 Logs

Los logs se guardan en:
- Console (desarrollo)
- Archivo (producción, configurar)

Niveles:
- INFO: Operaciones normales
- WARNING: Advertencias
- ERROR: Errores
- DEBUG: Debug (solo desarrollo)




