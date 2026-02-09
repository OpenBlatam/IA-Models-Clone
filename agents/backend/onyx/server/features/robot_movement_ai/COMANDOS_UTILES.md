# Comandos Útiles - Robot Movement AI

## ⚠️ IMPORTANTE: Usar `python -m pip` en lugar de `pip`

En Windows, cuando Python está instalado desde Microsoft Store, usa:
```powershell
python -m pip install <paquete>
```

En lugar de:
```powershell
pip install <paquete>  # ❌ Puede no funcionar
```

## 📦 Instalar Dependencias

### Dependencias Esenciales (mínimas para que funcione la API):
```powershell
python -m pip install fastapi uvicorn python-dotenv pydantic httpx websockets aiofiles
```

### Todas las Dependencias (puede fallar con algunas opcionales):
```powershell
python -m pip install -r requirements.txt
```

## 🚀 Ejecutar la API

### Opción 1: Usar el script batch
```powershell
.\START_API.bat
```

### Opción 2: Comando directo
```powershell
cd C:\blatam-academy\agents\backend\onyx\server\features
python -m robot_movement_ai.main --host 127.0.0.1 --port 8010
```

### Opción 3: Con opciones personalizadas
```powershell
python -m robot_movement_ai.main --host 0.0.0.0 --port 8010 --debug
```

## 🔍 Verificar que la API funciona

Abre tu navegador y visita:
- **Health Check**: http://127.0.0.1:8010/health
- **Documentación**: http://127.0.0.1:8010/docs
- **Status**: http://127.0.0.1:8010/api/v1/status

O desde PowerShell:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/health"
```

## 🛠️ Comandos Python Útiles

```powershell
# Verificar versión de Python
python --version

# Verificar pip
python -m pip --version

# Actualizar pip
python -m pip install --upgrade pip

# Listar paquetes instalados
python -m pip list

# Verificar si un paquete está instalado
python -m pip show fastapi
```

## 📝 Notas

- Si `pip` no funciona, siempre usa `python -m pip`
- El frontend está configurado para conectarse a `http://localhost:8010`
- Si el puerto 8010 está ocupado, cambia el puerto con `--port 8011`



