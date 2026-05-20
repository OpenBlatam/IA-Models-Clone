# 🚀 Solución Rápida - Error del Modelo Flux2

## ❌ Error que estás viendo:
```
Failed to build Flux2 model: Failed to load pipeline: Cannot load model 
black-forest-labs/flux2-dev: model is not cached locally and an error 
occurred while trying to fetch metadata from the Hub.
```

## ✅ Solución en 3 Pasos (5 minutos)

### Paso 1: Obtener Token de HuggingFace (2 min)

1. **Ve a**: https://huggingface.co/settings/tokens
2. **Haz clic en**: "New token"
3. **Nombre**: `clothing-changer-ai` (o cualquier nombre)
4. **Permisos**: Selecciona "Read"
5. **Haz clic en**: "Generate token"
6. **Copia el token** (empieza con `hf_...`)

### Paso 2: Aceptar Términos del Modelo (1 min)

1. **Ve a**: https://huggingface.co/black-forest-labs/flux2-dev
2. **Haz clic en**: "Agree and access repository"
3. **Acepta los términos** (si es necesario)

### Paso 3: Configurar el Token (2 min)

#### Opción A: Script Automático (Recomendado)

**Windows:**
```bash
SETUP_TOKEN.bat
```

**Linux/Mac:**
```bash
chmod +x SETUP_TOKEN.sh
./SETUP_TOKEN.sh
```

#### Opción B: Manual

**Windows (PowerShell):**
```powershell
$env:HUGGINGFACE_TOKEN="tu_token_aqui"
python run_server.py
```

**Windows (CMD):**
```cmd
set HUGGINGFACE_TOKEN=tu_token_aqui
python run_server.py
```

**Linux/Mac:**
```bash
export HUGGINGFACE_TOKEN=tu_token_aqui
python run_server.py
```

#### Opción C: Permanente (Para no tener que configurarlo cada vez)

**Windows:**
```cmd
setx HUGGINGFACE_TOKEN "tu_token_aqui"
```
Luego reinicia la terminal.

**Linux/Mac:**
Agrega a `~/.bashrc` o `~/.zshrc`:
```bash
export HUGGINGFACE_TOKEN=tu_token_aqui
```
Luego ejecuta:
```bash
source ~/.bashrc  # o source ~/.zshrc
```

## 🎯 Verificar que Funciona

1. **Inicia el servidor:**
   ```bash
   python run_server.py
   ```

2. **Verifica el token:**
   ```bash
   # Windows
   echo %HUGGINGFACE_TOKEN%
   
   # Linux/Mac
   echo $HUGGINGFACE_TOKEN
   ```

3. **Abre el navegador:**
   - Ve a: http://localhost:8002
   - Deberías ver la interfaz web

4. **Prueba el health check:**
   - Ve a: http://localhost:8002/api/v1/health
   - Debería mostrar: `{"status":"healthy","model_initialized":true}`

## 🔍 Si Aún No Funciona

### Verificar Token
```python
import os
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    print(f"✅ Token encontrado: {token[:10]}...")
else:
    print("❌ Token no configurado")
```

### Verificar Conexión
- Asegúrate de tener internet
- Verifica que no haya firewall bloqueando
- Prueba acceder a: https://huggingface.co

### Verificar Modelo
- Asegúrate de haber aceptado los términos en:
  https://huggingface.co/black-forest-labs/flux2-dev

### Logs del Servidor
Revisa los logs del servidor para ver errores específicos:
```bash
python run_server.py
```

## 📚 Más Ayuda

- **Guía Completa**: Ver `TROUBLESHOOTING.md`
- **Documentación**: Ver `README.md`
- **Script de Ayuda**: Ejecuta `python setup_hf_token.py`

## 💡 Tips

1. **Guarda el token de forma segura**: No lo compartas públicamente
2. **Usa tokens de solo lectura**: Para mayor seguridad
3. **Configura permanentemente**: Para no tener que hacerlo cada vez
4. **Verifica la expiración**: Algunos tokens tienen fecha de expiración

## ✅ Listo!

Una vez configurado el token, el modelo se descargará automáticamente la primera vez que lo uses. Esto puede tomar varios minutos dependiendo de tu conexión.

¡Disfruta cambiando la ropa de tus personajes! 👔✨


