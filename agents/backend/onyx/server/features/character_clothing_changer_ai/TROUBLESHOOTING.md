# 🔧 Solución de Problemas - Character Clothing Changer AI

## ❌ Error: "Failed to load pipeline: model is not cached locally"

Este error ocurre cuando el modelo Flux2 no se puede descargar desde HuggingFace Hub.

### 🔍 Causas Comunes

1. **Falta de conexión a internet**
2. **El modelo requiere autenticación** (token de HuggingFace)
3. **No has aceptado los términos del modelo**
4. **El modelo no está disponible públicamente**

### ✅ Soluciones

#### Solución 1: Obtener Token de HuggingFace (Recomendado)

El modelo `black-forest-labs/flux2-dev` requiere autenticación en HuggingFace.

**Pasos:**

1. **Crear cuenta en HuggingFace** (si no tienes una):
   - Ve a: https://huggingface.co/join

2. **Obtener un token de acceso**:
   - Ve a: https://huggingface.co/settings/tokens
   - Haz clic en "New token"
   - Selecciona "Read" como permiso
   - Copia el token generado

3. **Aceptar términos del modelo**:
   - Ve a: https://huggingface.co/black-forest-labs/flux2-dev
   - Haz clic en "Agree and access repository"
   - Acepta los términos y condiciones

4. **Configurar el token en Windows**:
   ```powershell
   set HUGGINGFACE_TOKEN=tu_token_aqui
   ```

   O en Linux/Mac:
   ```bash
   export HUGGINGFACE_TOKEN=tu_token_aqui
   ```

5. **Reiniciar el servidor**:
   ```bash
   python run_server.py
   ```

#### Solución 2: Configurar Token Permanente

**Windows (PowerShell):**
```powershell
[System.Environment]::SetEnvironmentVariable('HUGGINGFACE_TOKEN', 'tu_token', 'User')
```

**Windows (CMD):**
```cmd
setx HUGGINGFACE_TOKEN "tu_token"
```

**Linux/Mac:**
Agrega a tu `~/.bashrc` o `~/.zshrc`:
```bash
export HUGGINGFACE_TOKEN=tu_token
```

Luego ejecuta:
```bash
source ~/.bashrc  # o source ~/.zshrc
```

#### Solución 3: Descargar Modelo Manualmente

Si prefieres descargar el modelo manualmente:

```python
from huggingface_hub import snapshot_download
import os

token = os.getenv("HUGGINGFACE_TOKEN")
snapshot_download(
    repo_id="black-forest-labs/flux2-dev",
    token=token,
    local_dir="./models/flux2-dev"
)
```

Luego configura:
```bash
set CLOTHING_CHANGER_MODEL_ID=./models/flux2-dev
```

#### Solución 4: Usar Modelo Alternativo

Si no puedes acceder al modelo Flux2, puedes intentar con otros modelos:

```bash
# Usar Stable Diffusion como alternativa
set CLOTHING_CHANGER_MODEL_ID=runwayml/stable-diffusion-inpainting
```

**Nota:** Esto requerirá ajustes en el código ya que la API puede ser diferente.

### 🔍 Verificar Configuración

Para verificar que el token está configurado:

**Windows:**
```powershell
echo $env:HUGGINGFACE_TOKEN
```

**Linux/Mac:**
```bash
echo $HUGGINGFACE_TOKEN
```

### 📝 Verificar Conexión

Prueba la conexión a HuggingFace:

```python
from huggingface_hub import whoami

token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    try:
        user_info = whoami(token=token)
        print(f"✅ Autenticado como: {user_info['name']}")
    except Exception as e:
        print(f"❌ Error de autenticación: {e}")
else:
    print("❌ Token no configurado")
```

### 🚨 Otros Errores Comunes

#### Error: "401 Unauthorized"
- **Causa**: Token inválido o expirado
- **Solución**: Genera un nuevo token en HuggingFace

#### Error: "403 Forbidden"
- **Causa**: No has aceptado los términos del modelo
- **Solución**: Ve a https://huggingface.co/black-forest-labs/flux2-dev y acepta los términos

#### Error: "Connection timeout"
- **Causa**: Problemas de conexión a internet
- **Solución**: Verifica tu conexión o usa un VPN

#### Error: "Out of memory"
- **Causa**: No hay suficiente RAM/VRAM
- **Solución**: 
  - Usa CPU: `set CLOTHING_CHANGER_DEVICE=cpu`
  - Reduce el tamaño de imagen: `set CLOTHING_CHANGER_MAX_IMAGE_SIZE=512`

### 📚 Recursos Adicionales

- **Documentación de HuggingFace**: https://huggingface.co/docs/hub/security-tokens
- **Modelo Flux2**: https://huggingface.co/black-forest-labs/flux2-dev
- **Guía de Tokens**: https://huggingface.co/docs/hub/security-tokens

### 💡 Tips

1. **Guarda tu token de forma segura**: No lo compartas públicamente
2. **Usa tokens de solo lectura**: Para mayor seguridad
3. **Verifica la expiración**: Algunos tokens tienen fecha de expiración
4. **Usa variables de entorno**: No hardcodees el token en el código

### 🆘 ¿Aún tienes problemas?

1. Verifica los logs del servidor para más detalles
2. Asegúrate de tener la última versión de `diffusers` y `transformers`
3. Intenta actualizar las dependencias:
   ```bash
   pip install --upgrade diffusers transformers huggingface_hub
   ```


