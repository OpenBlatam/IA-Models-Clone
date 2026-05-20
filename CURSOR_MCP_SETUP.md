# 🚀 Configuración MCP para Cursor - Blatam Code Improvement

## 📋 Instrucciones de Configuración

### ⚠️ IMPORTANTE: Formato Correcto

Cursor espera un formato específico. El archivo `cursor_mcp_config.json` contiene el formato correcto que debes copiar.

### Opción 1: Configuración en Settings de Cursor (Recomendada)

1. **Abrir Cursor Settings**
   - Presiona `Ctrl+,` (o `Cmd+,` en Mac)
   - O ve a `File → Preferences → Settings`

2. **Buscar "MCP"** en la barra de búsqueda

3. **Hacer clic en "Edit in settings.json"** o buscar la sección `mcpServers`

4. **Copiar el contenido de `cursor_mcp_config.json`** y pegarlo en la configuración

   El contenido que debes copiar es:
   ```json
   {
     "mcpServers": {
       "blatam-code-improvement": {
         "command": "python",
         "args": [
           "-m",
           "mcp_code_improvement.server"
         ],
         "env": {
           "PYTHONPATH": "${workspaceFolder}/agents/backend/onyx/server/features"
         }
       }
     }
   }
   ```

### Opción 2: Configuración Manual en settings.json

Si necesitas configurarlo manualmente:

1. **Abrir el archivo de configuración de Cursor**

   **Windows:**
   ```
   %APPDATA%\Cursor\User\settings.json
   ```

   **macOS:**
   ```
   ~/Library/Application Support/Cursor/User/settings.json
   ```

   **Linux:**
   ```
   ~/.config/Cursor/User/settings.json
   ```

2. **Abrir el archivo `cursor_mcp_config.json`** en este directorio y copiar su contenido

3. **Agregar o fusionar** la configuración en `settings.json`:

   Si ya tienes `mcpServers`, agrega el nuevo servidor:
   ```json
   {
     "mcpServers": {
       "otro-servidor": { ... },
       "blatam-code-improvement": {
         "command": "python",
         "args": ["-m", "mcp_code_improvement.server"],
         "env": {
           "PYTHONPATH": "${workspaceFolder}/agents/backend/onyx/server/features"
         }
       }
     }
   }
   ```

   Si no tienes `mcpServers`, agrega toda la sección completa del archivo `cursor_mcp_config.json`

### Opción 3: Configuración a Nivel de Workspace

Puedes crear un archivo `.cursor/mcp.json` en la raíz del workspace con el contenido de `cursor_mcp_config.json`

## 🔧 Verificación

1. **Reinicia Cursor** después de agregar la configuración

2. **Verifica la conexión:**
   - Abre la paleta de comandos (`Ctrl+Shift+P` / `Cmd+Shift+P`)
   - Busca "MCP" para ver los comandos disponibles
   - Deberías ver herramientas como `analyze_code_quality`, `detect_code_duplication`, etc.

3. **Revisa el panel MCP:**
   - Ve a la vista de MCP en el panel lateral
   - Verifica que `blatam-code-improvement` aparezca como conectado

## 🐛 Solución de Problemas

### Error: "invalid config must be an object"

Este error ocurre cuando:
- El JSON tiene errores de sintaxis
- Falta la estructura `mcpServers`
- Hay comas adicionales o llaves desbalanceadas

**Solución:**
1. Valida el JSON con un validador online
2. Asegúrate de que el formato sea exactamente:
   ```json
   {
     "mcpServers": {
       "serverName": { ... }
     }
   }
   ```

### Error: "Command not found"

**Solución:**
1. Verifica que Python esté en el PATH
2. Asegúrate de estar en el directorio correcto
3. Prueba usar la ruta completa a Python:
   ```json
   "command": "C:\\Python\\python.exe"  // Windows
   "command": "/usr/bin/python3"        // Linux/Mac
   ```

### El servidor no se conecta

**Solución:**
1. Verifica que el módulo `mcp_code_improvement` esté instalado
2. Instala las dependencias:
   ```bash
   cd agents/backend/onyx/server/features
   pip install -r mcp_code_improvement/requirements.txt
   ```
3. Prueba ejecutar el servidor manualmente:
   ```bash
   python -m mcp_code_improvement.server
   ```

## 📝 Formato Correcto del JSON

El formato que Cursor espera es:

```json
{
  "mcpServers": {
    "nombre-del-servidor": {
      "command": "comando",
      "args": ["arg1", "arg2"],
      "env": {
        "VARIABLE": "valor"
      }
    }
  }
}
```

**Importante:**
- `mcpServers` debe ser un objeto (no un array)
- Cada servidor debe tener `command` y opcionalmente `args` y `env`
- No uses `default` en los esquemas de herramientas (eso es solo para documentación)

## ✅ Checklist de Configuración

- [ ] Archivo `.cursor/mcp.json` creado en el workspace
- [ ] JSON válido (sin errores de sintaxis)
- [ ] Python está en el PATH
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Cursor reiniciado después de la configuración
- [ ] Servidor aparece en el panel MCP de Cursor
- [ ] Las herramientas están disponibles en la paleta de comandos

## 🎯 Uso

Una vez configurado, puedes usar las herramientas desde:

1. **Paleta de comandos** (`Ctrl+Shift+P`):
   - Busca "MCP: Execute Tool"
   - Selecciona una herramienta como `analyze_code_quality`

2. **Chat de Cursor**:
   - Menciona las herramientas directamente
   - Ejemplo: "Usa analyze_code_quality en features/suno_clone_ai"

3. **Panel MCP**:
   - Ve al panel lateral de MCP
   - Explora las herramientas disponibles

## 📚 Herramientas Disponibles

- `analyze_code_quality` - Análisis de calidad de código
- `detect_code_duplication` - Detección de código duplicado
- `suggest_refactoring` - Sugerencias de refactorización
- `analyze_architecture` - Análisis arquitectónico
- `optimize_performance` - Optimización de performance
- `check_security` - Verificación de seguridad
- `improve_documentation` - Mejora de documentación
- Y 8 herramientas más...

Ver `MCP_SERVER_README.md` para documentación completa.

