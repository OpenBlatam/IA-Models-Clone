# 🚀 Configuraciones MCP Listas para Copiar

## 📋 Instrucciones Rápidas

1. Abre Cursor Settings (`Ctrl+,` o `Cmd+,`)
2. Busca "MCP" o "mcpServers"
3. Haz clic en "Edit in settings.json"
4. Copia y pega la configuración que necesites
5. **Reemplaza `YOUR-API-KEY` con tu API key real**
6. Guarda y reinicia Cursor

---

## 🔥 Firecrawl MCP Server

**Para web scraping y búsqueda web directamente en Cursor**

```json
{
  "mcpServers": {
    "firecrawl-mcp": {
      "command": "npx",
      "args": [
        "-y",
        "firecrawl-mcp"
      ],
      "env": {
        "FIRECRAWL_API_KEY": "YOUR-API-KEY"
      }
    }
  }
}
```

**Obtén tu API key:** https://firecrawl.dev/app

**Uso:**
- "Scrapea firecrawl.dev y dime de qué se trata"
- "Busca las mejores prácticas de TypeScript 2025"
- "Scrapea la documentación de React hooks"

---

## 🔧 Blatam Code Improvement Server

**Para análisis y mejora de código del proyecto**

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

**Herramientas disponibles:**
- `analyze_code_quality` - Análisis de calidad
- `detect_code_duplication` - Detección de duplicación
- `suggest_refactoring` - Sugerencias de refactorización
- `check_security` - Verificación de seguridad
- `optimize_performance` - Optimización de performance
- Y 11 herramientas más...

---

## 🎯 Configuración Completa (Ambos Servidores)

**Copia esto para tener ambos servidores activos:**

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
    },
    "firecrawl-mcp": {
      "command": "npx",
      "args": [
        "-y",
        "firecrawl-mcp"
      ],
      "env": {
        "FIRECRAWL_API_KEY": "YOUR-API-KEY"
      }
    }
  }
}
```

---

## 📝 Notas Importantes

### Para Firecrawl:
1. Necesitas Node.js y npx instalados
2. Obtén tu API key en https://firecrawl.dev/app
3. Reemplaza `YOUR-API-KEY` con tu clave real

### Para Blatam Code Improvement:
1. Necesitas Python 3.8+ instalado
2. Instala las dependencias:
   ```bash
   cd agents/backend/onyx/server/features
   pip install -r mcp_code_improvement/requirements.txt
   ```
3. Asegúrate de que Python esté en tu PATH

### Si ya tienes otros servidores MCP:
Simplemente agrega los nuevos servidores dentro del objeto `mcpServers`:

```json
{
  "mcpServers": {
    "otro-servidor-existente": { ... },
    "blatam-code-improvement": { ... },
    "firecrawl-mcp": { ... }
  }
}
```

---

## ✅ Verificación

Después de configurar:

1. **Reinicia Cursor**
2. Abre la paleta de comandos (`Ctrl+Shift+P` / `Cmd+Shift+P`)
3. Busca "MCP" - deberías ver las herramientas disponibles
4. Verifica en el panel MCP que ambos servidores estén conectados

---

## 🐛 Troubleshooting

### Error: "invalid config must be an object"
- Asegúrate de que el JSON sea válido
- Verifica que `mcpServers` sea un objeto (no un array)
- Usa un validador JSON online

### Error: "Command not found"
- Verifica que Node.js/Python estén en el PATH
- Prueba usar rutas completas:
  ```json
  "command": "C:\\Python\\python.exe"  // Windows
  "command": "/usr/bin/python3"        // Linux/Mac
  ```

### El servidor no se conecta
- Verifica que las dependencias estén instaladas
- Revisa los logs en la consola de Cursor
- Prueba ejecutar el servidor manualmente para ver errores

---

## 📚 Más Información

- **Firecrawl Docs:** https://docs.firecrawl.dev
- **MCP Protocol:** https://modelcontextprotocol.io
- **Cursor MCP Guide:** Ver `CURSOR_MCP_SETUP.md`










