# 🚀 Top 10 MCP Servers 2025 - Configuración Completa

Basado en el artículo de [dev.to](https://dev.to/therealmrmumba/top-10-cursor-mcp-servers-in-2025-1nm7), aquí tienes las configuraciones listas para copiar y pegar.

---

## ⚡ Configuración Esencial (Top 5 - Recomendada)

**Los 5 servidores más útiles sin configuración compleja:**

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
        "FIRECRAWL_API_KEY": "YOUR-FIRECRAWL-API-KEY"
      }
    },
    "apidog": {
      "command": "npx",
      "args": [
        "-y",
        "@apidog/mcp-server"
      ],
      "env": {
        "APIDOG_ACCESS_TOKEN": "YOUR-APIDOG-ACCESS-TOKEN",
        "APIDOG_PROJECT_ID": "YOUR-PROJECT-ID"
      }
    },
    "browserbase": {
      "command": "npx",
      "args": [
        "-y",
        "@browserbase/mcp-server"
      ],
      "env": {
        "BROWSERBASE_API_KEY": "YOUR-BROWSERBASE-API-KEY",
        "BROWSERBASE_PROJECT_ID": "YOUR-PROJECT-ID"
      }
    },
    "duckduckgo": {
      "command": "npx",
      "args": [
        "-y",
        "duckduckgo-mcp-server"
      ]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "${workspaceFolder}"
      ]
    }
  }
}
```

---

## 🔥 Configuración Completa (Top 10)

**Todos los servidores del top 10 listos para usar:**

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
        "FIRECRAWL_API_KEY": "YOUR-FIRECRAWL-API-KEY"
      }
    },
    "apidog": {
      "command": "npx",
      "args": [
        "-y",
        "@apidog/mcp-server"
      ],
      "env": {
        "APIDOG_ACCESS_TOKEN": "YOUR-APIDOG-ACCESS-TOKEN",
        "APIDOG_PROJECT_ID": "YOUR-PROJECT-ID"
      }
    },
    "browserbase": {
      "command": "npx",
      "args": [
        "-y",
        "@browserbase/mcp-server"
      ],
      "env": {
        "BROWSERBASE_API_KEY": "YOUR-BROWSERBASE-API-KEY",
        "BROWSERBASE_PROJECT_ID": "YOUR-PROJECT-ID"
      }
    },
    "magic-mcp": {
      "command": "npx",
      "args": [
        "-y",
        "magic-mcp-server"
      ],
      "env": {
        "OPENAI_API_KEY": "YOUR-OPENAI-API-KEY"
      }
    },
    "opik": {
      "command": "npx",
      "args": [
        "-y",
        "opik-mcp"
      ],
      "env": {
        "OPIK_API_KEY": "YOUR-OPIK-API-KEY"
      }
    },
    "figma-context": {
      "command": "npx",
      "args": [
        "-y",
        "@figma/mcp-server"
      ],
      "env": {
        "FIGMA_ACCESS_TOKEN": "YOUR-FIGMA-ACCESS-TOKEN"
      }
    },
    "duckduckgo": {
      "command": "npx",
      "args": [
        "-y",
        "duckduckgo-mcp-server"
      ]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "${workspaceFolder}"
      ]
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git"
      ]
    },
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR-GITHUB-TOKEN"
      }
    }
  }
}
```

---

## 📋 Descripción de Cada Servidor

### 1. 🔥 Firecrawl MCP Server
**Web scraping y búsqueda web avanzada**

- **API Key:** https://firecrawl.dev/app (gratis)
- **Uso:** Scraping de sitios web, búsqueda web inteligente
- **Ejemplos:**
  - "Scrapea firecrawl.dev y dime de qué se trata"
  - "Busca las mejores prácticas de TypeScript 2025"
  - "Extrae todos los links de una página"

### 2. 📡 Apidog MCP Server
**Integración con documentación de APIs**

- **API Key:** https://apidog.com (requiere cuenta)
- **Project ID:** Obtén de tu proyecto en Apidog
- **Uso:** Acceso a OpenAPI specs, generación de código desde APIs
- **Ejemplos:**
  - "Genera un cliente TypeScript para mi API"
  - "¿Cuál es la estructura de respuesta de /users?"
  - "Crea interfaces Python basadas en mi OpenAPI spec"

### 3. 🌐 Browserbase MCP Server
**Automatización de navegador en la nube**

- **API Key:** https://browserbase.io (requiere registro)
- **Project ID:** Obtén de tu proyecto en Browserbase
- **Uso:** Screenshots, automatización web, testing
- **Ejemplos:**
  - "Toma un screenshot de google.com"
  - "Navega a una URL y extrae el contenido"
  - "Automatiza el login en un sitio web"

### 4. ✨ Magic MCP Server
**Generación de contenido con IA**

- **API Key:** https://platform.openai.com/api-keys (requiere OpenAI)
- **Uso:** Generación de imágenes placeholder, transformación de texto, resúmenes
- **Ejemplos:**
  - "Genera una imagen placeholder de 800x600"
  - "Transforma este texto a formato markdown"
  - "Resume este contenido para documentación"

### 5. 🔍 Opik MCP Server
**Búsqueda web en tiempo real**

- **API Key:** https://opik.dev (requiere cuenta)
- **Uso:** Búsqueda web en tiempo real, resúmenes, citas de fuentes
- **Ejemplos:**
  - "Busca información sobre las últimas actualizaciones de React"
  - "Encuentra documentación sobre Python async/await"
  - "Resume los últimos artículos sobre AI"

### 6. 🎨 Figma Context MCP Server
**Integración con diseños de Figma**

- **Access Token:** https://www.figma.com/developers/api#access-tokens
- **Uso:** Acceso a diseños, componentes, assets de Figma
- **Ejemplos:**
  - "Muestra los componentes de diseño de mi proyecto"
  - "Extrae los colores del sistema de diseño"
  - "Lista todos los frames del archivo Figma"

### 7. 🦆 DuckDuckGo
**Búsqueda web privada (SIN API KEY)**

- **Sin configuración:** Funciona inmediatamente
- **Uso:** Búsqueda web privada sin tracking
- **Ejemplos:**
  - "Busca información sobre Python async/await"
  - "Encuentra documentación de React hooks"

### 8. 📁 Filesystem
**Navegación de archivos (SIN API KEY)**

- **Sin configuración:** Funciona inmediatamente
- **Uso:** Lectura/escritura de archivos, navegación de directorios
- **Ejemplos:**
  - "Lee el archivo package.json"
  - "Lista todos los archivos en src/"

### 9. 🔧 Git
**Operaciones Git (SIN API KEY)**

- **Sin configuración:** Funciona inmediatamente
- **Uso:** Commits, branches, diffs
- **Ejemplos:**
  - "Haz commit de los cambios"
  - "Muestra el diff actual"

### 10. 🐙 GitHub
**Integración con GitHub**

- **Token:** https://github.com/settings/tokens
- **Uso:** Issues, PRs, repositorios
- **Ejemplos:**
  - "Crea un issue en mi repositorio"
  - "Muestra los últimos commits"

---

## 🎯 Cómo Obtener las API Keys

### Firecrawl (Gratis)
1. Ve a https://firecrawl.dev/app
2. Regístrate (gratis)
3. Copia tu API key del dashboard

### Apidog
1. Ve a https://apidog.com
2. Crea una cuenta
3. Crea un proyecto
4. Obtén Access Token y Project ID desde la configuración

### Browserbase
1. Ve a https://browserbase.io
2. Regístrate (requiere cuenta)
3. Crea un proyecto
4. Obtén API Key y Project ID

### Magic MCP (OpenAI)
1. Ve a https://platform.openai.com/api-keys
2. Crea una API key
3. Copia la key (solo se muestra una vez)

### Opik
1. Ve a https://opik.dev
2. Regístrate
3. Obtén tu API key del dashboard

### Figma
1. Ve a https://www.figma.com/developers/api#access-tokens
2. Genera un Personal Access Token
3. Copia el token

### GitHub
1. Ve a https://github.com/settings/tokens
2. Genera un nuevo token (classic)
3. Selecciona los scopes necesarios (repo, read:org, etc.)

---

## 📝 Instrucciones de Instalación

### Paso 1: Verificar Node.js
```bash
node --version
npx --version
```
Si no tienes Node.js: https://nodejs.org

### Paso 2: Configurar en Cursor

1. **Abre Cursor Settings:**
   - `Ctrl+,` (Windows) o `Cmd+,` (Mac)
   - O `File → Preferences → Settings`

2. **Busca "MCP":**
   - Escribe "MCP" en la búsqueda
   - O busca "mcpServers"

3. **Edita settings.json:**
   - Haz clic en "Edit in settings.json"
   - O busca el archivo manualmente:
     - **Windows:** `%APPDATA%\Cursor\User\settings.json`
     - **Mac:** `~/Library/Application Support/Cursor/User/settings.json`
     - **Linux:** `~/.config/Cursor/User/settings.json`

4. **Copia y pega la configuración:**
   - Usa la configuración esencial o completa de arriba
   - Si ya tienes `mcpServers`, agrega los nuevos servidores

5. **Reemplaza las API Keys:**
   - Cambia todos los `YOUR-*-API-KEY` por tus keys reales
   - Los servidores sin API key funcionan inmediatamente

6. **Guarda y reinicia Cursor**

### Paso 3: Verificar

1. Reinicia Cursor completamente
2. Abre la paleta de comandos (`Ctrl+Shift+P` / `Cmd+Shift+P`)
3. Busca "MCP" - deberías ver todas las herramientas
4. Verifica en el panel MCP que los servidores estén conectados

---

## 💡 Recomendaciones

### Para empezar:
1. **Usa la configuración esencial** (Top 5)
2. **Solo configura Firecrawl** (gratis y muy útil)
3. **DuckDuckGo y Filesystem** funcionan sin API keys
4. Agrega más servidores según necesites

### Servidores que NO requieren API Key:
- ✅ **duckduckgo** - Búsqueda web privada
- ✅ **filesystem** - Sistema de archivos
- ✅ **git** - Operaciones Git

### Servidores recomendados para empezar:
1. **Firecrawl** - Web scraping (gratis)
2. **DuckDuckGo** - Búsqueda (sin API key)
3. **Filesystem** - Archivos (sin API key)
4. **Git** - Control de versiones (sin API key)

---

## 🐛 Solución de Problemas

### Error: "npx command not found"
**Solución:** Instala Node.js desde https://nodejs.org

### Error: "invalid config must be an object"
**Solución:**
- Verifica que el JSON sea válido
- Asegúrate de que `mcpServers` sea un objeto `{}`
- No uses arrays `[]`

### El servidor no se conecta
**Solución:**
- Verifica que la API key sea correcta
- Revisa los logs en la consola de Cursor
- Prueba ejecutar manualmente: `npx -y firecrawl-mcp`

### "Cannot find module" error
**Solución:**
- Los servidores se descargan automáticamente con `npx -y`
- Asegúrate de tener conexión a internet
- Espera unos segundos la primera vez

---

## 📚 Referencias

- **Artículo original:** https://dev.to/therealmrmumba/top-10-cursor-mcp-servers-in-2025-1nm7
- **MCP Protocol:** https://modelcontextprotocol.io
- **Lista completa de servidores:** https://github.com/modelcontextprotocol/servers

---

## ✅ Checklist

- [ ] Node.js instalado
- [ ] npx disponible
- [ ] Configuración copiada en settings.json
- [ ] API keys reemplazadas (al menos Firecrawl)
- [ ] Cursor reiniciado
- [ ] Servidores visibles en panel MCP
- [ ] Herramientas disponibles en paleta de comandos

---

**¡Listo! Ya tienes los Top 10 MCP Servers de 2025 configurados y listos para usar en Cursor!** 🚀










