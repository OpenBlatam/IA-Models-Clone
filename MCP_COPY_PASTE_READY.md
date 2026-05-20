# 🚀 Configuración MCP Completa - Lista para Copiar y Pegar

## ⚡ Configuración Mínima (Recomendada para empezar)

**Solo 3 servidores esenciales - Sin API keys requeridas (excepto Firecrawl)**

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

**Obtén Firecrawl API Key:** https://firecrawl.dev/app (gratis)

---

## 🎯 Configuración Esencial (6 servidores)

**Incluye búsqueda web, GitHub, Git, y más - Sin Python requerido**

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
    "duckduckgo": {
      "command": "npx",
      "args": [
        "-y",
        "duckduckgo-mcp-server"
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
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git"
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
    "puppeteer": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-puppeteer"
      ]
    }
  }
}
```

**Obtén GitHub Token:** https://github.com/settings/tokens (opcional)

---

## 🔥 Configuración Completa (10 servidores)

**Máxima funcionalidad - Todos los servidores populares**

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
    "brave-search": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-brave-search"
      ],
      "env": {
        "BRAVE_API_KEY": "YOUR-BRAVE-API-KEY"
      }
    },
    "duckduckgo": {
      "command": "npx",
      "args": [
        "-y",
        "duckduckgo-mcp-server"
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
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git"
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
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://user:password@localhost:5432/dbname"
      ]
    },
    "puppeteer": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-puppeteer"
      ]
    },
    "tavily": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-server-tavily"
      ],
      "env": {
        "TAVILY_API_KEY": "YOUR-TAVILY-API-KEY"
      }
    },
    "slack": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-slack"
      ],
      "env": {
        "SLACK_BOT_TOKEN": "YOUR-SLACK-BOT-TOKEN"
      }
    }
  }
}
```

---

## 📋 Instrucciones de Instalación

### Paso 1: Verificar Node.js

Abre una terminal y verifica que tienes Node.js instalado:

```bash
node --version
npx --version
```

Si no tienes Node.js:
- **Windows/Mac:** Descarga de https://nodejs.org
- **Linux:** `sudo apt install nodejs npm`

### Paso 2: Configurar en Cursor

1. **Abre Cursor Settings:**
   - Presiona `Ctrl+,` (Windows/Linux) o `Cmd+,` (Mac)
   - O ve a `File → Preferences → Settings`

2. **Busca "MCP":**
   - Escribe "MCP" en la barra de búsqueda
   - O busca "mcpServers" directamente

3. **Edita settings.json:**
   - Haz clic en "Edit in settings.json" (ícono de lápiz)
   - O busca el archivo manualmente:
     - **Windows:** `%APPDATA%\Cursor\User\settings.json`
     - **Mac:** `~/Library/Application Support/Cursor/User/settings.json`
     - **Linux:** `~/.config/Cursor/User/settings.json`

4. **Copia y pega la configuración:**
   - Si ya tienes `mcpServers`, agrega los nuevos servidores dentro del objeto
   - Si no tienes `mcpServers`, pega toda la configuración

5. **Reemplaza API Keys:**
   - Cambia `YOUR-FIRECRAWL-API-KEY` por tu API key real
   - Obtén tu key en: https://firecrawl.dev/app
   - Las otras API keys son opcionales

6. **Guarda y reinicia Cursor**

### Paso 3: Verificar

1. Reinicia Cursor completamente
2. Abre la paleta de comandos (`Ctrl+Shift+P` / `Cmd+Shift+P`)
3. Busca "MCP" - deberías ver las herramientas disponibles
4. Verifica en el panel MCP que los servidores estén conectados

---

## 🎯 Qué hace cada servidor

### 🔥 Firecrawl MCP
- **Web scraping** avanzado
- **Búsqueda web** inteligente
- **Extracción de datos** de sitios web
- **API Key requerida:** https://firecrawl.dev/app

**Ejemplos de uso:**
- "Scrapea firecrawl.dev y dime de qué se trata"
- "Busca las mejores prácticas de TypeScript 2025"
- "Extrae todos los links de una página"

### 🦆 DuckDuckGo
- **Búsqueda web** privada (sin API key)
- **Sin tracking** ni cookies
- **Búsquedas rápidas** desde Cursor

**Ejemplos:**
- "Busca información sobre Python async/await"
- "Encuentra documentación de React hooks"

### 📁 Filesystem
- **Navegación de archivos** desde Cursor
- **Lectura/escritura** de archivos
- **Operaciones de directorios**

**Ejemplos:**
- "Lee el archivo package.json"
- "Lista todos los archivos en src/"
- "Crea un nuevo archivo test.js"

### 🐙 GitHub
- **Acceso a repositorios** GitHub
- **Crear/editar issues** y PRs
- **Gestionar código** desde Cursor
- **API Key requerida:** https://github.com/settings/tokens

**Ejemplos:**
- "Crea un issue en mi repositorio"
- "Muestra los últimos commits"
- "Lista los pull requests abiertos"

### 🔧 Git
- **Operaciones Git** desde Cursor
- **Commits, branches, diffs**
- **Sin configuración adicional**

**Ejemplos:**
- "Haz commit de los cambios"
- "Muestra el diff actual"
- "Crea una nueva rama"

### 🎭 Puppeteer
- **Automatización de navegador**
- **Screenshots** y scraping
- **Testing** de aplicaciones web

**Ejemplos:**
- "Toma un screenshot de google.com"
- "Navega a una URL y extrae el contenido"

### 🦁 Brave Search
- **Búsqueda privada** con Brave
- **Resultados relevantes**
- **API Key requerida:** https://brave.com/search/api

### 🗄️ Postgres
- **Consultas a base de datos** PostgreSQL
- **Operaciones SQL** desde Cursor
- **Requiere conexión a DB**

### 🤖 Tavily
- **Búsqueda AI-powered**
- **Resultados semánticos**
- **API Key requerida:** https://tavily.com

### 💬 Slack
- **Integración con Slack**
- **Enviar mensajes** y leer canales
- **Bot Token requerido**

---

## ⚙️ Configuración por Servidor

### Servidores que NO requieren API Key (funcionan inmediatamente):
- ✅ **duckduckgo** - Búsqueda web privada
- ✅ **git** - Operaciones Git
- ✅ **filesystem** - Sistema de archivos
- ✅ **puppeteer** - Automatización de navegador

### Servidores que requieren API Key:
- 🔑 **firecrawl-mcp** - https://firecrawl.dev/app (gratis)
- 🔑 **github** - https://github.com/settings/tokens (opcional)
- 🔑 **brave-search** - https://brave.com/search/api (opcional)
- 🔑 **tavily** - https://tavily.com (opcional)
- 🔑 **slack** - Slack Bot Token (opcional)
- 🔑 **postgres** - Conexión a base de datos (opcional)

---

## 🐛 Solución de Problemas

### Error: "npx command not found"
**Solución:**
1. Instala Node.js: https://nodejs.org
2. Verifica instalación: `npx --version`
3. Reinicia Cursor después de instalar Node.js

### Error: "invalid config must be an object"
**Solución:**
1. Verifica que el JSON sea válido (usa un validador online)
2. Asegúrate de que `mcpServers` sea un objeto `{}` no un array `[]`
3. Revisa que no haya comas adicionales al final

### El servidor no se conecta
**Solución:**
1. Verifica que Node.js esté instalado
2. Prueba ejecutar manualmente: `npx -y firecrawl-mcp`
3. Revisa los logs en la consola de Cursor
4. Asegúrate de que la API key sea correcta (si es requerida)

### "Cannot find module" error
**Solución:**
1. Los servidores se descargan automáticamente con `npx -y`
2. Asegúrate de tener conexión a internet
3. Espera unos segundos la primera vez (descarga automática)

---

## 💡 Tips y Mejores Prácticas

1. **Empieza con la configuración mínima** y agrega servidores según necesites
2. **No necesitas todos los servidores** - solo los que uses
3. **Las API keys son opcionales** excepto Firecrawl (recomendado)
4. **Reinicia Cursor** después de cada cambio en la configuración
5. **Usa la paleta de comandos** (`Ctrl+Shift+P`) para acceder a herramientas MCP

---

## 📚 Recursos Adicionales

- **Firecrawl Docs:** https://docs.firecrawl.dev
- **MCP Protocol:** https://modelcontextprotocol.io
- **Cursor MCP Guide:** Ver `CURSOR_MCP_SETUP.md`
- **Lista de servidores MCP:** https://github.com/modelcontextprotocol/servers

---

## ✅ Checklist Rápido

- [ ] Node.js instalado (`node --version`)
- [ ] npx disponible (`npx --version`)
- [ ] Configuración copiada en `settings.json`
- [ ] API keys reemplazadas (si usas servidores que las requieren)
- [ ] Cursor reiniciado
- [ ] Servidores visibles en panel MCP
- [ ] Herramientas disponibles en paleta de comandos

---

**¡Listo! Ya puedes usar MCP en Cursor sin necesidad de Python. Todo funciona con npx (Node.js).** 🚀










