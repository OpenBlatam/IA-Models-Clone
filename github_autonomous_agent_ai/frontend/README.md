# GitHub Autonomous Agent AI - Frontend

Frontend de Next.js para el agente autónomo de GitHub. **Funciona de forma independiente sin necesidad de backend**. Permite conectar con GitHub, seleccionar repositorios y crear tareas automáticas.

## Características

- 🔐 **Autenticación con GitHub OAuth**: Conecta tu cuenta de GitHub de forma segura (sin backend)
- 📦 **Selector de Repositorios**: Navega y selecciona entre todos tus repositorios
- 🔍 **Búsqueda y Filtros**: Busca repositorios por nombre, filtra por tipo (público/privado) y ordena por fecha o nombre
- 📋 **Gestión de Tareas**: Crea y monitorea tareas para tus repositorios (almacenadas localmente)
- 📊 **Estado del Agente**: Visualiza el estado del agente y la cola de tareas
- 🚀 **Modo Standalone**: Funciona completamente sin backend API

## Instalación

```bash
# Instalar dependencias
yarn install

# O con npm
npm install
```

## Configuración

1. Copia el archivo `.env.example` a `.env.local`:
```bash
cp .env.example .env.local
```

2. Las credenciales ya están configuradas en `.env.example`. Si necesitas cambiarlas, edita `.env.local`:
```env
# GitHub OAuth Configuration
NEXT_PUBLIC_GITHUB_CLIENT_ID=Ov23liSy9XyIj3dD0dQc
GITHUB_CLIENT_SECRET=6ed948f00e7662bbba0eacd356e60747dd12f08f

# DeepSeek API Configuration
NEXT_PUBLIC_DEEPSEEK_API_KEY=sk-ae1c47feaa3e483b85a936430d1f494a
NEXT_PUBLIC_DEEPSEEK_API_BASE_URL=https://api.deepseek.com
NEXT_PUBLIC_DEEPSEEK_MODEL=deepseek-chat

# App Configuration
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

**Nota**: El `GITHUB_CLIENT_SECRET` se usa solo en las API routes del servidor de Next.js (no se expone al cliente).

## Desarrollo

```bash
# Iniciar servidor de desarrollo
yarn dev

# O con npm
npm run dev
```

La aplicación estará disponible en `http://localhost:3000`

## Uso

1. **Conectar con GitHub**: Haz clic en "Conectar con GitHub" y autoriza la aplicación. El token se guarda localmente en tu navegador.
2. **Seleccionar Repositorio**: Navega por tus repositorios y selecciona uno
3. **Crear Tarea**: Escribe una instrucción y crea una nueva tarea. Las tareas se guardan localmente en tu navegador.
4. **Monitorear**: Observa el estado de tus tareas en tiempo real

**Modo Standalone**: Esta aplicación funciona completamente sin backend. Todas las operaciones se realizan directamente con la API de GitHub y los datos se almacenan localmente en tu navegador.

## Estructura del Proyecto

```
app/
  api/
    github/
      auth/
        initiate/
          route.ts          # API route para iniciar OAuth
        callback/
          route.ts          # API route para procesar callback OAuth
  components/
    GithubAuth.tsx          # Componente de autenticación
    RepositorySelector.tsx   # Selector de repositorios
    GithubCallback.tsx       # Manejo de callback OAuth
  github/
    callback/
      page.tsx              # Página de callback OAuth
  lib/
    github-api.ts           # Cliente API para GitHub (usa API directamente)
  page.tsx                  # Página principal
  layout.tsx                # Layout raíz
  globals.css               # Estilos globales
```

## API Routes de Next.js

El frontend incluye API routes de Next.js para manejar OAuth de forma segura:

- `GET /api/github/auth/initiate` - Inicia el flujo OAuth de GitHub
- `GET /api/github/auth/callback` - Intercambia el código OAuth por un token de acceso

**Nota**: Estas API routes se ejecutan en el servidor de Next.js, por lo que el `GITHUB_CLIENT_SECRET` nunca se expone al cliente.

## Tecnologías

- **Next.js 14** - Framework React
- **TypeScript** - Tipado estático
- **Tailwind CSS** - Estilos
- **Axios** - Cliente HTTP
- **React Query** - Gestión de estado del servidor

## Build

```bash
# Crear build de producción
yarn build

# Iniciar servidor de producción
yarn start
```

