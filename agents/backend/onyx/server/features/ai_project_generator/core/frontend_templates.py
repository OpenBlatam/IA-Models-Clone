"""
Frontend Templates - Templates para generación de código frontend
==================================================================

Centraliza todos los templates de código para la generación de frontend,
reduciendo duplicación y mejorando mantenibilidad.
"""

from typing import Dict, Any


class FrontendTemplates:
    """Templates para generación de código frontend"""
    
    @staticmethod
    def package_json(project_name: str, version: str, description: str) -> Dict[str, Any]:
        """Template para package.json"""
        return {
            "name": project_name,
            "version": version,
            "description": description,
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.20.0",
                "axios": "^1.6.0",
                "@tanstack/react-query": "^5.8.0",
                "zustand": "^4.4.0",
                "tailwindcss": "^3.3.0",
                "autoprefixer": "^10.4.16",
                "postcss": "^8.4.32"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "@vitejs/plugin-react": "^4.2.0",
                "vite": "^5.0.0",
                "typescript": "^5.3.0",
                "eslint": "^8.54.0",
                "@typescript-eslint/eslint-plugin": "^6.12.0",
                "@typescript-eslint/parser": "^6.12.0"
            },
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview",
                "lint": "eslint src --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
            }
        }
    
    @staticmethod
    def vite_config() -> str:
        """Template para vite.config.ts"""
        return '''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
'''
    
    @staticmethod
    def tsconfig() -> Dict[str, Any]:
        """Template para tsconfig.json"""
        return {
            "compilerOptions": {
                "target": "ES2020",
                "useDefineForClassFields": True,
                "lib": ["ES2020", "DOM", "DOM.Iterable"],
                "module": "ESNext",
                "skipLibCheck": True,
                "moduleResolution": "bundler",
                "allowImportingTsExtensions": True,
                "resolveJsonModule": True,
                "isolatedModules": True,
                "noEmit": True,
                "jsx": "react-jsx",
                "strict": True,
                "noUnusedLocals": True,
                "noUnusedParameters": True,
                "noFallthroughCasesInSwitch": True
            },
            "include": ["src"],
            "references": [{"path": "./tsconfig.node.json"}]
        }
    
    @staticmethod
    def tsconfig_node() -> Dict[str, Any]:
        """Template para tsconfig.node.json"""
        return {
            "compilerOptions": {
                "composite": True,
                "skipLibCheck": True,
                "module": "ESNext",
                "moduleResolution": "bundler",
                "allowSyntheticDefaultImports": True
            },
            "include": ["vite.config.ts"]
        }
    
    @staticmethod
    def index_html(project_name: str) -> str:
        """Template para index.html"""
        title = project_name.replace('_', ' ').title()
        return f'''<!doctype html>
<html lang="es">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
'''
    
    @staticmethod
    def main_tsx() -> str:
        """Template para main.tsx"""
        return '''import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
'''
    
    @staticmethod
    def app_tsx(project_name: str) -> str:
        """Template para App.tsx"""
        title = project_name.replace('_', ' ').title()
        return f'''import {{ useState }} from 'react'
import {{ BrowserRouter as Router, Routes, Route, Link }} from 'react-router-dom'
import Home from './pages/Home'
import AIProcessor from './pages/AIProcessor'
import './App.css'

function App() {{
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <nav className="bg-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <h1 className="text-xl font-bold text-gray-800">
                  {title}
                </h1>
              </div>
              <div className="flex items-center space-x-4">
                <Link to="/" className="text-gray-600 hover:text-gray-800">
                  Inicio
                </Link>
                <Link to="/ai" className="text-gray-600 hover:text-gray-800">
                  Procesar IA
                </Link>
              </div>
            </div>
          </div>
        </nav>

        <main className="max-w-7xl mx-auto py-6 px-4">
          <Routes>
            <Route path="/" element={{<Home />}} />
            <Route path="/ai" element={{<AIProcessor />}} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}}

export default App
'''
    
    @staticmethod
    def home_tsx(project_name: str, description: str) -> str:
        """Template para Home.tsx"""
        title = project_name.replace('_', ' ').title()
        return f'''import {{ useEffect, useState }} from 'react'
import {{ aiService }} from '../services/aiService'

export default function Home() {{
  const [status, setStatus] = useState<any>(null)

  useEffect(() => {{
    aiService.getStatus().then(setStatus)
  }}, [])

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-md p-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">
          Bienvenido a {title}
        </h1>
        <p className="text-gray-600 mb-6">
          {description}
        </p>
        
        {{status && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <p className="text-green-800">
              Estado del servicio: <span className="font-semibold">{{status.status}}</span>
            </p>
          </div>
        )}}
      </div>
    </div>
  )
}}
'''
    
    @staticmethod
    def ai_processor_tsx() -> str:
        """Template para AIProcessor.tsx"""
        return '''import { useState } from 'react'
import { aiService } from '../services/aiService'

export default function AIProcessor() {
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await aiService.process(prompt)
      setResult(response.result)
    } catch (err: any) {
      setError(err.message || 'Error al procesar la solicitud')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-md p-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">
          Procesador de IA
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-2">
              Ingresa tu prompt:
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              rows={6}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Escribe aquí tu solicitud..."
            />
          </div>

          <button
            type="submit"
            disabled={loading || !prompt.trim()}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {loading ? 'Procesando...' : 'Procesar'}
          </button>
        </form>

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-6 bg-gray-50 border border-gray-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Resultado:</h3>
            <p className="text-gray-700 whitespace-pre-wrap">{result}</p>
          </div>
        )}
      </div>
    </div>
  )
}
'''
    
    @staticmethod
    def ai_service_ts() -> str:
        """Template para aiService.ts"""
        return '''import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface AIRequest {
  prompt: string
  context?: Record<string, any>
  parameters?: Record<string, any>
}

export interface AIResponse {
  result: string
  metadata?: Record<string, any>
}

export const aiService = {
  async process(prompt: string, context?: Record<string, any>): Promise<AIResponse> {
    const response = await api.post<AIResponse>('/api/v1/ai/process', {
      prompt,
      context,
    })
    return response.data
  },

  async getStatus() {
    const response = await api.get('/api/v1/ai/status')
    return response.data
  },
}
'''
    
    @staticmethod
    def index_css() -> str:
        """Template para index.css"""
        return '''@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
'''
    
    @staticmethod
    def app_css() -> str:
        """Template para App.css"""
        return "/* App styles */\n"
    
    @staticmethod
    def tailwind_config() -> str:
        """Template para tailwind.config.js"""
        return '''/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
'''
    
    @staticmethod
    def postcss_config() -> str:
        """Template para postcss.config.js"""
        return '''export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
'''
    
    @staticmethod
    def gitignore() -> str:
        """Template para .gitignore"""
        return '''# Dependencies
node_modules/
/.pnp
.pnp.js

# Testing
/coverage

# Production
/build
/dist

# Misc
.DS_Store
.env.local
.env.development.local
.env.test.local
.env.production.local

npm-debug.log*
yarn-debug.log*
yarn-error.log*
'''
    
    @staticmethod
    def env_example() -> str:
        """Template para .env.example"""
        return '''VITE_API_URL=http://localhost:8000
'''
    
    @staticmethod
    def dockerfile() -> str:
        """Template para Dockerfile"""
        return '''FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host"]
'''
    
    @staticmethod
    def readme(project_name: str, description: str) -> str:
        """Template para README.md"""
        title = project_name.replace('_', ' ').title()
        return f'''# {title} - Frontend

Frontend para {description}

## 🚀 Instalación

```bash
npm install
```

## 🏃 Ejecutar

```bash
npm run dev
npm run build
npm run preview
```

## 🛠️ Tecnologías

- React 18
- TypeScript
- Vite
- Tailwind CSS
- React Router
- Axios
'''

