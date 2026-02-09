# TruthGPT Model Builder

Frontend moderno de Next.js con TypeScript para crear modelos de IA adaptados con TruthGPT y publicarlos automáticamente en GitHub.

## 🚀 Características

- **Chat Interface**: Interfaz de chat intuitiva para describir modelos
- **Análisis Inteligente**: Analiza automáticamente tu descripción para determinar la mejor arquitectura
- **Generación Automática**: Crea modelos TruthGPT adaptados con arquitecturas optimizadas (Dense, CNN, LSTM, Transformer)
- **Progreso Detallado**: Monitoreo en tiempo real con barra de progreso y pasos actuales
- **Integración con GitHub**: Publica automáticamente los modelos en GitHub con CI/CD configurado
- **Templates Inteligentes**: Genera arquitecturas apropiadas según el tipo de tarea (clasificación, regresión, NLP, visión, etc.)
- **UI Moderna**: Interfaz hermosa con animaciones y diseño responsivo
- **Especificaciones de Modelo**: Cada modelo incluye especificaciones detalladas (tipo, arquitectura, optimizador, etc.)

## 📋 Requisitos Previos

- Node.js 18+ 
- npm o yarn
- Token de GitHub con permisos para crear repositorios
- TruthGPT instalado (en el directorio `../TruthGPT-main`)

## 🔧 Instalación

1. Instala las dependencias:

```bash
npm install
# o
yarn install
```

2. Configura las variables de entorno:

Copia `.env.example` a `.env` y configura:

```env
GITHUB_TOKEN=tu_token_de_github
GITHUB_OWNER=tu_usuario_o_organizacion  # Opcional, por defecto usa el usuario autenticado
TRUTHGPT_API_PATH=../TruthGPT-main  # Ruta al directorio TruthGPT
```

3. Ejecuta el servidor de desarrollo:

```bash
npm run dev
# o
yarn dev
```

Abre [http://localhost:3000](http://localhost:3000) en tu navegador.

## 🎯 Uso

1. **Describe tu modelo**: Escribe en el chat qué tipo de modelo de IA quieres crear
   - Ejemplo: "Un modelo para análisis de sentimientos en español"
   - Ejemplo: "Un clasificador de imágenes para detectar objetos"

2. **Generación automática**: El sistema creará un modelo TruthGPT adaptado basado en tu descripción

3. **Publicación en GitHub**: Una vez creado, el modelo se publicará automáticamente en un repositorio de GitHub

4. **Monitoreo**: Puedes ver el progreso en tiempo real en la barra de estado

## 📁 Estructura del Proyecto

```
truthgpt-model-builder/
├── app/
│   ├── api/
│   │   ├── create-model/      # API para crear modelos
│   │   └── model-status/      # API para verificar estado
│   ├── globals.css            # Estilos globales
│   ├── layout.tsx              # Layout principal
│   └── page.tsx               # Página principal
├── components/
│   ├── ChatInterface.tsx      # Componente de chat
│   ├── Header.tsx             # Header de la aplicación
│   ├── Message.tsx            # Componente de mensaje
│   └── ModelStatus.tsx        # Barra de estado del modelo
├── lib/
│   ├── truthgpt-service.ts   # Servicio para TruthGPT
│   └── github-service.ts     # Servicio para GitHub
├── store/
│   └── modelStore.ts          # Estado global (Zustand)
└── package.json
```

## 🔑 Obtención de Token de GitHub

1. Ve a GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Genera un nuevo token con los siguientes permisos:
   - `repo` (acceso completo a repositorios)
   - `workflow` (si quieres usar GitHub Actions)
3. Copia el token y úsalo en la variable `GITHUB_TOKEN`

## 🛠️ Tecnologías Utilizadas

- **Next.js 14**: Framework React con App Router
- **TypeScript**: Tipado estático
- **Tailwind CSS**: Estilos utilitarios
- **Framer Motion**: Animaciones
- **Zustand**: Gestión de estado
- **Octokit**: Cliente de GitHub API
- **Lucide React**: Iconos

## 📝 API Endpoints

### POST `/api/create-model`

Crea un nuevo modelo TruthGPT.

**Body:**
```json
{
  "description": "Descripción del modelo"
}
```

**Response:**
```json
{
  "modelId": "model-1234567890-abc123",
  "modelName": "truthgpt-mi-modelo",
  "description": "Descripción del modelo",
  "githubUrl": "https://github.com/user/truthgpt-mi-modelo",
  "status": "creating"
}
```

### GET `/api/model-status/[modelId]`

Obtiene el estado de un modelo.

**Response:**
```json
{
  "status": "completed",
  "githubUrl": "https://github.com/user/truthgpt-mi-modelo"
}
```

## 🚀 Despliegue

### Vercel

1. Conecta tu repositorio con Vercel
2. Configura las variables de entorno en Vercel
3. Deploy automático

### Docker

```bash
docker build -t truthgpt-model-builder .
docker run -p 3000:3000 --env-file .env truthgpt-model-builder
```

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la misma licencia que TruthGPT.

## 🆘 Soporte

Si encuentras algún problema o tienes preguntas:

1. Revisa los issues existentes
2. Crea un nuevo issue con detalles del problema
3. Incluye logs y pasos para reproducir el error

---

**¡Crea modelos de IA con TruthGPT de forma fácil y rápida!** 🎉

