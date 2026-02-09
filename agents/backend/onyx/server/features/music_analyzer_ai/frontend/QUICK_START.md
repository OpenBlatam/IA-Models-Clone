# 🚀 Inicio Rápido

## Instalación

```bash
cd agents/frontend
npm install
```

## Configuración

1. Copia `.env.local.example` a `.env.local`:
```bash
cp .env.local.example .env.local
```

2. Edita `.env.local` con las URLs de tus backends:
```env
NEXT_PUBLIC_MUSIC_API_URL=http://localhost:8010
NEXT_PUBLIC_ROBOT_API_URL=http://localhost:8010
```

## Ejecutar

```bash
# Desarrollo
npm run dev

# Producción
npm run build
npm start
```

## Verificar Backends

Asegúrate de que ambos backends estén corriendo:

### Music Analyzer AI
```bash
cd agents/backend/onyx/server/features/music_analyzer_ai
python main.py
```

### Robot Movement AI
```bash
cd agents/backend/onyx/server/features/robot_movement_ai
python main.py
```

## Estructura

- `/` - Página principal con selección de plataforma
- `/music` - Music Analyzer AI
- `/robot` - Robot Movement AI

## Características

✅ Búsqueda de canciones en tiempo real
✅ Análisis musical completo
✅ Coaching personalizado
✅ Control de robots mediante chat
✅ Monitoreo en tiempo real
✅ UI moderna y responsive

## Troubleshooting

### Error de conexión
- Verifica que los backends estén corriendo
- Revisa las URLs en `.env.local`
- Verifica los puertos (por defecto 8010)

### Errores de TypeScript
```bash
npm run type-check
```

### Errores de linting
```bash
npm run lint
```

