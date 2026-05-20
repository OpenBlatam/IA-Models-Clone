# Lista Completa de Tests

## Resumen de Tests Disponibles

### Tests de Utilidades (1 archivo)
- ✅ `utils.test.ts` - Funciones utilitarias (cn, formatDuration, formatBPM, formatPercentage, debounce)

### Tests de Hooks (2 archivos)
- ✅ `lib/hooks/use-debounce.test.ts` - Hook de debounce
- ✅ `lib/hooks/use-local-storage.test.ts` - Hook de localStorage

### Tests de Componentes Base (4 archivos)
- ✅ `components/api-status.test.tsx` - Componente de estado de API
- ✅ `components/error-boundary.test.tsx` - Error boundary
- ✅ `components/navigation.test.tsx` - Navegación
- ✅ `components/music/audio-player.test.tsx` - Reproductor de audio

### Tests de Componentes de Música (27 archivos)
- ✅ `components/music/AudioPlayer.test.tsx` - Reproductor de audio
- ✅ `components/music/CommentSection.test.tsx` - Sección de comentarios
- ✅ `components/music/Equalizer.test.tsx` - Ecualizador
- ✅ `components/music/FavoritesManager.test.tsx` - Gestor de favoritos
- ✅ `components/music/LyricsViewer.test.tsx` - Visor de letras
- ✅ `components/music/MusicActivity.test.tsx` - Actividad
- ✅ `components/music/MusicAchievements.test.tsx` - Logros
- ✅ `components/music/MusicChallenges.test.tsx` - Desafíos
- ✅ `components/music/MusicDiscoverWeekly.test.tsx` - Descubrimientos semanales
- ✅ `components/music/MusicLeaderboard.test.tsx` - Clasificación
- ✅ `components/music/MusicPlayer.test.tsx` - Reproductor principal
- ✅ `components/music/MusicRadio.test.tsx` - Radio
- ✅ `components/music/MusicSettings.test.tsx` - Configuración
- ✅ `components/music/MusicSocial.test.tsx` - Social
- ✅ `components/music/MusicStreak.test.tsx` - Racha
- ✅ `components/music/MusicTrendingNow.test.tsx` - Tendencias
- ✅ `components/music/NotesManager.test.tsx` - Gestor de notas
- ✅ `components/music/PlaylistManager.test.tsx` - Gestor de playlists
- ✅ `components/music/PlaylistQueue.test.tsx` - Cola de reproducción
- ✅ `components/music/Recommendations.test.tsx` - Recomendaciones
- ✅ `components/music/TagManager.test.tsx` - Gestor de etiquetas
- ✅ `components/music/ThemeToggle.test.tsx` - Toggle de tema
- ✅ `components/music/TrackAnalysis.test.tsx` - Análisis de tracks
- ✅ `components/music/TrackComparison.test.tsx` - Comparación de tracks
- ✅ `components/music/TrackRating.test.tsx` - Calificación de tracks
- ✅ `components/music/TrackSearch.test.tsx` - Búsqueda de tracks
- ✅ `components/music/WaveformVisualizer.test.tsx` - Visualizador de onda

### Tests de Librerías (3 archivos)
- ✅ `lib/errors.test.ts` - Clases de error y utilidades
- ✅ `lib/api/client.test.ts` - Cliente API
- ✅ `lib/api/music-api.test.ts` - API de música

## Estadísticas Totales

- **Total de archivos de test**: 37
- **Tests de componentes**: 31
- **Tests de hooks**: 2
- **Tests de utilidades**: 1
- **Tests de librerías**: 3

## Comandos para Ejecutar Tests

```bash
# Ejecutar todos los tests
npm test

# Modo watch
npm run test:watch

# Con cobertura
npm run test:coverage

# Test específico
npm test -- utils.test.ts
npm test -- components/api-status.test.tsx
npm test -- lib/hooks/use-debounce.test.ts
```

## Solución al Problema de SWC

Si encuentras el error:
```
Failed to load SWC binary for win32/x64
```

### Solución 1: Reinstalar dependencias
```bash
rm -rf node_modules package-lock.json
npm install
```

### Solución 2: Usar ts-jest en lugar de @swc/jest

Si el problema persiste, puedes modificar `jest.config.js` para usar `ts-jest`:

```javascript
// Instalar ts-jest
npm install --save-dev ts-jest

// Modificar jest.config.js para usar ts-jest en lugar de @swc/jest
transform: {
  '^.+\\.(ts|tsx)$': 'ts-jest',
},
```

### Solución 3: Ejecutar tests sin Next.js SWC

Crear un script alternativo en `package.json`:

```json
"test:direct": "jest --config jest.config.js --no-cache"
```

## Verificación de Tests

Para verificar que los tests están correctamente estructurados sin ejecutarlos:

```bash
# Verificar sintaxis de TypeScript
npm run type-check

# Verificar linting
npm run lint
```

