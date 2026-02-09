# Guía de Testing

## Configuración

El proyecto está configurado con:
- **Jest** - Framework de testing
- **React Testing Library** - Utilidades para testing de componentes React
- **@swc/jest** - Compilador rápido para Jest
- **jest-environment-jsdom** - Entorno de testing para DOM

## Instalación

```bash
npm install
```

## Ejecutar Tests

```bash
# Ejecutar todos los tests
npm test

# Ejecutar tests en modo watch
npm run test:watch

# Ejecutar tests con cobertura
npm run test:coverage
```

## Estructura de Tests

```
__tests__/
├── components/
│   └── music/
│       ├── MusicSocial.test.tsx          # Tests del feed social
│       ├── MusicTrendingNow.test.tsx     # Tests de tendencias
│       ├── MusicActivity.test.tsx        # Tests de actividad
│       ├── MusicRadio.test.tsx           # Tests de radio
│       ├── MusicDiscoverWeekly.test.tsx   # Tests de descubrimiento
│       ├── MusicAchievements.test.tsx    # Tests de logros
│       ├── MusicStreak.test.tsx          # Tests de racha
│       ├── TrackSearch.test.tsx          # Tests de búsqueda
│       └── TrackAnalysis.test.tsx        # Tests de análisis
├── lib/
│   └── api/
│       └── music-api.test.ts             # Tests del servicio API
└── utils.test.ts                          # Tests de utilidades
```

## Tests Creados

### Componentes de Música

1. **MusicSocial.test.tsx**
   - Renderizado del feed social
   - Interacciones (like, comentar, compartir)
   - Botón de seguir usuarios
   - Timestamps

2. **MusicTrendingNow.test.tsx**
   - Renderizado de tracks en tendencia
   - Visualización de porcentajes de tendencia
   - Ranking visual
   - Selección de tracks

3. **MusicActivity.test.tsx**
   - Feed de actividad reciente
   - Diferentes tipos de actividad
   - Información de usuarios y tracks

4. **MusicRadio.test.tsx**
   - Renderizado de estaciones de radio
   - Reproducción/pausa
   - Información de estaciones

5. **MusicDiscoverWeekly.test.tsx**
   - Renderizado de descubrimientos
   - Estado vacío
   - Selección de tracks

6. **TrackSearch.test.tsx**
   - Búsqueda de tracks
   - Debounce de búsqueda
   - Renderizado de resultados
   - Selección de tracks

7. **TrackAnalysis.test.tsx**
   - Renderizado de análisis
   - Análisis musical y técnico
   - Manejo de datos faltantes

### Servicios

1. **music-api.test.ts**
   - Búsqueda de tracks
   - Análisis de tracks
   - Recomendaciones
   - Comparación de tracks
   - Favoritos (agregar/remover/obtener)
   - Manejo de errores

### Utilidades

1. **utils.test.ts**
   - Formateo de duración
   - Formateo de números
   - Debounce

## Mocks Configurados

- `next/navigation` - Router de Next.js
- `react-hot-toast` - Notificaciones
- `framer-motion` - Animaciones
- `axios` - Cliente HTTP
- `window.matchMedia` - Media queries
- `IntersectionObserver` - Observer API
- `Audio` - Audio API del navegador

## Cobertura

Los tests cubren:
- ✅ Renderizado de componentes
- ✅ Interacciones de usuario
- ✅ Llamadas a API
- ✅ Manejo de errores
- ✅ Funciones utilitarias
- ✅ Estados de carga
- ✅ Validaciones

## Mejores Prácticas

1. **Nombres descriptivos**: Usa nombres claros para tests
2. **AAA Pattern**: Arrange, Act, Assert
3. **Un test, una cosa**: Cada test verifica una funcionalidad
4. **Mocks apropiados**: Mockea dependencias externas
5. **Tests independientes**: Cada test debe poder ejecutarse solo

## Ejemplos

### Test de Componente

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { MyComponent } from '@/components/MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('handles click', () => {
    const handleClick = jest.fn();
    render(<MyComponent onClick={handleClick} />);
    
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalled();
  });
});
```

### Test de API

```typescript
import axios from 'axios';
import { myApiService } from '@/lib/api/my-api';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('MyApiService', () => {
  it('should make API call', async () => {
    const mockResponse = { data: { success: true } };
    mockedAxios.get.mockResolvedValue(mockResponse);

    const result = await myApiService.getData();
    expect(result).toEqual({ success: true });
  });
});
```

## Próximos Pasos

- [ ] Agregar más tests de integración
- [ ] Tests E2E con Playwright
- [ ] Tests de accesibilidad
- [ ] Tests de rendimiento
- [ ] Aumentar cobertura a 80%+

