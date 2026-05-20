# Contributing Guide - Test Suite

## 🤝 Guía para Contribuir a los Tests

### Antes de Contribuir

1. ✅ Leer `README.md` para entender la estructura
2. ✅ Revisar `best-practices.md` para mejores prácticas
3. ✅ Ver `example-tests.md` para ejemplos
4. ✅ Ejecutar tests existentes: `npm test`

## 📝 Agregar Nuevos Tests

### 1. Para Nuevos Componentes

```typescript
// __tests__/components/nuevo-component.test.tsx
import { render, screen } from '@testing-library/react';
import { NuevoComponent } from '@/components/NuevoComponent';

describe('NuevoComponent', () => {
  it('should render correctly', () => {
    render(<NuevoComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

### 2. Para Nuevos Hooks

```typescript
// __tests__/lib/hooks/use-nuevo-hook.test.ts
import { renderHook, act } from '@testing-library/react';
import { useNuevoHook } from '@/lib/hooks/use-nuevo-hook';

describe('useNuevoHook', () => {
  it('should work correctly', () => {
    const { result } = renderHook(() => useNuevoHook());
    expect(result.current).toBeDefined();
  });
});
```

### 3. Para Nuevas Utilidades

```typescript
// __tests__/lib/utils/nueva-util.test.ts
import { nuevaUtil } from '@/lib/utils/nueva-util';

describe('nuevaUtil', () => {
  it('should work correctly', () => {
    expect(nuevaUtil()).toBe(expected);
  });
});
```

## ✅ Checklist para Pull Requests

- [ ] Tests pasan: `npm test`
- [ ] Cobertura mantenida: `npm run test:coverage`
- [ ] Tests siguen mejores prácticas
- [ ] Nombres descriptivos
- [ ] Tests independientes
- [ ] Casos edge incluidos
- [ ] Documentación actualizada (si aplica)
- [ ] Sin warnings de linting

## 🎯 Estándares de Calidad

### Nombres de Tests
✅ **Bueno:**
```typescript
it('should display error message when API call fails', () => {});
```

❌ **Malo:**
```typescript
it('test 1', () => {});
```

### Estructura AAA
```typescript
it('should do something', () => {
  // Arrange
  const props = { name: 'Test' };
  
  // Act
  render(<Component {...props} />);
  
  // Assert
  expect(screen.getByText('Test')).toBeInTheDocument();
});
```

### Casos Edge
Siempre incluir:
- Valores normales
- Valores límite (0, null, undefined)
- Valores extremos
- Errores

## 📚 Recursos

- [README.md](./README.md) - Guía principal
- [best-practices.md](./best-practices.md) - Mejores prácticas
- [example-tests.md](./examples/example-tests.md) - Ejemplos
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Solución de problemas

## 🚀 Proceso de Contribución

1. Fork el repositorio
2. Crear branch: `git checkout -b feature/nuevos-tests`
3. Agregar tests siguiendo estándares
4. Ejecutar tests: `npm test`
5. Verificar cobertura: `npm run test:coverage`
6. Commit: `git commit -m "Add tests for X"`
7. Push: `git push origin feature/nuevos-tests`
8. Crear Pull Request

## ✨ Gracias por Contribuir!

Tu contribución ayuda a mantener la calidad del código. 🎉

