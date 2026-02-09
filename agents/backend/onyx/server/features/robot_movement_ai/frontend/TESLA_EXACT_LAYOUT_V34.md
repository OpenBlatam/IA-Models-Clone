# Sistema de Layout Exacto Tesla - V34

## 🎯 Sistema de Layout Completo

### Archivos Creados

1. **`tesla-exact-layout.ts`** - Sistema de layout exacto
2. **`tesla-exact-responsive.ts`** - Valores responsive exactos

## 📐 Layout System

### Container Max Widths (Exactos)
```typescript
container: {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
  full: '100%',
}
```

### Grid Gaps (Exactos)
```typescript
gridGap: {
  xs: '8px',
  sm: '12px',
  md: '16px',
  lg: '24px',
  xl: '32px',
  '2xl': '48px',
  '3xl': '64px',
}
```

### Section Padding (Exactos)
```typescript
sectionPadding: {
  mobile: {
    top: '64px',
    bottom: '64px',
    horizontal: '16px',
  },
  tablet: {
    top: '96px',
    bottom: '96px',
    horizontal: '24px',
  },
  desktop: {
    top: '128px',
    bottom: '128px',
    horizontal: '32px',
  },
}
```

### Component Spacing (Exactos)
```typescript
componentSpacing: {
  card: {
    padding: '24px',
    gap: '16px',
    marginBottom: '24px',
  },
  form: {
    fieldGap: '16px',
    labelMarginBottom: '8px',
    errorMarginTop: '4px',
    groupGap: '24px',
  },
  buttonGroup: {
    gap: '12px',
    marginTop: '24px',
  },
  list: {
    itemGap: '12px',
    itemPadding: '12px',
    sectionGap: '24px',
  },
  grid: {
    gap: '24px',
    itemPadding: '24px',
  },
}
```

### Z-Index Scale (Exactos)
```typescript
zIndex: {
  base: 0,
  dropdown: 1000,
  sticky: 1020,
  fixed: 1030,
  modalBackdrop: 1040,
  modal: 1050,
  popover: 1060,
  tooltip: 1070,
  notification: 1080,
}
```

### Aspect Ratios (Exactos)
```typescript
aspectRatio: {
  square: '1 / 1',
  video: '16 / 9',
  photo: '4 / 3',
  wide: '21 / 9',
  portrait: '3 / 4',
}
```

## 📱 Responsive System

### Breakpoints (Exactos)
```typescript
breakpoints: {
  xs: '475px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
  '3xl': '1920px',
}
```

### Responsive Font Sizes (Exactos)
```typescript
fontSize: {
  hero: {
    mobile: '40px',
    tablet: '60px',
    desktop: '96px',
    clamp: 'clamp(40px, 8vw, 96px)',
  },
  display: {
    mobile: '30px',
    tablet: '45px',
    desktop: '60px',
    clamp: 'clamp(30px, 5vw, 60px)',
  },
  h1: {
    mobile: '28px',
    tablet: '32px',
    desktop: '36px',
  },
  h2: {
    mobile: '24px',
    tablet: '28px',
    desktop: '30px',
  },
  h3: {
    mobile: '20px',
    tablet: '22px',
    desktop: '24px',
  },
  body: {
    mobile: '14px',
    tablet: '15px',
    desktop: '16px',
  },
}
```

### Responsive Spacing (Exactos)
```typescript
spacing: {
  section: {
    mobile: '64px',
    tablet: '96px',
    desktop: '128px',
  },
  container: {
    mobile: '16px',
    tablet: '24px',
    desktop: '32px',
  },
  card: {
    mobile: '16px',
    tablet: '20px',
    desktop: '24px',
  },
  button: {
    mobile: '12px 24px',
    tablet: '12px 24px',
    desktop: '16px 32px',
  },
}
```

### Responsive Grid Columns (Exactos)
```typescript
gridColumns: {
  mobile: 1,
  tablet: 2,
  desktop: 3,
  wide: 4,
}
```

### Responsive Gaps (Exactos)
```typescript
gaps: {
  mobile: '16px',
  tablet: '24px',
  desktop: '32px',
}
```

## 🔧 Helper Functions

### Layout Helpers
```typescript
getTeslaContainerWidth('xl') // '1280px'
getTeslaGridGap('lg') // '24px'
getTeslaSectionPadding('desktop') // { top: '128px', bottom: '128px', horizontal: '32px' }
getTeslaComponentSpacing('card') // { padding: '24px', gap: '16px', marginBottom: '24px' }
getTeslaZIndex('modal') // 1050
getTeslaAspectRatio('video') // '16 / 9'
```

### Responsive Helpers
```typescript
getTeslaBreakpoint('lg') // '1024px'
getTeslaResponsiveFontSize('hero', 'clamp') // 'clamp(40px, 8vw, 96px)'
getTeslaResponsiveSpacing('section', 'desktop') // '128px'
```

## 📊 Valores Exactos por Categoría

### Spacing
- **Container**: 640px - 1536px
- **Grid Gaps**: 8px - 64px
- **Section Padding**: 64px - 128px (vertical), 16px - 32px (horizontal)
- **Component Padding**: 12px - 24px

### Typography
- **Hero**: 40px - 96px (responsive)
- **Display**: 30px - 60px (responsive)
- **Headings**: 20px - 36px (responsive)
- **Body**: 14px - 16px (responsive)

### Layout
- **Z-Index**: 0 - 1080
- **Aspect Ratios**: 5 variantes
- **Grid Columns**: 1 - 4 (responsive)

## 🎯 Uso en Componentes

### Ejemplo con Layout
```tsx
import { getTeslaComponentSpacing, getTeslaGridGap } from '@/lib/utils/tesla-exact-layout';

const cardSpacing = getTeslaComponentSpacing('card');
// { padding: '24px', gap: '16px', marginBottom: '24px' }

<div style={{ padding: cardSpacing.padding, gap: cardSpacing.gap }}>
  Content
</div>
```

### Ejemplo con Responsive
```tsx
import { getTeslaResponsiveFontSize, getTeslaResponsiveSpacing } from '@/lib/utils/tesla-exact-responsive';

const heroSize = getTeslaResponsiveFontSize('hero', 'clamp');
// 'clamp(40px, 8vw, 96px)'

<h1 style={{ fontSize: heroSize }}>
  Hero Title
</h1>
```

## ✅ Características

1. **Valores Exactos**: Todos los valores en pixels
2. **Responsive**: Sistema completo responsive
3. **Helper Functions**: Acceso fácil a valores
4. **Type Safety**: TypeScript completo
5. **Consistencia**: Valores consistentes en todo el sistema

## 📦 Archivos Creados

1. ✅ `tesla-exact-layout.ts` - Sistema de layout
2. ✅ `tesla-exact-responsive.ts` - Sistema responsive
3. ✅ `TESLA_EXACT_LAYOUT_V34.md` - Documentación

## 🎉 Estado Final

**Sistema completo de layout exacto implementado con valores precisos de Tesla para spacing, grid, responsive, y todos los aspectos de layout.**



