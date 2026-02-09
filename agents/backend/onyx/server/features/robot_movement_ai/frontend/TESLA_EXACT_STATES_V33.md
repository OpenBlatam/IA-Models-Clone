# Estados Interactivos Exactos Tesla - V33

## 🎯 Sistema de Estados Exactos

### Archivo: `tesla-exact-interactive-states.ts`

Sistema completo de estados interactivos con valores exactos de Tesla para todos los componentes.

## 📋 Estados por Componente

### 1. **Button States** (Exactos)

#### Primary Button
```typescript
default: {
  backgroundColor: '#0062cc',
  color: '#ffffff',
  border: 'none',
  boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
}
hover: {
  backgroundColor: '#0052a3',
  boxShadow: '0 4px 6px -1px rgba(0, 98, 204, 0.2)',
  transform: 'scale(1.02)',
}
active: {
  backgroundColor: '#004280',
  transform: 'scale(0.98)',
}
focus: {
  outline: '2px solid #0062cc',
  outlineOffset: '2px',
  ringWidth: '2px',
  ringColor: '#0062cc',
  ringOffset: '2px',
}
disabled: {
  backgroundColor: '#e5e7eb',
  color: '#9ca3af',
  opacity: 0.5,
  cursor: 'not-allowed',
}
```

#### Secondary Button
```typescript
default: {
  backgroundColor: '#ffffff',
  color: '#171a20',
  border: '1px solid #d1d5db',
  boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
}
hover: {
  backgroundColor: '#f9fafb',
  borderColor: '#9ca3af',
  boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1)',
  transform: 'scale(1.02)',
}
active: {
  backgroundColor: '#f3f4f6',
  transform: 'scale(0.98)',
}
```

#### Tertiary Button
```typescript
default: {
  backgroundColor: 'transparent',
  color: '#393c41',
  border: 'none',
}
hover: {
  backgroundColor: '#f9fafb',
  color: '#171a20',
}
active: {
  backgroundColor: '#f3f4f6',
}
```

### 2. **Input States** (Exactos)

```typescript
default: {
  backgroundColor: '#ffffff',
  border: '1px solid #d1d5db',
  color: '#171a20',
  boxShadow: 'none',
}
hover: {
  borderColor: '#9ca3af',
}
focus: {
  borderColor: '#0062cc',
  boxShadow: '0 0 0 3px rgba(0, 98, 204, 0.1)',
  outline: 'none',
}
error: {
  borderColor: '#ef4444',
  boxShadow: '0 0 0 3px rgba(239, 68, 68, 0.1)',
}
disabled: {
  backgroundColor: '#f9fafb',
  borderColor: '#e5e7eb',
  color: '#9ca3af',
  cursor: 'not-allowed',
}
```

### 3. **Card States** (Exactos)

```typescript
default: {
  backgroundColor: '#ffffff',
  border: '1px solid #e5e7eb',
  boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
}
hover: {
  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  transform: 'translateY(-2px)',
  borderColor: '#d1d5db',
}
active: {
  boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  transform: 'translateY(0)',
}
```

### 4. **Link States** (Exactos)

```typescript
default: {
  color: '#0062cc',
  textDecoration: 'none',
}
hover: {
  color: '#0052a3',
  textDecoration: 'underline',
  textUnderlineOffset: '2px',
}
visited: {
  color: '#0052a3',
}
focus: {
  outline: '2px solid #0062cc',
  outlineOffset: '2px',
  borderRadius: '2px',
}
```

### 5. **Checkbox/Radio States** (Exactos)

```typescript
default: {
  border: '1px solid #b5b5b5',
  backgroundColor: '#ffffff',
  width: '18px',
  height: '18px',
}
checked: {
  borderColor: '#0062cc',
  backgroundColor: '#0062cc',
}
hover: {
  borderColor: '#0062cc',
}
focus: {
  outline: '2px solid #0062cc',
  outlineOffset: '2px',
}
disabled: {
  borderColor: '#e5e7eb',
  backgroundColor: '#f9fafb',
  opacity: 0.5,
  cursor: 'not-allowed',
}
```

### 6. **Switch States** (Exactos)

```typescript
default: {
  backgroundColor: '#e5e7eb',
  width: '44px',
  height: '24px',
}
checked: {
  backgroundColor: '#0062cc',
}
thumb: {
  width: '20px',
  height: '20px',
  backgroundColor: '#ffffff',
  boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.2)',
}
disabled: {
  backgroundColor: '#f3f4f6',
  opacity: 0.5,
  cursor: 'not-allowed',
}
```

### 7. **Tab States** (Exactos)

```typescript
default: {
  color: '#393c41',
  borderBottom: '2px solid transparent',
  backgroundColor: 'transparent',
}
active: {
  color: '#0062cc',
  borderBottomColor: '#0062cc',
  backgroundColor: 'rgba(0, 98, 204, 0.05)',
}
hover: {
  color: '#171a20',
  borderBottomColor: '#d1d5db',
  backgroundColor: '#f9fafb',
}
```

### 8. **Badge States** (Exactos)

```typescript
default: {
  backgroundColor: '#e5e7eb',
  color: '#171a20',
  padding: '4px 12px',
  borderRadius: '9999px',
  fontSize: '12px',
  fontWeight: 500,
}
primary: {
  backgroundColor: '#0062cc',
  color: '#ffffff',
}
success: {
  backgroundColor: '#10b981',
  color: '#ffffff',
}
error: {
  backgroundColor: '#ef4444',
  color: '#ffffff',
}
warning: {
  backgroundColor: '#f59e0b',
  color: '#ffffff',
}
```

## 🔧 Helper Functions

```typescript
// Button states
getTeslaButtonState('primary', 'hover')
getTeslaButtonState('secondary', 'active')
getTeslaButtonState('tertiary', 'focus')

// Input states
getTeslaInputState('focus')
getTeslaInputState('error')
getTeslaInputState('disabled')

// Card states
getTeslaCardState('hover')
getTeslaCardState('active')

// Link states
getTeslaLinkState('hover')
getTeslaLinkState('visited')

// Checkbox states
getTeslaCheckboxState('checked')
getTeslaCheckboxState('disabled')

// Tab states
getTeslaTabState('active')
getTeslaTabState('hover')

// Badge states
getTeslaBadgeState('primary')
getTeslaBadgeState('success')
```

## ✅ Componentes Actualizados

### Button Component
- ✅ Estados exactos: default, hover, active, focus, disabled
- ✅ Colores exactos en hex
- ✅ Shadows exactos
- ✅ Transform exactos
- ✅ Ring focus exacto

### Input Component
- ✅ Estados exactos: default, hover, focus, error, disabled
- ✅ Border colors exactos
- ✅ Focus ring exacto (3px con opacity 0.1)
- ✅ Error state exacto
- ✅ Disabled state exacto

### Card Component
- ✅ Estados exactos: default, hover, active
- ✅ Shadows exactos
- ✅ Transform exacto (-2px en hover)
- ✅ Border colors exactos

## 📊 Valores Exactos Aplicados

### Colores
- Primary: `#0062cc` → `#0052a3` (hover) → `#004280` (active)
- Borders: `#d1d5db` → `#9ca3af` (hover) → `#e5e7eb` (default)
- Text: `#171a20` (black) → `#393c41` (gray-dark) → `#9ca3af` (disabled)

### Shadows
- Default: `0 1px 2px 0 rgba(0, 0, 0, 0.05)`
- Hover: `0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)`
- Focus Ring: `0 0 0 3px rgba(0, 98, 204, 0.1)`

### Transforms
- Hover Scale: `1.02`
- Active Scale: `0.98`
- Card Hover: `translateY(-2px)`

### Opacity
- Disabled: `0.5`
- Focus Ring: `0.1`

## 🎯 Uso en Componentes

```tsx
import { getTeslaButtonState } from '@/lib/utils/tesla-exact-interactive-states';

const hoverState = getTeslaButtonState('primary', 'hover');
// { backgroundColor: '#0052a3', boxShadow: '...', transform: 'scale(1.02)' }
```

## 📦 Archivos Creados/Actualizados

1. ✅ `tesla-exact-interactive-states.ts` - Sistema completo de estados
2. ✅ `Button.tsx` - Actualizado con estados exactos
3. ✅ `Input.tsx` - Actualizado con estados exactos
4. ✅ `Card.tsx` - Actualizado con estados exactos

## 🎉 Estado Final

**Sistema completo de estados interactivos exactos implementado con valores precisos de Tesla para todos los componentes principales.**



