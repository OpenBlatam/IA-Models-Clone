# Tesla Design System - Valores Exactos Implementados - V23

## 🎨 Colores Exactos de Tesla

```typescript
// Colores Primarios
black: '#171a20'           // RGB(23, 26, 32)
gray-dark: '#393c41'       // RGB(57, 60, 65)
gray-medium: '#5c5e62'     // RGB(92, 94, 98)
gray-light: '#b5b5b5'      // RGB(181, 181, 181)
gray-lighter: '#e5e7eb'    // RGB(229, 231, 235)
white: '#ffffff'           // RGB(255, 255, 255)
blue: '#0062cc'            // RGB(0, 98, 204) - Tesla Blue
blue-hover: '#0052a3'      // RGB(0, 82, 163)
blue-light: '#e6f2ff'      // RGB(230, 242, 255)
```

## 📏 Espaciado Exacto (Spacing Scale)

```typescript
xs: '0.5rem'    // 8px
sm: '0.75rem'   // 12px
md: '1rem'      // 16px
lg: '1.5rem'    // 24px
xl: '2rem'      // 32px
2xl: '3rem'     // 48px
3xl: '4rem'     // 64px
4xl: '6rem'     // 96px
5xl: '8rem'     // 128px
```

## ✍️ Tipografía Exacta

### Tamaños de Fuente
```typescript
xs: '0.75rem'      // 12px
sm: '0.875rem'     // 14px
base: '1rem'       // 16px
lg: '1.125rem'     // 18px
xl: '1.25rem'      // 20px
2xl: '1.5rem'      // 24px
3xl: '1.875rem'    // 30px
4xl: '2.25rem'     // 36px
5xl: '3rem'        // 48px
6xl: '3.75rem'     // 60px
7xl: '4.5rem'      // 72px
8xl: '6rem'        // 96px
```

### Pesos de Fuente
```typescript
light: 300
normal: 400
medium: 500
semibold: 600
bold: 700
```

### Line Height
```typescript
tight: 1.25
snug: 1.375
normal: 1.5
relaxed: 1.75
loose: 2
```

### Letter Spacing
```typescript
tighter: '-0.05em'
tight: '-0.025em'
normal: '0'
wide: '0.025em'
wider: '0.05em'
```

### Font Family
```typescript
sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif']
display: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif']
```

## 🔲 Border Radius Exacto

```typescript
none: '0'
xs: '2px'
sm: '4px'
md: '6px'
lg: '8px'
xl: '12px'
2xl: '16px'
3xl: '24px'
full: '9999px'
```

## 🌑 Sombras Exactas (Box Shadows)

```typescript
xs: '0 1px 2px 0 rgba(0, 0, 0, 0.05)'
sm: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
2xl: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)'
```

## ⏱️ Transiciones Exactas

### Duración
```typescript
fast: '150ms'
base: '200ms'
slow: '300ms'
slower: '400ms'
```

### Timing Functions (Easing Curves)
```typescript
ease-in: 'cubic-bezier(0.4, 0, 1, 1)'
ease-out: 'cubic-bezier(0, 0, 0.2, 1)'
ease-in-out: 'cubic-bezier(0.4, 0, 0.2, 1)'
spring: 'cubic-bezier(0.16, 1, 0.3, 1)'
```

## 📐 Z-Index Scale

```typescript
base: 0
dropdown: 1000
sticky: 1020
fixed: 1030
modal-backdrop: 1040
modal: 1050
popover: 1060
tooltip: 1070
```

## 📱 Breakpoints Exactos

```typescript
xs: '475px'
sm: '640px'
md: '768px'
lg: '1024px'
xl: '1280px'
2xl: '1536px'
3xl: '1920px'
```

## 👆 Touch Targets (WCAG 2.1 AA)

```typescript
minimum: '44px'      // Mínimo requerido
recommended: '48px'  // Recomendado
comfortable: '56px'  // Cómodo
```

## 🎯 Componentes con Valores Exactos

### Botones
- **Padding**: `12px 24px` (vertical horizontal)
- **Border Radius**: `4px`
- **Font Size**: `14px`
- **Font Weight**: `500`
- **Line Height**: `1.5rem`
- **Min Height**: `44px`
- **Transition**: `200ms cubic-bezier(0.4, 0, 0.2, 1)`

### Cards
- **Padding**: `24px`
- **Border Radius**: `8px`
- **Border**: `1px solid rgb(229, 231, 235)`
- **Shadow**: `0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)`
- **Transition**: `200ms cubic-bezier(0.4, 0, 0.2, 1)`

### Inputs
- **Padding**: `12px 16px` (vertical horizontal)
- **Border Radius**: `4px`
- **Border**: `1px solid rgb(209, 213, 219)`
- **Font Size**: `14px`
- **Min Height**: `44px`

## 📊 Uso en Tailwind

### Colores
```tsx
className="bg-tesla-black text-tesla-white border-tesla-blue"
```

### Espaciado
```tsx
className="p-tesla-lg m-tesla-xl gap-tesla-md"
```

### Tipografía
```tsx
className="text-tesla-2xl font-tesla-semibold leading-tesla-tight tracking-tesla-tight"
```

### Border Radius
```tsx
className="rounded-tesla-lg"
```

### Sombras
```tsx
className="shadow-tesla-md hover:shadow-tesla-lg"
```

### Transiciones
```tsx
className="transition-tesla-base duration-tesla-base ease-tesla-in-out"
```

## 🎨 Ejemplos de Uso

### Botón Primario Exacto
```tsx
<button className="
  bg-tesla-blue 
  text-white 
  px-6 py-3 
  rounded-tesla-sm 
  text-tesla-sm 
  font-medium 
  min-h-touch 
  transition-tesla-base 
  duration-tesla-base 
  ease-tesla-in-out
  hover:bg-tesla-blue-hover
  hover:shadow-tesla-md
  active:scale-[0.98]
">
  Click Me
</button>
```

### Card Exacta
```tsx
<div className="
  bg-white 
  border border-gray-200 
  rounded-tesla-lg 
  p-tesla-lg 
  shadow-tesla-sm 
  transition-tesla-base 
  duration-tesla-base 
  ease-tesla-in-out
  hover:shadow-tesla-md
  hover:-translate-y-0.5
">
  Content
</div>
```

## 📝 Notas Importantes

1. **Todos los valores están en el archivo**: `lib/utils/tesla-design-tokens.ts`
2. **Tailwind config extendido** con todos los valores exactos
3. **CSS global** actualizado con estilos exactos de Tesla
4. **TypeScript** con tipos seguros para todos los tokens
5. **Helpers functions** para acceso programático a los tokens

## ✅ Validación

- ✅ Colores exactos de Tesla
- ✅ Espaciado consistente
- ✅ Tipografía precisa
- ✅ Sombras exactas
- ✅ Transiciones con timing correcto
- ✅ Z-index scale apropiado
- ✅ Touch targets WCAG compliant
- ✅ Breakpoints responsive



