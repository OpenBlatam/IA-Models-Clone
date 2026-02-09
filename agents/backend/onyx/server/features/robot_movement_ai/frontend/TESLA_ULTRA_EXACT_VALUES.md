# Tesla Design System - Valores Ultra Exactos - V24

## 🎯 Valores Exactos Implementados

### Opacidad Exacta
```typescript
0: '0'
5: '0.05'
10: '0.1'
20: '0.2'
30: '0.3'
40: '0.4'
50: '0.5'
60: '0.6'
70: '0.7'
80: '0.8'
90: '0.9'
95: '0.95'
100: '1'
```

**Uso en Tailwind:**
```tsx
className="opacity-10 opacity-20 opacity-90"
```

### Blur Exacto
```typescript
none: '0'
sm: '4px'
base: '8px'
md: '12px'
lg: '16px'
xl: '20px'
2xl: '24px'
3xl: '40px'
```

**Uso en Tailwind:**
```tsx
className="blur-tesla-sm blur-tesla-xl backdrop-blur-tesla-lg"
```

### Transform Scale Exacto
```typescript
95: '0.95'   // Pressed state
98: '0.98'   // Active state
100: '1.0'   // Normal
102: '1.02'  // Hover subtle
105: '1.05'  // Hover normal
110: '1.1'   // Hover prominent
```

**Uso en Tailwind:**
```tsx
className="scale-95 scale-98 scale-102 scale-105"
```

### Transform TranslateY Exacto
```typescript
-1: '-1px'   // Subtle hover
-2: '-2px'   // Normal hover
-4: '-4px'   // Prominent hover
-8: '-8px'   // Large hover
-12: '-12px' // Extra large hover
```

**Uso en Tailwind:**
```tsx
className="-translate-tesla-1 -translate-tesla-2 -translate-tesla-4"
```

### Transform TranslateX Exacto
```typescript
-1: '-1px'
-2: '-2px'
-4: '-4px'
-8: '-8px'
1: '1px'
2: '2px'
4: '4px'
8: '8px'
```

### Transform Rotate Exacto
```typescript
-180: '-180deg'
-90: '-90deg'
-45: '-45deg'
0: '0deg'
45: '45deg'
90: '90deg'
180: '180deg'
```

### Estados Hover Exactos

#### Opacidad Hover
```typescript
default: '0.9'   // Hover normal
light: '0.8'     // Hover ligero
heavy: '0.95'    // Hover pesado
```

#### Scale Hover
```typescript
subtle: '1.01'     // Hover muy sutil
normal: '1.02'     // Hover normal
prominent: '1.05'  // Hover prominente
```

#### TranslateY Hover
```typescript
subtle: '-1px'     // Hover muy sutil
normal: '-2px'     // Hover normal
prominent: '-4px'  // Hover prominente
```

#### Shadow Hover
```typescript
sm: '0 2px 4px 0 rgba(0, 0, 0, 0.1)'
md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
```

### Estados Focus Exactos

#### Ring Focus
```typescript
width: '2px'
offset: '2px'
color: '#0062cc'
```

#### Outline Focus
```typescript
width: '2px'
style: 'solid'
color: '#0062cc'
offset: '2px'
```

### Estados Active Exactos

#### Scale Active
```typescript
pressed: '0.98'  // Presionado normal
heavy: '0.95'    // Presionado fuerte
```

#### Opacity Active
```typescript
pressed: '0.8'   // Opacidad al presionar
```

### Grid Exacto

#### Columnas
```typescript
1, 2, 3, 4, 6, 12
```

#### Gaps
```typescript
xs: '0.5rem'   // 8px
sm: '0.75rem'  // 12px
md: '1rem'     // 16px
lg: '1.5rem'   // 24px
xl: '2rem'     // 32px
2xl: '3rem'    // 48px
```

**Uso en Tailwind:**
```tsx
className="grid grid-cols-12 gap-tesla-md gap-tesla-lg"
```

### Border Width Exacto
```typescript
0: '0'
1: '1px'
2: '2px'
4: '4px'
8: '8px'
```

**Uso en Tailwind:**
```tsx
className="border-tesla-1 border-tesla-2"
```

### Line Height Exacto
```typescript
none: '1'
tight: '1.25'
snug: '1.375'
normal: '1.5'
relaxed: '1.75'
loose: '2'
3: '0.75rem'   // 12px
4: '1rem'      // 16px
5: '1.25rem'   // 20px
6: '1.5rem'    // 24px
7: '1.75rem'   // 28px
8: '2rem'      // 32px
```

## 🎨 CSS Variables Exactas

### Colores
```css
--tesla-black: 23 26 32
--tesla-gray-dark: 57 60 65
--tesla-gray-medium: 92 94 98
--tesla-gray-light: 181 181 181
--tesla-gray-lighter: 229 231 235
--tesla-white: 255 255 255
--tesla-blue: 0 98 204
--tesla-blue-hover: 0 82 163
--tesla-blue-light: 230 242 255
```

### Opacidad
```css
--tesla-opacity-5: 0.05
--tesla-opacity-10: 0.1
--tesla-opacity-20: 0.2
--tesla-opacity-30: 0.3
--tesla-opacity-40: 0.4
--tesla-opacity-50: 0.5
--tesla-opacity-60: 0.6
--tesla-opacity-70: 0.7
--tesla-opacity-80: 0.8
--tesla-opacity-90: 0.9
--tesla-opacity-95: 0.95
```

### Blur
```css
--tesla-blur-sm: 4px
--tesla-blur-base: 8px
--tesla-blur-md: 12px
--tesla-blur-lg: 16px
--tesla-blur-xl: 20px
--tesla-blur-2xl: 24px
--tesla-blur-3xl: 40px
```

### Transform
```css
--tesla-scale-95: 0.95
--tesla-scale-98: 0.98
--tesla-scale-102: 1.02
--tesla-scale-105: 1.05
--tesla-scale-110: 1.1
```

### Transiciones
```css
--tesla-transition-fast: 150ms
--tesla-transition-base: 200ms
--tesla-transition-slow: 300ms
--tesla-transition-slower: 400ms
```

### Easing Curves
```css
--tesla-ease-in: cubic-bezier(0.4, 0, 1, 1)
--tesla-ease-out: cubic-bezier(0, 0, 0.2, 1)
--tesla-ease-in-out: cubic-bezier(0.4, 0, 0.2, 1)
--tesla-ease-spring: cubic-bezier(0.16, 1, 0.3, 1)
```

## 📐 Ejemplos de Uso Exacto

### Botón con Valores Exactos
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
  hover:opacity-90
  hover:scale-102
  hover:-translate-tesla-1
  hover:shadow-tesla-md
  active:scale-98
  focus:ring-2
  focus:ring-tesla-blue
  focus:ring-offset-2
">
  Click Me
</button>
```

### Card con Valores Exactos
```tsx
<div className="
  bg-white 
  border-tesla-1 
  border-gray-200 
  rounded-tesla-lg 
  p-tesla-lg 
  shadow-tesla-sm 
  transition-tesla-base 
  duration-tesla-base 
  ease-tesla-in-out
  hover:shadow-tesla-md
  hover:-translate-tesla-2
  hover:scale-102
  card-hover
">
  Content
</div>
```

### Input con Valores Exactos
```tsx
<input className="
  w-full 
  px-4 py-3 
  bg-white 
  border-tesla-1 
  border-gray-300 
  rounded-tesla-sm 
  text-tesla-black 
  placeholder-tesla-gray-light 
  focus:outline-none 
  focus:ring-2 
  focus:ring-tesla-blue 
  focus:border-transparent 
  transition-tesla-base 
  duration-tesla-base 
  ease-tesla-in-out
  min-h-touch
" />
```

### Grid con Valores Exactos
```tsx
<div className="
  grid 
  grid-cols-1 
  md:grid-cols-3 
  gap-tesla-md 
  lg:gap-tesla-lg
">
  {/* Items */}
</div>
```

## 🎯 Valores de Animación Exactos

### Fade In
```css
Duration: 300ms
Easing: cubic-bezier(0, 0, 0.2, 1)
Transform: translateY(-10px) → translateY(0)
```

### Slide In
```css
Duration: 300ms
Easing: cubic-bezier(0, 0, 0.2, 1)
From: translateY(100%) / translateX(100%)
To: translateY(0) / translateX(0)
```

### Scale In
```css
Duration: 200ms
Easing: cubic-bezier(0, 0, 0.2, 1)
From: scale(0.9)
To: scale(1)
```

### Pulse
```css
Duration: 2s
Easing: cubic-bezier(0.4, 0, 0.2, 1)
Opacity: 1 → 0.5 → 1
```

### Spin
```css
Duration: 1s
Easing: linear
Transform: rotate(0deg) → rotate(360deg)
```

## 🔧 Helpers Functions

```typescript
// Obtener color Tesla
getTeslaColor('blue') // '#0062cc'

// Obtener spacing Tesla
getTeslaSpacing('lg') // '1.5rem'

// Obtener shadow Tesla
getTeslaShadow('md') // '0 4px 6px -1px...'

// Obtener opacity Tesla
getTeslaOpacity('90') // '0.9'

// Obtener blur Tesla
getTeslaBlur('xl') // '20px'

// Obtener scale Tesla
getTeslaScale('102') // '1.02'

// Obtener translateY Tesla
getTeslaTranslateY('-2') // '-2px'
```

## ✅ Validación de Valores

- ✅ Todos los valores están documentados
- ✅ CSS variables para fácil mantenimiento
- ✅ Tailwind utilities para uso rápido
- ✅ TypeScript types para seguridad
- ✅ Helpers functions para acceso programático
- ✅ Ejemplos de uso incluidos
- ✅ Valores exactos de Tesla implementados

## 📊 Resumen de Valores

- **Colores**: 15+ valores exactos
- **Opacidad**: 12 niveles (0-100)
- **Blur**: 8 niveles (0-40px)
- **Scale**: 6 niveles (0.95-1.1)
- **Translate**: 9 valores (-12px a 8px)
- **Rotate**: 7 valores (-180° a 180°)
- **Hover States**: 4 tipos (opacity, scale, translateY, shadow)
- **Focus States**: 2 tipos (ring, outline)
- **Active States**: 2 tipos (scale, opacity)
- **Grid**: 6 columnas, 6 gaps
- **Border Width**: 5 valores (0-8px)
- **Line Height**: 12 valores (1-2, 12px-32px)

**Total: 100+ valores exactos implementados**



