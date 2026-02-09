# Sistema de Animaciones y Efectos Visuales Tesla - V31

## 🎨 Nuevos Componentes de Animación

### 1. **ParallaxSection** (`ParallaxSection.tsx`)
Efecto parallax basado en scroll:

```tsx
<ParallaxSection speed={0.5} direction="up">
  <div>Contenido con efecto parallax</div>
</ParallaxSection>
```

**Características**:
- ✅ Velocidad configurable
- ✅ 4 direcciones: up, down, left, right
- ✅ Offset personalizable
- ✅ Usa Framer Motion scroll

**ParallaxImage**:
- ✅ Imagen con efecto parallax
- ✅ Opacidad y escala animadas
- ✅ Transición suave

### 2. **ScrollReveal** (`ScrollReveal.tsx`)
Revelación de contenido al hacer scroll:

```tsx
<ScrollReveal direction="up" delay={0.2} duration={0.6}>
  <div>Contenido que se revela</div>
</ScrollReveal>
```

**Características**:
- ✅ 6 direcciones: up, down, left, right, fade, scale
- ✅ Delay y duración configurables
- ✅ Threshold personalizable
- ✅ Modo "once" o repetible

**StaggerReveal**:
- ✅ Revelación escalonada de múltiples elementos
- ✅ Delay entre elementos configurable
- ✅ Perfecto para listas y grids

### 3. **TextReveal** (`TextReveal.tsx`)
Animación de texto palabra por palabra o carácter por carácter:

```tsx
<TextReveal splitBy="word" delay={0.1}>
  Texto animado
</TextReveal>
```

**Características**:
- ✅ 3 modos: word, char, line
- ✅ Delay y duración configurables
- ✅ Easing Tesla spring

**GradientText**:
- ✅ Texto con gradiente animado
- ✅ Colores personalizables
- ✅ Animación opcional

### 4. **HoverGlow** (`HoverGlow.tsx`)
Efecto de brillo que sigue el cursor:

```tsx
<HoverGlow glowColor="#0062cc" intensity={0.3}>
  <div>Contenido con glow</div>
</HoverGlow>
```

**Características**:
- ✅ Sigue el cursor
- ✅ Color e intensidad configurables
- ✅ Tamaño del glow personalizable

**MagneticButton**:
- ✅ Botón con efecto magnético
- ✅ Sigue el cursor suavemente
- ✅ Strength configurable
- ✅ Hover y tap animations

### 5. **PageTransition** (`PageTransition.tsx`)
Transiciones de página suaves:

```tsx
<PageTransition>
  <div>Contenido de página</div>
</PageTransition>
```

**Características**:
- ✅ Transición automática entre páginas
- ✅ Usa pathname para detectar cambios
- ✅ Easing Tesla spring
- ✅ AnimatePresence para salidas

**FadeTransition**:
- ✅ Fade in/out simple
- ✅ Control de visibilidad
- ✅ Duración configurable

**SlideTransition**:
- ✅ Slide en 4 direcciones
- ✅ Distancia personalizable
- ✅ Control de visibilidad

### 6. **ParticleBackground** (`ParticleBackground.tsx`)
Fondo con partículas animadas:

```tsx
<ParticleBackground count={50} color="#0062cc" speed={1} />
```

**Características**:
- ✅ Número de partículas configurable
- ✅ Color y tamaño personalizables
- ✅ Velocidad ajustable
- ✅ Movimiento suave

**FloatingElements**:
- ✅ Elementos flotantes
- ✅ Múltiples elementos
- ✅ Rotación y opacidad animadas

### 7. **ShimmerEffect** (`ShimmerEffect.tsx`)
Efecto shimmer sobre contenido:

```tsx
<ShimmerEffect duration={2} gradient="linear-gradient(...)">
  <div>Contenido con shimmer</div>
</ShimmerEffect>
```

**Características**:
- ✅ Gradiente personalizable
- ✅ Duración configurable
- ✅ Animación infinita

**ShimmerText**:
- ✅ Shimmer específico para texto
- ✅ Efecto de brillo deslizante
- ✅ Animación suave

**GlowEffect**:
- ✅ Efecto de brillo estático
- ✅ Color e intensidad configurables
- ✅ Tamaño del glow personalizable

## 🎯 Utilidades de Animación

### **tesla-animations.ts**
Sistema completo de utilidades de animación:

```typescript
// Duraciones exactas
duration: {
  instant: '0ms',
  fast: '150ms',
  base: '200ms',
  slow: '300ms',
  slower: '400ms',
  slowest: '600ms',
}

// Easing curves
easing: {
  linear: 'linear',
  easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
  easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
  easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
  spring: 'cubic-bezier(0.16, 1, 0.3, 1)',
  bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
}

// Spring configurations
spring: {
  gentle: { tension: 200, friction: 25 },
  normal: { tension: 280, friction: 60 },
  stiff: { tension: 400, friction: 80 },
  wobbly: { tension: 180, friction: 12 },
}

// Variants predefinidos
variants: {
  fadeIn, slideUp, slideDown, slideLeft, slideRight, scale, rotate
}

// Presets de animación
animationPresets: {
  pageTransition,
  cardEntrance,
  buttonHover,
  buttonTap,
  modalEntrance,
  toastEntrance,
}
```

## 🎨 Nuevas Animaciones CSS

### Animaciones de Efecto
- ✅ `gradient` - Gradiente animado
- ✅ `float` - Flotación suave
- ✅ `glow-pulse` - Pulso de brillo
- ✅ `shimmer-text` - Shimmer en texto
- ✅ `loading-dots` - Puntos de carga
- ✅ `wave` - Onda animada
- ✅ `morph` - Blob morfing
- ✅ `text-gradient` - Gradiente de texto animado

### Efectos de Hover
- ✅ `hover-lift` - Elevación al hover
- ✅ `hover-scale` - Escala al hover
- ✅ `hover-rotate` - Rotación al hover
- ✅ `hover-blur` - Desenfoque al hover
- ✅ `hover-brightness` - Brillo al hover
- ✅ `hover-contrast` - Contraste al hover
- ✅ `hover-saturate` - Saturación al hover
- ✅ `hover-backdrop-blur` - Backdrop blur al hover
- ✅ `hover-text-shadow` - Sombra de texto al hover

### Efectos Especiales
- ✅ `magnetic` - Efecto magnético
- ✅ `parallax-container` - Contenedor parallax
- ✅ `scroll-snap` - Scroll snap
- ✅ `reveal-on-scroll` - Revelación al scroll
- ✅ `border-glow` - Borde brillante
- ✅ `ripple` - Efecto ripple

## 📊 Estadísticas

- **Componentes nuevos**: 7
- **Utilidades de animación**: 1 archivo completo
- **Animaciones CSS**: 20+
- **Efectos de hover**: 10+
- **Presets de animación**: 6
- **Variants predefinidos**: 7
- **Spring configs**: 4

## 🚀 Uso

### Ejemplo Completo
```tsx
import ParallaxSection from '@/components/ui/ParallaxSection';
import ScrollReveal from '@/components/ui/ScrollReveal';
import TextReveal from '@/components/ui/TextReveal';
import HoverGlow from '@/components/ui/HoverGlow';
import { MagneticButton } from '@/components/ui/HoverGlow';

export default function Example() {
  return (
    <div>
      <ParallaxSection speed={0.5} direction="up">
        <ScrollReveal direction="fade" delay={0.2}>
          <HoverGlow glowColor="#0062cc">
            <TextReveal splitBy="word">
              <h1>Título Animado</h1>
            </TextReveal>
          </HoverGlow>
        </ScrollReveal>
      </ParallaxSection>
      
      <MagneticButton strength={0.3}>
        <button>Botón Magnético</button>
      </MagneticButton>
    </div>
  );
}
```

## ✨ Características Destacadas

1. **Performance Optimizado**
   - ✅ Usa `will-change` donde corresponde
   - ✅ Transform en lugar de position
   - ✅ GPU acceleration

2. **Accesibilidad**
   - ✅ Respeta `prefers-reduced-motion`
   - ✅ Animaciones opcionales
   - ✅ Transiciones suaves

3. **Personalización**
   - ✅ Todos los parámetros configurables
   - ✅ Variantes múltiples
   - ✅ Presets listos para usar

4. **Tesla Design System**
   - ✅ Valores exactos de Tesla
   - ✅ Easing curves de Tesla
   - ✅ Colores de Tesla
   - ✅ Timing exacto

## 🎯 Próximos Pasos

1. ✅ Sistema de animaciones completo
2. ✅ Efectos visuales avanzados
3. ✅ Microinteracciones
4. ⏳ Integrar en componentes existentes
5. ⏳ Crear ejemplos de uso
6. ⏳ Optimizar performance
7. ⏳ Añadir más variantes

## 📦 Archivos Creados

1. `ParallaxSection.tsx` - Efecto parallax
2. `ScrollReveal.tsx` - Revelación al scroll
3. `TextReveal.tsx` - Animación de texto
4. `HoverGlow.tsx` - Efecto glow al hover
5. `PageTransition.tsx` - Transiciones de página
6. `ParticleBackground.tsx` - Fondo de partículas
7. `ShimmerEffect.tsx` - Efecto shimmer
8. `tesla-animations.ts` - Utilidades de animación
9. Actualizaciones en `globals.css` - 20+ animaciones CSS

## 🎉 Estado Final

**Sistema completo de animaciones y efectos visuales Tesla implementado con 7 componentes nuevos, 20+ animaciones CSS, y utilidades completas de animación.**



