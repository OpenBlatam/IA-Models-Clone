# Librerías y Dependencias

Este documento describe todas las librerías utilizadas en el proyecto y cómo usarlas.

## Dependencias Principales

### React y Next.js
- **react** (^18.2.0) - Biblioteca principal de React
- **react-dom** (^18.2.0) - React DOM renderer
- **next** (^14.0.0) - Framework React con App Router

### HTTP y Estado
- **axios** (^1.6.0) - Cliente HTTP para llamadas API
- **react-query** (^3.39.3) - Gestión de estado del servidor y caché

### Formularios
- **react-hook-form** (^7.48.2) - Gestión de formularios performante
- **zod** (^3.22.4) - Validación de esquemas TypeScript-first
- **@hookform/resolvers** (^3.3.2) - Resolvers para react-hook-form

### Estilos
- **clsx** (^2.0.0) - Utilidad para construir className condicionales
- **tailwind-merge** (^2.0.0) - Merge de clases Tailwind

### Animaciones
- **framer-motion** (^10.16.4) - Biblioteca de animaciones para React

### Iconos
- **lucide-react** (^0.294.0) - Iconos modernos y consistentes

### Gráficos
- **recharts** (^2.10.3) - Biblioteca de gráficos responsive

### Fechas
- **date-fns** (^2.30.0) - Utilidades modernas para manejo de fechas

### Notificaciones
- **react-hot-toast** (^2.4.1) - Notificaciones toast elegantes

### Observadores
- **react-intersection-observer** (^9.5.2) - Hook para Intersection Observer API

### Virtualización
- **react-virtual** (^2.10.4) - Virtualización de listas
- **react-window** (^1.8.10) - Ventanas virtuales para listas grandes

### Drag and Drop
- **react-beautiful-dnd** (^13.1.1) - Drag and drop accesible

### Select Avanzado
- **react-select** (^5.8.0) - Componente select mejorado

### Date Picker
- **react-datepicker** (^4.21.0) - Selector de fechas

### Markdown
- **react-markdown** (^9.0.1) - Renderizado de Markdown
- **remark-gfm** (^4.0.0) - Soporte para GitHub Flavored Markdown
- **react-syntax-highlighter** (^15.5.0) - Resaltado de sintaxis

### Utilidades
- **copy-to-clipboard** (^3.3.3) - Copiar al portapapeles
- **qrcode.react** (^3.1.0) - Generación de códigos QR
- **react-confetti** (^6.1.0) - Efectos de confeti
- **react-use** (^17.4.2) - Colección de hooks útiles
- **use-debounce** (^10.0.0) - Hook para debounce
- **use-local-storage-state** (^18.4.0) - Hook para localStorage

## Integraciones Creadas

### React Hook Form
Ubicación: `lib/integrations/react-hook-form.ts`

```typescript
import { createForm, getFieldError, isFieldValid } from '@/lib/integrations/react-hook-form';
import { z } from 'zod';

const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

const form = createForm(schema, { email: '', password: '' });
```

### Framer Motion
Ubicación: `lib/integrations/framer-motion.ts`

```typescript
import { MotionDiv, fadeIn, slideIn, defaultTransition } from '@/lib/integrations/framer-motion';

<MotionDiv
  initial="hidden"
  animate="visible"
  variants={fadeIn}
  transition={defaultTransition}
>
  Content
</MotionDiv>
```

### Recharts
Ubicación: `lib/integrations/recharts.ts`

```typescript
import { LineChartComponent } from '@/components/UI/Chart';

<LineChartComponent
  data={chartData}
  dataKey="value"
  strokeColor="#3b82f6"
/>
```

### Date-fns
Ubicación: `lib/integrations/date-fns.ts`

```typescript
import { formatDateShort, formatTimeAgo, dateUtils } from '@/lib/integrations/date-fns';

const short = formatDateShort(new Date());
const ago = formatTimeAgo(date);
```

### React Hot Toast
Ubicación: `lib/integrations/react-hot-toast.ts`

```typescript
import { showToast } from '@/lib/integrations/react-hot-toast';

showToast.success('Operation successful!');
showToast.error('Something went wrong');
showToast.promise(apiCall(), {
  loading: 'Loading...',
  success: 'Done!',
  error: 'Failed!',
});
```

## Componentes UI Creados

### Chart Components
- `LineChartComponent` - Gráfico de líneas
- `BarChartComponent` - Gráfico de barras
- `PieChartComponent` - Gráfico de pastel
- `AreaChartComponent` - Gráfico de área

### Markdown
- `Markdown` - Renderizado de Markdown con syntax highlighting

### QRCode
- `QRCode` - Generación de códigos QR

## Instalación

```bash
npm install
```

## Uso

Todas las librerías están configuradas y listas para usar. Los wrappers e integraciones están en `lib/integrations/` y los componentes UI en `components/UI/`.



