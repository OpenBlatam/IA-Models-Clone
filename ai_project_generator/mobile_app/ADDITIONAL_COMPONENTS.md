# Componentes Adicionales - AI Project Generator Mobile

## Nuevos Componentes UI

### 1. ImageLoader
Componente para cargar imágenes con estados de carga y error.

**Características:**
- Estado de carga con spinner
- Manejo de errores con placeholder
- Soporte para callbacks onLoad y onError
- Tema dinámico integrado

**Uso:**
```tsx
<ImageLoader
  uri="https://example.com/image.jpg"
  style={styles.image}
  placeholder={<CustomPlaceholder />}
  onLoad={() => console.log('Loaded')}
/>
```

### 2. Badge
Componente de badge con múltiples variantes y tamaños.

**Características:**
- Variantes: default, primary, success, warning, error, info
- Tamaños: small, medium, large
- Tema dinámico
- Personalizable

**Uso:**
```tsx
<Badge label="Nuevo" variant="success" size="medium" />
<Badge label="Error" variant="error" size="small" />
```

### 3. Divider
Componente divisor horizontal o vertical.

**Características:**
- Orientación horizontal o vertical
- Espaciado personalizable
- Tema dinámico

**Uso:**
```tsx
<Divider orientation="horizontal" spacing={16} />
<Divider orientation="vertical" spacing={8} />
```

### 4. Chip
Componente chip interactivo con múltiples variantes.

**Características:**
- Variantes: default, outlined, filled
- Estado seleccionado
- Soporte para iconos
- Haptic feedback
- Tema dinámico

**Uso:**
```tsx
<Chip
  label="React"
  selected={isSelected}
  onPress={() => setSelected(!isSelected)}
  variant="outlined"
  icon={<Icon />}
/>
```

### 5. Tooltip
Componente tooltip con delay configurable.

**Características:**
- Posiciones: top, bottom, left, right
- Delay configurable
- Modal transparente
- Tema dinámico

**Uso:**
```tsx
<Tooltip message="Información útil" position="top" delay={500}>
  <TouchableOpacity>
    <Text>Presiona para ver tooltip</Text>
  </TouchableOpacity>
</Tooltip>
```

### 6. CopyButton
Botón para copiar texto al portapapeles.

**Características:**
- Feedback visual al copiar
- Toast notification
- Haptic feedback
- Estado de copiado
- Tema dinámico

**Uso:**
```tsx
<CopyButton
  text="Texto a copiar"
  label="Copiar"
  showLabel={true}
  onCopy={() => console.log('Copied')}
/>
```

## Nuevas Utilidades

### clipboard.ts
Utilidades para interactuar con el portapapeles.

**Funciones:**
- `copyToClipboard(text: string)`: Copia texto al portapapeles
- `getFromClipboard()`: Obtiene texto del portapapeles
- `hasClipboardContent()`: Verifica si hay contenido en el portapapeles

**Uso:**
```tsx
import { copyToClipboard, getFromClipboard } from '../utils/clipboard';

const success = await copyToClipboard('Texto');
const text = await getFromClipboard();
```

## Integración

Todos los componentes nuevos:
- ✅ Usan tema dinámico
- ✅ Son accesibles
- ✅ Tienen TypeScript completo
- ✅ Siguen las convenciones del proyecto
- ✅ Son reutilizables

## Ejemplos de Uso en Pantallas

### ProjectDetailScreen
```tsx
import { CopyButton } from '../components/CopyButton';
import { Badge } from '../components/Badge';
import { Divider } from '../components/Divider';

<View>
  <Badge label={project.status} variant="success" />
  <Divider />
  <CopyButton text={project.project_id} showLabel />
</View>
```

### ProjectsScreen
```tsx
import { Chip } from '../components/Chip';

<Chip
  label="Favoritos"
  selected={showFavoritesOnly}
  onPress={() => setShowFavoritesOnly(!showFavoritesOnly)}
  variant="outlined"
/>
```

## Dependencias Agregadas

- `expo-clipboard`: ~5.0.0

## Notas

- Todos los componentes están listos para usar
- No requieren configuración adicional
- Son completamente compatibles con el tema existente
- Siguen las mejores prácticas de React Native

