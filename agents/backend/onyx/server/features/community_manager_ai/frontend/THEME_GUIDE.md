# Guía de Sistema de Temas (Dark Mode)

## 🌓 Características

- **3 Modos de Tema**: Light, Dark, System (sigue preferencia del sistema)
- **Persistencia**: La preferencia se guarda en localStorage
- **Transiciones Suaves**: Cambios de tema con animaciones
- **Soporte Completo**: Todos los componentes soportan dark mode
- **Multiidioma**: Traducciones para todos los idiomas

## 🎨 Componentes

### ThemeToggle
Botón simple para alternar entre temas (Light → Dark → System → Light)

```typescript
import { ThemeToggle } from '@/components/ui/ThemeToggle';

<ThemeToggle />
```

### ThemeSelect
Selector completo con opciones visuales para elegir tema

```typescript
import { ThemeSelect } from '@/components/ui/ThemeToggle';

<ThemeSelect />
```

## 🎣 Hook useTheme

```typescript
import { useTheme } from '@/hooks/useTheme';

const { theme, setTheme, toggleTheme, effectiveTheme } = useTheme();

// theme: 'light' | 'dark' | 'system'
// effectiveTheme: 'light' | 'dark' (tema efectivo aplicado)
```

## 📦 Store Zustand

El tema se almacena en el store global:

```typescript
import { useAppStore } from '@/lib/store';

const theme = useAppStore((state) => state.theme);
const setTheme = useAppStore((state) => state.setTheme);
const toggleTheme = useAppStore((state) => state.toggleTheme);
```

## 🎨 Clases Tailwind para Dark Mode

Todos los componentes usan clases `dark:` para dark mode:

```tsx
<div className="bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100">
  Contenido
</div>
```

## 🔧 Configuración

### Tailwind Config
```typescript
darkMode: 'class' // Usa clase en lugar de media query
```

### CSS Global
Transiciones suaves en `app/globals.css`:
```css
body {
  transition: background-color 0.3s ease, color 0.3s ease;
}
```

## 📱 Componentes Actualizados

Todos los componentes UI soportan dark mode:
- Button
- Card
- Input
- Textarea
- Select
- Modal
- Alert
- Badge
- Dropdown
- SearchInput
- Skeleton
- EmptyState
- Header
- Sidebar
- Layout

## 🌐 Traducciones

Sección `theme` en todos los archivos de traducción:
- `theme.theme` - "Tema"
- `theme.light` - "Claro"
- `theme.dark` - "Oscuro"
- `theme.system` - "Sistema"
- `theme.toggleTheme` - "Cambiar tema"

## 🎯 Uso

### En Componentes
```typescript
'use client';

import { useTheme } from '@/hooks/useTheme';

const MyComponent = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="bg-white dark:bg-gray-800">
      <button onClick={toggleTheme}>
        Tema actual: {theme}
      </button>
    </div>
  );
};
```

### En Settings
El selector de tema está disponible en `/settings`

## 🔄 Flujo de Tema

1. Usuario selecciona tema (Light/Dark/System)
2. Se guarda en Zustand store (persistido)
3. `ThemeInitializer` aplica clase al `<html>`
4. Tailwind aplica estilos dark mode
5. Todos los componentes se actualizan automáticamente

## ✨ Características Técnicas

- **No Flash**: Tema aplicado antes de renderizar
- **Sincronización**: Cambios del sistema detectados automáticamente
- **Performance**: Solo actualiza clases CSS, sin re-renders innecesarios
- **Accesibilidad**: Mantiene contraste en ambos modos

## 🎨 Paleta de Colores Dark Mode

- Background: `gray-900` / `gray-800`
- Text: `gray-100` / `gray-200`
- Borders: `gray-700` / `gray-800`
- Cards: `gray-800`
- Inputs: `gray-800` con `gray-700` borders

## 📝 Notas

- El tema se aplica a nivel de `<html>` elemento
- Las transiciones son suaves (0.3s)
- El modo "system" detecta cambios automáticamente
- La preferencia persiste entre sesiones



