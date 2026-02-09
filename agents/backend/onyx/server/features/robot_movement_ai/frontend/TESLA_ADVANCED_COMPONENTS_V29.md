# Componentes Avanzados Inspirados en Tesla - V29

## 🎨 Nuevos Componentes de Interacción

### 1. **Drawer** - Panel Lateral
Panel lateral deslizable estilo Tesla con:
- ✅ 4 posiciones: left, right, top, bottom
- ✅ 5 tamaños: sm, md, lg, xl, full
- ✅ Overlay con backdrop blur
- ✅ Animaciones con spring physics
- ✅ Cierre con click fuera o Escape
- ✅ Bloqueo de scroll del body
- ✅ Header con título y botón de cierre

**Características**:
- Animaciones suaves con Framer Motion
- Spring physics para movimiento natural
- Overlay opcional
- Click outside para cerrar
- Escape key support
- Body scroll lock

### 2. **ConfirmDialog** - Diálogo de Confirmación
Diálogo de confirmación mejorado:
- ✅ 3 variantes: default, danger, warning
- ✅ Iconos contextuales
- ✅ Botones de acción
- ✅ Estado de loading
- ✅ Diseño centrado
- ✅ Estilo Tesla aplicado

**Variantes**:
- **default**: Confirmación normal
- **danger**: Confirmación peligrosa (rojo)
- **warning**: Advertencia (amarillo)

### 3. **Notification** - Notificación Individual
Notificación individual estilo Tesla:
- ✅ 4 tipos: success, error, warning, info
- ✅ Iconos automáticos
- ✅ Auto-dismiss configurable
- ✅ 6 posiciones
- ✅ Animaciones de entrada/salida
- ✅ Botón de cierre

**Tipos**:
- **success**: Verde con check
- **error**: Rojo con X
- **warning**: Amarillo con alerta
- **info**: Azul con info

### 4. **NotificationContainer** - Contenedor de Notificaciones
Contenedor para múltiples notificaciones:
- ✅ Stack de notificaciones
- ✅ Posición configurable
- ✅ Animaciones escalonadas
- ✅ Gestión de estado

### 5. **Tabs** - Pestañas Mejoradas
Componente de pestañas mejorado:
- ✅ Radix UI base
- ✅ Estilo Tesla aplicado
- ✅ Estados activos claros
- ✅ Focus states
- ✅ Animaciones suaves

### 6. **SplitView** - Vista Dividida
Vista dividida redimensionable:
- ✅ División configurable
- ✅ Resizable opcional
- ✅ Mínimos configurables
- ✅ Handle visual
- ✅ Feedback durante drag

**Características**:
- División inicial configurable
- Resize con drag
- Mínimos para cada lado
- Handle visual con icono
- Feedback visual durante drag

## 📐 Patrones de Uso

### Drawer
```tsx
<Drawer
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="Configuración"
  side="right"
  size="md"
>
  <Content />
</Drawer>
```

### ConfirmDialog
```tsx
<ConfirmDialog
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  onConfirm={handleConfirm}
  title="¿Estás seguro?"
  description="Esta acción no se puede deshacer"
  variant="danger"
  confirmLabel="Eliminar"
/>
```

### Notifications
```tsx
const [notifications, setNotifications] = useState<NotificationData[]>([]);

<NotificationContainer
  notifications={notifications}
  onClose={(id) => setNotifications(prev => prev.filter(n => n.id !== id))}
  position="top-right"
/>
```

### Tabs
```tsx
<Tabs defaultValue="tab1">
  <TabsList>
    <TabsTrigger value="tab1">Tab 1</TabsTrigger>
    <TabsTrigger value="tab2">Tab 2</TabsTrigger>
  </TabsList>
  <TabsContent value="tab1">Content 1</TabsContent>
  <TabsContent value="tab2">Content 2</TabsContent>
</Tabs>
```

### SplitView
```tsx
<SplitView
  left={<LeftContent />}
  right={<RightContent />}
  defaultSplit={50}
  minLeft={20}
  minRight={20}
  resizable={true}
/>
```

## 🎯 Características de Diseño

### Drawer
- **Posiciones**: 4 (left, right, top, bottom)
- **Tamaños**: 5 (sm, md, lg, xl, full)
- **Animaciones**: Spring physics
- **Overlay**: Backdrop blur opcional
- **Accesibilidad**: Escape key, click outside

### ConfirmDialog
- **Variantes**: 3 (default, danger, warning)
- **Iconos**: Contextuales por variante
- **Loading**: Estado de carga
- **Centrado**: Diseño centrado

### Notification
- **Tipos**: 4 (success, error, warning, info)
- **Posiciones**: 6
- **Auto-dismiss**: Configurable
- **Animaciones**: Entrada/salida suaves

### SplitView
- **Resizable**: Opcional
- **Mínimos**: Configurables
- **Handle**: Visual con feedback
- **Smooth**: Transiciones suaves

## 📊 Estadísticas

- **Componentes nuevos**: 6
- **Variantes de diseño**: 10+
- **Posiciones**: 6 (Notification)
- **Tamaños**: 5 (Drawer)
- **Tipos**: 4 (Notification)
- **Total de componentes UI**: 55+

## 🚀 Casos de Uso Completos

### Drawer para Configuración
```tsx
const [isSettingsOpen, setIsSettingsOpen] = useState(false);

<Button onClick={() => setIsSettingsOpen(true)}>
  Configuración
</Button>

<Drawer
  isOpen={isSettingsOpen}
  onClose={() => setIsSettingsOpen(false)}
  title="Configuración"
  side="right"
  size="lg"
>
  <SettingsForm />
</Drawer>
```

### Confirmación de Eliminación
```tsx
<ConfirmDialog
  isOpen={showConfirm}
  onClose={() => setShowConfirm(false)}
  onConfirm={handleDelete}
  title="Eliminar Item"
  description="¿Estás seguro de que deseas eliminar este item? Esta acción no se puede deshacer."
  variant="danger"
  confirmLabel="Eliminar"
  isLoading={isDeleting}
/>
```

### Sistema de Notificaciones
```tsx
const addNotification = (type: NotificationType, title: string, message?: string) => {
  const id = Date.now().toString();
  setNotifications(prev => [...prev, { id, type, title, message }]);
};

// Uso
addNotification('success', 'Guardado exitoso', 'Los cambios se han guardado correctamente');
addNotification('error', 'Error', 'No se pudo completar la operación');
```

### Vista Dividida para Editor
```tsx
<SplitView
  left={
    <CodeEditor code={code} onChange={setCode} />
  }
  right={
    <Preview code={code} />
  }
  defaultSplit={60}
  minLeft={30}
  minRight={30}
/>
```

## 🎨 Paleta de Colores para Notificaciones

```css
/* Success */
--success-bg: #d1fae5
--success-border: #10b981
--success-icon: #059669
--success-text: #065f46

/* Error */
--error-bg: #fee2e2
--error-border: #ef4444
--error-icon: #dc2626
--error-text: #991b1b

/* Warning */
--warning-bg: #fef3c7
--warning-border: #f59e0b
--warning-icon: #d97706
--warning-text: #92400e

/* Info */
--info-bg: #dbeafe
--info-border: #3b82f6
--info-icon: #0062cc
--info-text: #1e40af
```

## ✨ Mejoras de UX

1. **Drawer**:
   - Animaciones naturales con spring
   - Overlay con blur
   - Múltiples posiciones
   - Tamaños flexibles

2. **ConfirmDialog**:
   - Variantes contextuales
   - Iconos claros
   - Estados de loading
   - Diseño centrado

3. **Notifications**:
   - Auto-dismiss
   - Múltiples posiciones
   - Stack visual
   - Animaciones suaves

4. **SplitView**:
   - Resize intuitivo
   - Feedback visual
   - Mínimos configurables
   - Handle claro

## 📦 Resumen de Componentes

**Total de componentes UI**: 55+
- **Navegación**: Navigation, Breadcrumbs, Footer
- **Layout**: HeroBanner, CTASection, SplitView
- **Modales**: Dialog, Drawer, ConfirmDialog
- **Notificaciones**: Notification, NotificationContainer
- **Productos**: ProductCard, ProductGrid
- **Filtros**: CategoryFilter, PriceFilter
- **Estados**: LoadingSpinner, EmptyState, SkeletonLoader
- **Feedback**: StatusBadge, ProgressRing, StatCard
- **Formularios**: Input, Textarea, Select, Button, Tabs
- **Datos**: DataTable, VirtualizedList
- **Utilidades**: Badge, Avatar, Tooltip, etc.

## 🎯 Próximos Pasos

1. ✅ Drawer creado
2. ✅ ConfirmDialog mejorado
3. ✅ Sistema de notificaciones
4. ✅ Tabs mejorados
5. ✅ SplitView redimensionable
6. ⏳ Integrar en componentes existentes
7. ⏳ Añadir más variantes
8. ⏳ Crear ejemplos de uso completos



