# Diseño Inspirado en Tesla Shop - V26

## 🛍️ Nuevos Componentes de Tienda

### 1. **ProductCard** - Tarjeta de Producto
Componente de tarjeta de producto estilo Tesla Shop con:
- ✅ Imagen con hover effect (scale)
- ✅ Badge opcional
- ✅ Botón de favoritos
- ✅ Quick view overlay
- ✅ Precio formateado
- ✅ Botón de añadir al carrito
- ✅ Animaciones con Framer Motion
- ✅ Diseño responsive

**Características**:
- Hover effect con scale y translateY
- Overlay gradient en hover
- Quick view button que aparece en hover
- Favoritos con animación de corazón
- Precio formateado con separadores de miles

### 2. **ProductGrid** - Grid de Productos
Grid responsive para mostrar productos:
- ✅ Columnas configurables (2, 3, 4)
- ✅ Animaciones escalonadas
- ✅ Responsive design
- ✅ Integración con ProductCard

**Uso**:
```tsx
<ProductGrid
  products={products}
  columns={4}
  onAddToCart={handleAddToCart}
  onView={handleView}
  onFavorite={handleFavorite}
  favorites={favorites}
/>
```

### 3. **HeroBanner** - Banner Hero
Banner hero estilo Tesla con:
- ✅ Imagen de fondo opcional
- ✅ Overlay gradient configurable
- ✅ Título grande con tipografía hero
- ✅ Subtítulo opcional
- ✅ Descripción opcional
- ✅ Botones de acción (primario y secundario)
- ✅ Scroll indicator animado
- ✅ Altura configurable (full height o fija)

**Características**:
- Animaciones de entrada escalonadas
- Scroll indicator con animación infinita
- Overlay gradient para legibilidad
- Botones con estilos Tesla

### 4. **CategoryFilter** - Filtro de Categorías
Filtro de categorías con 3 variantes:
- ✅ Variant `default`: Lista vertical
- ✅ Variant `pills`: Pills horizontales
- ✅ Variant `tabs`: Tabs horizontales
- ✅ Contador de items por categoría
- ✅ Animaciones en hover

**Variantes**:
- **default**: Lista vertical con hover effect
- **pills**: Pills redondeadas con contador
- **tabs**: Tabs con borde inferior

### 5. **PriceFilter** - Filtro de Precio
Filtro de rango de precios:
- ✅ Inputs de min y max
- ✅ Range slider dual
- ✅ Formato de precio
- ✅ Validación de rangos
- ✅ Estilo Tesla aplicado

**Características**:
- Dos inputs para min y max
- Range slider con dos handles
- Validación automática de rangos
- Formato de precio con separadores

## 🎨 Elementos de Diseño Tesla Shop

### Colores
- **Fondo**: Blanco puro (#ffffff)
- **Texto principal**: Negro Tesla (#171a20)
- **Texto secundario**: Gris oscuro (#393c41)
- **Acento**: Azul Tesla (#0062cc)
- **Bordes**: Gris claro (#e5e7eb)

### Tipografía
- **Títulos**: Inter, 600 weight, letter-spacing -0.02em
- **Texto**: Inter, 400 weight
- **Hero**: clamp(2.5rem, 8vw, 6rem), 700 weight

### Espaciado
- **Cards**: padding 24px (1.5rem)
- **Grid gap**: 24px (1.5rem)
- **Section padding**: 64px-128px (4rem-8rem)

### Sombras
- **Card default**: `0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)`
- **Card hover**: `0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)`

### Animaciones
- **Hover scale**: 1.02
- **Hover translateY**: -8px
- **Transition**: 300ms ease-in-out
- **Image scale**: 1.1 en hover

## 📐 Layout Patterns

### Product Grid
```tsx
<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
  {products.map(product => (
    <ProductCard key={product.id} {...product} />
  ))}
</div>
```

### Hero Section
```tsx
<HeroBanner
  title="Título Principal"
  subtitle="Subtítulo"
  description="Descripción del producto o servicio"
  primaryAction={{ label: 'Comprar', onClick: handleBuy }}
  secondaryAction={{ label: 'Ver Más', onClick: handleView }}
  backgroundImage="/hero-bg.jpg"
  fullHeight={true}
/>
```

### Filter Sidebar
```tsx
<div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
  <aside className="lg:col-span-1">
    <CategoryFilter categories={categories} />
    <PriceFilter onPriceChange={handlePriceChange} />
  </aside>
  <main className="lg:col-span-3">
    <ProductGrid products={filteredProducts} />
  </main>
</div>
```

## 🎯 Características Implementadas

### ProductCard
- ✅ Imagen con hover scale
- ✅ Badge de oferta/nuevo
- ✅ Botón de favoritos
- ✅ Quick view overlay
- ✅ Precio formateado
- ✅ Botón de añadir al carrito
- ✅ Animaciones suaves

### ProductGrid
- ✅ Grid responsive
- ✅ Columnas configurables
- ✅ Animaciones escalonadas
- ✅ Integración completa

### HeroBanner
- ✅ Imagen de fondo
- ✅ Overlay gradient
- ✅ Tipografía hero
- ✅ Botones de acción
- ✅ Scroll indicator
- ✅ Animaciones de entrada

### CategoryFilter
- ✅ 3 variantes de diseño
- ✅ Contador de items
- ✅ Animaciones
- ✅ Estado activo claro

### PriceFilter
- ✅ Inputs de min/max
- ✅ Range slider dual
- ✅ Validación
- ✅ Formato de precio

## 📊 Estadísticas

- **Componentes nuevos**: 5
- **Variantes de diseño**: 3 (CategoryFilter)
- **Animaciones**: 10+
- **Responsive breakpoints**: 3 (sm, md, lg)
- **Estados interactivos**: hover, active, focus

## 🚀 Próximos Pasos

1. ✅ Componentes de tienda creados
2. ⏳ Integrar con carrito de compras
3. ⏳ Añadir página de detalle de producto
4. ⏳ Implementar búsqueda de productos
5. ⏳ Añadir filtros adicionales (marca, rating, etc.)
6. ⏳ Crear checkout flow
7. ⏳ Añadir wishlist/favoritos persistente

## 💡 Ejemplos de Uso

### Página de Productos Completa
```tsx
<HeroBanner
  title="Explora Nuestros Productos"
  description="Encuentra todo lo que necesitas"
  backgroundImage="/hero-products.jpg"
/>

<div className="container-tesla mx-auto px-4 py-12">
  <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
    <aside>
      <CategoryFilter
        categories={categories}
        variant="default"
        onCategoryChange={setCategory}
      />
      <PriceFilter
        onPriceChange={(min, max) => setPriceRange({ min, max })}
      />
    </aside>
    <main className="lg:col-span-3">
      <ProductGrid
        products={filteredProducts}
        columns={3}
        onAddToCart={addToCart}
        favorites={favorites}
      />
    </main>
  </div>
</div>
```

## 🎨 Paleta de Colores Tesla Shop

```css
--tesla-white: #ffffff
--tesla-black: #171a20
--tesla-gray-dark: #393c41
--tesla-gray-light: #b5b5b5
--tesla-gray-lighter: #e5e7eb
--tesla-blue: #0062cc
--tesla-blue-hover: #0052a3
```

## ✨ Mejoras de UX

1. **Hover Effects**: Todos los elementos tienen hover effects suaves
2. **Animaciones**: Entrada escalonada para mejor percepción
3. **Feedback Visual**: Estados claros para todas las interacciones
4. **Responsive**: Diseño adaptativo para todos los dispositivos
5. **Accesibilidad**: ARIA labels y navegación por teclado



